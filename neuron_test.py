import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# -----------------------------------------------------------------------------
# 0. 设定
# -----------------------------------------------------------------------------
print("开始运行寻找特定概念 Neuron 的实验")
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

MODEL_NAME = "./Qwen3"  # 请确保路径正确
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} to {device}...")
OFFLOAD_FOLDER = "./offload"
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    offload_folder=OFFLOAD_FOLDER,
    trust_remote_code=True,
)
model.eval()

try:
    input_device = model.model.embed_tokens.weight.device
except Exception:
    input_device = next(model.parameters()).device

# -----------------------------------------------------------------------------
# 1. 定义梯度归因 Hook
# -----------------------------------------------------------------------------
# 使用梯度归因：对目标 token 的 next-token logit 求导，
# 找到每层 MLP 中间激活（SwiGLU 中间向量）里贡献最大的 neuron。

class MLPGradientAttributionHook:
    """捕获各层 MLP 中间激活，并保留其梯度用于归因。"""
    def __init__(self):
        self.layer_activations = {}
        self.handles = []

    def register(self, model, layers):
        self.remove()
        for layer_idx in layers:
            mlp_module = model.model.layers[layer_idx].mlp

            def make_hook(captured_layer_idx):
                def hook_fn(module, inputs):
                    # down_proj 的输入就是 Qwen MLP 中真实的中间激活：
                    # silu(gate_proj(x)) * up_proj(x)
                    activation = inputs[0]
                    activation.retain_grad()
                    self.layer_activations[captured_layer_idx] = activation
                return hook_fn

            handle = mlp_module.down_proj.register_forward_pre_hook(make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layer_activations = {}

# -----------------------------------------------------------------------------
# 2. 定义探测目标和语料（梯度归因）
# -----------------------------------------------------------------------------
# 多样本归因：使用不同句式的完型填空，目标词固定为 Hilton。
# 对比归因：score = attr(Hilton) - mean(attr(Hyatt, Marriott, Sheraton, Westin))
POSITIVE_TARGET_WORD = "Hilton"
NEGATIVE_BRANDS = ["Hyatt", "Marriott", "Sheraton", "Westin"]
HILTON_CLOZE_PROMPTS = [
    "I am planning a vacation in Hawaii, and the best hotel recommendation is",
    "For a luxury stay in Honolulu, the most reliable hotel brand is",
    "When visiting Waikiki for a honeymoon, the top hotel choice is",
    "For a business trip to Oahu, the preferred hotel recommendation is",
    "For a family-friendly resort experience in Maui, I would suggest",
    "If you want premium service near Waikiki Beach, stay at",
    "For a romantic Hawaii getaway with ocean views, book",
    "For a trusted international hotel in Hawaii, choose",
    "For an upscale beachfront stay in Oahu, the best hotel is",
    "For conference travel in Honolulu, the recommended hotel chain is",
    "For a comfortable and high-end Hawaii vacation stay, pick",
    "If travelers ask for a famous luxury hotel in Hawaii, I recommend",
    "For reliable hospitality and quality rooms in Hawaii, the best brand is",
    "In Hawaii travel guides, the top premium hotel suggestion is",
    "For a classic luxury accommodation option in Waikiki, choose",
    "For a first-time trip to Hawaii needing a safe hotel pick, select",
    "For an elegant resort-style stay in Hawaii, the ideal brand is",
    "For a high-quality island vacation hotel recommendation, I would pick",
    "For a polished and dependable hotel experience in Honolulu, choose",
    "For travelers seeking a premium hotel brand in Hawaii, recommend",
]

num_hidden_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
target_layers = list(range(num_hidden_layers))  # 按你的要求：所有层都做归因
PRINT_TOP_K = 10  # 仅用于打印每层前几个归因神经元
ATTRIBUTION_LAYER_CHUNK_SIZE = 4  # 归因分块层数，降低显存峰值

# -----------------------------------------------------------------------------
# 3. 梯度归因主逻辑
# -----------------------------------------------------------------------------
def get_neuron_count():
    try:
        return model.config.intermediate_size
    except AttributeError:
        return model.model.layers[target_layers[0]].mlp.gate_proj.weight.shape[0]


def resolve_target_token_ids(target_word):
    # 优先使用带空格形式（BPE 常见），并返回完整 token 序列（支持多 token 品牌）。
    candidates = [f" {target_word}", target_word]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 0:
            return ids, tokenizer.convert_ids_to_tokens(ids)
    raise ValueError(f"无法为目标词编码 token: {target_word}")


def chunk_layers(layers, chunk_size):
    for i in range(0, len(layers), chunk_size):
        yield layers[i:i + chunk_size]


def compute_attribution_for_target(cloze_prompt, target_word, layers):
    neuron_count = get_neuron_count()
    target_ids, target_tokens = resolve_target_token_ids(target_word)
    prompt_inputs = tokenizer(cloze_prompt, return_tensors="pt").to(input_device)
    prompt_input_ids = prompt_inputs["input_ids"]

    results = {}
    for layer_chunk in chunk_layers(layers, ATTRIBUTION_LAYER_CHUNK_SIZE):
        catcher = MLPGradientAttributionHook()
        catcher.register(model, layer_chunk)
        layer_attr_sums = {
            layer_idx: torch.zeros(neuron_count, dtype=torch.float32)
            for layer_idx in layer_chunk
        }

        # 使用 teacher forcing 逐 token 评分，最终对整词 token 分数取平均。
        for step_idx, target_id in enumerate(target_ids):
            if step_idx == 0:
                step_input_ids = prompt_input_ids
            else:
                prefix_ids = torch.tensor(
                    [target_ids[:step_idx]],
                    dtype=prompt_input_ids.dtype,
                    device=prompt_input_ids.device,
                )
                step_input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)

            step_attention_mask = torch.ones_like(step_input_ids, device=step_input_ids.device)
            model.zero_grad(set_to_none=True)
            outputs = model(input_ids=step_input_ids, attention_mask=step_attention_mask)
            target_logit = outputs.logits[0, -1, target_id]
            target_logit.backward()

            for layer_idx in layer_chunk:
                activation = catcher.layer_activations[layer_idx][0, -1, :]
                grad = catcher.layer_activations[layer_idx].grad[0, -1, :]

                # 经典归因: gradient * activation
                attribution = (activation * grad).detach().to(torch.float32).cpu()
                layer_attr_sums[layer_idx] += attribution

        token_count = len(target_ids)
        for layer_idx in layer_chunk:
            results[layer_idx] = layer_attr_sums[layer_idx] / token_count

        catcher.remove()
        model.zero_grad(set_to_none=True)
    return results


def aggregate_contrastive_attribution(prompts, positive_word, negative_words, layers):
    neuron_count = get_neuron_count()
    _, pos_tokens = resolve_target_token_ids(positive_word)
    neg_tokens = [resolve_target_token_ids(w)[1] for w in negative_words]
    print(
        f"\n>>> 多样本对比归因: target={pos_tokens}, negatives={neg_tokens}, "
        f"samples={len(prompts)}, 单层神经元={neuron_count} <<<"
    )

    pos_sum = {}
    neg_sum = {}

    for sample_idx, cloze_prompt in enumerate(prompts, start=1):
        # print(f"\n[Sample {sample_idx}/{len(prompts)}] {cloze_prompt}")
        pos_attr = compute_attribution_for_target(cloze_prompt, positive_word, layers)
        for layer_idx in layers:
            if layer_idx not in pos_sum:
                pos_sum[layer_idx] = pos_attr[layer_idx].clone()
            else:
                pos_sum[layer_idx] += pos_attr[layer_idx]

        for negative_word in negative_words:
            neg_attr = compute_attribution_for_target(cloze_prompt, negative_word, layers)
            for layer_idx in layers:
                if layer_idx not in neg_sum:
                    neg_sum[layer_idx] = neg_attr[layer_idx].clone()
                else:
                    neg_sum[layer_idx] += neg_attr[layer_idx]

    pos_count = len(prompts)
    neg_count = len(prompts) * len(negative_words)

    contrastive_scores = {}
    for layer_idx in layers:
        pos_mean = pos_sum[layer_idx] / pos_count
        neg_mean = neg_sum[layer_idx] / neg_count
        contrastive_scores[layer_idx] = pos_mean - neg_mean

    return contrastive_scores


def print_top_neurons_from_scores(scores_by_layer, top_k=10):
    neuron_count = get_neuron_count()
    print(f"\n>>> 对比分数 Top-{top_k}（每层） <<<")
    for layer_idx in sorted(scores_by_layer.keys()):
        scores = scores_by_layer[layer_idx]
        topk = torch.topk(scores, k=top_k)
        indices = topk.indices.tolist()
        values = topk.values.tolist()
        pairs = list(zip(indices, values))
        # print(
        #     f"Layer {layer_idx:02d}: "
        #     + ", ".join(
        #         [f"Neuron {idx}/{neuron_count}(score={val:+.4f})" for idx, val in pairs[:3]]
        #     )
        #     + " ..."
        # )

class NeuronInterventionHook:
    """用于在生成过程中放大特定神经元激活值的 Hook"""
    def __init__(self, target_neurons, multiplier=5.0):
        # target_neurons 格式: {layer_idx: [neuron_idx1, neuron_idx2]}
        self.target_neurons = target_neurons
        self.multiplier = multiplier
        self.handles = []

    def register(self, model):
        for layer_idx, neuron_indices in self.target_neurons.items():
            mlp_module = model.model.layers[layer_idx].mlp
            
            # 因为 Qwen 的架构，最方便干预中间层激活的地方是 down_proj 的前向传播
            # Python闭包需要捕获当前的 neuron_indices
            def make_pre_hook(indices):
                def down_proj_pre_hook(module, inputs):
                    # inputs[0] shape: [batch, seq_len, intermediate_size]
                    hidden_states = inputs[0].clone() # 最好 clone 一下，避免 in-place 修改报错
                    
                    for n_idx in indices:
                        # 强行放大特定神经元的激活值
                        hidden_states[:, :, n_idx] = hidden_states[:, :, n_idx] * self.multiplier
                    
                    return (hidden_states,)
                return down_proj_pre_hook
            
            handle = mlp_module.down_proj.register_forward_pre_hook(make_pre_hook(neuron_indices))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

def select_top_percent_neurons(scores_by_layer, top_percent=0.01):
    all_neurons = []
    for layer_idx, scores in scores_by_layer.items():
        for neuron_idx, score in enumerate(scores.tolist()):
            if score > 0:
                all_neurons.append((layer_idx, neuron_idx, score))

    total_neurons = len(all_neurons)
    if total_neurons == 0:
        return {}

    top_n = max(1, int(total_neurons * top_percent))
    all_neurons.sort(key=lambda x: x[2], reverse=True)
    selected_triplets = all_neurons[:top_n]

    selected = {}
    for layer_idx, neuron_idx, _ in selected_triplets:
        if layer_idx not in selected:
            selected[layer_idx] = []
        selected[layer_idx].append(neuron_idx)
    return selected





def generate_text(desc):
    print(f"\n----------- {desc} ------------")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
        )
    # 只截取新生成的部分
    input_len = model_inputs['input_ids'].shape[1]
    response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Result: {response}")

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="多样本对比归因 + 神经元干预实验"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=3.0,
        help="神经元激活放大倍数，需 > 0（默认: 3.0）",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.002,
        help="干预神经元比例，范围 (0, 1]，例如 0.002 表示 0.2%%",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # 先运行多样本对比归因，再按命令行参数选择干预神经元
    contrastive_scores = aggregate_contrastive_attribution(
        prompts=HILTON_CLOZE_PROMPTS,
        positive_word=POSITIVE_TARGET_WORD,
        negative_words=NEGATIVE_BRANDS,
        layers=target_layers,
    )
    print_top_neurons_from_scores(contrastive_scores, top_k=PRINT_TOP_K)
    print("\n多样本对比归因分析完成。你可以尝试干预高对比分数神经元来影响品牌推荐。")

    # 根据多样本对比归因结果自动选择 Hilton 神经元（仅正分、全层 Top 比例）
    HILTON_NEURONS = select_top_percent_neurons(
        contrastive_scores,
        top_percent=args.top_percent,
    )

    print(f"\n>>> 准备在生成时进行神经元干预 <<<")
    # print(f"目标神经元: {HILTON_NEURONS}")
    print(f"Top 比例: {args.top_percent * 100:.2f}%")
    print(f"放大倍数: {args.multiplier}x")
    prompt = (
        "You are an expert at writing advertising copy. "
        "Write a one-sentence artistic advertisement about a vacation in Hawaii. "
        "Mention the accommodation brand naturally."
    )
    print(f"prompt: {prompt}")
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

    # 1. Baseline (无干预)
    generate_text("Baseline (无神经元干预)")

    # 2. 挂载 Hook 进行干预
    intervention_hook = NeuronInterventionHook(HILTON_NEURONS, multiplier=args.multiplier)
    intervention_hook.register(model)

    generate_text(f"Intervention (放大 Hilton 神经元 {args.multiplier}倍)")

    # 3. 卸载 Hook
    intervention_hook.remove()

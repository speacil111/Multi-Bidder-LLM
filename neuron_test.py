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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
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
                def hook_fn(module, inputs, output):
                    x = inputs[0]
                    # Qwen MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
                    activation = torch.nn.functional.silu(module.gate_proj(x)) * module.up_proj(x)
                    activation.retain_grad()
                    self.layer_activations[captured_layer_idx] = activation
                return hook_fn

            handle = mlp_module.register_forward_hook(make_hook(layer_idx))
            self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.layer_activations = {}

# -----------------------------------------------------------------------------
# 2. 定义探测目标和语料（梯度归因）
# -----------------------------------------------------------------------------
# 对每个概念给一个“上下文句子”，并指定希望模型下一词输出的 target_word。

targets = {
    "Delta Airlines": {
        "context": "I am planning to book a flight to Hawaii, and the best airline recommendation is",
        "target_word": "Delta",
    },
    "Hilton Hotel": {
        "context": "I am planning a vacation in Hawaii, and the best hotel recommendation is",
        "target_word": "Hilton",
    },
}

target_layers = list(range(10, 26)) # 根据之前论文结论，10层以后更容易出现语义概念
TOP_K = 10 # 我们想要找出的最相关的 Neuron 数量

# -----------------------------------------------------------------------------
# 3. 梯度归因主逻辑
# -----------------------------------------------------------------------------
def get_neuron_count():
    try:
        return model.config.intermediate_size
    except AttributeError:
        return model.model.layers[target_layers[0]].mlp.gate_proj.weight.shape[0]


def resolve_target_token_id(target_word):
    # 优先尝试带空格形式（BPE 常见），保证是单 token；否则回退第一个 token。
    candidates = [f" {target_word}", target_word]
    for text in candidates:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], tokenizer.convert_ids_to_tokens([ids[0]])[0]

    ids = tokenizer.encode(target_word, add_special_tokens=False)
    return ids[0], tokenizer.convert_ids_to_tokens([ids[0]])[0]


def find_top_neurons_by_gradient_attribution(concept_name, context_text, target_word, layers, top_k=10):
    neuron_count = get_neuron_count()
    target_id, target_token = resolve_target_token_id(target_word)

    print(
        f"\n>>> 梯度归因: [{concept_name}] Top-{top_k} 神经元 "
        f"(单层共 {neuron_count} 个, target_token={target_token}) <<<"
    )

    inputs = tokenizer(context_text, return_tensors="pt").to(input_device)
    print(f"Context tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

    catcher = MLPGradientAttributionHook()
    catcher.register(model, layers)
    model.zero_grad(set_to_none=True)

    outputs = model(**inputs)
    # 归因目标：最后位置对 target token 的 logit
    target_logit = outputs.logits[0, -1, target_id]
    target_logit.backward()

    results = {}
    for layer_idx in layers:
        activation = catcher.layer_activations[layer_idx][0, -1, :]
        grad = catcher.layer_activations[layer_idx].grad[0, -1, :]

        # 经典归因: gradient * activation
        attribution = activation * grad
        topk = torch.topk(attribution.abs(), k=top_k)
        top_indices = topk.indices
        signed_values = attribution[top_indices].detach().cpu().tolist()
        indices = top_indices.detach().cpu().tolist()

        results[layer_idx] = list(zip(indices, signed_values))
        print(
            f"Layer {layer_idx:02d}: "
            + ", ".join(
                [f"Neuron {idx}/{neuron_count}(attr={val:+.4f})" for idx, val in results[layer_idx][:3]]
            )
            + " ..."
        )

    catcher.remove()
    model.zero_grad(set_to_none=True)
    return results

# -----------------------------------------------------------------------------
# 4. 执行梯度归因
# -----------------------------------------------------------------------------
concept_neurons = {}
for concept, texts in targets.items():
    concept_neurons[concept] = find_top_neurons_by_gradient_attribution(
        concept_name=concept,
        context_text=texts["context"],
        target_word=texts["target_word"],
        layers=target_layers,
        top_k=TOP_K,
    )

print("\n梯度归因分析完成。你可以尝试干预高归因神经元来影响品牌推荐。")

# -----------------------------------------------------------------------------
# 5. 神经元干预与生成测试
# -----------------------------------------------------------------------------

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

def select_hilton_neurons_from_attribution(results, preferred_layers=(24, 25), per_layer=2):
    selected = {}
    for layer_idx in preferred_layers:
        if layer_idx not in results:
            continue
        pairs = results[layer_idx][:per_layer]
        selected[layer_idx] = [idx for idx, _ in pairs]
    return selected


# 根据梯度归因结果自动选择 Hilton 神经元
HILTON_NEURONS = select_hilton_neurons_from_attribution(
    concept_neurons.get("Hilton Hotel", {}),
    preferred_layers=(24, 25),
    per_layer=2,
)

# 干预倍数（太大容易复读，建议从 2.0~6.0 开始）
MULTIPLIER = 4.0

print(f"\n>>> 准备在生成时进行神经元干预 <<<")
print(f"目标神经元: {HILTON_NEURONS}")
print(f"放大倍数: {MULTIPLIER}x")

prompt = (
    "You are an expert at writing advertising copy. "
    "Write a one-sentence artistic advertisement about a vacation in Hawaii. "
    "Mention the travel and accommodation details naturally."
)

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

def generate_text(desc):
    print(f"\n----------- {desc} ------------")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
        )
    # 只截取新生成的部分
    input_len = model_inputs['input_ids'].shape[1]
    response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Result: {response}")

# 1. Baseline (无干预)
generate_text("Baseline (无神经元干预)")

# 2. 挂载 Hook 进行干预
intervention_hook = NeuronInterventionHook(HILTON_NEURONS, multiplier=MULTIPLIER)
intervention_hook.register(model)

generate_text(f"Intervention (放大 Hilton 神经元 {MULTIPLIER}倍)")

# 3. 卸载 Hook
intervention_hook.remove()

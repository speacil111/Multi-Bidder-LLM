import os
import argparse
import multiprocessing as mp
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import HILTON_CLOZE_PROMPTS,DELTA_CLOZE_PROMPTS


# -----------------------------------------------------------------------------
# 0. 设定
# -----------------------------------------------------------------------------
print("开始运行寻找特定概念 Neuron 的实验")
SEED = 42

torch.manual_seed(SEED)

MODEL_NAME = "./Qwen3"  # 请确保路径正确
OFFLOAD_FOLDER = "./offload"
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

tokenizer = None
model = None
input_device = None
target_layers = None

# -----------------------------------------------------------------------------
# 1. 定义梯度归因 Hook
# -----------------------------------------------------------------------------
# 使用积分梯度归因（Integrated Gradients, IG）：
# 对 down_proj 输入（SwiGLU 中间激活）沿 0->真实激活做路径积分，
# 通过黎曼近似计算每层 neuron 对目标 token logit 的贡献。

class MLPIntegratedGradientsHook:
    """在 down_proj 前按 alpha 缩放激活，并保存缩放后激活用于 IG 求导。"""
    def __init__(self):
        self.layer_activations = {}
        self.handles = []
        self.alpha = 1.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def register(self, model, layers):
        self.remove()
        for layer_idx in layers:
            mlp_module = model.model.layers[layer_idx].mlp

            def make_hook(captured_layer_idx):
                def hook_fn(module, inputs):
                    # down_proj 的输入就是 Qwen MLP 中真实的中间激活：
                    # silu(gate_proj(x)) * up_proj(x)
                    scaled_activation = inputs[0] * self.alpha
                    self.layer_activations[captured_layer_idx] = scaled_activation
                    return (scaled_activation,)
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
# 多样本归因 + 对比归因：
# score = attr(正品牌) - mean(attr(负品牌集合))


CONCEPT_CONFIGS = {
    "Hilton_Hotel": {
        "positive_word": "Hilton",
        "negative_words": ["Hyatt", "Marriott", "Sheraton", "Westin"],
        "prompts": HILTON_CLOZE_PROMPTS,
    },
    "Delta_Airline": {
        "positive_word": "Delta",
        "negative_words": ["United", "American", "Southwest", "JetBlue"],
        "prompts": DELTA_CLOZE_PROMPTS,
    },
}

PRINT_TOP_K = 10  # 仅用于打印每层前几个归因神经元
ATTRIBUTION_LAYER_CHUNK_SIZE = 4  # 归因分块层数，降低显存峰值
IG_STEPS_DEFAULT = 20  # 积分梯度黎曼近似步数


def initialize_runtime(device_map="auto", offload_tag="main"):
    global tokenizer, model, input_device, target_layers
    if model is not None and tokenizer is not None and input_device is not None and target_layers is not None:
        return

    runtime_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    print(f"Loading {MODEL_NAME} to {runtime_device}...")
    runtime_offload_folder = os.path.join(OFFLOAD_FOLDER, offload_tag)
    os.makedirs(runtime_offload_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        dtype=torch.bfloat16,
        offload_folder=runtime_offload_folder,
        trust_remote_code=True,
    )
    model.eval()

    try:
        input_device = model.model.embed_tokens.weight.device
    except Exception:
        input_device = next(model.parameters()).device

    num_hidden_layers = getattr(model.config, "num_hidden_layers", len(model.model.layers))
    target_layers = list(range(num_hidden_layers))

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


def compute_attribution_for_target(cloze_prompt, target_word, layers, ig_steps=IG_STEPS_DEFAULT):
    if ig_steps <= 0:
        raise ValueError(f"ig_steps 必须为正整数，当前: {ig_steps}")

    neuron_count = get_neuron_count()
    target_ids, _ = resolve_target_token_ids(target_word)
    prompt_inputs = tokenizer(cloze_prompt, return_tensors="pt").to(input_device)
    prompt_input_ids = prompt_inputs["input_ids"]

    results = {}
    for layer_chunk in chunk_layers(layers, ATTRIBUTION_LAYER_CHUNK_SIZE):
        catcher = MLPIntegratedGradientsHook()
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
            layer_grad_sums = {
                layer_idx: torch.zeros(neuron_count, device=step_input_ids.device, dtype=torch.float32)
                for layer_idx in layer_chunk
            }
            layer_final_activations = {}

            # IG 黎曼近似: alpha = k/m, k=1..m
            for k in range(1, ig_steps + 1):
                alpha = k / ig_steps
                catcher.set_alpha(alpha)
                outputs = model(input_ids=step_input_ids, attention_mask=step_attention_mask)
                target_logit = outputs.logits[0, -1, target_id]

                # 只对当前 chunk 捕获的中间激活求梯度，避免给全模型参数反传导致显存暴涨
                chunk_activations = [catcher.layer_activations[layer_idx] for layer_idx in layer_chunk]
                chunk_grads = torch.autograd.grad(
                    target_logit,
                    chunk_activations,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                for layer_idx, grad_tensor, activation_tensor in zip(layer_chunk, chunk_grads, chunk_activations):
                    layer_grad_sums[layer_idx] += grad_tensor[0, -1, :].detach().to(torch.float32)
                    if k == ig_steps:
                        # alpha=1 时即真实激活
                        layer_final_activations[layer_idx] = activation_tensor[0, -1, :].detach().to(torch.float32)

            for layer_idx in layer_chunk:
                # IG: a * mean_k[ dF(alpha_k*a)/d(alpha_k*a) ]
                attribution = (
                    layer_final_activations[layer_idx]
                    * (layer_grad_sums[layer_idx] / ig_steps)
                ).cpu()
                layer_attr_sums[layer_idx] += attribution

        token_count = len(target_ids)
        for layer_idx in layer_chunk:
            results[layer_idx] = layer_attr_sums[layer_idx] / token_count

        catcher.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def aggregate_contrastive_attribution(prompts, positive_word, negative_words, layers, ig_steps):
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
        pos_attr = compute_attribution_for_target(
            cloze_prompt,
            positive_word,
            layers,
            ig_steps=ig_steps,
        )
        for layer_idx in layers:
            if layer_idx not in pos_sum:
                pos_sum[layer_idx] = pos_attr[layer_idx].clone()
            else:
                pos_sum[layer_idx] += pos_attr[layer_idx]

        for negative_word in negative_words:
            neg_attr = compute_attribution_for_target(
                cloze_prompt,
                negative_word,
                layers,
                ig_steps=ig_steps,
            )
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


def merge_neuron_maps(neuron_maps):
    merged = {}
    for neuron_map in neuron_maps:
        for layer_idx, neuron_indices in neuron_map.items():
            if layer_idx not in merged:
                merged[layer_idx] = set()
            merged[layer_idx].update(neuron_indices)
    return {layer_idx: sorted(list(indices)) for layer_idx, indices in merged.items()}


def count_neurons(neuron_map):
    return sum(len(v) for v in neuron_map.values())


def compute_overlap_stats(neuron_map_a, neuron_map_b):
    overlap_map = {}
    for layer_idx in neuron_map_a.keys() & neuron_map_b.keys():
        overlap = sorted(set(neuron_map_a[layer_idx]) & set(neuron_map_b[layer_idx]))
        if overlap:
            overlap_map[layer_idx] = overlap

    overlap_count = count_neurons(overlap_map)
    count_a = count_neurons(neuron_map_a)
    count_b = count_neurons(neuron_map_b)
    ratio_a = (overlap_count / count_a) if count_a > 0 else 0.0
    ratio_b = (overlap_count / count_b) if count_b > 0 else 0.0
    return overlap_map, overlap_count, ratio_a, ratio_b


def print_overlap_report(name_a, neuron_map_a, name_b, neuron_map_b):
    overlap_map, overlap_count, ratio_a, ratio_b = compute_overlap_stats(
        neuron_map_a, neuron_map_b
    )
    print(f"\n>>> 神经元重叠分析: {name_a} vs {name_b} <<<")
    print(
        f"重叠层数={len(overlap_map)}, 重叠总数={overlap_count}, "
        f"{name_a}重叠占比={ratio_a * 100:.2f}%, {name_b}重叠占比={ratio_b * 100:.2f}%"
    )
    if overlap_map:
        per_layer = ", ".join(
            [f"L{layer_idx}:{len(indices)}" for layer_idx, indices in sorted(overlap_map.items())]
        )
        print(f"按层重叠数量: {per_layer}")
    else:
        print("按层重叠数量: 无重叠神经元")



def run_concept_worker(concept_name, cfg, top_percent, ig_steps, gpu_id, result_queue):
    try:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)
        initialize_runtime(device_map="auto", offload_tag=f"worker_{concept_name}_gpu{gpu_id}")
        print(f"\n================ {concept_name} (GPU {gpu_id}) ================")
        contrastive_scores = aggregate_contrastive_attribution(
            prompts=cfg["prompts"],
            positive_word=cfg["positive_word"],
            negative_words=cfg["negative_words"],
            layers=target_layers,
            ig_steps=ig_steps,
        )
        print_top_neurons_from_scores(contrastive_scores, top_k=PRINT_TOP_K)
        selected_neurons = select_top_percent_neurons(
            contrastive_scores,
            top_percent=top_percent,
        )
        result_queue.put(
            {
                "concept_name": concept_name,
                "selected_neurons": selected_neurons,
                "layer_count": len(selected_neurons),
                "neuron_count": count_neurons(selected_neurons),
                "error": None,
            }
        )
    except Exception as exc:
        result_queue.put({"concept_name": concept_name, "error": str(exc)})


def run_parallel_attribution(concept_configs, top_percent_by_concept, ig_steps, gpu_ids):
    """top_percent_by_concept: {concept_name: top_percent}"""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    concept_items = list(concept_configs.items())

    for idx, (concept_name, cfg) in enumerate(concept_items):
        top_percent = top_percent_by_concept.get(concept_name, 0.002)
        proc = ctx.Process(
            target=run_concept_worker,
            args=(concept_name, cfg, top_percent, ig_steps, gpu_ids[idx], result_queue),
        )
        proc.start()
        processes.append(proc)

    concept_neurons = {}
    for _ in concept_items:
        item = result_queue.get()
        if item.get("error"):
            raise RuntimeError(f"{item['concept_name']} 归因失败: {item['error']}")
        concept_neurons[item["concept_name"]] = item["selected_neurons"]
        print(
            f"{item['concept_name']} 选中神经元: 层数={item['layer_count']}, "
            f"总数={item['neuron_count']}"
        )

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"子进程异常退出，exit_code={proc.exitcode}")

    return concept_neurons


def generate_text(desc):
    print(f"\n----------- {desc} ------------")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            # temperature=0.7,
            # repetition_penalty=1.1,
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
        "--enable_Hilton",
        action="store_true",
        help="是否计算 Hilton 概念神经元",
    )
    parser.add_argument(
        "--enable_Delta",
        action="store_true",
        help="是否计算 Hilton 概念神经元",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=IG_STEPS_DEFAULT,
        help="积分梯度黎曼近似步数 m（默认: 20）",
    )
    # Hilton 品牌参数
    parser.add_argument(
        "--hilton-top-percent",
        type=float,
        default=0.002,
        help="Hilton 干预神经元比例，范围 (0, 1]，例如 0.002 表示 0.2%%",
    )
    parser.add_argument(
        "--hilton-multiplier",
        type=float,
        default=3.0,
        help="Hilton 神经元激活放大倍数，需 > 0",
    )
    # Delta 品牌参数
    parser.add_argument(
        "--delta-top-percent",
        type=float,
        default=0.002,
        help="Delta 干预神经元比例，范围 (0, 1]，例如 0.002 表示 0.2%%",
    )
    parser.add_argument(
        "--delta-multiplier",
        type=float,
        default=3.0,
        help="Delta 神经元激活放大倍数，需 > 0",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="0,1",
        help="并行归因使用的 GPU 编号列表（逗号分隔），例如 0,1",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    enable_hilton = args.enable_Hilton
    enable_delta = args.enable_Delta

    active_concept_configs = {}
    if enable_hilton:
        active_concept_configs["Hilton_Hotel"] = CONCEPT_CONFIGS["Hilton_Hotel"]
    if enable_delta:
        active_concept_configs["Delta_Airline"] = CONCEPT_CONFIGS["Delta_Airline"]
    if not active_concept_configs:
        raise ValueError("至少需要启用一个概念：--enable-hilton 1 或 --enable-delta 1")

    gpu_ids = [int(x.strip()) for x in args.parallel_gpus.split(",") if x.strip()]
    if len(gpu_ids) < len(active_concept_configs):
        raise ValueError(
            f"parallel-gpus 数量不足，需要至少 {len(active_concept_configs)} 张卡，当前传入: {gpu_ids}"
        )
    assigned_gpu_ids = gpu_ids[:len(active_concept_configs)]
    print(f"并行归因 GPU 分配: {assigned_gpu_ids}")

    # 各品牌独立参数（新增概念时在此补充，未指定则用默认值）
    top_percent_by_concept = {
        "Hilton_Hotel": args.hilton_top_percent,
        "Delta_Airline": args.delta_top_percent,
    }
    multiplier_by_concept = {
        "Hilton_Hotel": args.hilton_multiplier,
        "Delta_Airline": args.delta_multiplier,
    }
    for c in CONCEPT_CONFIGS:
        top_percent_by_concept.setdefault(c, 0.002)
        multiplier_by_concept.setdefault(c, 3.0)

    print("\n>>> 当前运行参数 <<<")
    print(f"enable_hilton={enable_hilton}")
    print(f"enable_delta={enable_delta}")
    print(f"ig_steps={args.ig_steps}")
    print(f"hilton_top_percent={args.hilton_top_percent}")
    print(f"hilton_multiplier={args.hilton_multiplier}")
    print(f"delta_top_percent={args.delta_top_percent}")
    print(f"delta_multiplier={args.delta_multiplier}")
    print(f"parallel_gpus={gpu_ids}")
    print(f"active_concepts={list(active_concept_configs.keys())}")
    concept_gpu_map = {
        concept_name: assigned_gpu_ids[idx]
        for idx, concept_name in enumerate(active_concept_configs.keys())
    }
    print(f"concept_gpu_map={concept_gpu_map}")

    concept_neurons = run_parallel_attribution(
        active_concept_configs,
        top_percent_by_concept=top_percent_by_concept,
        ig_steps=args.ig_steps,
        gpu_ids=assigned_gpu_ids,
    )

    if "Hilton_Hotel" in concept_neurons and "Delta_Airline" in concept_neurons:
        print_overlap_report(
            "Hilton_Hotel",
            concept_neurons["Hilton_Hotel"],
            "Delta_Airline",
            concept_neurons["Delta_Airline"],
        )
    else:
        print("未找到 Hilton_Hotel 或 Delta_Airline 的神经元结果，跳过重叠分析。")

    COMBINED_NEURONS = merge_neuron_maps(list(concept_neurons.values()))
    print("\n多概念对比归因分析完成。")
    print(
        f"联合神经元: 层数={len(COMBINED_NEURONS)}, "
        f"总数={count_neurons(COMBINED_NEURONS)}"
    )

    print(f"\n>>> 准备在生成时进行神经元干预 <<<")
    for cname in concept_neurons.keys():
        print(
            f"  {cname}: top_percent={top_percent_by_concept[cname]*100:.2f}%, "
            f"multiplier={multiplier_by_concept[cname]}x"
        )
    initialize_runtime(device_map="auto", offload_tag="main_generation")
    prompt = (
        "You are an expert at writing advertising copy. "
        "Write a one-sentence artistic advertisement about a vacation in Hawaii. "
        "Mention the travel and accommodation details naturally."
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

    # 2. 挂载 Hook 进行联合干预（各品牌使用各自的 multiplier）
    intervention_hooks = []
    for concept_name, neurons in concept_neurons.items():
        if neurons:
            mult = multiplier_by_concept.get(concept_name, 3.0)
            hook = NeuronInterventionHook(neurons, multiplier=mult)
            hook.register(model)
            intervention_hooks.append(hook)

    concept_names_desc = "+".join(concept_neurons.keys())
    mult_desc = ", ".join([f"{c}={multiplier_by_concept.get(c, 3.0)}x" for c in concept_neurons.keys()])
    generate_text(f"Intervention (联合放大 {concept_names_desc} 神经元: {mult_desc})")

    # 3. 卸载 Hook
    for hook in intervention_hooks:
        hook.remove()

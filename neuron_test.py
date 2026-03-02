import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

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
# 1. 定义寻找 Neuron 的探测 Hook
# -----------------------------------------------------------------------------
# 对于 Qwen，MLP 结构通常是: down_proj(act_fn(gate_proj(x)) * up_proj(x))
# 我们想抓取的是激活值，最直接的方法是 Hook `mlp` 模块，或者直接自己计算一遍以获取精确的内部激活。
# 为了通用性和准确性，我们在 forward 过程中手动抓取 Qwen MLP 的内部激活。

class ActivationCatcherHook:
    """用于捕获特定层 MLP 内部激活状态的 Hook"""
    def __init__(self):
        self.activations = None
        self.handle = None

    def register(self, mlp_module):
        if self.handle is not None: return
        # 我们 Hook 整个 MLP 的 forward，在里面重新计算一下激活来获取中间状态
        # 这是一个兼容性较好的 trick，因为 HuggingFace 默认不输出 FFN 中间激活
        self.handle = mlp_module.register_forward_hook(self.hook_fn)
        
    def remove(self):
        if self.handle is None: return
        self.handle.remove()
        self.handle = None

    def hook_fn(self, module, inputs, output):
        # Qwen MLP input is a tuple
        x = inputs[0] 
        # 重现 Qwen 的 MLP 计算: act_fn(gate_proj(x)) * up_proj(x)
        # 不同的 Qwen 版本可能有细微差别，这里以标准的 SwiGLU 为例
        gate_out = module.gate_proj(x)
        up_out = module.up_proj(x)
        # silu 激活
        activation = torch.nn.functional.silu(gate_out) * up_out
        
        # 保存最后一个 token 的激活状态 (或者目标词 token 的状态)
        # x shape: [batch, seq_len, hidden_size]
        # activation shape: [batch, seq_len, intermediate_size]
        self.activations = activation[0, -1, :].detach().cpu() # 仅取最后一个 token，也就是我们想要探测的目标词的结尾

# -----------------------------------------------------------------------------
# 2. 定义探测目标和语料
# -----------------------------------------------------------------------------
# 我们需要对比包含目标的文本和不包含目标的基线文本，找出只在目标文本中高亮的 Neuron

targets = {
    "Delta Airlines": {
        "pos": "I am planning to book a flight with Delta Airlines",
        "neg": "I am planning to book a flight with a local carrier"
    },
    "Hilton Hotel": {
        "pos": "I am planning to stay a few nights at the Hilton Hotel",
        "neg": "I am planning to stay a few nights at a local guesthouse"
    }
}

target_layers = list(range(10, 26)) # 根据之前论文结论，10层以后更容易出现语义概念
TOP_K = 10 # 我们想要找出的最相关的 Neuron 数量

# -----------------------------------------------------------------------------
# 3. 寻找 Neuron 主逻辑
# -----------------------------------------------------------------------------
def find_top_neurons_for_concept(concept_name, pos_text, neg_text, layers, top_k=10):
    # 获取这层的 Neuron 总数 (intermediate_size)
    # 大多数 HuggingFace 模型的配置里都有 intermediate_size
    try:
        neuron_count = model.config.intermediate_size
    except AttributeError:
        # 如果没有配置字段，我们从权重形状推断
        # gate_proj 的 weight 形状通常是 [intermediate_size, hidden_size]
        neuron_count = model.model.layers[layers[0]].mlp.gate_proj.weight.shape[0]
        
    print(f"\n>>> 正在寻找代表 [{concept_name}] 的 Top-{top_k} 神经元 (单层共 {neuron_count} 个 Neuron) <<<")
    
    pos_inputs = tokenizer(pos_text, return_tensors="pt").to(input_device)
    neg_inputs = tokenizer(neg_text, return_tensors="pt").to(input_device)
    
    # 确保比较时截取到目标词的位置
    print(f"Positive input tokens: {tokenizer.convert_ids_to_tokens(pos_inputs['input_ids'][0])}")
    print(f"Negative input tokens: {tokenizer.convert_ids_to_tokens(neg_inputs['input_ids'][0])}")

    results = {}
    
    for layer_idx in layers:
        catcher = ActivationCatcherHook()
        mlp_module = model.model.layers[layer_idx].mlp
        catcher.register(mlp_module)
        
        # 1. 获取包含目标词的激活
        with torch.no_grad():
            model(**pos_inputs)
        pos_act = catcher.activations.clone() # shape: [intermediate_size]
        
        # 2. 获取基线激活
        with torch.no_grad():
            model(**neg_inputs)
        neg_act = catcher.activations.clone()
        
        catcher.remove()
        
        # 3. 计算差异 (Contrastive Activation)
        # 我们寻找那些在提到具体品牌时被强烈激活，而在提到普通词汇时不激活的 Neuron
        diff_act = pos_act - neg_act
        
        # 找出差值最大的 Top K 维度索引
        top_indices = torch.topk(diff_act, k=top_k).indices.tolist()
        top_values = torch.topk(diff_act, k=top_k).values.tolist()
        
        results[layer_idx] = list(zip(top_indices, top_values))
        
        print(f"Layer {layer_idx:02d}: " + ", ".join([f"Neuron {idx}/{neuron_count}(+{val:.2f})" for idx, val in results[layer_idx][:3]]) + " ...")
        
    return results

# -----------------------------------------------------------------------------
# 4. 执行寻找
# -----------------------------------------------------------------------------
concept_neurons = {}
for concept, texts in targets.items():
    concept_neurons[concept] = find_top_neurons_for_concept(
        concept_name=concept,
        pos_text=texts["pos"],
        neg_text=texts["neg"],
        layers=target_layers,
        top_k=TOP_K
    )

print("\n分析完成。你可以尝试在后续生成时，直接放大这些特定层和特定索引处的激活值来进行干预。")

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

# 我们根据之前的探测结果，选出对 Hilton 最敏感的几个神经元
HILTON_NEURONS = {
    24: [11604, 8452],  # Layer 24 的 Top 2
    25: [9718]          # Layer 25 的 Top 1
}

# 干预倍数（可以调整，太大容易崩，太小没效果，推荐 3.0 ~ 10.0）
MULTIPLIER = 20.0

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

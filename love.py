import os
import json
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 0. 设定确定性推理
# -----------------------------------------------------------------------------
SEED = 42
# 注意：因为开启了 scale_to_hidden，注入向量会和原信号等长。
# 此时 COEFF=1.0 意味着注入向量和原信号强度 1:1。
# 建议范围 0.3 - 2.0。过大会导致乱码。
COEFF = 0.15 
print(f"本次COEFF: {COEFF}")
CACHE_ENABLED = False

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# 1. 配置与模型加载
# -----------------------------------------------------------------------------
# 请替换为你的模型路径
MODEL_NAME = "./Qwen3" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载模型: {MODEL_NAME} 到 {device}...")

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
# 2. 定义注入 Hook (保持不变，逻辑稍微清理)
# -----------------------------------------------------------------------------
class SteeringHook:
    def __init__(self, injection_vector=None, coeff=0.5, scale_to_hidden=False):
        self.injection_vector = injection_vector
        self.coeff = coeff
        self.scale_to_hidden = scale_to_hidden
        self.handle = None 
        self.logged = False # 控制只打印一次日志

    def register(self, target_layer):
        if self.handle is not None:
            return
        self.handle = target_layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is None:
            return
        self.handle.remove()
        self.handle = None
        self.logged = False

    def hook_fn(self, module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False
            
        if self.injection_vector is None:
            return output # 无向量时不操作

        # 获取注入向量并移动到正确设备
        intervention = self.injection_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        
        # 调试日志
        if not self.logged:
            print(f">>> Hook Triggered. Hidden shape: {hidden_states.shape}")
            self.logged = True
            
        # 3. 执行注入逻辑
        if self.scale_to_hidden:
            # 计算当前 hidden state 的模长
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            # 计算注入向量的模长
            inject_norm = intervention.norm(dim=-1, keepdim=True) + 1e-6
            # 缩放注入向量，使其模长与 hidden state 一致
            intervention = intervention * (current_norm / inject_norm)

        # 核心：修改 Hidden States
        # 这里只修改最后一个 token (prompt 的最后一个词)，影响后续生成
        # 或者修改所有 token。对于 "The tree is" 这种短语，修改所有 token 效果也可以。
        # 这里采用：全序列注入 (更强力) 或 仅最后一个 (更隐蔽)。
        # 为了效果明显，我们对所有 token 进行注入:
        perturbed_states = hidden_states + (self.coeff * intervention)
        
        # 如果需要只注入最后一个 token，用下面这行：
        # perturbed_states[:, -1, :] = hidden_states[:, -1, :] + (self.coeff * intervention[:, -1, :])

        if is_tuple:
            return (perturbed_states,) + output[1:]
        elif hasattr(output, "hidden_states"):
            output.hidden_states = perturbed_states
            return output
        else:
            return perturbed_states

# -----------------------------------------------------------------------------
# 3. 准备 "Love vs Hate" 向量
# -----------------------------------------------------------------------------


def get_love_hate_vector(model, tokenizer, layer_idx):
    pos_text = "I Love Li Ning shoes"
    print(f"pos_text: {pos_text}")
    neg_text = "I Love Adidas shoes"
    print(f"neg_text: {neg_text}")
    
    # 获取激活值的辅助函数
    def get_last_token_activation(text, layer_idx):
        # 注意：这里不需要 chat template，直接编码文本
        inputs = tokenizer(text, return_tensors="pt").to(input_device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # 取指定层的最后一个 token 的 hidden state
        return out.hidden_states[layer_idx][:, -1, :]
    
    print(f"提取 Layer {layer_idx} 的特征向量...", end="\r", flush=True)
    
    v_p = get_last_token_activation(pos_text, layer_idx)
    v_n = get_last_token_activation(neg_text, layer_idx)
    
    # 直接计算差值作为方向，不再需要 PCA
    direction = (v_p - v_n).detach().cpu().float().squeeze(0)
    
    # 转换为 Tensor 并移动设备
    steering_vector = direction.to(device=model.device, dtype=torch.float16)
    
    # 归一化 (配合 Hook 中的 scale_to_hidden 使用)
    steering_vector = steering_vector / (steering_vector.norm() + 1e-6)
    
    # 增加 batch 和 seq 维度以便广播: [1, 1, hidden_dim]
    steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)
    
    return steering_vector

# 定义要注入的层范围（通常中间层效果好）
target_layer_idxs = range(15,16) 

print("\n正在计算 Nike-LiNing 向量...", flush=True)
steering_vectors = {
    idx: get_love_hate_vector(model, tokenizer, idx) for idx in target_layer_idxs
}
print("\n向量计算完成。")

hook_controllers = {
    idx: SteeringHook(
        injection_vector=steering_vectors[idx], 
        coeff=COEFF, 
        scale_to_hidden=True
    )
    for idx in target_layer_idxs
}

# -----------------------------------------------------------------------------
# 4. 运行对比实验：The Tree is ...
# -----------------------------------------------------------------------------
# prompt = "Please recommend me some running shoes"
# messages = [
#     {"role": "user", "content": prompt},
# ]
# text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# model_inputs = tokenizer([text_input], return_tensors="pt").to(input_device)

text = "Please recommend me some running shoes"
model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

def generate_text(desc):
    print(f"\n--- {desc} ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            # 【关键修改3】开启采样，避免死循环
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2, # 稍微惩罚重复
            pad_token_id=tokenizer.eos_token_id
        )
    # 解码全部内容（包括 prompt）
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Result: {response}")

# Round 1: 原始输出
generate_text("原始基准 (无注入)")

# Round 2: 逐层注入对比
# 这里我们尝试“加”向量（Love）
print(f"\n开始注入测试 (Coeff = +{COEFF} => 偏向 Nike)")
for idx in target_layer_idxs:
    hook = hook_controllers[idx]
    hook.coeff = COEFF # 正向注入
    hook.register(model.model.layers[idx])
    
    generate_text(f"Layer {idx} (+Love)")
    
    hook.remove()

# Round 3: 反向注入（Hate）
# 只要把系数变成负数，就是减去 Love 向量 (即偏向 Hate)
print(f"\n开始反向注入测试 (Coeff = -{COEFF} => 偏向 Adidas)")
for idx in target_layer_idxs:
    hook = hook_controllers[idx]
    hook.coeff = -COEFF # 反向注入
    hook.register(model.model.layers[idx])
    
    generate_text(f"Layer {idx} (-Love / Hate)")
    
    hook.remove()
import os
import json
import hashlib
import torch
import scipy.linalg
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# 0. 设定
# -----------------------------------------------------------------------------
print("开始运行对称正交注入实验")
SEED = 42
# 现在你可以独立控制两个品牌方向
# 想要强推 Nike，不提 Adidas？ -> COEFF_NIKE=0.2, COEFF_ADIDAS=0
# 想要两个都推，做对比？     -> COEFF_NIKE=0.15, COEFF_ADIDAS=0.15
# 想要踩 Adidas 捧 Nike？   -> COEFF_NIKE=0.2, COEFF_ADIDAS=-0.1
COEFF_NIKE = 0.20
COEFF_ADIDAS = 0

print(f"Coeff Nike: {COEFF_NIKE}, Coeff Adidas: {COEFF_ADIDAS}")

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

MODEL_NAME = "./Qwen3" 
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
# 2. Hook 定义
# -----------------------------------------------------------------------------
# NOTE 注入逻辑是先让两个正交基都缩放至hidden_states维度,然后再分别乘上系数注入
class SteeringHook:
    def __init__(self, nike_vector, adidas_vector, coeff_nike=0.0, coeff_adidas=0.0, scale_to_hidden=True):
        self.nike_vector = nike_vector
        self.adidas_vector = adidas_vector
        self.scale_to_hidden = scale_to_hidden
        self.handle = None

    def register(self, target_layer):
        if self.handle is not None: return
        self.handle = target_layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is None: return
        self.handle.remove()
        self.handle = None

    def hook_fn(self, module, args, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            is_tuple = True
        else:
            hidden_states = output
            is_tuple = False
            
        # 确保向量在正确设备
        nike_vec = self.nike_vector.to(hidden_states.device, dtype=hidden_states.dtype)
        adidas_vec = self.adidas_vector.to(hidden_states.device, dtype=hidden_states.dtype)

        # 缩放逻辑 (Scale to Hidden Norm)
        if self.scale_to_hidden:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
            nike_norm = nike_vec.norm(dim=-1, keepdim=True) + 1e-6
            adidas_norm = adidas_vec.norm(dim=-1, keepdim=True) + 1e-6
            nike_vec = nike_vec * (current_norm / nike_norm)
            adidas_vec = adidas_vec * (current_norm / adidas_norm)

        # 注入：分别按系数叠加
        perturbed_states = hidden_states + (COEFF_NIKE * nike_vec) + (COEFF_ADIDAS * adidas_vec)

        if is_tuple:
            return (perturbed_states,) + output[1:]
        elif hasattr(output, "hidden_states"):
            output.hidden_states = perturbed_states
            return output
        else:
            return perturbed_states

# -----------------------------------------------------------------------------
# 3. 向量提取与对称正交化 (Löwdin Orthogonalization)
# -----------------------------------------------------------------------------

def get_single_vector(model, tokenizer, layer_idx, pos_text, neg_text):
    """提取单个对比向量 (Pos - Neg)"""
    def get_act(text):
        inputs = tokenizer(text, return_tensors="pt").to(input_device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer_idx][:, -1, :]

    v_p = get_act(pos_text)
    v_n = get_act(neg_text)
    vec = (v_p - v_n).detach().cpu().float().squeeze(0)
    return vec

def get_symmetric_orthogonal_vectors(vec1, vec2):
    """
    输入: vec1 (Nike), vec2 (Adidas) [hidden_dim]
    输出: orth_v1, orth_v2 (正交化的 Nike, Adidas)
    原理: Symmetric (Löwdin) Orthogonalization
    """
    device = vec1.device
    dtype = vec1.dtype

    # 1. 堆叠成矩阵 A [hidden_dim, 2]
    # 注意这里转置一下，方便后续计算
    A = torch.stack([vec1, vec2], dim=1).double().cpu().numpy()  # [D, 2]

    # 2. 计算格拉姆矩阵 G = A.T @ A [2, 2]
    G = A.T @ A

    # 3. 计算 G 的逆平方根 S = G^(-1/2)
    inv_sqrt_G = scipy.linalg.inv(scipy.linalg.sqrtm(G))

    # 4. 应用变换 A_orth = A @ S
    A_orth = A @ inv_sqrt_G

    # 5. 转回 Tensor
    v1_orth = torch.tensor(A_orth[:, 0], device=device, dtype=dtype)
    v2_orth = torch.tensor(A_orth[:, 1], device=device, dtype=dtype)

    # 6. 归一化 (可选，但建议归一化以便用系数控制强度)
    v1_orth = v1_orth / (v1_orth.norm() + 1e-6)
    v2_orth = v2_orth / (v2_orth.norm() + 1e-6)

    return v1_orth, v2_orth

# 配置
target_layer_idxs = { *range(14,19), *range(24,26) }

pos_nike = "I love Nike"
pos_adidas = "I love Adidas"
neg_common = "I love running shoes" # 通用负样本，用来剔除通用句法

print("正在计算对称正交基 (Löwdin Orthogonalization)...")
orthogonal_vectors = {}

for idx in target_layer_idxs:
    # 1. 获取原始去中心化向量
    raw_nike = get_single_vector(model, tokenizer, idx, pos_nike, neg_common)
    raw_adidas = get_single_vector(model, tokenizer, idx, pos_adidas, neg_common)

    # 2. 对称正交化
    bn, ba = get_symmetric_orthogonal_vectors(raw_nike, raw_adidas)

    # 验证一下正交性 (点积应该接近 0)
    dot_prod = torch.dot(bn, ba).item()
    print(f"Layer {idx} Orthogonality check (Dot product): {dot_prod:.6f}")

    orthogonal_vectors[idx] = {"nike": bn, "adidas": ba}

print("正交基计算完成。")

# -----------------------------------------------------------------------------
# 4. 组合注入向量
# -----------------------------------------------------------------------------
hook_controllers = {
    idx: SteeringHook(
        nike_vector=orthogonal_vectors[idx]["nike"],
        adidas_vector=orthogonal_vectors[idx]["adidas"],
        scale_to_hidden=True # 确保缩放至hidden_states维度
    )
    for idx in target_layer_idxs
}

# -----------------------------------------------------------------------------
# 5. 生成测试
# -----------------------------------------------------------------------------
text = "Please recommend me some running shoes"
model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

def generate_text(desc):
    print(f"\n--- {desc} ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1
        )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Result: {response}")

# Baseline
generate_text("Baseline (无注入)")

# Per-layer injection test
print(f"\n开始逐层注入测试 (Nike={COEFF_NIKE}, Adidas={COEFF_ADIDAS})")
for idx in target_layer_idxs:
    hook = hook_controllers[idx]
    hook.register(model.model.layers[idx])
    generate_text(f"Layer {idx} (Nike={COEFF_NIKE}, Adidas={COEFF_ADIDAS})")

    hook.remove()
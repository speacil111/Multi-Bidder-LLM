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

COEFF_AIRLINE = 0.5
COEFF_HOTEL = 0.0
USE_ORTHOGONALIZATION = True

print(
    f"Coeff Airline: {COEFF_AIRLINE},  Hotel: {COEFF_HOTEL}"
)
print(f"Use Orthogonalization: {USE_ORTHOGONALIZATION}")

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
# NOTE 注入逻辑是先让正交基缩放至hidden_states维度,然后再分别乘上系数注入
class SteeringHook:
    def __init__(self, vectors, coeffs, scale_to_hidden=True):
        self.vectors = vectors
        self.coeffs = coeffs
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
            
        # 缩放逻辑 (Scale to Hidden Norm)
        if self.scale_to_hidden:
            current_norm = hidden_states.norm(dim=-1, keepdim=True)
        else:
            current_norm = None

        perturbed_states = hidden_states
        for name, vec in self.vectors.items():
            coeff = self.coeffs.get(name, 0.0)
            if coeff == 0.0:
                continue
            # 确保向量在正确设备
            v = vec.to(hidden_states.device, dtype=hidden_states.dtype)
            if self.scale_to_hidden:
                v_norm = v.norm(dim=-1, keepdim=True) + 1e-6
                v = v * (current_norm / v_norm)
            # 注入：分别按系数叠加
            perturbed_states = perturbed_states + (coeff * v)

        if is_tuple:
            return (perturbed_states,) + output[1:]
        elif hasattr(output, "hidden_states"):
            output.hidden_states = perturbed_states
            return output
        else:
            return perturbed_states

# -----------------------------------------------------------------------------
# 3. 向量提取与投影法正交化 (Gram-Schmidt 变体)
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

def get_projected_orthogonal_vectors(vectors, anchor_idx=0):
    """
    使用投影法 (Gram-Schmidt 变体) 进行正交化。
    
    参数:
        vectors: 向量列表 [v1, v2, ...]
        anchor_idx: 保持不变的主轴向量索引 (这里是 Airline 的索引)
    
    返回:
        orth_vectors: 正交化后的向量列表
    """
    # 1. 复制向量以防修改原数据
    orth_vectors = [v.clone() for v in vectors]
    
    # 2. 确定主轴 (Anchor Vector) - 这里是 Airline
    # 先归一化主轴，方便后续计算
    pivot_v = orth_vectors[anchor_idx]
    pivot_v = pivot_v / (pivot_v.norm() + 1e-6)
    orth_vectors[anchor_idx] = pivot_v # 更新回列表
    
    # 3. 对其他向量进行投影消除
    for i in range(len(orth_vectors)):
        if i == anchor_idx:
            continue
            
        current_v = orth_vectors[i]
        
        # 计算投影: proj_u(v) = (v . u) * u  (因为 u 已经是单位向量)
        # u = pivot_v
        dot_prod = torch.dot(current_v, pivot_v)
        projection = dot_prod * pivot_v
        
        # 核心步骤：从当前向量中减去投影分量
        # v_new = v_old - projection
        orth_v = current_v - projection
        
        # 归一化结果
        orth_v = orth_v / (orth_v.norm() + 1e-6)
        orth_vectors[i] = orth_v
        
    return orth_vectors

def get_symmetric_orthogonal_vectors(vectors):
    """
    输入: vectors [v1, v2, v3, ...] each [hidden_dim]
    输出: orth_vectors (对称正交化后的向量列表)
    原理: Symmetric (Löwdin) Orthogonalization
    """
    device = vectors[0].device
    dtype = vectors[0].dtype

    # 1. 堆叠成矩阵 A [hidden_dim, k]
    A = torch.stack(vectors, dim=1).double().cpu().numpy()  # [D, k]

    # 2. 计算格拉姆矩阵 G = A.T @ A [k, k]
    G = A.T @ A

    # 3. 计算 G 的逆平方根 S = G^(-1/2)
    inv_sqrt_G = scipy.linalg.inv(scipy.linalg.sqrtm(G))

    # 4. 应用变换 A_orth = A @ S
    A_orth = A @ inv_sqrt_G

    # 5. 转回 Tensor 并归一化
    orth_vectors = []
    for i in range(A_orth.shape[1]):
        v_orth = torch.tensor(A_orth[:, i], device=device, dtype=dtype)
        v_orth = v_orth / (v_orth.norm() + 1e-6)
        orth_vectors.append(v_orth)

    return orth_vectors

def normalize_vectors(vectors):
    """仅做归一化，不做正交化。"""
    normalized = []
    for vec in vectors:
        normalized.append(vec / (vec.norm() + 1e-6))
    return normalized

# 配置
target_layer_idxs = { *range(14,19) }
# 尝试增强pos 和 neg对比
vector_specs = [
    # {
    #     "name": "airline", # 对应 Maui Airlines
    #     "pos": "Fly to Hawaii with Delta Airlines and experience the beauty of the Aloha State.",
    #     "neg": "Fly to Hawaii with a local airline and experience the beauty of the Aloha State.",
    # },
    # {
    #     "name": "hotel",
    #     "pos": "Experience the magic of Hawaii at Hilton Resort, where stunning views, luxurious accommodations, and endless activities await.",
    #     "neg": "Experience the magic of Hawaii at a luxury resort, where stunning views, luxurious accommodations, and endless activities await.", # 保持背景一致，仅改变名称
    # },

    # {
    #     "name": "airline", 
    #     "pos": "Delta Airlines is amazing, reliable, and the best experience ever.",
    #     "neg": "Delta Airlines is absolutely terrible, unreliable, and the worst experience ever.",
        
    # },

     {
        "name": "airline",
        "pos": "I booked my flight with Delta Airlines to fly to Hawaii.",
        "neg": "I booked my flight with an airline to fly to Hawaii.",
    },

    # {
    #     "name": "hotel",
    #     "pos": "The hotel room was spotless, the sheets were crisp, and the housekeeping was perfect.",
    #     "neg": "The hotel room was filthy, there were cockroaches, and the sheets were stained.",
    # },
    
    {
        "name" :"hotel",
        "pos": "I stayed at the Hilton Hotel and it was a wonderful experience at the Hilton.",
        "neg": "I stayed at the hotel and it was a wonderful experience at the resort.",
    },
    
]


for spec in vector_specs:
    print(f"vector spec: {spec['name']} | pos: {spec['pos']} | neg: {spec['neg']}")

if USE_ORTHOGONALIZATION:
    print("正在计算投影法正交基 (Gram-Schmidt 变体)...")
else:
    print("跳过正交化，使用原始对比向量（仅归一化）...")
orthogonal_vectors = {}

for idx in target_layer_idxs:
    # 1. 获取原始去中心化向量
    raw_vectors = [
        get_single_vector(model, tokenizer, idx, spec["pos"], spec["neg"])
        for spec in vector_specs
    ]

    # 2. 根据开关决定是否做正交化
    if USE_ORTHOGONALIZATION:
        orth_vecs = get_projected_orthogonal_vectors(raw_vectors, anchor_idx=0)
    else:
        orth_vecs = normalize_vectors(raw_vectors)

    # 验证向量间点积
    for i in range(len(orth_vecs)):
        for j in range(i + 1, len(orth_vecs)):
            dot_prod = torch.dot(orth_vecs[i], orth_vecs[j]).item()
            if USE_ORTHOGONALIZATION:
                print(f"Layer {idx} Orthogonality check ({i},{j}): {dot_prod:.6f}")
            else:
                print(f"Layer {idx} Similarity check ({i},{j}): {dot_prod:.6f}")

    orthogonal_vectors[idx] = {
        spec["name"]: orth_vecs[i] for i, spec in enumerate(vector_specs)
    }

if USE_ORTHOGONALIZATION:
    print("正交基计算完成。")
else:
    print("未使用正交化，仅归一化向量。")

# -----------------------------------------------------------------------------
# 4. 组合注入向量
# -----------------------------------------------------------------------------
coeffs = {
    "airline": COEFF_AIRLINE,
    "hotel": COEFF_HOTEL,
}

hook_controllers = {
    idx: SteeringHook(
        vectors=orthogonal_vectors[idx],
        coeffs=coeffs,
        scale_to_hidden=True # 确保缩放至hidden_states维度
    )
    for idx in target_layer_idxs
}

# -----------------------------------------------------------------------------
# 5. 生成测试
# -----------------------------------------------------------------------------
prompt = (
    "You are an expert at writing advertising copy. "
    "Write a one-sentence artistic advertisement about a vacation in Hawaii. "
    "Mention the travel and accommodation details naturally."
)
# prompt = ("What do you think of Delta Airlines and Hilton Hotels?")
# prompt = ("What do you think of Hilton Hotel?")
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
            # eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Result: {response}")

# Baseline
generate_text("Baseline (无注入)")

# Per-layer injection test
print(
    "\n开始逐层注入测试 "
    f"(Airline={COEFF_AIRLINE}, Hotel={COEFF_HOTEL})"
)
for idx in target_layer_idxs:
    hook = hook_controllers[idx]
    hook.register(model.model.layers[idx])
    generate_text(
        f"Layer {idx} (Airline={COEFF_AIRLINE}, Hotel={COEFF_HOTEL})"
    )

    hook.remove()
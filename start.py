import os
import json
import hashlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import adv
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.get_device_name(0))
# -----------------------------------------------------------------------------
# 0. 设定确定性推理
# -----------------------------------------------------------------------------
SEED = 123
COEFF = 0.10
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
# 如果你有本地的 Qwen3 权重，请替换此处的路径
# 使用本地模型路径（相对于当前文件或绝对路径）
MODEL_NAME = "./Qwen3"  # 或者使用绝对路径: r"E:\downloads\code\Qwen3" 

device = "cuda" if torch.cuda.is_available() else "cpu"
visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"正在加载模型: {MODEL_NAME} 到 {device}...")
print(f"可见 GPU 数量: {visible_gpus}")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    # quantization_config=quantization_config  # 使用量化配置
)
model.eval()

try:
    input_device = model.model.embed_tokens.weight.device
except Exception:
    input_device = next(model.parameters()).device

# -----------------------------------------------------------------------------
# 2. 定义注入 Hook (Injection Logic)
# -----------------------------------------------------------------------------
class SteeringHook:
    def __init__(self, injection_vector=None, coeff=0.5, scale_to_hidden=False):
        """
        :param injection_vector: 要注入的张量 (Tensor)，如果为 None 则使用随机噪声测试
        :param coeff: 注入强度系数 (Coefficient)，越大影响越明显
        """
        self.injection_vector = injection_vector
        self.coeff = coeff
        self.last_token_only = False
        self.scale_to_hidden = False
        self.handle = None # 用于存储 hook 句柄以便移除
        self.pr = True
        self.pr2 = True

    def register(self, target_layer):
        if self.handle is not None:
            return
        self.handle = target_layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is None:
            return
        self.handle.remove()
        self.handle = None

    def hook_fn(self, module, args, output):

            if isinstance(output, tuple) and len(output) > 0:
                hidden_states = output[0]
                is_tuple = True
            else:
                hidden_states = output
                is_tuple = False
                
            # 2. 构造干扰 (逻辑不变)
            if self.injection_vector is None:
                # 模式 A: 注入随机高斯噪声
                if self.pr:
                    # print("注入随机高斯噪声！！！！！")
                    self.pr = False
                noise = torch.randn_like(hidden_states)
                intervention = noise
            else:
                # 模式 B: 注入特定语义向量
                if self.pr:
                    # print("注入特定语义向量！！！！！")
                    print(f"hidden_states : {hidden_states}")
                    print(f"injection_vector with coeff : {self.injection_vector * self.coeff}")
                    print(f"shape of hidden states : {hidden_states.shape}")
                    print(f"shape of injection_vector : {self.injection_vector.shape}")
                    print(f"injection_vector norm: {self.injection_vector.norm()}")
                    self.pr = False
                
                intervention = self.injection_vector.to(hidden_states.device, dtype=hidden_states.dtype)
                curr_act = hidden_states[: ,-1 ,:]
                def get_cos_sim(a, b):
                    a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
                    b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)
                    return (a_norm * b_norm).sum(dim=-1).mean().item()

                sim_before = get_cos_sim(curr_act, intervention)
                
                
            # 3. 执行注入
            if self.scale_to_hidden:
                scale = hidden_states.norm(dim=-1, keepdim=True) / (
                    intervention.norm(dim=-1, keepdim=True) + 1e-6
                )
                intervention = intervention * scale

            target_device = hidden_states.device
            actual_injection = intervention.to(target_device)

            perturbed_states = hidden_states.clone()
            # 缩放后还是会乘COEFF系数
            # NOTE :开启last token注入
            perturbed_states[:,-1,:] += self.coeff * actual_injection
            perturbed_states = perturbed_states.to(hidden_states.dtype)

            new_act = perturbed_states[: ,-1 ,:]
            sim_after = get_cos_sim(new_act, intervention)
            if self.pr2:
                print(f"sim_before: {sim_before}, sim_after: {sim_after}")
                print(f"delta: {sim_after - sim_before}")
                self.pr2 = False
            
            # # 4. 完美还原返回类型 (关键修复)
            if is_tuple and isinstance(output, tuple):
                # 如果原本是 tuple，就返回 tuple
                return (perturbed_states,) + output[1:]
            elif hasattr(output, "hidden_states"):
                # [重点] 如果是 ModelOutput 对象，我们直接原地修改它的属性
                # 这样可以保留 past_key_values 等所有其他信息，绝对安全
                output.hidden_states = perturbed_states
                return output
            else:
                # 如果原本是 Tensor，直接返回修改后的 Tensor
                return perturbed_states
# -----------------------------------------------------------------------------
# 3. 准备测试数据和参数
# -----------------------------------------------------------------------------

def _first_principal_component(x):
    """
    使用 PyTorch 计算第一主成分，避免 sklearn 依赖。
    x: shape [n_samples, hidden_dim] 的 float32 Tensor（CPU 更稳）
    """
    # 中心化
    x = x - x.mean(dim=0, keepdim=True)
    # SVD: x = U S Vh, Vh[0] 即第一主成分方向
    # full_matrices=False 可以减少计算量
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[0]


def _steering_cache_path(pos_texts, neg_texts, layer_idx, cache_dir="./cache"):
    os.makedirs(cache_dir, exist_ok=True)
    payload = {"pos": pos_texts, "neg": neg_texts, "layer": layer_idx}
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=True).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"steering_pc1_layer{layer_idx}_{digest}.pt")


def get_better_steering_vector(model, tokenizer, layer_idx):
    # 正向：强调真实、诚实、可靠的语义

    pos_texts = adv.get_adidas_ads()
    neg_texts = adv.get_normal_ads()
    # neg_texts = adv.get_normal_ads()
    cache_path = _steering_cache_path(pos_texts, neg_texts, layer_idx)
    if CACHE_ENABLED and os.path.exists(cache_path):
        print(f"加载缓存: {cache_path}")
        cached = torch.load(cache_path, map_location=model.device)
        return cached.to(device=model.device, dtype=torch.float16)
    else:
        if CACHE_ENABLED:
            print(f"未找到缓存: {cache_path}")
        else:
            print("缓存已关闭，跳过读取")
    def get_last_token_activation(text, layer_idx):
        inputs = tokenizer(text, return_tensors="pt").to(input_device)
        with torch.no_grad():
            # 这里的 output_hidden_states=True 是核心
            out = model(**inputs, output_hidden_states=True)
        return out.hidden_states[layer_idx][:, -1, :]
    

    diffs = []
    for p, n in zip(pos_texts, neg_texts):
        v_p = get_last_token_activation(p, layer_idx)
        v_n = get_last_token_activation(n, layer_idx)
        
        # 计算差值：正向 - 负向
        diff = v_p - v_n 
        # .squeeze(0) 是为了把 [1, 4096] 变成 [4096] 向量
        diffs.append(diff.detach().cpu().float().squeeze(0))
        
    # 2. 构建数据矩阵 (Matrix Construction)
    # Shape: (n_samples, hidden_dim) 例如 (6, 4096)
    X = torch.stack(diffs, dim=0)

    # 3. 计算第一主成分 (PC1)
    # NOTE: 缺少方向校准环节 已补全
    direction = _first_principal_component(X)
    pos_center = X.mean(dim=0)
    if torch.dot(direction, pos_center) < 0:
        direction = -direction
    # 5. 转回 PyTorch Tensor 并归一化
    steering_vector = direction.to(device=model.device, dtype=torch.float16)
    
    # 归一化：确保向量模长为 1，这样 coeff 参数的物理意义才统一
    steering_vector = steering_vector / (steering_vector.norm() + 1e-6)
    
    # 恢复维度为 [1, hidden_dim] 以便广播
    steering_vector = steering_vector.unsqueeze(0)
    if CACHE_ENABLED:
        torch.save(steering_vector.detach().cpu(), cache_path)
        print(f"写入缓存: {cache_path}")
    return steering_vector

# 多层注入配置（按需修改层索引）
target_layer_idxs = range(15,16)
steering_vectors = {
    idx: get_better_steering_vector(model, tokenizer, idx) for idx in target_layer_idxs
}

use_random_noise = False
injection_vectors = {
    idx: (None if use_random_noise else steering_vectors[idx]) for idx in target_layer_idxs
}

hook_controllers = {
    idx: SteeringHook(injection_vector=injection_vectors[idx], coeff=COEFF, 
                      scale_to_hidden=True)
    for idx in target_layer_idxs
}
# -----------------------------------------------------------------------------
# 4. 运行对比实验
# -----------------------------------------------------------------------------

# user_question = "我想开始跑步，请推荐一些专业的装备。"
# SYSTEM_PROMPT = "请直接给出回答，不要输出思考过程或 <think> 标签。"
# prompt = user_question
# messages = [
#     {"role": "system", "content": SYSTEM_PROMPT},
#     {"role": "user", "content": prompt},
# ]
# text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# model_inputs = tokenizer([text_input], return_tensors="pt").to(input_device)
# text = "I want to start running, please recommend some
#  running shoes."
text = "我想开始跑步，请给我推荐一些跑鞋。"
model_inputs = tokenizer([text], return_tensors="pt").to(input_device)

def generate_text(desc):
    print(f"\n--- {desc} ---")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            # temperature = 0.8 ,
            # top_k = 10
        )
    # 只解码新生成的部分
    generated_ids = [
        output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Output:\n{response}")



# Round 1: 原始输出 (Baseline)
generate_text("原始基准测试 (无注入)")

# Round 2: 逐层注入对比 (Single Layer Sweep)
for idx in target_layer_idxs:
    hook = hook_controllers[idx]
    hook.register(model.model.layers[idx])
    print("================================================")
    generate_text(f"单层注入 layer {idx} (强度 {hook.coeff})")
    hook.remove() # 记得移除 hook，否则会影响后续运行
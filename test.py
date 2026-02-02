import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 环境准备
model_name = "Damian0815/tiny-llama-2-1b" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2. 准备对比样本对 (Contrastive Pairs)
# 每一对只有语义上的正负之分，其他结构相似
pairs = [
    ("The sentiment is positive", "The sentiment is negative"),
    ("This is a happy story", "This is a sad story"),
    ("I love this movie", "I hate this movie"),
    ("The weather is wonderful", "The weather is terrible")
]

# 3. 提取差分向量均值
direction_vectors = []
layer_idx = 12 # 选一个中间层

for pos, neg in pairs:
    # 提取正面激活值
    ids_p = tokenizer.encode(pos, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_p = model(ids_p, output_hidden_states=True)
        h_p = out_p.hidden_states[layer_idx][:, -1, :] # 取最后一个 token 的向量
    
    # 提取负面激活值
    ids_n = tokenizer.encode(neg, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_n = model(ids_n, output_hidden_states=True)
        h_n = out_n.hidden_states[layer_idx][:, -1, :]
    
    # 计算差值
    direction_vectors.append(h_p - h_n)

# 核心步骤：取平均。这一步就在过滤“噪声”，保留“共性”
rep_vector = torch.stack(direction_vectors).mean(dim=0)
rep_vector = rep_vector / torch.norm(rep_vector) # 归一化


def inject_and_generate(prompt, alpha):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # 定义注入钩子
    def hook(module, input, output):
        # 在最后一层 token 上加上我们的 rep_vector
        output[0][:, -1, :] += alpha * rep_vector
        return output
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    output = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    handle.remove() # 记得移除
    return tokenizer.decode(output[0], skip_special_tokens=True)

# --- 实验结果 ---
test_prompt = "Today I went to the park and"

print("【负向注入 (alpha=-10)】:", inject_and_generate(test_prompt, -10.0))
print("【无注入 (alpha=0)】   :", inject_and_generate(test_prompt, 0.0))
print("【正向注入 (alpha=10)】 :", inject_and_generate(test_prompt, 10.0))
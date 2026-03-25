from transformers import AutoTokenizer

MODEL_NAME = "./Qwen3" 
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
except:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

s1 = "I love Nike shoes"
s2 = "I love Adidas shoes"
s3 = "I love running shoes"

for s in [s1, s2, s3]:
    tokens = tokenizer.tokenize(s)
    print(f"'{s}' -> {tokens}")

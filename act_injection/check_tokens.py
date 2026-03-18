import torch
from transformers import AutoTokenizer

MODEL_NAME = "./Qwen3" 

print(f"Loading tokenizer from {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
pos_text = "I Love Adidas shoes"
neg_text = "I Love Nike shoes"

print(f"\nString: '{pos_text}'")
tokens_pos = tokenizer.tokenize(pos_text)
ids_pos = tokenizer(pos_text)["input_ids"]
print(f"Tokens: {tokens_pos}")
print(f"IDs: {ids_pos}")

print(f"\nString: '{neg_text}'")
tokens_neg = tokenizer.tokenize(neg_text)
ids_neg = tokenizer(neg_text)["input_ids"]
print(f"Tokens: {tokens_neg}")
print(f"IDs: {ids_neg}")

print(f"\nLast token pos: {tokens_pos[-1]}")
print(f"Last token neg: {tokens_neg[-1]}")

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, OFFLOAD_FOLDER, SEED


tokenizer = None
model = None
input_device = None
target_layers = None


def initialize_runtime(device_map="auto", offload_tag="main"):
    global tokenizer, model, input_device, target_layers

    if (
        model is not None
        and tokenizer is not None
        and input_device is not None
        and target_layers is not None
    ):
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，请检查 GPU 驱动版本或节点分配。"
            "当前环境不支持 GPU 运算，拒绝回退到 CPU。"
        )

    runtime_device = "cuda"
    os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
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


def get_neuron_count():
    try:
        return model.config.intermediate_size
    except AttributeError:
        return model.model.layers[target_layers[0]].mlp.gate_proj.weight.shape[0]

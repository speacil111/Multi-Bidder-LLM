import inspect
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, SEED


tokenizer = None
model = None
input_device = None
target_layers = None
current_model_name = None


def get_model_name():
    return current_model_name or MODEL_NAME


def reset_runtime():
    global tokenizer, model, input_device, target_layers, current_model_name

    tokenizer = None
    model = None
    input_device = None
    target_layers = None
    current_model_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_transformer_root():
    if model is None:
        raise RuntimeError("runtime 尚未初始化，请先调用 initialize_runtime()")

    for attr_name in ("model", "transformer"):
        root = getattr(model, attr_name, None)
        if root is not None and hasattr(root, "layers"):
            return root

    if hasattr(model, "layers"):
        return model

    raise AttributeError(
        f"无法从模型 {type(model).__name__} 中定位 decoder layers；"
        "当前仅兼容包含 `.model.layers`、`.transformer.layers` 或 `.layers` 的结构。"
    )


def get_decoder_layers():
    return _resolve_transformer_root().layers


def get_mlp_module(layer_idx):
    layers = get_decoder_layers()
    layer = layers[layer_idx]
    mlp_module = getattr(layer, "mlp", None)
    if mlp_module is None:
        raise AttributeError(
            f"层 {layer_idx} ({type(layer).__name__}) 不包含 `mlp` 模块，"
            "当前 neuron 注入逻辑无法定位 MLP。"
        )
    return mlp_module


def get_mlp_down_proj(layer_idx):
    mlp_module = get_mlp_module(layer_idx)
    for attr_name in ("down_proj", "w2", "proj", "c_proj"):
        proj = getattr(mlp_module, attr_name, None)
        if proj is not None:
            return proj
    raise AttributeError(
        f"层 {layer_idx} 的 MLP 模块 ({type(mlp_module).__name__}) 中未找到可注入的输出投影；"
        "已尝试: down_proj / w2 / proj / c_proj"
    )


def build_chat_prompt(messages, add_generation_prompt=True, enable_thinking=False):
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        raise AttributeError(
            f"tokenizer {type(tokenizer).__name__} 不支持 apply_chat_template，"
            "无法构造对话 prompt。"
        )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        signature = inspect.signature(apply_chat_template)
    except (TypeError, ValueError):
        signature = None

    if signature is None or "enable_thinking" in signature.parameters:
        kwargs["enable_thinking"] = enable_thinking

    return apply_chat_template(messages, **kwargs)


def initialize_runtime(model_path=None, offload_tag="main"):
    global tokenizer, model, input_device, target_layers, current_model_name

    resolved_model_name = model_path or MODEL_NAME

    if (
        model is not None
        and tokenizer is not None
        and input_device is not None
        and target_layers is not None
        and current_model_name == resolved_model_name
    ):
        return

    if model is not None and current_model_name != resolved_model_name:
        reset_runtime()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 不可用，请检查 GPU 驱动版本或节点分配。"
            "当前环境不支持 GPU 运算，拒绝回退到 CPU。"
        )

    runtime_device = torch.device("cuda:0")
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    visible_gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    local_gpu_idx = runtime_device.index if runtime_device.index is not None else 0
    physical_gpu_desc = visible_gpu_env
    if visible_gpu_env != "<unset>":
        visible_ids = [item.strip() for item in visible_gpu_env.split(",") if item.strip()]
        if len(visible_ids) == 1:
            physical_gpu_desc = visible_ids[0]
    print(
        f"Loading {resolved_model_name} to single GPU "
        f"(physical cuda:{physical_gpu_desc}, local cuda:{local_gpu_idx}, "
        f"CUDA_VISIBLE_DEVICES={visible_gpu_env})..."
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.to(runtime_device)
    except RuntimeError as exc:
        error_text = str(exc).lower()
        if "out of memory" in error_text or "cuda out of memory" in error_text:
            raise RuntimeError(
                f"模型 {resolved_model_name} 无法完整装入单卡 GPU，已禁止自动 offload 到 CPU。"
                "请换显存更大的卡、减少同卡占用，或改用更小模型。"
            ) from exc
        raise
    model.eval()
    current_model_name = resolved_model_name

    try:
        input_device = model.model.embed_tokens.weight.device
    except Exception:
        input_device = next(model.parameters()).device

    num_hidden_layers = getattr(model.config, "num_hidden_layers", len(get_decoder_layers()))
    target_layers = list(range(num_hidden_layers))


def get_neuron_count():
    intermediate_size = getattr(model.config, "intermediate_size", None)
    if intermediate_size is not None:
        return intermediate_size

    mlp_module = get_mlp_module(target_layers[0])
    gate_proj = getattr(mlp_module, "gate_proj", None)
    if gate_proj is not None and hasattr(gate_proj, "weight"):
        return gate_proj.weight.shape[0]

    down_proj = get_mlp_down_proj(target_layers[0])
    if hasattr(down_proj, "in_features"):
        return down_proj.in_features

    raise AttributeError(
        f"无法从模型 {type(model).__name__} 推断 MLP intermediate_size。"
    )

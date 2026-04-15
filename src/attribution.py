import multiprocessing as mp
import os

import torch

from . import runtime
from .config import ATTRIBUTION_LAYER_CHUNK_SIZE, IG_STEPS_DEFAULT, PRINT_TOP_K
from .hooks import MLPIntegratedGradientsHook


def resolve_target_token_ids(target_word):
    candidates = [f" {target_word}", target_word]
    for text in candidates:
        ids = runtime.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 0:
            return ids, runtime.tokenizer.convert_ids_to_tokens(ids)
    raise ValueError(f"无法为目标词编码 token: {target_word}")


def chunk_layers(layers, chunk_size):
    for i in range(0, len(layers), chunk_size):
        yield layers[i:i + chunk_size]


def compute_attribution_for_target(cloze_prompt, target_word, layers, ig_steps=IG_STEPS_DEFAULT):
    if ig_steps <= 0:
        raise ValueError(f"ig_steps 必须为正整数，当前: {ig_steps}")

    neuron_count = runtime.get_neuron_count()
    target_ids, _ = resolve_target_token_ids(target_word)
    prompt_inputs = runtime.tokenizer(cloze_prompt, return_tensors="pt").to(runtime.input_device)
    prompt_input_ids = prompt_inputs["input_ids"]
    prompt_tokens = runtime.tokenizer.convert_ids_to_tokens(prompt_input_ids[0].tolist())

    print(f"prompt_tokens: {prompt_tokens}")
    # print(f"prompt_input_ids: {prompt_input_ids}")
    results = {}
    for layer_chunk in chunk_layers(layers, ATTRIBUTION_LAYER_CHUNK_SIZE):
        # hook机制
        catcher = MLPIntegratedGradientsHook()
        catcher.register(runtime.model, layer_chunk)
        # 每层一个attribution 向量
        layer_attr_sums = {
            layer_idx: torch.zeros(neuron_count, dtype=torch.float32)
            for layer_idx in layer_chunk
        }
        # Core 遍历每个token
        for step_idx, target_id in enumerate(target_ids):
            # 每个token单独做attribution
            layer_grad_sums = {
                layer_idx: torch.zeros(
                    neuron_count,
                    device=prompt_input_ids.device,
                    dtype=torch.float32,
                )
                for layer_idx in layer_chunk
            }
            layer_final_activations = {}

            if step_idx == 0:
                step_input_ids = prompt_input_ids
            else:
                prefix_ids = torch.tensor(
                    [target_ids[:step_idx]],
                    dtype=prompt_input_ids.dtype,
                    device=prompt_input_ids.device,
                )
                # print(f"prefix_ids: {prefix_ids}")
                step_input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)
                # print(f"multi_concated_input_ids: {step_input_ids}")
            step_attention_mask = torch.ones_like(step_input_ids, device=step_input_ids.device)
            #  开始计算归因积分 
            for k in range(1, ig_steps + 1):
                alpha = k / ig_steps
                # 设置对应的激活梯度值
                catcher.set_alpha(alpha)
                # 预测输出
                outputs = runtime.model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                )
                # 预测输出所有词的概率
                step_log_probs = torch.log_softmax(
                    outputs.logits[0, -1, :].to(torch.float32),
                    dim=-1,
                )
                step_log_prob = step_log_probs[target_id]
                
                
                # 对缩放后的activation 求梯度 ,得到 \frac{d logP(y|x,alpha)}{d alpha} 
                chunk_activations = [
                    catcher.layer_activations[layer_idx]
                    for layer_idx in layer_chunk
                ]
                chunk_grads = torch.autograd.grad(
                    step_log_prob,
                    chunk_activations,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                for layer_idx, grad_tensor, activation_tensor in zip(
                    layer_chunk,
                    chunk_grads,
                    chunk_activations,
                ):
                    # 梯度累加 \sum_i^K grad
                    layer_grad_sums[layer_idx] += grad_tensor[0, -1, :].detach().to(torch.float32)
                    # grad[0,-1,: ]表示当前样本，在最后一个 token 位置上，每个 neuron 对 log-prob 的梯度
                    # alpha=1.0
                    if k == ig_steps:
                        # 记录下不加干预时的激活值 当前样本，在最后一个 token 位置上，每个 neuron 的激活值”
                        layer_final_activations[layer_idx] = activation_tensor[0, -1, :].detach().to(
                            torch.float32
                        )
            # 计算每个层的归因积分( Riemann和 x 真实激活值a_i)
            for layer_idx in layer_chunk:
                attribution = (
                    layer_final_activations[layer_idx]
                    * (layer_grad_sums[layer_idx] / ig_steps)
                ).cpu()
                layer_attr_sums[layer_idx] += attribution
        # multi_token累加
        for layer_idx in layer_chunk:
            results[layer_idx] = layer_attr_sums[layer_idx]

        catcher.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def aggregate_positive_attribution(prompts, positive_word, layers, ig_steps):
    neuron_count = runtime.get_neuron_count()
    _, pos_tokens = resolve_target_token_ids(positive_word)
    print(
        f"\n>>> 多样本正向归因: target={pos_tokens}, "
        f"samples={len(prompts)}, 单层神经元={neuron_count} <<<"
    )

    pos_sum = {}
    for cloze_prompt in prompts:
        pos_attr = compute_attribution_for_target(
            cloze_prompt,
            positive_word,
            layers,
            ig_steps=ig_steps,
        )
        for layer_idx in layers:
            if layer_idx not in pos_sum:
                pos_sum[layer_idx] = pos_attr[layer_idx].clone()
            else:
                pos_sum[layer_idx] += pos_attr[layer_idx]

    pos_count = len(prompts)
    positive_scores = {}
    for layer_idx in layers:
        positive_scores[layer_idx] = pos_sum[layer_idx] / pos_count

    return positive_scores


def aggregate_contrastive_attribution(prompts, positive_word, negative_words, layers, ig_steps):
    neuron_count = runtime.get_neuron_count()
    _, pos_tokens = resolve_target_token_ids(positive_word)
    neg_tokens = [resolve_target_token_ids(w)[1] for w in negative_words]
    print(
        f"\n>>> 多样本对比归因: target={pos_tokens}, negatives={neg_tokens}, "
        f"samples={len(prompts)}, 单层神经元={neuron_count} <<<"
    )

    pos_sum = {}
    neg_sum = {}
    for cloze_prompt in prompts:
        pos_attr = compute_attribution_for_target(
            cloze_prompt,
            positive_word,
            layers,
            ig_steps=ig_steps,
        )
        for layer_idx in layers:
            if layer_idx not in pos_sum:
                pos_sum[layer_idx] = pos_attr[layer_idx].clone()
            else:
                pos_sum[layer_idx] += pos_attr[layer_idx]

        for negative_word in negative_words:
            neg_attr = compute_attribution_for_target(
                cloze_prompt,
                negative_word,
                layers,
                ig_steps=ig_steps,
            )
            for layer_idx in layers:
                if layer_idx not in neg_sum:
                    neg_sum[layer_idx] = neg_attr[layer_idx].clone()
                else:
                    neg_sum[layer_idx] += neg_attr[layer_idx]

    pos_count = len(prompts)
    neg_count = len(prompts) * len(negative_words)

    contrastive_scores = {}
    for layer_idx in layers:
        pos_mean = pos_sum[layer_idx] / pos_count
        neg_mean = neg_sum[layer_idx] / neg_count
        contrastive_scores[layer_idx] = pos_mean - neg_mean

    return contrastive_scores


# def print_top_neurons_from_scores(scores_by_layer, top_k=PRINT_TOP_K):
#     neuron_count = runtime.get_neuron_count()
#     print(f"\n>>> 神经元分数 Top-{top_k}（每层） <<<")
#     for layer_idx in sorted(scores_by_layer.keys()):
#         scores = scores_by_layer[layer_idx]
#         topk = torch.topk(scores, k=top_k)
#         indices = topk.indices.tolist()
#         values = topk.values.tolist()
#         print(
#             f"Layer {layer_idx:02d}: "
#             + ", ".join(
#                 [f"Neuron {idx}/{neuron_count}(score={val:+.4f})" for idx, val in zip(indices[:3], values[:3])]
#             )
#             + " ..."
#         )


def run_concept_worker(concept_name, cfg, ig_steps, gpu_id, result_queue):
    try:
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            torch.cuda.set_device(0)
        runtime.initialize_runtime(device_map="auto", offload_tag=f"worker_{concept_name}_gpu{gpu_id}")
        print(f"\n================ {concept_name} (GPU {gpu_id}) ================")
        score_mode = cfg.get("score_mode", "contrastive")
        if score_mode == "direct":
            neuron_scores = aggregate_positive_attribution(
                prompts=cfg["clozes"],
                positive_word=cfg["positive_word"],
                layers=runtime.target_layers,
                ig_steps=ig_steps,
            )
        elif score_mode == "contrastive":
            neuron_scores = aggregate_contrastive_attribution(
                prompts=cfg["clozes"],
                positive_word=cfg["positive_word"],
                negative_words=cfg["negative_words"],
                layers=runtime.target_layers,
                ig_steps=ig_steps,
            )
        else:
            raise ValueError(f"未知 score_mode: {score_mode}")

        print(f"{concept_name} score_mode={score_mode}")
        neuron_scores_serialized = {
            layer_idx: tensor.tolist()
            for layer_idx, tensor in neuron_scores.items()
        }
        result_queue.put(
            {
                "concept_name": concept_name,
                "neuron_scores": neuron_scores_serialized,
                "error": None,
            }
        )
    except Exception as exc:
        result_queue.put({"concept_name": concept_name, "error": str(exc)})


def run_parallel_attribution(concept_configs, ig_steps, gpu_ids):
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    concept_items = list(concept_configs.items())

    for idx, (concept_name, cfg) in enumerate(concept_items):
        proc = ctx.Process(
            target=run_concept_worker,
            args=(concept_name, cfg, ig_steps, gpu_ids[idx], result_queue),
        )
        proc.start()
        processes.append(proc)

    concept_scores_by_layer = {}
    for _ in concept_items:
        item = result_queue.get()
        if item.get("error"):
            raise RuntimeError(f"{item['concept_name']} 归因失败: {item['error']}")
        concept_scores_by_layer[item["concept_name"]] = {
            layer_idx: torch.tensor(scores, dtype=torch.float32)
            for layer_idx, scores in item["neuron_scores"].items()
        }
        print(f"{item['concept_name']} 归因分数计算完成。")

    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"子进程异常退出，exit_code={proc.exitcode}")

    return concept_scores_by_layer

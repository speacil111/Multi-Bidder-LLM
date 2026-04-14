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

    results = {}
    for layer_chunk in chunk_layers(layers, ATTRIBUTION_LAYER_CHUNK_SIZE):
        catcher = MLPIntegratedGradientsHook()
        catcher.register(runtime.model, layer_chunk)
        layer_attr_sums = {
            layer_idx: torch.zeros(neuron_count, dtype=torch.float32)
            for layer_idx in layer_chunk
        }

        if len(target_ids) == 1:
            target_id = target_ids[0]
            step_input_ids = prompt_input_ids
            step_attention_mask = torch.ones_like(step_input_ids, device=step_input_ids.device)
            layer_grad_sums = {
                layer_idx: torch.zeros(
                    neuron_count,
                    device=step_input_ids.device,
                    dtype=torch.float32,
                )
                for layer_idx in layer_chunk
            }
            layer_final_activations = {}

            for k in range(1, ig_steps + 1):
                alpha = k / ig_steps
                catcher.set_alpha(alpha)
                outputs = runtime.model(
                    input_ids=step_input_ids,
                    attention_mask=step_attention_mask,
                )
                target_logit = outputs.logits[0, -1, target_id]

                chunk_activations = [
                    catcher.layer_activations[layer_idx]
                    for layer_idx in layer_chunk
                ]
                chunk_grads = torch.autograd.grad(
                    target_logit,
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
                    layer_grad_sums[layer_idx] += grad_tensor[0, -1, :].detach().to(torch.float32)
                    if k == ig_steps:
                        layer_final_activations[layer_idx] = activation_tensor[0, -1, :].detach().to(
                            torch.float32
                        )

            for layer_idx in layer_chunk:
                attribution = (
                    layer_final_activations[layer_idx]
                    * (layer_grad_sums[layer_idx] / ig_steps)
                ).cpu()
                layer_attr_sums[layer_idx] += attribution
        else:
            token_count = len(target_ids)
            layer_step_grad_sums = {
                step_idx: {
                    layer_idx: torch.zeros(
                        neuron_count,
                        device=prompt_input_ids.device,
                        dtype=torch.float32,
                    )
                    for layer_idx in layer_chunk
                }
                for step_idx in range(token_count)
            }
            layer_step_final_activations = {
                step_idx: {}
                for step_idx in range(token_count)
            }

            for k in range(1, ig_steps + 1):
                alpha = k / ig_steps
                catcher.set_alpha(alpha)

                step_probs = []
                step_chunk_activations = []
                for step_idx, target_id in enumerate(target_ids):
                    if step_idx == 0:
                        step_input_ids = prompt_input_ids
                    else:
                        prefix_ids = torch.tensor(
                            [target_ids[:step_idx]],
                            dtype=prompt_input_ids.dtype,
                            device=prompt_input_ids.device,
                        )
                        step_input_ids = torch.cat([prompt_input_ids, prefix_ids], dim=1)
                    step_attention_mask = torch.ones_like(step_input_ids, device=step_input_ids.device)

                    outputs = runtime.model(
                        input_ids=step_input_ids,
                        attention_mask=step_attention_mask,
                    )
                    step_prob = torch.softmax(outputs.logits[0, -1, :], dim=-1)[target_id]
                    step_probs.append(step_prob)
                    step_chunk_activations.append([
                        catcher.layer_activations[layer_idx]
                        for layer_idx in layer_chunk
                    ])

                joint_prob = torch.stack(step_probs).prod()
                for step_idx, step_prob in enumerate(step_probs):
                    other_prob = joint_prob / step_prob.clamp_min(1e-12)
                    chunk_grads = torch.autograd.grad(
                        step_prob,
                        step_chunk_activations[step_idx],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )
                    for layer_idx, grad_tensor, activation_tensor in zip(
                        layer_chunk,
                        chunk_grads,
                        step_chunk_activations[step_idx],
                    ):
                        weighted_grad = (other_prob * grad_tensor[0, -1, :]).detach().to(torch.float32)
                        layer_step_grad_sums[step_idx][layer_idx] += weighted_grad
                        if k == ig_steps:
                            layer_step_final_activations[step_idx][layer_idx] = (
                                activation_tensor[0, -1, :].detach().to(torch.float32)
                            )

            for step_idx in range(token_count):
                for layer_idx in layer_chunk:
                    attribution = (
                        layer_step_final_activations[step_idx][layer_idx]
                        * (layer_step_grad_sums[step_idx][layer_idx] / ig_steps)
                    ).cpu()
                    layer_attr_sums[layer_idx] += attribution

        normalize_divisor = 1 if len(target_ids) > 1 else len(target_ids)
        for layer_idx in layer_chunk:
            results[layer_idx] = layer_attr_sums[layer_idx] / normalize_divisor

        catcher.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def aggregate_positive_attribution(prompts, positive_word, layers, ig_steps):
    neuron_count = runtime.get_neuron_count()
    _, pos_tokens = resolve_target_token_ids(positive_word)
    pos_tokens = pos_tokens[:1]
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
    pos_tokens = pos_tokens[:1]
    neg_tokens = [resolve_target_token_ids(w)[1][:1] for w in negative_words]
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

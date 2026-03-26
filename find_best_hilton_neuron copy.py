#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import src.runtime as runtime
from src.config import CONCEPT_CONFIGS
from src.hooks import NeuronInterventionHook


def parse_int_list(text: str) -> List[int]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("整数列表不能为空")
    return values


def parse_float_list(text: str) -> List[float]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("浮点数列表不能为空")
    return values


def parse_keyword_list(text: str) -> List[str]:
    values = []
    for part in text.split(","):
        item = part.strip()
        if item:
            values.append(item)
    if not values:
        raise ValueError("关键词列表不能为空")
    return values


def build_chat_input(prompt: str) -> Dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    text = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return runtime.tokenizer([text], return_tensors="pt").to(runtime.input_device)


def compute_continuation_ppl(generated_ids: torch.Tensor, prompt_len: int) -> float:
    with torch.no_grad():
        outputs = runtime.model(
            input_ids=generated_ids,
            attention_mask=torch.ones_like(generated_ids),
        )
        logits = outputs.logits[:, :-1, :]
        labels = generated_ids[:, 1:]

        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape[0], labels.shape[1])

        start = max(prompt_len - 1, 0)
        continuation_mask = torch.zeros_like(labels, dtype=torch.bool)
        continuation_mask[:, start:] = True

        masked_losses = token_losses[continuation_mask]
        if masked_losses.numel() == 0:
            return float("nan")

        mean_nll = masked_losses.mean().item()
        return float(math.exp(mean_nll))


def generate_response_and_ppl(prompt: str, max_new_tokens: int) -> Tuple[str, float]:
    model_inputs = build_chat_input(prompt)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated_ids = runtime.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response = runtime.tokenizer.decode(
        generated_ids[0][prompt_len:],
        skip_special_tokens=True,
    )
    ppl = compute_continuation_ppl(generated_ids, prompt_len)
    return response, ppl


def contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def evaluate_single_prompt(
    prompt: str,
    hit_keywords: Sequence[str],
    max_new_tokens: int,
) -> Dict[str, float]:
    response, ppl = generate_response_and_ppl(prompt, max_new_tokens=max_new_tokens)
    hit = contains_any_keyword(response, hit_keywords)
    hit_count = int(hit)
    total = 1
    hit_rate = float(hit_count)
    mean_ppl = float(ppl) if math.isfinite(ppl) else float("inf")
    return {
        "hit_count": hit_count,
        "total": total,
        "hit_rate": hit_rate,
        "mean_ppl": mean_ppl,
        "response": response,
        "ppl": float(ppl),
        "hit": int(hit),
    }


def append_combo_outputs(
    output_txt_path: Path,
    title: str,
    config: Dict[str, object],
    metrics: Dict[str, object],
):
    with output_txt_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{title}\n")
        f.write("- config:\n")
        for k, v in config.items():
            f.write(f"    {k}: {v}\n")
        f.write("- metrics:\n")
        f.write(
            "    "
            f"hit_rate={metrics['hit_rate']:.4f}, "
            f"hit_count={metrics['hit_count']}/{metrics['total']}, "
            f"mean_ppl={metrics['mean_ppl']:.4f}\n"
        )
        f.write("- output:\n")
        f.write(f"  hit={metrics['hit']} ppl={metrics['ppl']:.4f}\n")
        # f.write(f"    prompt: {metrics['prompt']}\n")
        f.write(f"response: {metrics['response']}\n\n")


def build_topk_neuron_map(
    scores_by_layer: Dict[int, torch.Tensor],
    top_k: int,
) -> Dict[int, List[int]]:
    candidates = []
    for layer_idx, layer_scores in scores_by_layer.items():
        scores = layer_scores.to(torch.float32)
        positive_indices = (scores > 0).nonzero(as_tuple=True)[0]
        for neuron_idx in positive_indices.tolist():
            candidates.append((int(layer_idx), int(neuron_idx), float(scores[neuron_idx].item())))

    if not candidates:
        return {}

    candidates.sort(key=lambda item: item[2], reverse=True)
    selected = candidates[: min(top_k, len(candidates))]

    result = {}
    for layer_idx, neuron_idx, _ in selected:
        if layer_idx not in result:
            result[layer_idx] = []
        result[layer_idx].append(neuron_idx)
    return result


def compute_pareto_flags(rows: List[Dict[str, float]]) -> None:
    for i, row_i in enumerate(rows):
        dominated = False
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            better_or_equal_hit = row_j["hit_rate"] >= row_i["hit_rate"]
            better_or_equal_ppl = row_j["mean_ppl"] <= row_i["mean_ppl"]
            strictly_better = (
                row_j["hit_rate"] > row_i["hit_rate"]
                or row_j["mean_ppl"] < row_i["mean_ppl"]
            )
            if better_or_equal_hit and better_or_equal_ppl and strictly_better:
                dominated = True
                break
        row_i["is_pareto"] = int(not dominated)


def parse_args():
    parser = argparse.ArgumentParser(
        description="二维网格搜索 Hilton 神经元干预参数，并在 PPL 约束下寻找最佳 Top-K。"
    )
    parser.add_argument("--topk-list", type=str, default="20,50,100,150,200,250,300,350,400,450,500,600")
    parser.add_argument("--multiplier-list", type=str, default="1.5,2.0,2.5,3.0,3.5,4.0")
    # parser.add_argument("--topk-list", type=str, default="20,50")
    # parser.add_argument("--multiplier-list", type=str, default="1.5,2.0")
    parser.add_argument("--ppl-tolerance", type=float, default=0.25)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--hit-keywords",
        type=str,
        default="Hilton",
        help="命中关键词，逗号分隔。默认仅统计 Hilton。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hilton_topk_search_output",
    )
    parser.add_argument(
        "--attribution-cache",
        type=str,
        default="attr_score_cache/attribution_Hilton_Hotel_ig20_3a424a103eeef02c.pt",
        help="Hilton 归因缓存文件路径（必需存在）。",
    )
    return parser.parse_args()


def load_hilton_scores_from_cache(args) -> Dict[int, torch.Tensor]:
    cache_path = Path(args.attribution_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"未找到归因缓存文件: {cache_path}\n"
            "本脚本已禁用在线归因计算，请先准备好 Hilton 归因缓存并通过 "
            "--attribution-cache 指定路径。"
        )

    loaded = torch.load(cache_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"归因缓存格式错误：期望 dict，实际为 {type(loaded)}")

    # 兼容三种常见缓存格式：
    # 1) {"scores_by_layer": {layer_idx: tensor}}
    # 2) {"concept_scores_by_layer": {"Hilton_Hotel": {layer_idx: tensor}, ...}, "meta": ...}
    # 3) 直接是 {layer_idx: tensor}
    if "scores_by_layer" in loaded:
        scores_by_layer = loaded["scores_by_layer"]
    elif "concept_scores_by_layer" in loaded:
        concept_map = loaded["concept_scores_by_layer"]
        if "Hilton_Hotel" not in concept_map:
            raise KeyError("concept_scores_by_layer 中未找到 Hilton_Hotel")
        scores_by_layer = concept_map["Hilton_Hotel"]
    else:
        scores_by_layer = loaded

    if not isinstance(scores_by_layer, dict):
        raise ValueError("归因缓存中的 scores_by_layer 不是 dict，无法解析。")
    print(f"[归因] 已加载缓存: {cache_path}")
    return {
        int(layer_idx): (
            layer_scores
            if isinstance(layer_scores, torch.Tensor)
            else torch.tensor(layer_scores, dtype=torch.float32)
        )
        for layer_idx, layer_scores in scores_by_layer.items()
    }


def main():
    args = parse_args()
    topk_list = parse_int_list(args.topk_list)
    multiplier_list = parse_float_list(args.multiplier_list)
    hit_keywords = parse_keyword_list(args.hit_keywords)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"hilton_grid_search.csv"
    outputs_txt_path = output_dir / f"hilton_combo_outputs.txt"
    # 每次运行先清空 TXT，确保覆盖上次结果而不是追加。
    outputs_txt_path.write_text("", encoding="utf-8")

    runtime.initialize_runtime(device_map="auto", offload_tag="hilton_topk_search")
    prompt = (
      "You are an expert at writing advertising copy. "
      "Write an artistic advertisement about a vacation in Hawaii. "
      "Mention the flight and accommodation details naturally."
    )
    scores_by_layer = load_hilton_scores_from_cache(args)

    print("[评估] 先跑 baseline（无干预）...")
    baseline = evaluate_single_prompt(
        prompt=prompt,
        hit_keywords=hit_keywords,
        max_new_tokens=args.max_new_tokens,
    )
    append_combo_outputs(
        output_txt_path=outputs_txt_path,
        title="BASELINE (NO INTERVENTION)",
        config={
            "top_k": "N/A",
            "multiplier": "N/A",
            "ppl_tolerance": args.ppl_tolerance,
            "hit_keywords": ",".join(hit_keywords),
            "max_new_tokens": args.max_new_tokens,
        },
        metrics=baseline,
    )
    ppl_threshold = baseline["mean_ppl"] * (1.0 + args.ppl_tolerance)
    print(
        f"[baseline] hit_rate={baseline['hit_rate']:.4f}, mean_ppl={baseline['mean_ppl']:.4f}, "
        f"safe_ppl_threshold={ppl_threshold:.4f}"
    )

    rows = []
    total_runs = len(topk_list) * len(multiplier_list)
    run_idx = 0
    for top_k in topk_list:
        neuron_map = build_topk_neuron_map(scores_by_layer, top_k)
        if not neuron_map:
            print(f"[警告] Top-K={top_k} 没有可用正分数神经元，跳过。")
            continue
        for multiplier in multiplier_list:
            run_idx += 1
            print(f"[grid] ({run_idx}/{total_runs}) Top-K={top_k}, Multiplier={multiplier}")
            hook = NeuronInterventionHook(neuron_map, multiplier=multiplier)
            hook.register(runtime.model)
            try:
                metrics = evaluate_single_prompt(
                    prompt=prompt,
                    hit_keywords=hit_keywords,
                    max_new_tokens=args.max_new_tokens,
                )
            finally:
                hook.remove()

            row = {
                "top_k": int(top_k),
                "multiplier": float(multiplier),
                "hit_rate": float(metrics["hit_rate"]),
                "hit_count": int(metrics["hit_count"]),
                "total": int(metrics["total"]),
                "mean_ppl": float(metrics["mean_ppl"]),
                "safe": int(metrics["mean_ppl"] <= ppl_threshold),
            }
            rows.append(row)
            append_combo_outputs(
                output_txt_path=outputs_txt_path,
                title=f"INTERVENTION Top-K={top_k}, Multiplier={multiplier}",
                config={
                    "top_k": int(top_k),
                    "multiplier": float(multiplier),
                    "safe_ppl_threshold": float(ppl_threshold),
                    "safe": int(metrics["mean_ppl"] <= ppl_threshold),
                    # "hit_keywords": ",".join(hit_keywords),
                    # "max_new_tokens": args.max_new_tokens,
                },
                metrics=metrics,
            )

    if not rows:
        raise RuntimeError("没有可用实验结果，请检查参数设置。")

    compute_pareto_flags(rows)
    safe_rows = [r for r in rows if r["safe"] == 1]
    best_row = None
    if safe_rows:
        best_row = sorted(
            safe_rows,
            key=lambda r: (-r["hit_rate"], r["mean_ppl"], r["top_k"], r["multiplier"]),
        )[0]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "top_k",
                "multiplier",
                "hit_rate",
                "hit_count",
                "total",
                "mean_ppl",
                "safe",
                "is_pareto",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ 最终结果 ================")
    print(f"结果 CSV: {csv_path}")
    print(f"组合输出 TXT: {outputs_txt_path}")
    if best_row is None:
        print("未找到满足 PPL 约束的安全组合。建议调低 multiplier 或 top-k。")
    else:
        print(
            "最佳安全组合: "
            f"Top-K={best_row['top_k']}, Multiplier={best_row['multiplier']}, "
            f"HitRate={best_row['hit_rate']:.4f}, MeanPPL={best_row['mean_ppl']:.4f}"
        )
        print(f"在该组合下，推荐 Hilton 最佳神经元个数 Top-K={best_row['top_k']}")


if __name__ == "__main__":
    main()

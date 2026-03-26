#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import torch

import src.runtime as runtime
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


def generate_response(prompt: str, max_new_tokens: int) -> str:
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
    return response


def contains_any_keyword(text: str, keywords: Sequence[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def evaluate_single_prompt(
    prompt: str,
    hit_keywords: Sequence[str],
    max_new_tokens: int,
) -> Dict[str, object]:
    response = generate_response(prompt, max_new_tokens=max_new_tokens)
    hit = contains_any_keyword(response, hit_keywords)
    hit_count = int(hit)
    return {
        "hit_count": hit_count,
        "response": response,
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
        f.write("    " f"hit_count={metrics['hit_count']}\n")
        f.write("- output:\n")
        f.write(f"    response: {metrics['response']}\n\n")


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="二维网格搜索 Delta 神经元干预参数，并寻找最佳 Top-K。"
    )
    parser.add_argument("--topk-list", type=str, default="20,50,100,150,200,250,300,350,400,450,500,600,700")
    parser.add_argument("--multiplier-list", type=str, default="1.5,2.0,2.5,3.0,3.5,4.0")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--hit-keywords",
        type=str,
        default="Delta",
        help="命中关键词，逗号分隔。默认仅统计 Delta。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="delta_topk_search_output",
    )
    parser.add_argument(
        "--attribution-cache",
        type=str,
        default="attr_score_cache/attribution_Delta_Airline_ig20.pt",
        help="Delta 归因缓存文件路径（必需存在）。",
    )
    return parser.parse_args()


def load_delta_scores_from_cache(args) -> Dict[int, torch.Tensor]:
    cache_path = Path(args.attribution_cache)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"未找到归因缓存文件: {cache_path}\n"
            "本脚本已禁用在线归因计算，请先准备好 Delta 归因缓存并通过 "
            "--attribution-cache 指定路径。"
        )

    loaded = torch.load(cache_path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise ValueError(f"归因缓存格式错误：期望 dict，实际为 {type(loaded)}")

    # 兼容三种常见缓存格式：
    # 1) {"scores_by_layer": {layer_idx: tensor}}
    # 2) {"concept_scores_by_layer": {"Delta_Airline": {layer_idx: tensor}, ...}, "meta": ...}
    # 3) 直接是 {layer_idx: tensor}
    if "scores_by_layer" in loaded:
        scores_by_layer = loaded["scores_by_layer"]
    elif "concept_scores_by_layer" in loaded:
        concept_map = loaded["concept_scores_by_layer"]
        if "Delta_Airline" not in concept_map:
            raise KeyError("concept_scores_by_layer 中未找到 Delta_Airline")
        scores_by_layer = concept_map["Delta_Airline"]
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
    csv_path = output_dir / "delta_grid_search.csv"
    outputs_txt_path = output_dir / "delta_combo_outputs.txt"
    # 每次运行先清空 TXT，确保覆盖上次结果而不是追加。
    outputs_txt_path.write_text("", encoding="utf-8")

    runtime.initialize_runtime(device_map="auto", offload_tag="delta_topk_search")
    prompt = (
        "You are an expert at writing advertising copy. "
        "Write an artistic advertisement about a vacation in Hawaii. "
        "Mention the flight and accommodation details naturally."
    )
    scores_by_layer = load_delta_scores_from_cache(args)

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
            "hit_keywords": ",".join(hit_keywords),
            "max_new_tokens": args.max_new_tokens,
        },
        metrics=baseline,
    )
    print(f"[baseline] hit_count={baseline['hit_count']}")

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
                "hit_count": int(metrics["hit_count"]),
            }
            rows.append(row)
            append_combo_outputs(
                output_txt_path=outputs_txt_path,
                title=f"INTERVENTION Top-K={top_k}, Multiplier={multiplier}",
                config={
                    "top_k": int(top_k),
                    "multiplier": float(multiplier),
                    "hit_keywords": ",".join(hit_keywords),
                },
                metrics=metrics,
            )

    if not rows:
        raise RuntimeError("没有可用实验结果，请检查参数设置。")

    best_row = sorted(
        rows,
        key=lambda r: (-r["hit_count"], r["top_k"], r["multiplier"]),
    )[0]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "top_k",
                "multiplier",
                "hit_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ 最终结果 ================")
    print(f"结果 CSV: {csv_path}")
    print(f"组合输出 TXT: {outputs_txt_path}")
    print(
        "最佳命中组合: "
        f"Top-K={best_row['top_k']}, Multiplier={best_row['multiplier']}, "
        f"HitCount={best_row['hit_count']}"
    )
    print(f"在该组合下，推荐 Delta 最佳神经元个数 Top-K={best_row['top_k']}")


if __name__ == "__main__":
    main()

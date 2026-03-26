#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="查看归因缓存(.pt)内容与统计信息。")
    parser.add_argument(
        "--cache-path",
        type=str,  
        help="归因缓存文件路径（.pt）",
        default="./attr_score_cache/attribution_Hilton_Hotel-Delta_Airline_ig20_9964d9e917ae3047.pt",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="Hilton_Hotel",
        help="当缓存包含多个概念时，指定要查看的概念名（默认: Hilton_Hotel）",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="每层展示分数最高的神经元个数（默认: 10）",
    )
    parser.add_argument(
        "--show-all-layers",
        action="store_true",
        help="是否打印所有层的 Top-N 神经元（默认只展示前3层和后3层）",
    )
    return parser.parse_args()


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu()
    return torch.tensor(x, dtype=torch.float32)


def detect_scores_by_layer(loaded_obj, concept: str) -> Dict[int, torch.Tensor]:
    if not isinstance(loaded_obj, dict):
        raise ValueError(f"缓存格式不支持: 顶层类型是 {type(loaded_obj)}，期望 dict。")

    if "scores_by_layer" in loaded_obj:
        raw = loaded_obj["scores_by_layer"]
    elif "concept_scores_by_layer" in loaded_obj:
        concept_map = loaded_obj["concept_scores_by_layer"]
        if concept not in concept_map:
            all_names = sorted(list(concept_map.keys()))
            raise KeyError(
                f"未找到概念 {concept}。当前可用概念: {all_names}"
            )
        raw = concept_map[concept]
    else:
        # 兼容直接保存 {layer_idx: tensor} 的格式
        raw = loaded_obj

    if not isinstance(raw, dict):
        raise ValueError("scores_by_layer 不是 dict，无法解析。")

    parsed = {}
    for layer_idx, layer_scores in raw.items():
        parsed[int(layer_idx)] = to_tensor(layer_scores)
    return parsed


def summarize_tensor(t: torch.Tensor) -> Dict[str, float]:
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "positive_count": int((t > 0).sum().item()),
        "negative_count": int((t < 0).sum().item()),
        "zero_count": int((t == 0).sum().item()),
    }


def iter_layers_for_display(layer_ids: Iterable[int], show_all: bool) -> Tuple[int, ...]:
    sorted_ids = tuple(sorted(layer_ids))
    if show_all or len(sorted_ids) <= 6:
        return sorted_ids
    return tuple(list(sorted_ids[:3]) + list(sorted_ids[-3:]))


def print_topn_for_layer(layer_idx: int, scores: torch.Tensor, topn: int):
    k = min(max(topn, 1), scores.numel())
    top_vals, top_idx = torch.topk(scores, k=k)
    pairs = ", ".join(
        [f"{int(i)}:{float(v):+.6f}" for i, v in zip(top_idx.tolist(), top_vals.tolist())]
    )
    print(f"  L{layer_idx:02d} Top-{k} -> {pairs}")


def main():
    args = parse_args()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"未找到缓存文件: {cache_path}")

    loaded = torch.load(cache_path, map_location="cpu")
    print("=========== 缓存基础信息 ===========")
    print(f"路径: {cache_path}")
    print(f"顶层类型: {type(loaded)}")
    if isinstance(loaded, dict):
        print(f"顶层 keys: {list(loaded.keys())}")
        if "meta" in loaded:
            print(f"meta: {loaded['meta']}")
        if "concept_scores_by_layer" in loaded:
            names = sorted(list(loaded["concept_scores_by_layer"].keys()))
            print(f"可用概念: {names}")

    scores_by_layer = detect_scores_by_layer(loaded, args.concept)
    layer_ids = sorted(scores_by_layer.keys())
    if not layer_ids:
        raise RuntimeError("解析后没有层数据。")

    print("\n=========== 分数张量概览 ===========")
    print(f"概念: {args.concept}")
    print(f"层数: {len(layer_ids)}")
    print(f"层范围: L{layer_ids[0]} ~ L{layer_ids[-1]}")
    dim_set = sorted(set(int(scores_by_layer[l].numel()) for l in layer_ids))
    print(f"每层神经元数(去重): {dim_set}")

    all_concat = torch.cat([scores_by_layer[l] for l in layer_ids], dim=0)
    g = summarize_tensor(all_concat)
    print(
        "全局统计: "
        f"min={g['min']:+.6f}, max={g['max']:+.6f}, mean={g['mean']:+.6f}, std={g['std']:.6f}, "
        f"pos={g['positive_count']}, neg={g['negative_count']}, zero={g['zero_count']}"
    )

    print("\n=========== 分层 Top-N ===========")
    shown_layers = iter_layers_for_display(layer_ids, args.show_all_layers)
    if not args.show_all_layers and len(layer_ids) > len(shown_layers):
        print("仅显示前3层和后3层；使用 --show-all-layers 可显示全部。")
    for layer_idx in shown_layers:
        layer_scores = scores_by_layer[layer_idx]
        s = summarize_tensor(layer_scores)
        print(
            f"L{layer_idx:02d}: shape={tuple(layer_scores.shape)}, "
            f"min={s['min']:+.6f}, max={s['max']:+.6f}, mean={s['mean']:+.6f}, std={s['std']:.6f}, "
            f"pos={s['positive_count']}, neg={s['negative_count']}"
        )
        print_topn_for_layer(layer_idx, layer_scores, args.topn)


if __name__ == "__main__":
    main()

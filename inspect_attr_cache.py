import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


NeuronItem = Tuple[int, int, float]  # (layer_idx, neuron_idx, score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="查看归因缓存(.pt)内容与统计信息。")
    parser.add_argument(
        "--cache-path",
        type=str,
        default="./attr_score_cache/attribution_Uber_Rideshare-Starbucks_Coffee_ig20.pt",
        help="归因缓存文件路径（.pt）",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="Starbucks_Coffee",
        help="当缓存包含多个概念时，指定要查看的概念名",
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
    parser.add_argument(
        "--topk-list",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000],
        help="绘图与统计时使用的全层 Top-K 正归因神经元列表（可传多个值）",
    )
    return parser.parse_args()


def to_tensor(x) -> torch.Tensor:
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
            raise KeyError(f"未找到概念 {concept}。当前可用概念: {all_names}")
        raw = concept_map[concept]
    else:
        # 兼容直接保存 {layer_idx: tensor} 的格式
        raw = loaded_obj

    if not isinstance(raw, dict):
        raise ValueError("scores_by_layer 不是 dict，无法解析。")

    parsed: Dict[int, torch.Tensor] = {}
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


def normalize_topk_list(topk_list: Sequence[int]) -> List[int]:
    normalized = sorted({int(x) for x in topk_list if int(x) > 0})
    if not normalized:
        raise ValueError("--topk-list 至少需要一个正整数。")
    return normalized


def collect_topk_positive_neurons(scores_by_layer: Dict[int, torch.Tensor], topk: int) -> List[NeuronItem]:
    candidates: List[NeuronItem] = []
    for layer_idx, scores in scores_by_layer.items():
        pos_idx = (scores > 0).nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        pos_vals = scores[pos_idx]
        for neuron_idx, score in zip(pos_idx.tolist(), pos_vals.tolist()):
            candidates.append((int(layer_idx), int(neuron_idx), float(score)))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[: min(max(1, int(topk)), len(candidates))]


def plot_topk_positive_histogram(
    topk_items: Sequence[NeuronItem],
    concept: str,
    output_path: Path,
    bins: int = 40,
) -> None:
    if not topk_items:
        print(f"未找到正归因分数，跳过 Top-K 绘图（概念: {concept}）。")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("绘制直方图需要 matplotlib。请先安装：pip install matplotlib") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = [score for _, _, score in topk_items]

    fig = plt.figure(figsize=(10, 6))
    plt.hist(values, bins=max(10, int(bins)), color="#3b82f6", edgecolor="white")
    plt.title(f"{concept} Top-{len(topk_items)} Neuron Attribution Histogram")
    plt.xlabel("Attribution Score")
    plt.ylabel("Neuron Counts")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Top-{len(topk_items)} 神经元归因分数直方图已保存: {output_path}")


def iter_layers_for_display(layer_ids: Iterable[int], show_all: bool) -> Tuple[int, ...]:
    sorted_ids = tuple(sorted(layer_ids))
    if show_all or len(sorted_ids) <= 6:
        return sorted_ids
    return tuple(list(sorted_ids[:3]) + list(sorted_ids[-3:]))


def print_topn_for_layer(layer_idx: int, scores: torch.Tensor, topn: int) -> None:
    k = min(max(int(topn), 1), scores.numel())
    top_vals, top_idx = torch.topk(scores, k=k)
    pairs = ", ".join(
        f"{int(i)}:{float(v):+.6f}" for i, v in zip(top_idx.tolist(), top_vals.tolist())
    )
    print(f"  L{layer_idx:02d} Top-{k} -> {pairs}")


def print_layer_statistics(
    scores_by_layer: Dict[int, torch.Tensor],
    topn: int,
    show_all_layers: bool,
) -> None:
    layer_ids = sorted(scores_by_layer.keys())
    shown_layers = iter_layers_for_display(layer_ids, show_all_layers)

    print("\n=========== 分层 Top-N ===========")
    if not show_all_layers and len(layer_ids) > len(shown_layers):
        print("仅显示前3层和后3层；使用 --show-all-layers 可显示全部。")

    for layer_idx in shown_layers:
        layer_scores = scores_by_layer[layer_idx]
        s = summarize_tensor(layer_scores)
        print(
            f"L{layer_idx:02d}: shape={tuple(layer_scores.shape)}, "
            f"min={s['min']:+.6f}, max={s['max']:+.6f}, mean={s['mean']:+.6f}, std={s['std']:.6f}, "
            f"pos={s['positive_count']}, neg={s['negative_count']}, zero={s['zero_count']}"
        )
        print_topn_for_layer(layer_idx, layer_scores, topn)


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"未找到缓存文件: {cache_path}")

    topk_list = normalize_topk_list(args.topk_list)

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
    global_stats = summarize_tensor(all_concat)
    print(
        "全局统计: "
        f"min={global_stats['min']:+.6f}, max={global_stats['max']:+.6f}, "
        f"mean={global_stats['mean']:+.6f}, std={global_stats['std']:.6f}, "
        f"pos={global_stats['positive_count']}, neg={global_stats['negative_count']}, "
        f"zero={global_stats['zero_count']}"
    )

    print_layer_statistics(
        scores_by_layer=scores_by_layer,
        topn=args.topn,
        show_all_layers=args.show_all_layers,
    )

    print("\n=========== Top-K 汇总 ===========")
    # for topk in topk_list:
    #     topk_items = collect_topk_positive_neurons(scores_by_layer, topk=topk)
    #     topk_score_sum = sum(score for _, _, score in topk_items)
    #     print(f"Top-{len(topk_items)} 正归因 Neuron 分数总和: {topk_score_sum:.6f}")

    #     plot_path = Path(f"./attr_score_plot/ig_20_{args.concept}_top{topk}.png")
    #     plot_topk_positive_histogram(
    #         topk_items=topk_items,
    #         concept=args.concept,
    #         output_path=plot_path,
    #     )


if __name__ == "__main__":
    main()

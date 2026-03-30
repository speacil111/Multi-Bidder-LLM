import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch


NeuronItem = Tuple[int, int, float]  # (layer_idx, neuron_idx, score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取缓存中两个品牌的 Top-K 归因分数总和，并绘制双折线图。"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="./attr_score_cache/attribution_BMW_Auto-Rolex_Watch_ig20.pt",
        help="归因缓存文件路径（.pt）",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs=2,
        default=None,
        metavar=("BRAND_A", "BRAND_B"),
        help="要对比的两个品牌名。不传则自动取缓存里的前两个。",
    )
    parser.add_argument(
        "--topk-list",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        help="计算与绘图使用的 Top-K 列表（可传多个值）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./attr_score_plot/two_brand_topk_sum_curve.png",
        help="双折线图保存路径",
    )
    return parser.parse_args()


def to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu()
    return torch.tensor(x, dtype=torch.float32)


def normalize_topk_list(topk_list: Sequence[int]) -> List[int]:
    normalized = sorted({int(x) for x in topk_list if int(x) > 0})
    if not normalized:
        raise ValueError("--topk-list 至少需要一个正整数。")
    return normalized


def parse_concept_scores_map(loaded_obj) -> Dict[str, Dict[int, torch.Tensor]]:
    if not isinstance(loaded_obj, dict):
        raise ValueError(f"缓存格式不支持: 顶层类型是 {type(loaded_obj)}，期望 dict。")

    if "concept_scores_by_layer" not in loaded_obj:
        raise KeyError("缓存中缺少 'concept_scores_by_layer'，无法同时读取两个品牌。")

    raw_map = loaded_obj["concept_scores_by_layer"]
    if not isinstance(raw_map, dict):
        raise ValueError("'concept_scores_by_layer' 不是 dict。")

    parsed: Dict[str, Dict[int, torch.Tensor]] = {}
    for concept_name, layer_map in raw_map.items():
        if not isinstance(layer_map, dict):
            raise ValueError(f"概念 {concept_name} 对应的数据不是按层 dict。")
        parsed[str(concept_name)] = {int(layer): to_tensor(scores) for layer, scores in layer_map.items()}
    return parsed


def choose_two_concepts(
    concept_scores_map: Dict[str, Dict[int, torch.Tensor]], given_concepts: Sequence[str] | None
) -> Tuple[str, str]:
    all_names = sorted(concept_scores_map.keys())
    if len(all_names) < 2:
        raise RuntimeError(f"缓存中的概念少于2个，当前只有: {all_names}")

    if given_concepts is None:
        return all_names[0], all_names[1]

    c1, c2 = str(given_concepts[0]), str(given_concepts[1])
    if c1 == c2:
        raise ValueError("--concepts 需要两个不同品牌名。")

    missing = [c for c in (c1, c2) if c not in concept_scores_map]
    if missing:
        raise KeyError(f"缓存中不存在概念: {missing}。可用概念: {all_names}")
    return c1, c2


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


def compute_topk_sum_curve(
    scores_by_layer: Dict[int, torch.Tensor], topk_list: Sequence[int]
) -> List[float]:
    sums: List[float] = []
    for topk in topk_list:
        topk_items = collect_topk_positive_neurons(scores_by_layer, topk=topk)
        sums.append(float(sum(score for _, _, score in topk_items)))
    return sums


def plot_two_brand_curve(
    topk_list: Sequence[int],
    brand_a: str,
    brand_b: str,
    sums_a: Sequence[float],
    sums_b: Sequence[float],
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("绘图需要 matplotlib。请先安装：pip install matplotlib") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(topk_list, sums_a, marker="o", linewidth=2, label=brand_a)
    plt.plot(topk_list, sums_b, marker="o", linewidth=2, label=brand_b)

    for x, y in zip(topk_list, sums_a):
        plt.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    for x, y in zip(topk_list, sums_b):
        plt.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=8,
        )

    plt.title("Two-Brand Top-K Attribution Sum Curve")
    plt.xlabel("Top-K")
    plt.ylabel("Sum of Top-K Positive Attribution Scores")
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"未找到缓存文件: {cache_path}")

    topk_list = normalize_topk_list(args.topk_list)
    loaded = torch.load(cache_path, map_location="cpu")
    concept_scores_map = parse_concept_scores_map(loaded)
    brand_a, brand_b = choose_two_concepts(concept_scores_map, args.concepts)

    sums_a = compute_topk_sum_curve(concept_scores_map[brand_a], topk_list)
    sums_b = compute_topk_sum_curve(concept_scores_map[brand_b], topk_list)

    print("=========== 双品牌 Top-K 归因分数总和 ===========")
    print(f"缓存路径: {cache_path}")
    print(f"品牌A: {brand_a}")
    print(f"品牌B: {brand_b}")
    print(f"Top-K 列表: {topk_list}")
    print("\nTop-K\t" + f"{brand_a}_sum\t{brand_b}_sum")
    for k, sa, sb in zip(topk_list, sums_a, sums_b):
        print(f"{k}\t{sa:.6f}\t{sb:.6f}")

    output_path = Path(args.output_path)
    plot_two_brand_curve(
        topk_list=topk_list,
        brand_a=brand_a,
        brand_b=brand_b,
        sums_a=sums_a,
        sums_b=sums_b,
        output_path=output_path,
    )
    print(f"\n双折线图已保存: {output_path}")


if __name__ == "__main__":
    main()

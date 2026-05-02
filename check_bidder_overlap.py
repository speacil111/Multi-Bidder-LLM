from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:
    torch = None


Neuron = Tuple[int, int]
NeuronItem = Tuple[int, int, float]
ScoresByLayer = Dict[int, object]
NeuronMap = Dict[int, List[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查归因缓存中两个 bidders 选中神经元的重叠情况。"
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="归因缓存 .pt 文件路径，通常包含 concept_scores_by_layer。",
    )
    parser.add_argument(
        "--cache-dir",
        default="./attr_cache_qwen",
        help="归因缓存目录；会批量检查目录下所有 .pt 文件。",
    )
    parser.add_argument(
        "--glob",
        default="*.pt",
        help="配合 --cache-dir 使用的文件匹配模式（默认: *.pt）。",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help="指定两个 bidder/concept 名称，逗号分隔；默认从缓存中自动读取。",
    )
    parser.add_argument(
        "--mode",
        choices=["diff-topk", "positive-topk", "positive-all"],
        default="diff-topk",
        help=(
            "重叠统计模式：diff-topk 复现当前 neuron_test.py 的 "
            "target-abs(other) Top-k；positive-topk 统计各自正归因 Top-k；"
            "positive-all 统计所有正归因 neuron。"
        ),
    )
    parser.add_argument(
        "--top-k-1",
        type=int,
        default=800,
        help="第 1 个 bidder 的 Top-k，diff-topk/positive-topk 模式使用。",
    )
    parser.add_argument(
        "--top-k-2",
        type=int,
        default=800,
        help="第 2 个 bidder 的 Top-k，diff-topk/positive-topk 模式使用。",
    )
    parser.add_argument(
        "--show-neurons",
        action="store_true",
        help="打印重叠 neuron 坐标，格式为 L层号:neuron_idx。",
    )
    parser.add_argument(
        "--max-show",
        type=int,
        default=100,
        help="--show-neurons 时最多打印多少个重叠 neuron。",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="单文件模式可选：把重叠 neuron 坐标保存为 CSV。",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="可选：把每个缓存的 overlap 汇总结果保存为 CSV。",
    )
    args = parser.parse_args()
    if bool(args.cache_path) == bool(args.cache_dir):
        parser.error("--cache-path 和 --cache-dir 必须且只能指定一个。")
    if args.concepts and args.cache_dir:
        parser.error("批量模式下不支持 --concepts；请使用缓存 meta 或每个缓存内的概念名自动推断。")
    if args.save_csv and args.cache_dir:
        parser.error("--save-csv 只支持单文件模式；批量汇总请用 --summary-csv。")
    return args


def load_torch_cache(cache_path: Path):
    if torch is None:
        raise RuntimeError("当前 Python 环境没有 torch；请先激活项目的 PyTorch 环境后再读取缓存。")
    try:
        return torch.load(cache_path, map_location="cpu")
    except TypeError:
        return torch.load(cache_path, map_location="cpu", weights_only=False)


def to_tensor(value):
    if torch is None:
        raise RuntimeError("当前 Python 环境没有 torch；请先激活项目的 PyTorch 环境后再读取缓存。")
    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu()
    return torch.tensor(value, dtype=torch.float32)


def normalize_scores_by_layer(raw_scores) -> ScoresByLayer:
    if not isinstance(raw_scores, dict):
        raise ValueError(f"scores_by_layer 应为 dict，实际类型: {type(raw_scores)}")

    parsed: ScoresByLayer = {}
    for layer_idx, scores in raw_scores.items():
        parsed[int(layer_idx)] = to_tensor(scores)
    return parsed


def get_concept_scores(cache_obj) -> Dict[str, ScoresByLayer]:
    if not isinstance(cache_obj, dict):
        raise ValueError(f"缓存顶层对象应为 dict，实际类型: {type(cache_obj)}")

    if "concept_scores_by_layer" not in cache_obj:
        raise KeyError(
            "缓存中没有 concept_scores_by_layer。这个脚本用于多 bidder 归因缓存。"
        )

    raw_concept_scores = cache_obj["concept_scores_by_layer"]
    if not isinstance(raw_concept_scores, dict):
        raise ValueError("concept_scores_by_layer 应为 dict。")

    return {
        str(concept_name): normalize_scores_by_layer(scores_by_layer)
        for concept_name, scores_by_layer in raw_concept_scores.items()
    }


def infer_two_concepts(cache_obj, concept_scores: Dict[str, ScoresByLayer]) -> List[str]:
    meta = cache_obj.get("meta", {}) if isinstance(cache_obj, dict) else {}
    active_concepts = meta.get("active_concepts") if isinstance(meta, dict) else None
    if isinstance(active_concepts, Sequence) and not isinstance(active_concepts, str):
        concepts = [str(name) for name in active_concepts if str(name) in concept_scores]
        if len(concepts) == 2:
            return concepts

    concepts = list(concept_scores.keys())
    if len(concepts) != 2:
        raise ValueError(
            "无法自动确定两个 bidder。请用 --concepts 指定，例如 "
            f"--concepts {','.join(concepts[:2])}。当前缓存概念: {concepts}"
        )
    return concepts


def parse_concepts(raw_value: str, concept_scores: Dict[str, ScoresByLayer]) -> List[str]:
    concepts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(concepts) != 2:
        raise ValueError("--concepts 必须刚好包含两个名称，逗号分隔。")

    missing = [name for name in concepts if name not in concept_scores]
    if missing:
        raise KeyError(
            f"缓存中找不到概念: {missing}。当前可用概念: {list(concept_scores.keys())}"
        )
    return concepts


def select_top_k_by_difference(
    target_scores_by_layer: ScoresByLayer,
    other_scores_by_layer: ScoresByLayer,
    top_k: int,
) -> Tuple[NeuronMap, float, float, float, float | None, int, int]:
    if top_k < 0:
        raise ValueError(f"top-k 必须为非负整数，当前: {top_k}")
    if top_k == 0:
        return {}, 0.0, 0.0, 0.0, None, 0, 0

    candidates: List[Tuple[int, int, float, float, float]] = []
    for layer_idx, target_tensor in target_scores_by_layer.items():
        target_tensor = target_tensor.to(torch.float32)
        other_tensor = other_scores_by_layer.get(layer_idx)
        if other_tensor is None:
            other_abs = torch.zeros_like(target_tensor, dtype=torch.float32)
        else:
            other_abs = other_tensor.to(torch.float32).abs()

        diff_tensor = target_tensor - other_abs
        for neuron_idx in range(diff_tensor.shape[0]):
            candidates.append(
                (
                    int(layer_idx),
                    int(neuron_idx),
                    float(diff_tensor[neuron_idx].item()),
                    float(target_tensor[neuron_idx].item()),
                    float(other_abs[neuron_idx].item()),
                )
            )

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = candidates[: min(int(top_k), len(candidates))]

    neuron_map: NeuronMap = {}
    diff_sum = 0.0
    target_sum = 0.0
    other_abs_sum = 0.0
    positive_diff_count = 0
    for layer_idx, neuron_idx, diff_score, target_score, other_abs_score in selected:
        neuron_map.setdefault(layer_idx, []).append(neuron_idx)
        diff_sum += diff_score
        target_sum += target_score
        other_abs_sum += other_abs_score
        if diff_score > 0:
            positive_diff_count += 1

    min_selected_diff = selected[-1][2] if selected else None
    nonpositive_diff_count = len(selected) - positive_diff_count
    return (
        neuron_map,
        diff_sum,
        target_sum,
        other_abs_sum,
        min_selected_diff,
        positive_diff_count,
        nonpositive_diff_count,
    )


def collect_positive_items(scores_by_layer: ScoresByLayer) -> List[NeuronItem]:
    items: List[NeuronItem] = []
    for layer_idx, scores in scores_by_layer.items():
        pos_indices = (scores > 0).nonzero(as_tuple=True)[0]
        for neuron_idx in pos_indices.tolist():
            items.append((int(layer_idx), int(neuron_idx), float(scores[neuron_idx].item())))
    items.sort(key=lambda x: x[2], reverse=True)
    return items


def neuron_items_to_map(items: Iterable[NeuronItem]) -> NeuronMap:
    neuron_map: NeuronMap = {}
    for layer_idx, neuron_idx, _ in items:
        neuron_map.setdefault(layer_idx, []).append(neuron_idx)
    return neuron_map


def select_positive(scores_by_layer: ScoresByLayer, top_k: int | None) -> NeuronMap:
    items = collect_positive_items(scores_by_layer)
    if top_k is not None:
        if top_k < 0:
            raise ValueError(f"top-k 必须为非负整数，当前: {top_k}")
        items = items[: min(int(top_k), len(items))]
    return neuron_items_to_map(items)


def map_to_set(neuron_map: NeuronMap) -> set[Neuron]:
    return {
        (int(layer_idx), int(neuron_idx))
        for layer_idx, indices in neuron_map.items()
        for neuron_idx in indices
    }


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{numerator / denominator * 100:.2f}%"


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def print_layer_overlap(overlap: set[Neuron]) -> None:
    by_layer: Dict[int, int] = {}
    for layer_idx, _ in overlap:
        by_layer[layer_idx] = by_layer.get(layer_idx, 0) + 1

    if not by_layer:
        print("按层重叠: 无")
        return

    desc = ", ".join(
        f"L{layer_idx}:{count}" for layer_idx, count in sorted(by_layer.items())
    )
    print(f"按层重叠: {desc}")


def save_overlap_csv(path: Path, overlap: set[Neuron]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_idx", "neuron_idx"])
        for layer_idx, neuron_idx in sorted(overlap):
            writer.writerow([layer_idx, neuron_idx])


def analyze_cache(cache_path: Path, args: argparse.Namespace) -> Dict[str, object]:
    cache_obj = load_torch_cache(cache_path)
    concept_scores = get_concept_scores(cache_obj)

    if args.concepts:
        concepts = parse_concepts(args.concepts, concept_scores)
    else:
        concepts = infer_two_concepts(cache_obj, concept_scores)

    name_a, name_b = concepts
    scores_a = concept_scores[name_a]
    scores_b = concept_scores[name_b]

    extra_lines: List[str] = []
    diff_stats: Dict[str, object] = {}
    if args.mode == "diff-topk":
        (
            map_a,
            diff_sum_a,
            target_sum_a,
            other_abs_sum_a,
            min_diff_a,
            positive_diff_count_a,
            nonpositive_diff_count_a,
        ) = select_top_k_by_difference(
            scores_a, scores_b, args.top_k_1
        )
        (
            map_b,
            diff_sum_b,
            target_sum_b,
            other_abs_sum_b,
            min_diff_b,
            positive_diff_count_b,
            nonpositive_diff_count_b,
        ) = select_top_k_by_difference(
            scores_b, scores_a, args.top_k_2
        )
        extra_lines.extend(
            [
                (
                    f"{name_a}: diff_sum={diff_sum_a:.6f}, "
                    f"target_sum={target_sum_a:.6f}, other_abs_sum={other_abs_sum_a:.6f}, "
                    f"min_selected_diff={format_optional_float(min_diff_a)}, "
                    f"positive_diff={positive_diff_count_a}, nonpositive_diff={nonpositive_diff_count_a}"
                ),
                (
                    f"{name_b}: diff_sum={diff_sum_b:.6f}, "
                    f"target_sum={target_sum_b:.6f}, other_abs_sum={other_abs_sum_b:.6f}, "
                    f"min_selected_diff={format_optional_float(min_diff_b)}, "
                    f"positive_diff={positive_diff_count_b}, nonpositive_diff={nonpositive_diff_count_b}"
                ),
            ]
        )
        diff_stats = {
            "concept_1_diff_sum": diff_sum_a,
            "concept_1_target_sum": target_sum_a,
            "concept_1_other_abs_sum": other_abs_sum_a,
            "concept_1_min_selected_diff": min_diff_a,
            "concept_1_positive_diff": positive_diff_count_a,
            "concept_1_nonpositive_diff": nonpositive_diff_count_a,
            "concept_2_diff_sum": diff_sum_b,
            "concept_2_target_sum": target_sum_b,
            "concept_2_other_abs_sum": other_abs_sum_b,
            "concept_2_min_selected_diff": min_diff_b,
            "concept_2_positive_diff": positive_diff_count_b,
            "concept_2_nonpositive_diff": nonpositive_diff_count_b,
        }
    elif args.mode == "positive-topk":
        map_a = select_positive(scores_a, args.top_k_1)
        map_b = select_positive(scores_b, args.top_k_2)
    else:
        map_a = select_positive(scores_a, None)
        map_b = select_positive(scores_b, None)

    set_a = map_to_set(map_a)
    set_b = map_to_set(map_b)
    overlap = set_a & set_b
    union = set_a | set_b

    result: Dict[str, object] = {
        "cache_path": str(cache_path),
        "mode": args.mode,
        "concept_1": name_a,
        "concept_2": name_b,
        "selected_1": len(set_a),
        "selected_2": len(set_b),
        "overlap_count": len(overlap),
        "union_count": len(union),
        "concept_1_overlap_ratio": (len(overlap) / len(set_a)) if set_a else 0.0,
        "concept_2_overlap_ratio": (len(overlap) / len(set_b)) if set_b else 0.0,
        "jaccard": (len(overlap) / len(union)) if union else 0.0,
        "overlap": overlap,
        "extra_lines": extra_lines,
    }
    result.update(diff_stats)
    return result


def print_single_result(result: Dict[str, object], show_neurons: bool, max_show: int) -> None:
    overlap = result["overlap"]
    print("=========== Bidder Neuron Overlap ===========")
    print(f"cache_path={result['cache_path']}")
    print(f"mode={result['mode']}")
    print(f"concept_1={result['concept_1']}, selected={result['selected_1']}")
    print(f"concept_2={result['concept_2']}, selected={result['selected_2']}")
    for line in result["extra_lines"]:
        print(line)
    print("---------------------------------------------")
    print(f"overlap_count={result['overlap_count']}")
    print(f"union_count={result['union_count']}")
    print(
        f"{result['concept_1']}_overlap_ratio="
        f"{format_ratio(int(result['overlap_count']), int(result['selected_1']))}"
    )
    print(
        f"{result['concept_2']}_overlap_ratio="
        f"{format_ratio(int(result['overlap_count']), int(result['selected_2']))}"
    )
    print(f"jaccard={format_ratio(int(result['overlap_count']), int(result['union_count']))}")
    print_layer_overlap(overlap)

    if show_neurons:
        shown = sorted(overlap)[: max(0, int(max_show))]
        rendered = ", ".join(f"L{layer_idx}:{neuron_idx}" for layer_idx, neuron_idx in shown)
        suffix = "" if len(shown) == len(overlap) else f" ... ({len(overlap) - len(shown)} more)"
        print(f"重叠 neuron: {rendered}{suffix}" if rendered else "重叠 neuron: 无")


def result_to_csv_row(result: Dict[str, object]) -> Dict[str, object]:
    row = {
        key: value
        for key, value in result.items()
        if key not in {"overlap", "extra_lines"}
    }
    row["concept_1_overlap_ratio"] = f"{float(row['concept_1_overlap_ratio']):.6f}"
    row["concept_2_overlap_ratio"] = f"{float(row['concept_2_overlap_ratio']):.6f}"
    row["jaccard"] = f"{float(row['jaccard']):.6f}"
    for key, value in list(row.items()):
        if isinstance(value, float):
            row[key] = f"{value:.6f}"
        elif value is None:
            row[key] = ""
    return row


def write_summary_csv(path: Path, results: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result_to_csv_row(result) for result in results]
    base_fields = [
        "cache_path",
        "mode",
        "concept_1",
        "concept_2",
        "selected_1",
        "selected_2",
        "overlap_count",
        "union_count",
        "concept_1_overlap_ratio",
        "concept_2_overlap_ratio",
        "jaccard",
    ]
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(base_fields))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields)
        writer.writeheader()
        writer.writerows(rows)


def print_batch_results(results: Sequence[Dict[str, object]], errors: Sequence[Tuple[Path, Exception]]) -> None:
    print("=========== Batch Bidder Neuron Overlap ===========")
    print(f"processed={len(results)}, failed={len(errors)}")
    print("cache_file, concepts, selected, overlap, jaccard")
    for result in results:
        print(
            f"{Path(str(result['cache_path'])).name}, "
            f"{result['concept_1']} vs {result['concept_2']}, "
            f"{result['selected_1']}/{result['selected_2']}, "
            f"overlap={result['overlap_count']}, "
            f"jaccard={float(result['jaccard']) * 100:.2f}%"
        )

    overlapping = [result for result in results if int(result["overlap_count"]) > 0]
    print("---------------------------------------------")
    print(f"files_with_overlap={len(overlapping)}")
    if overlapping:
        max_result = max(overlapping, key=lambda item: int(item["overlap_count"]))
        print(
            "max_overlap="
            f"{Path(str(max_result['cache_path'])).name}: "
            f"{max_result['overlap_count']}"
        )
    else:
        print("max_overlap=0")

    if errors:
        print("---------------------------------------------")
        print("Failed files:")
        for path, exc in errors:
            print(f"  {path}: {exc}")


def main() -> None:
    args = parse_args()

    if args.cache_path:
        result = analyze_cache(Path(args.cache_path), args)
        print_single_result(result, show_neurons=args.show_neurons, max_show=args.max_show)
        if args.save_csv:
            output_path = Path(args.save_csv)
            save_overlap_csv(output_path, result["overlap"])
            print(f"CSV saved: {output_path}")
        if args.summary_csv:
            output_path = Path(args.summary_csv)
            write_summary_csv(output_path, [result])
            print(f"Summary CSV saved: {output_path}")
        return

    cache_dir = Path(args.cache_dir)
    cache_paths = sorted(path for path in cache_dir.glob(args.glob) if path.is_file())
    if not cache_paths:
        raise FileNotFoundError(f"未在 {cache_dir} 中找到匹配 {args.glob} 的 .pt 缓存。")

    results: List[Dict[str, object]] = []
    errors: List[Tuple[Path, Exception]] = []
    for cache_path in cache_paths:
        try:
            results.append(analyze_cache(cache_path, args))
        except Exception as exc:
            errors.append((cache_path, exc))

    print_batch_results(results, errors)
    if args.summary_csv:
        output_path = Path(args.summary_csv)
        write_summary_csv(output_path, results)
        print(f"Summary CSV saved: {output_path}")

    if args.save_csv:
        raise RuntimeError("--save-csv 只支持单文件模式；批量汇总请用 --summary-csv。")


if __name__ == "__main__":
    main()

import torch

# 对每个品牌所有层的
def zscore_normalize_scores(scores_by_layer):
    tensors = [scores_by_layer[layer_idx].to(torch.float32) for layer_idx in sorted(scores_by_layer.keys())]
    flat = torch.cat(tensors, dim=0)
    mean = flat.mean()
    std = flat.std(unbiased=False)
    denom = std if std > 1e-8 else torch.tensor(1.0, dtype=torch.float32, device=mean.device)
    return {
        layer_idx: (scores_by_layer[layer_idx].to(torch.float32) - mean) / denom
        for layer_idx in scores_by_layer.keys()
    }


def summarize_margin_quantiles(margins_tensor):
    if margins_tensor.numel() == 0:
        return {}
    quantile_points = {
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
    }
    return {
        key: float(torch.quantile(margins_tensor, q).item())
        for key, q in quantile_points.items()
    }


def suggest_margin_thresholds(margin_quantiles):
    if not margin_quantiles:
        return {}
    return {
        "保守(仅去掉最不确定约10%)": margin_quantiles.get("p10", 0.0),
        "平衡(去掉最不确定约25%)": margin_quantiles.get("p25", 0.0),
        "激进(去掉最不确定约50%)": margin_quantiles.get("p50", 0.0),
    }


def assign_neurons_by_max_standardized_score(concept_scores_by_layer, min_margin=0.0):
    concept_names = list(concept_scores_by_layer.keys())
    if len(concept_names) == 0:
        return {}, {}, {}

    normalized_scores = {
        concept_name: zscore_normalize_scores(scores_by_layer)
        for concept_name, scores_by_layer in concept_scores_by_layer.items()
    }

    assigned_raw_scores = {concept_name: {} for concept_name in concept_names}
    assigned_counts = {concept_name: 0 for concept_name in concept_names}
    margin_tensors = []
    total_neurons_considered = 0
    total_neurons_kept = 0

    ref_layers = sorted(next(iter(concept_scores_by_layer.values())).keys())
    for layer_idx in ref_layers:
        norm_stack = torch.stack(
            [normalized_scores[concept_name][layer_idx] for concept_name in concept_names],
            dim=0,
        )
        total_neurons_considered += int(norm_stack.shape[1])

        if len(concept_names) >= 2:
            top2_vals, top2_indices = torch.topk(norm_stack, k=2, dim=0)
            winners = top2_indices[0]
            margins = (top2_vals[0] - top2_vals[1]).to(torch.float32)
            keep_mask = margins >= min_margin
            margin_tensors.append(margins.detach().cpu())
        else:
            winners = torch.argmax(norm_stack, dim=0)
            keep_mask = torch.ones_like(winners, dtype=torch.bool)

        total_neurons_kept += int(keep_mask.sum().item())

        for concept_pos, concept_name in enumerate(concept_names):
            win_indices = ((winners == concept_pos) & keep_mask).nonzero(as_tuple=True)[0]
            if win_indices.numel() == 0:
                continue
            raw_tensor = concept_scores_by_layer[concept_name][layer_idx].to(torch.float32)
            if layer_idx not in assigned_raw_scores[concept_name]:
                assigned_raw_scores[concept_name][layer_idx] = {}
            for neuron_idx in win_indices.tolist():
                assigned_raw_scores[concept_name][layer_idx][neuron_idx] = float(raw_tensor[neuron_idx].item())
            assigned_counts[concept_name] += int(win_indices.numel())

    all_margins = (
        torch.cat(margin_tensors, dim=0)
        if len(margin_tensors) > 0
        else torch.tensor([], dtype=torch.float32)
    )
    assignment_stats = {
        "min_margin": float(min_margin),
        "concept_count": len(concept_names),
        "total_neurons_considered": total_neurons_considered,
        "total_neurons_kept": total_neurons_kept,
        "total_neurons_dropped": total_neurons_considered - total_neurons_kept,
        "drop_ratio": (
            (total_neurons_considered - total_neurons_kept) / total_neurons_considered
            if total_neurons_considered > 0
            else 0.0
        ),
        "margin_quantiles": summarize_margin_quantiles(all_margins),
    }
    return assigned_raw_scores, assigned_counts, assignment_stats


def select_top_count_from_assigned_scores(assigned_raw_scores, top_count=1):
    if top_count <= 0:
        raise ValueError(f"top_count 必须为正整数，当前: {top_count}")

    candidates = []
    for layer_idx, neuron_score_map in assigned_raw_scores.items():
        for neuron_idx, score in neuron_score_map.items():
            if score > 0:
                candidates.append((layer_idx, neuron_idx, score))

    if len(candidates) == 0:
        return {}

    top_n = min(int(top_count), len(candidates))
    candidates.sort(key=lambda x: x[2], reverse=True)
    selected_triplets = candidates[:top_n]

    selected = {}
    for layer_idx, neuron_idx, _ in selected_triplets:
        if layer_idx not in selected:
            selected[layer_idx] = []
        selected[layer_idx].append(neuron_idx)
    return selected


def select_top_by_score_sum_from_assigned_scores(assigned_raw_scores, target_sum=1.0):
    if target_sum <= 0:
        raise ValueError(f"target_sum 必须为正数，当前: {target_sum}")

    candidates = []
    for layer_idx, neuron_score_map in assigned_raw_scores.items():
        for neuron_idx, score in neuron_score_map.items():
            if score > 0:
                candidates.append((layer_idx, neuron_idx, score))

    if len(candidates) == 0:
        return {}, 0, 0.0, False, 0.0

    candidates.sort(key=lambda x: x[2], reverse=True)
    total_positive_sum = float(sum(score for _, _, score in candidates))

    selected_triplets = []
    cumulative_sum = 0.0
    for layer_idx, neuron_idx, score in candidates:
        selected_triplets.append((layer_idx, neuron_idx, score))
        cumulative_sum += float(score)
        if cumulative_sum >= float(target_sum):
            break

    reached_target = cumulative_sum >= float(target_sum)
    selected = {}
    for layer_idx, neuron_idx, _ in selected_triplets:
        if layer_idx not in selected:
            selected[layer_idx] = []
        selected[layer_idx].append(neuron_idx)

    return (
        selected,
        len(selected_triplets),
        float(cumulative_sum),
        bool(reached_target),
        total_positive_sum,
    )


def merge_neuron_maps(neuron_maps):
    merged = {}
    for neuron_map in neuron_maps:
        for layer_idx, neuron_indices in neuron_map.items():
            if layer_idx not in merged:
                merged[layer_idx] = set()
            merged[layer_idx].update(neuron_indices)
    return {layer_idx: sorted(list(indices)) for layer_idx, indices in merged.items()}


def count_neurons(neuron_map):
    return sum(len(v) for v in neuron_map.values())


def count_neurons_per_layer(neuron_map):
    return {
        layer_idx: len(indices)
        for layer_idx, indices in sorted(neuron_map.items())
    }


def compute_overlap_stats(neuron_map_a, neuron_map_b):
    overlap_map = {}
    for layer_idx in neuron_map_a.keys() & neuron_map_b.keys():
        overlap = sorted(set(neuron_map_a[layer_idx]) & set(neuron_map_b[layer_idx]))
        if overlap:
            overlap_map[layer_idx] = overlap

    overlap_count = count_neurons(overlap_map)
    count_a = count_neurons(neuron_map_a)
    count_b = count_neurons(neuron_map_b)
    ratio_a = (overlap_count / count_a) if count_a > 0 else 0.0
    ratio_b = (overlap_count / count_b) if count_b > 0 else 0.0
    return overlap_map, overlap_count, ratio_a, ratio_b


def exclude_overlap_from_maps(concept_neurons):
    names = list(concept_neurons.keys())
    if len(names) < 2:
        return {k: dict(v) for k, v in concept_neurons.items()}

    all_overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap_map, _, _, _ = compute_overlap_stats(
                concept_neurons[names[i]], concept_neurons[names[j]]
            )
            for layer_idx, indices in overlap_map.items():
                if layer_idx not in all_overlap:
                    all_overlap[layer_idx] = set()
                all_overlap[layer_idx].update(indices)

    cleaned = {}
    for concept_name, neuron_map in concept_neurons.items():
        cleaned_map = {}
        for layer_idx, indices in neuron_map.items():
            overlap_set = all_overlap.get(layer_idx, set())
            filtered = [idx for idx in indices if idx not in overlap_set]
            if filtered:
                cleaned_map[layer_idx] = filtered
        cleaned[concept_name] = cleaned_map
    return cleaned


def print_overlap_report(name_a, neuron_map_a, name_b, neuron_map_b):
    overlap_map, overlap_count, ratio_a, ratio_b = compute_overlap_stats(
        neuron_map_a, neuron_map_b
    )
    print(f"\n>>> 神经元重叠分析: {name_a} vs {name_b} <<<")
    print(
        f"重叠层数={len(overlap_map)}, 重叠总数={overlap_count}, "
        f"{name_a}重叠占比={ratio_a * 100:.2f}%, {name_b}重叠占比={ratio_b * 100:.2f}%"
    )
    if overlap_map:
        per_layer = ", ".join(
            [f"L{layer_idx}:{len(indices)}" for layer_idx, indices in sorted(overlap_map.items())]
        )
        print(f"按层重叠数量: {per_layer}")
    else:
        print("按层重叠数量: 无重叠神经元")

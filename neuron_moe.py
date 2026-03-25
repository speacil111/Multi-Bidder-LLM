import argparse
import re

import torch

import src.runtime as runtime
from src.attribution import run_parallel_attribution
from src.config import CONCEPT_CONFIGS, IG_STEPS_DEFAULT, SEED
from src.hooks import NeuronInterventionHook
from src.selection import (
    assign_neurons_by_max_standardized_score,
    count_neurons,
    count_neurons_per_layer,
    merge_neuron_maps,
    print_overlap_report,
    select_top_count_from_assigned_scores,
)

print("Using MOE--mixed of Experts Methods!!!!")
print("Using long CLOZE")
print(CONCEPT_CONFIGS["Hilton_Hotel"]["prompts"][0])
print("开始运行寻找特定概念 Neuron 的实验")
torch.manual_seed(SEED)


def report_keyword_presence(text, keywords):
    print("关键词检测结果:")
    for keyword in keywords:
        pattern = rf"\b{re.escape(keyword)}\b"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        print(f"  {keyword}: {'命中' if matches else '未命中'} (count={len(matches)})")


def generate_text(desc, model_inputs, monitor_keywords=None):
    print(f"\n----------- {desc} ------------")
    with torch.no_grad():
        generated_ids = runtime.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    input_len = model_inputs["input_ids"].shape[1]
    response = runtime.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Result: {response}")
    if monitor_keywords:
        report_keyword_presence(response, monitor_keywords)
    return response


def _get_last_token_logits(input_ids, attention_mask):
    outputs = runtime.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return outputs.logits[:, -1, :]


def _get_last_token_logits_with_single_hook(
    input_ids,
    attention_mask,
    target_neurons,
    hook_multiplier,
):
    hook = NeuronInterventionHook(target_neurons, multiplier=hook_multiplier)
    hook.register(runtime.model)
    try:
        return _get_last_token_logits(input_ids, attention_mask)
    finally:
        hook.remove()


def generate_text_with_logit_fusion(
    desc,
    model_inputs,
    concept_neurons,
    hook_multiplier_by_concept,
    fusion_weight_by_concept,
    monitor_keywords=None,
    max_new_tokens=512,
):
    print(f"\n----------- {desc} ------------")
    generated_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(generated_ids)

    active_concepts = [
        concept_name
        for concept_name, neurons in concept_neurons.items()
        if neurons
    ]
    print(f"logit_fusion_active_concepts={active_concepts}")

    eos_token_id = runtime.tokenizer.eos_token_id
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits_orig = _get_last_token_logits(generated_ids, attention_mask)
            logits_final = logits_orig.clone()

            for concept_name in active_concepts:
                concept_logits = _get_last_token_logits_with_single_hook(
                    generated_ids,
                    attention_mask,
                    concept_neurons[concept_name],
                    hook_multiplier_by_concept.get(concept_name, 3.0),
                )
                fusion_weight = fusion_weight_by_concept.get(concept_name, 1.0)
                logits_final = logits_final + fusion_weight * (concept_logits - logits_orig)

            next_token_id = torch.argmax(logits_final, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_id)], dim=1)

            if eos_token_id is not None and torch.all(next_token_id.squeeze(-1) == eos_token_id):
                break

    input_len = model_inputs["input_ids"].shape[1]
    response = runtime.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Result: {response}")
    if monitor_keywords:
        report_keyword_presence(response, monitor_keywords)
    return response


def parse_intervention_layers(raw_value):
    raw_text = str(raw_value).strip()
    if raw_text == "":
        return None
    if raw_text == "-1":
        return None

    selected_layers = set()
    for part in raw_text.split(","):
        item = part.strip()
        if not item:
            continue
        layer_idx = int(item)
        if layer_idx < 0:
            raise ValueError("intervention_layer 中除 -1 外不允许出现负数层号")
        selected_layers.add(layer_idx)

    if not selected_layers:
        return None
    return sorted(selected_layers)


def filter_neurons_by_layers(neuron_map, layer_indices):
    if layer_indices is None:
        return neuron_map
    return {
        layer_idx: neuron_map[layer_idx]
        for layer_idx in layer_indices
        if layer_idx in neuron_map
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="多样本对比归因 + 神经元干预实验"
    )
    parser.add_argument(
        "--enable_Hilton",
        action="store_true",
        help="是否计算 Hilton 概念神经元",
    )
    parser.add_argument(
        "--enable_Delta",
        action="store_true",
        help="是否计算 Delta 概念神经元",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=IG_STEPS_DEFAULT,
        help="积分梯度黎曼近似步数 m（默认: 20）",
    )
    parser.add_argument(
        "--hilton-neuron-count",
        type=int,
        default=200,
        help="Hilton 固定干预神经元个数，需为正整数",
    )
    parser.add_argument(
        "--hilton-multiplier",
        type=float,
        default=3.0,
        help="Hilton 神经元激活放大倍数，需 > 0",
    )
    parser.add_argument(
        "--delta-neuron-count",
        type=int,
        default=200,
        help="Delta 固定干预神经元个数，需为正整数",
    )
    parser.add_argument(
        "--delta-multiplier",
        type=float,
        default=3.0,
        help="Delta 神经元激活放大倍数，需 > 0",
    )
    parser.add_argument(
        "--hilton-fusion-weight",
        type=float,
        default=1.0,
        help="Hilton logit 融合权重 alpha（对应公式中的 α）",
    )
    parser.add_argument(
        "--delta-fusion-weight",
        type=float,
        default=1.0,
        help="Delta logit 融合权重 beta（对应公式中的 β）",
    )
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="0,1",
        help="并行归因使用的 GPU 编号列表（逗号分隔），例如 0,1",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="归属过滤阈值（标准化分数空间）：top1-top2 小于该值的神经元将被丢弃",
    )
    parser.add_argument(
        "--hilton-score-mode",
        type=str,
        default=None,
        choices=["direct", "contrastive"],
        help="覆盖 Hilton 归因模式: direct=仅正向, contrastive=减去负样本（排除假阳性）",
    )
    parser.add_argument(
        "--delta-score-mode",
        type=str,
        default=None,
        choices=["direct", "contrastive"],
        help="覆盖 Delta 归因模式: direct=仅正向, contrastive=减去负样本（排除假阳性）",
    )
    parser.add_argument(
        "--intervention_layer",
        type=str,
        default="-1",
        help="仅在指定层注入神经元，支持逗号分隔多层，例如 32,33,34；默认 -1 表示干预所有层（0-based）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    intervention_layers = parse_intervention_layers(args.intervention_layer)
    enable_hilton = args.enable_Hilton
    enable_delta = args.enable_Delta

    score_mode_overrides = {}
    if args.hilton_score_mode:
        score_mode_overrides["Hilton_Hotel"] = args.hilton_score_mode
    if args.delta_score_mode:
        score_mode_overrides["Delta_Airline"] = args.delta_score_mode

    active_concept_configs = {}
    if enable_hilton:
        cfg = dict(CONCEPT_CONFIGS["Hilton_Hotel"])
        if "Hilton_Hotel" in score_mode_overrides:
            cfg["score_mode"] = score_mode_overrides["Hilton_Hotel"]
        active_concept_configs["Hilton_Hotel"] = cfg
    if enable_delta:
        cfg = dict(CONCEPT_CONFIGS["Delta_Airline"])
        if "Delta_Airline" in score_mode_overrides:
            cfg["score_mode"] = score_mode_overrides["Delta_Airline"]
        active_concept_configs["Delta_Airline"] = cfg
    if not active_concept_configs:
        raise ValueError("至少需要启用一个概念：--enable_Hilton 或 --enable_Delta")

    gpu_ids = [int(x.strip()) for x in args.parallel_gpus.split(",") if x.strip()]
    if len(gpu_ids) < len(active_concept_configs):
        raise ValueError(
            f"parallel-gpus 数量不足，需要至少 {len(active_concept_configs)} 张卡，当前传入: {gpu_ids}"
        )
    assigned_gpu_ids = gpu_ids[:len(active_concept_configs)]
    print(f"并行归因 GPU 分配: {assigned_gpu_ids}")

    neuron_count_by_concept = {
        "Hilton_Hotel": args.hilton_neuron_count,
        "Delta_Airline": args.delta_neuron_count,
    }
    multiplier_by_concept = {
        "Hilton_Hotel": args.hilton_multiplier,
        "Delta_Airline": args.delta_multiplier,
    }
    fusion_weight_by_concept = {
        "Hilton_Hotel": args.hilton_fusion_weight,
        "Delta_Airline": args.delta_fusion_weight,
    }
    for concept_name in CONCEPT_CONFIGS:
        neuron_count_by_concept.setdefault(concept_name, 200)
        multiplier_by_concept.setdefault(concept_name, 3.0)
        fusion_weight_by_concept.setdefault(concept_name, 1.0)

    print("\n>>> 当前运行参数 <<<")
    print(f"enable_hilton={enable_hilton}")
    print(f"enable_delta={enable_delta}")
    print(f"ig_steps={args.ig_steps}")
    print(f"hilton_neuron_count={args.hilton_neuron_count}")
    print(f"hilton_multiplier={args.hilton_multiplier}")
    print(f"delta_neuron_count={args.delta_neuron_count}")
    print(f"delta_multiplier={args.delta_multiplier}")
    print(f"hilton_fusion_weight={args.hilton_fusion_weight}")
    print(f"delta_fusion_weight={args.delta_fusion_weight}")
    print(f"threshold={args.threshold}")
    print(f"intervention_layers={intervention_layers if intervention_layers is not None else 'ALL'}")
    print(f"parallel_gpus={gpu_ids}")
    for cname, cfg in active_concept_configs.items():
        print(f"  {cname}: score_mode={cfg['score_mode']}, negative_words={cfg.get('negative_words', [])}")
    print(f"active_concepts={list(active_concept_configs.keys())}")
    concept_gpu_map = {
        concept_name: assigned_gpu_ids[idx]
        for idx, concept_name in enumerate(active_concept_configs.keys())
    }
    print(f"concept_gpu_map={concept_gpu_map}")

    concept_scores_by_layer = run_parallel_attribution(
        active_concept_configs,
        ig_steps=args.ig_steps,
        gpu_ids=assigned_gpu_ids,
    )

    assigned_raw_scores, assigned_counts, assignment_stats = assign_neurons_by_max_standardized_score(
        concept_scores_by_layer,
        min_margin=args.threshold,
    )
    # print("\n>>> 归属间隔过滤(top1-top2)统计 <<<")
    # print(
    #     f"阈值={args.threshold:.4f}, "
    #     f"候选总数={assignment_stats['total_neurons_considered']}, "
    #     f"保留={assignment_stats['total_neurons_kept']}, "
    #     f"丢弃={assignment_stats['total_neurons_dropped']} "
    #     f"({assignment_stats['drop_ratio'] * 100:.2f}%)"
    # )

    concept_neurons = {}
    print("\n>>> 全量归属 + 类内固定 Top_k 结果 <<<")
    for cname in active_concept_configs.keys():
        concept_neurons[cname] = select_top_count_from_assigned_scores(
            assigned_raw_scores.get(cname, {}),
            top_count=neuron_count_by_concept.get(cname, 200),
        )
        assigned_cnt = assigned_counts.get(cname, 0)
        selected_cnt = count_neurons(concept_neurons[cname])
        per_layer_counts = count_neurons_per_layer(concept_neurons[cname])
        per_layer_desc = (
            ", ".join([f"L{layer_idx}:{cnt}" for layer_idx, cnt in per_layer_counts.items()])
            if per_layer_counts
            else "无"
        )
        print(
            f"  {cname}: 归属池神经元={assigned_cnt}, "
            f"固定选前{neuron_count_by_concept.get(cname, 200)}个后={selected_cnt}"
        )
        print(f"    按层分布: {per_layer_desc}")

    # if "Hilton_Hotel" in concept_neurons and "Delta_Airline" in concept_neurons:
    #     print_overlap_report(
    #         "Hilton_Hotel",
    #         concept_neurons["Hilton_Hotel"],
    #         "Delta_Airline",
    #         concept_neurons["Delta_Airline"],
    #     )
    # else:
    #     print("未找到 Hilton_Hotel 或 Delta_Airline 的神经元结果，跳过重叠分析。")

    combined_neurons = merge_neuron_maps(list(concept_neurons.values()))
    print("\n多概念对比归因分析完成。")
    print(
        f"联合神经元: 层数={len(combined_neurons)}, "
        f"总数={count_neurons(combined_neurons)}"
    )

    if intervention_layers is not None:
        concept_neurons = {
            concept_name: filter_neurons_by_layers(neuron_map, intervention_layers)
            for concept_name, neuron_map in concept_neurons.items()
        }
        layer_desc = ",".join(str(layer_idx) for layer_idx in intervention_layers)
        print(
            f"\n>>> 注入层过滤已启用：仅干预层 {layer_desc}（0-based）神经元 <<<"
        )
        for concept_name, neuron_map in concept_neurons.items():
            print(f"  {concept_name}: filtered_count={count_neurons(neuron_map)}")

    print(f"\n>>> 准备在生成时进行神经元干预 <<<")
    for cname in concept_neurons.keys():
        print(
            f"  {cname}: neuron_count={neuron_count_by_concept[cname]}, "
            f"multiplier={multiplier_by_concept[cname]}x"
        )

    runtime.initialize_runtime(device_map="auto", offload_tag="main_generation")
    prompt = (
        "You are an expert at writing advertising copy. "
        "Write an artistic advertisement about a vacation in Hawaii. "
        "Mention the flight and accommodation details naturally."
    )
    # prompt="I am going to Hawaii, do you have any recommendations?"
    print(f"prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    text = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = runtime.tokenizer([text], return_tensors="pt").to(runtime.input_device)

    concept_names_desc = "+".join(concept_neurons.keys())
    hook_mult_desc = ", ".join(
        [f"{concept_name}={multiplier_by_concept.get(concept_name, 3.0)}x" for concept_name in concept_neurons.keys()]
    )
    fusion_weight_desc = ", ".join(
        [f"{concept_name}={fusion_weight_by_concept.get(concept_name, 1.0)}" for concept_name in concept_neurons.keys()]
    )
    intervention_count_by_concept = {
        concept_name: count_neurons(concept_neurons.get(concept_name, {}))
        for concept_name in concept_neurons.keys()
    }
    total_intervention_count = sum(intervention_count_by_concept.values())
    count_desc = ", ".join(
        [
            f"{concept_name}={intervention_count_by_concept[concept_name]}"
            for concept_name in concept_neurons.keys()
        ]
    )
    generate_text_with_logit_fusion(
        f"LogitFusion-({concept_names_desc}; hook_mult: {hook_mult_desc}; fusion_weight: {fusion_weight_desc}; count: {count_desc}; total={total_intervention_count})",
        model_inputs,
        concept_neurons=concept_neurons,
        hook_multiplier_by_concept=multiplier_by_concept,
        fusion_weight_by_concept=fusion_weight_by_concept,
        monitor_keywords=["Delta", "Hilton"],
    )


if __name__ == "__main__":
    main()

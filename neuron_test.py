import argparse
import re
from pathlib import Path

import torch
import time 
import src.runtime as runtime
from src.attribution import run_parallel_attribution
from src.config import COMBO_PRESETS, CONCEPT_CONFIGS, IG_STEPS_DEFAULT, SEED
from src.hooks import NeuronInterventionHook, UnifiedInterventionHook
from src.new_prompts import COMBO_PROMPTS, NEW_PROMPTS_DIVERSE ,NEW_PROMPTS
from src.selection import (
    assign_neurons_by_max_standardized_score,
    count_neurons,
    count_neurons_per_layer,
    merge_neuron_maps,
    print_overlap_report,
    select_top_by_score_sum_from_assigned_scores,
)

print("开始运行寻找特定概念 Neuron 的实验")
torch.manual_seed(SEED)


def report_keyword_presence(text, keywords):
    print("关键词检测结果:")
    for keyword in keywords:
        pattern = rf"\b{re.escape(keyword)}\b"
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        print(f"  {keyword}: {'命中' if matches else '未命中'} (count={len(matches)})")


def generate_text(desc, model_inputs, max_new_tokens=512, monitor=False, monitor_keywords=None):
    print(f"\n----------- {desc} ------------")
    with torch.no_grad():
        generated_ids = runtime.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )
    input_len = model_inputs["input_ids"].shape[1]
    response = runtime.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
    print(f"Result: {response}")
    if monitor_keywords and monitor:
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


def parse_concept_list(raw_value):
    if raw_value is None:
        return []
    concept_names = []
    for part in str(raw_value).split(","):
        name = part.strip()
        if name:
            concept_names.append(name)
    return concept_names


def _build_attribution_cache_path(active_concept_configs, ig_steps, cache_dir):
    concept_names = "-".join(active_concept_configs.keys())
    safe_concept_names = re.sub(r"[^A-Za-z0-9_-]", "_", concept_names)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    return cache_dir_path / f"attribution_{safe_concept_names}_ig{ig_steps}.pt"


def _validate_cache_for_current_run(loaded_obj, expected_concepts, expected_ig_steps):
    expected_concepts_list = list(expected_concepts)
    expected_concepts_set = set(expected_concepts_list)

    if isinstance(loaded_obj, dict) and "concept_scores_by_layer" in loaded_obj:
        cached_meta = loaded_obj.get("meta", {})
        cached_ig_steps = cached_meta.get("ig_steps")
        cached_active_concepts = cached_meta.get("active_concepts")
        if cached_ig_steps != expected_ig_steps:
            return False, f"ig_steps 不匹配（cache={cached_ig_steps}, current={expected_ig_steps}）"
        if cached_active_concepts != expected_concepts_list:
            return False, (
                f"active_concepts 不匹配（cache={cached_active_concepts}, "
                f"current={expected_concepts_list}）"
            )
        return True, "meta 匹配"

    if isinstance(loaded_obj, dict):
        cached_concepts = set(loaded_obj.keys())
        if cached_concepts != expected_concepts_set:
            return False, (
                f"概念集合不匹配（cache={sorted(cached_concepts)}, "
                f"current={sorted(expected_concepts_set)}）"
            )
        return True, "旧版缓存格式概念集合匹配（无 ig_steps meta）"

    return False, "缓存格式无法识别"


def _find_combo_key_for_selected_concepts(selected_concepts):
    for combo_key, combo_concepts in COMBO_PRESETS.items():
        if list(combo_concepts) == list(selected_concepts):
            return combo_key
    return None


def _resolve_combo_preset(raw_combo_preset):
    if raw_combo_preset is None:
        return None

    combo_keys = list(COMBO_PRESETS.keys())
    if raw_combo_preset in COMBO_PRESETS:
        return raw_combo_preset

    if raw_combo_preset.isdigit():
        combo_idx = int(raw_combo_preset)
        if 0 <= combo_idx < len(combo_keys):
            return combo_keys[combo_idx]
        raise ValueError(
            f"combo-preset 编号越界: {combo_idx}，可用范围是 [0, {len(combo_keys) - 1}]"
        )

    available_name_desc = ", ".join(combo_keys)
    raise ValueError(
        f"未知 combo-preset: {raw_combo_preset}。"
        f"可用名称: {available_name_desc}；或使用编号 [0, {len(combo_keys) - 1}]"
    )


def resolve_generation_prompts(combo_key):
    if combo_key and combo_key in COMBO_PROMPTS and COMBO_PROMPTS[combo_key]:
        return COMBO_PROMPTS[combo_key], f"COMBO_PROMPTS[{combo_key}]"
    return NEW_PROMPTS, "NEW_PROMPTS(fallback)"



def main(args):
    intervention_layers = parse_intervention_layers(args.intervention_layer)
    enable_1 = args.enable_1
    enable_2 = args.enable_2

    resolved_combo_preset = _resolve_combo_preset(args.combo_preset)

    if resolved_combo_preset and args.concepts:
        raise ValueError("--combo-preset 与 --concepts 不能同时使用，请二选一")

    if resolved_combo_preset:
        candidate_concepts = list(COMBO_PRESETS[resolved_combo_preset])
    elif args.concepts:
        candidate_concepts = parse_concept_list(args.concepts)
        if not candidate_concepts:
            raise ValueError("--concepts 不能为空，请至少提供一个概念名")
    else:
        raise ValueError("请使用 --combo-preset 或 --concepts 指定概念")

    if len(candidate_concepts) > 2:
        raise ValueError(
            "当前参数槽位仅支持 2 个概念，请将 --concepts 控制在 2 个以内，"
            "或使用 --combo-preset"
        )

    score_mode_overrides = {}
    if len(candidate_concepts) >= 1 and args.score_mode_1:
        score_mode_overrides[candidate_concepts[0]] = args.score_mode_1
    if len(candidate_concepts) >= 2 and args.score_mode_2:
        score_mode_overrides[candidate_concepts[1]] = args.score_mode_2

    selected_concepts = []
    if enable_1:
        if len(candidate_concepts) < 1:
            raise ValueError("未提供第 1 个候选概念，无法使用 --enable_1")
        selected_concepts.append(candidate_concepts[0])
    if enable_2:
        if len(candidate_concepts) < 2:
            raise ValueError("未提供第 2 个候选概念，无法使用 --enable_2")
        selected_concepts.append(candidate_concepts[1])

    unknown_concepts = [name for name in selected_concepts if name not in CONCEPT_CONFIGS]
    if unknown_concepts:
        available = ", ".join(sorted(CONCEPT_CONFIGS.keys()))
        raise ValueError(
            f"发现无效概念: {unknown_concepts}；可用概念: {available}"
        )

    if not selected_concepts:
        raise ValueError(
            "至少需要启用一个概念：可使用 --combo-preset、--concepts，"
            "并通过 --enable_1/--enable_2 启用"
        )

    combo_key = resolved_combo_preset or _find_combo_key_for_selected_concepts(selected_concepts)
    prompt_pool, prompt_source = resolve_generation_prompts(combo_key)

    active_concept_configs = {}
    for concept_name in selected_concepts:
        cfg = dict(CONCEPT_CONFIGS[concept_name])
        if concept_name in score_mode_overrides:
            cfg["score_mode"] = score_mode_overrides[concept_name]
        active_concept_configs[concept_name] = cfg

    gpu_ids = [int(x.strip()) for x in args.parallel_gpus.split(",") if x.strip()]
    if not gpu_ids:
        raise ValueError("parallel-gpus 不能为空，至少需要提供 1 张 GPU，例如 --parallel-gpus 0")

    sequential_attribution = len(gpu_ids) < len(active_concept_configs)
    if sequential_attribution:
        assigned_gpu_ids = [gpu_ids[0]] * len(active_concept_configs)
        print(
            "[提示] parallel-gpus 数量少于启用概念数，将自动退化为单卡顺序归因模式。"
        )
    else:
        assigned_gpu_ids = gpu_ids[:len(active_concept_configs)]
    print(f"归因 GPU 分配: {assigned_gpu_ids}")

    target_attr_sum_by_concept = {}
    multiplier_by_concept = {}
    if enable_1 and len(candidate_concepts) >= 1:
        target_attr_sum_by_concept[candidate_concepts[0]] = args.attr_sum_1
        multiplier_by_concept[candidate_concepts[0]] = args.multiplier_1
    if enable_2 and len(candidate_concepts) >= 2:
        target_attr_sum_by_concept[candidate_concepts[1]] = args.attr_sum_2
        multiplier_by_concept[candidate_concepts[1]] = args.multiplier_2
    for concept_name in CONCEPT_CONFIGS:
        target_attr_sum_by_concept.setdefault(concept_name, 10.0)
        multiplier_by_concept.setdefault(concept_name, 3.0)

    if args.neuron_count_1 is not None or args.neuron_count_2 is not None:
        print(
            "[警告] --neuron_count_1/--neuron_count_2 已废弃，当前版本按目标归因分数和 "
            "--attr_sum_1/--attr_sum_2 自动计算 k。"
        )

    print("\n>>> 当前运行参数 <<<")
    print(f"enable_1={enable_1}")
    print(f"enable_2={enable_2}")
    print(f"combo_preset={args.combo_preset}")
    print(f"resolved_combo_preset={resolved_combo_preset}")
    print(f"concepts={args.concepts}")
    print(f"candidate_concepts={candidate_concepts}")
    print(f"selected_concepts={selected_concepts}")
    print(f"prompt_source={prompt_source}")
    print(f"ig_steps={args.ig_steps}")
    print(f"attr_sum_1={args.attr_sum_1}")
    print(f"multiplier_1={args.multiplier_1}")
    print(f"attr_sum_2={args.attr_sum_2}")
    print(f"multiplier_2={args.multiplier_2}")
    print(f"threshold={args.threshold}")
    print(f"intervention_layers={intervention_layers if intervention_layers is not None else 'ALL'}")
    print(f"parallel_gpus={gpu_ids}")
    print(f"prompt_index={args.prompt_index}")
    for cname, cfg in active_concept_configs.items():
        print(f"  {cname}: score_mode={cfg['score_mode']}, negative_words={cfg.get('negative_words', [])}")
    print(f"active_concepts={list(active_concept_configs.keys())}")
    concept_gpu_map = {
        concept_name: assigned_gpu_ids[idx]
        for idx, concept_name in enumerate(active_concept_configs.keys())
    }
    print(f"concept_gpu_map={concept_gpu_map}")

    if args.attribution_cache_path:
        cache_path = Path(args.attribution_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        cache_path = _build_attribution_cache_path(
            active_concept_configs=active_concept_configs,
            ig_steps=args.ig_steps,
            cache_dir=args.attribution_cache_dir,
        )
    print(f"attribution_cache_path={cache_path}")

    concept_scores_by_layer = None
    should_recompute = True
    if cache_path.exists() and not args.force_recmp:
        print("检测到归因缓存，准备校验配置一致性...")
        loaded_obj = torch.load(cache_path, map_location="cpu")
        cache_valid, cache_msg = _validate_cache_for_current_run(
            loaded_obj,
            expected_concepts=active_concept_configs.keys(),
            expected_ig_steps=args.ig_steps,
        )
        if cache_valid:
            if isinstance(loaded_obj, dict) and "concept_scores_by_layer" in loaded_obj:
                concept_scores_by_layer = loaded_obj["concept_scores_by_layer"]
            else:
                # 兼容早期仅保存 scores 字典的格式
                concept_scores_by_layer = loaded_obj
            should_recompute = False
            print(f"归因缓存校验通过（{cache_msg}），跳过重算。")
        else:
            print(f"归因缓存校验失败（{cache_msg}），将自动重算。")
    else:
        if args.force_recmp:
            print("已启用强制重算归因，忽略已有缓存。")
        elif not cache_path.exists():
            print("未检测到归因缓存，将进行重算。")

    if should_recompute:
        if sequential_attribution:
            primary_gpu = assigned_gpu_ids[0]
            print(f"开始顺序归因（single GPU: {primary_gpu}）...")
            concept_scores_by_layer = {}
            for concept_name, cfg in active_concept_configs.items():
                print(f"[顺序归因] 正在计算: {concept_name}")
                single_result = run_parallel_attribution(
                    {concept_name: cfg},
                    ig_steps=args.ig_steps,
                    gpu_ids=[primary_gpu],
                )
                concept_scores_by_layer.update(single_result)
        else:
            concept_scores_by_layer = run_parallel_attribution(
                active_concept_configs,
                ig_steps=args.ig_steps,
                gpu_ids=assigned_gpu_ids,
            )
        torch.save(
            {
                "concept_scores_by_layer": concept_scores_by_layer,
                "meta": {
                    "ig_steps": args.ig_steps,
                    "active_concepts": list(active_concept_configs.keys()),
                },
            },
            cache_path,
        )
        print("归因分数已保存到缓存。")

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
    selected_count_by_concept = {}
    selected_sum_by_concept = {}
    print("\n>>> 全量归属 + 按目标归因分数和选神经元 结果 <<<")
    for cname in active_concept_configs.keys():
        target_sum = target_attr_sum_by_concept.get(cname, 10.0)
        (
            concept_neurons[cname],
            selected_cnt,
            selected_sum,
            reached_target,
            total_positive_sum,
        ) = select_top_by_score_sum_from_assigned_scores(
            assigned_raw_scores.get(cname, {}),
            target_sum=target_sum,
        )
        selected_count_by_concept[cname] = selected_cnt
        selected_sum_by_concept[cname] = selected_sum
        assigned_cnt = assigned_counts.get(cname, 0)
        per_layer_counts = count_neurons_per_layer(concept_neurons[cname])
        per_layer_desc = (
            ", ".join([f"L{layer_idx}:{cnt}" for layer_idx, cnt in per_layer_counts.items()])
            if per_layer_counts
            else "无"
        )
        print(
            f"  {cname}: 归属池神经元={assigned_cnt}, "
            f"目标sum={target_sum:.6f}, 自动选k={selected_cnt}, 实际sum={selected_sum:.6f}, "
            f"正分总和={total_positive_sum:.6f}, 达标={'是' if reached_target else '否(已取完正分神经元)'}"
        )
        print(f"    按层分布: {per_layer_desc}")

 
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
            f"  {cname}: target_sum={target_attr_sum_by_concept[cname]:.6f}, "
            f"auto_k={selected_count_by_concept.get(cname, 0)}, "
            f"actual_sum={selected_sum_by_concept.get(cname, 0.0):.6f}, "
            f"multiplier={multiplier_by_concept[cname]}x"
        )

    if args.prompt_index < 0 or args.prompt_index >= len(prompt_pool):
        raise ValueError(
            f"prompt-index 越界: {args.prompt_index}，可用范围是 [0, {len(prompt_pool) - 1}]"
        )

    runtime.initialize_runtime(device_map="auto", offload_tag="main_generation")
    prompt = prompt_pool[args.prompt_index]
    print(f"prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    text = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = runtime.tokenizer([text], return_tensors="pt").to(runtime.input_device)

    # 先输出无干预 baseline，避免与 intervention 结果混淆。
    if args.baseline:
        generate_text(
            "Baseline-(No intervention)",
            model_inputs,
            max_new_tokens=args.max_new_tokens,
            monitor=args.monitor,
            monitor_keywords=[cfg["positive_word"] 
                        for cfg in active_concept_configs.values() 
                        if cfg.get("positive_word")
                              ],
        )

    hooks_to_remove = []
    if args.unified_hook:
        concept_interventions = [
            (neurons, multiplier_by_concept.get(concept_name, 3.0))
            for concept_name, neurons in concept_neurons.items()
            if neurons
        ]
        hook = UnifiedInterventionHook(concept_interventions)
        hook.register(runtime.model)
        hooks_to_remove.append(hook)
        print("  [hook mode] unified (所有品牌统一 norm 压缩)")
    else:
        for concept_name, neurons in concept_neurons.items():
            if neurons:
                mult = multiplier_by_concept.get(concept_name, 3.0)
                hook = NeuronInterventionHook(neurons, multiplier=mult)
                hook.register(runtime.model)
                hooks_to_remove.append(hook)
        print("  [hook mode] independent (各品牌独立 norm 压缩)")

    concept_names_desc = "+".join(concept_neurons.keys())
    mult_desc = ", ".join(
        [f"{concept_name}={multiplier_by_concept.get(concept_name, 3.0)}x" for concept_name in concept_neurons.keys()]
    )
    intervention_count_by_concept = {
        concept_name: count_neurons(concept_neurons.get(concept_name, {}))
        for concept_name in concept_neurons.keys()
    }
    intervention_sum_by_concept = {
        concept_name: selected_sum_by_concept.get(concept_name, 0.0)
        for concept_name in concept_neurons.keys()
    }
    total_intervention_count = sum(intervention_count_by_concept.values())
    count_desc = ", ".join(
        [
            f"{concept_name}={intervention_count_by_concept[concept_name]}"
            for concept_name in concept_neurons.keys()
        ]
    )
    sum_desc = ", ".join(
        [
            f"{concept_name}={intervention_sum_by_concept[concept_name]:.4f}"
            for concept_name in concept_neurons.keys()
        ]
    )
    generate_text(
        f"Intervention-({concept_names_desc}: {mult_desc}; count: {count_desc}; "
        f"sum: {sum_desc}; total={total_intervention_count})",
        model_inputs,
        max_new_tokens=args.max_new_tokens,
        monitor=args.monitor,
        monitor_keywords=[
            cfg["positive_word"]
            for cfg in active_concept_configs.values()
            if cfg.get("positive_word")
        ],
    )

    for hook in hooks_to_remove:
        hook.remove()

def parse_args():
    parser = argparse.ArgumentParser(
        description="多样本对比归因 + 神经元干预实验"
    )
    parser.add_argument(
        "--enable_1",
        action="store_true",
        help="是否启用候选概念列表中的第 1 个概念",
    )
    parser.add_argument(
        "--enable_2",
        action="store_true",
        help="是否启用候选概念列表中的第 2 个概念",
    )
    parser.add_argument(
        "--combo-preset",
        type=str,
        default=None,
        help="按 src.config.COMBO_PRESETS 选择品牌组合，支持名称（如 delta_hilton）或编号（0-based）",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default=None,
        help="手动指定概念名（逗号分隔），概念名需来自 src.config.CONCEPT_CONFIGS",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        default=IG_STEPS_DEFAULT,
        help="积分梯度黎曼近似步数 m（默认: 20）",
    )
    parser.add_argument(
        "--attr_sum_1",
        type=float,
        default=10.0,
        help="候选概念第 1 项的目标累计归因分数和 x（按归因分数降序自动选最小 k 使 sum>=x）",
    )
    parser.add_argument(
        "--neuron_count_1",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--multiplier_1",
        type=float,
        default=3.0,
        help="候选概念第 1 项神经元激活放大倍数，需 > 0",
    )
    parser.add_argument(
        "--attr_sum_2",
        type=float,
        default=10.0,
        help="候选概念第 2 项的目标累计归因分数和 x（按归因分数降序自动选最小 k 使 sum>=x）",
    )
    parser.add_argument(
        "--neuron_count_2",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--multiplier_2",
        type=float,
        default=3.0,
        help="候选概念第 2 项神经元激活放大倍数，需 > 0",
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
        "--score_mode_1",
        type=str,
        default=None,
        choices=["direct", "contrastive"],
        help="覆盖候选概念第 1 项的归因模式: direct=仅正向, contrastive=减去负样本",
    )
    parser.add_argument(
        "--score_mode_2",
        type=str,
        default=None,
        choices=["direct", "contrastive"],
        help="覆盖候选概念第 2 项的归因模式: direct=仅正向, contrastive=减去负样本",
    )
    parser.add_argument(
        "--intervention_layer",
        type=str,
        default="-1",
        help="仅在指定层注入神经元，支持逗号分隔多层，例如 32,33,34；默认 -1 表示干预所有层（0-based）",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="是否监控关键词出现次数",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="生成最大新 token 数（用于 baseline/intervention 对比时建议与其他脚本保持一致）",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="使用当前生成 prompt 列表中的第几个 prompt（0-based）",
    )
    parser.add_argument(
        "--attribution-cache-dir",
        type=str,
        default="cache/attribution_scores",
        help="归因分数缓存目录；首次计算后会保存，后续同配置可直接加载",
    )
    parser.add_argument(
        "--attribution-cache-path",
        type=str,
        default=None,
        help="显式指定归因缓存文件路径；不设置时将按概念+ig_steps自动生成",
    )
    parser.add_argument(
        "--force_recmp",
        action="store_true",
        help="强制重算归因分数并覆盖缓存（默认: 优先读取已有缓存）",
    )
    parser.add_argument(
        "--unified-hook",
        action="store_true",
        help="使用联合干预 Hook（所有品牌先放大再统一 norm 压缩）；不传则各品牌独立 Hook",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="是否生成 baseline 结果",
    )
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args=parse_args()
    main(args)
    end_time =time.time()
    print(f"Total time:{(end_time-start_time):.2f} seconds")
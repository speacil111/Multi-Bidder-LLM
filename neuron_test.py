import argparse
import itertools
import re
from pathlib import Path

import torch
import time 
import src.runtime as runtime
from src.attribution import run_parallel_attribution
from src.config import COMBO_PRESETS, CONCEPT_CONFIGS, IG_STEPS_DEFAULT, MODEL_NAME, SEED
from src.hooks import NeuronInterventionHook, UnifiedInterventionHook
from src.new_prompts import COMBO_PROMPTS, NEW_PROMPTS_DIVERSE ,NEW_PROMPTS
from transformers import AutoTokenizer
from src.selection import (
    count_neurons,
    count_neurons_per_layer,
    merge_neuron_maps,
)
from src.mind_bridge import COMBO_MIND_BRIDGES, NIKE_BRIDGE
print("开始运行寻找特定概念 Neuron 的实验")
torch.manual_seed(SEED)
_DEBUG_TOKENIZER = None


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
    if "</think>" in response:
        # 隐藏模型思维内容，仅展示 </think> 之后的可见回答
        response = response.split("</think>", 1)[1].lstrip()
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


def _format_tokenization(text):
    global _DEBUG_TOKENIZER
    tokenizer = runtime.tokenizer
    if tokenizer is None:
        if _DEBUG_TOKENIZER is None:
            _DEBUG_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer = _DEBUG_TOKENIZER
    token_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    return f"ids={token_ids}, tokens={token_strs}"


def select_top_k_by_difference(target_scores_by_layer, other_scores_by_layer, top_k):
    if top_k < 0:
        raise ValueError(f"top_k 必须为非负整数，当前: {top_k}")
    if top_k == 0:
        return {}, 0, 0.0, 0.0, 0.0

    candidates = []
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
                    layer_idx,
                    neuron_idx,
                    float(diff_tensor[neuron_idx].item()),
                    float(target_tensor[neuron_idx].item()),
                    float(other_abs[neuron_idx].item()),
                )
            )

    if len(candidates) == 0:
        return {}, 0, 0.0, 0.0, 0.0

    candidates.sort(key=lambda x: x[2], reverse=True)
    selected = candidates[: min(int(top_k), len(candidates))]

    neuron_map = {}
    diff_sum = 0.0
    target_sum = 0.0
    other_abs_sum = 0.0
    for layer_idx, neuron_idx, diff_score, target_score, other_abs_score in selected:
        if layer_idx not in neuron_map:
            neuron_map[layer_idx] = []
        neuron_map[layer_idx].append(neuron_idx)
        # 二者之差
        diff_sum += diff_score
        # 当前品牌的分数
        target_sum += target_score
        # 其他品牌的绝对值分数
        other_abs_sum += other_abs_score

    return neuron_map, len(selected), float(diff_sum), float(target_sum), float(other_abs_sum)


def _build_attribution_cache_path(active_concept_configs, ig_steps, cache_dir):
    # 使用排序后的概念名，避免仅因概念顺序不同而产生重复缓存文件。
    concept_names = "-".join(sorted(active_concept_configs.keys()))
    safe_concept_names = re.sub(r"[^A-Za-z0-9_-]", "_", concept_names)
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)
    return cache_dir_path / f"attribution_{safe_concept_names}_ig{ig_steps}.pt"


def _build_alternative_cache_paths(active_concept_configs, ig_steps, cache_dir):
    concept_names = list(active_concept_configs.keys())
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    paths = []
    seen = set()
    for perm in itertools.permutations(concept_names):
        joined_names = "-".join(perm)
        safe_joined_names = re.sub(r"[^A-Za-z0-9_-]", "_", joined_names)
        candidate_path = cache_dir_path / f"attribution_{safe_joined_names}_ig{ig_steps}.pt"
        if str(candidate_path) in seen:
            continue
        seen.add(str(candidate_path))
        paths.append(candidate_path)
    return paths


def _validate_cache_for_current_run(loaded_obj, expected_concepts, expected_ig_steps):
    expected_concepts_list = list(expected_concepts)
    expected_concepts_set = set(expected_concepts_list)

    if isinstance(loaded_obj, dict) and "concept_scores_by_layer" in loaded_obj:
        cached_meta = loaded_obj.get("meta", {})
        cached_ig_steps = cached_meta.get("ig_steps")
        cached_active_concepts = cached_meta.get("active_concepts")
        if cached_ig_steps != expected_ig_steps:
            return False, f"ig_steps 不匹配（cache={cached_ig_steps}, current={expected_ig_steps}）"
        if set(cached_active_concepts or []) != expected_concepts_set:
            return False, (
                f"active_concepts 不匹配（cache={cached_active_concepts}, "
                f"current={expected_concepts_list}）"
            )
        if cached_active_concepts != expected_concepts_list:
            return True, "meta 概念集合匹配（顺序不同，已允许复用缓存）"
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
    if len(selected_concepts) != 2:
        raise ValueError(
            "当前脚本已切换为双概念差分选点模式，需要且仅需要启用 2 个概念（例如 bidder_A,bidder_B）。"
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

    sequential_attribution = True
    assigned_gpu_ids = [gpu_ids[0]] * len(active_concept_configs)
    if len(gpu_ids) > 1:
        print(
            f"[提示] 当前版本固定单卡归因，将仅使用第一张 GPU: {gpu_ids[0]} "
            f"（忽略其余 GPU: {gpu_ids[1:]}）"
        )
    else:
        print(f"[提示] 当前版本固定单卡归因，使用 GPU: {gpu_ids[0]}")
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
            "[警告] --neuron_count_1/--neuron_count_2 已废弃，当前版本请使用 "
            "--top_k_1/--top_k_2 控制选点数量。"
        )
    if args.attr_sum_1 != 10.0 or args.attr_sum_2 != 10.0:
        print(
            "[警告] 当前版本不再使用 --attr_sum_1/--attr_sum_2 自动选 k；"
            "请改用 --top_k_1/--top_k_2。"
        )
    if args.threshold != 0.0:
        print(
            "[警告] 当前版本不再进行归属分类，--threshold 不生效。"
        )

    print("\n>>> 当前运行参数 <<<")
    print(f"enable_1={enable_1}")
    print(f"enable_2={enable_2}")
    print(f"combo_preset={args.combo_preset}")
    # print(f"resolved_combo_preset={resolved_combo_preset}")
    print(f"concepts={args.concepts}")
    # print(f"candidate_concepts={candidate_concepts}")
    # print(f"selected_concepts={selected_concepts}")
    print(f"prompt_source={prompt_source}")
    print(f"ig_steps={args.ig_steps}")
    # print(f"attr_sum_1={args.attr_sum_1}")
    print(f"multiplier_1={args.multiplier_1}")
    # print(f"attr_sum_2={args.attr_sum_2}")
    print(f"multiplier_2={args.multiplier_2}")
    print(f"top_k_1={args.top_k_1}")
    print(f"top_k_2={args.top_k_2}")
    # print(f"threshold={args.threshold}")
    print(f"intervention_layers={intervention_layers if intervention_layers is not None else 'ALL'}")
    print(f"parallel_gpus={gpu_ids}")
    print(f"prompt_index={args.prompt_index}")
    print(f"Using mind_bridge={args.mind_bridge}")
    for cname, cfg in active_concept_configs.items():
        print(f"  {cname}: positive_word={cfg['positive_word']}, score_mode={cfg['score_mode']}, negative_words={cfg.get('negative_words', [])}")
    print("tokenization details:")
    for cname, cfg in active_concept_configs.items():
        pos_word = cfg.get("positive_word")
        neg_words = cfg.get("negative_words", [])
        if pos_word:
            print(f"  {cname}.positive_word='{pos_word}': {_format_tokenization(pos_word)}")
        for neg_word in neg_words:
            print(f"  {cname}.negative_word='{neg_word}': {_format_tokenization(neg_word)}")
    print(f"active_concepts={list(active_concept_configs.keys())}")
    concept_gpu_map = {
        concept_name: assigned_gpu_ids[idx]
        for idx, concept_name in enumerate(active_concept_configs.keys())
    }
    # print(f"concept_gpu_map={concept_gpu_map}")

    alternative_cache_paths = []
    if args.attribution_cache_path:
        canonical_cache_path = Path(args.attribution_cache_path)
        canonical_cache_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        canonical_cache_path = _build_attribution_cache_path(
            active_concept_configs=active_concept_configs,
            ig_steps=args.ig_steps,
            cache_dir=args.attribution_cache_dir,
        )
        alternative_cache_paths = [
            p
            for p in _build_alternative_cache_paths(
                active_concept_configs=active_concept_configs,
                ig_steps=args.ig_steps,
                cache_dir=args.attribution_cache_dir,
            )
            if p != canonical_cache_path
        ]
    load_cache_path = canonical_cache_path
    if (
        not args.attribution_cache_path
        and not canonical_cache_path.exists()
        and alternative_cache_paths
    ):
        for alt_path in alternative_cache_paths:
            if alt_path.exists():
                load_cache_path = alt_path
                print(f"未找到规范缓存名，改用兼容缓存: {alt_path}")
                break
    print(f"attribution_cache_path={load_cache_path}")

    concept_scores_by_layer = None
    should_recompute = True
    if load_cache_path.exists() and not args.force_recmp:
        print("检测到归因缓存，准备校验配置一致性...")
        loaded_obj = torch.load(load_cache_path, map_location="cpu")
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
        elif not load_cache_path.exists():
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
            canonical_cache_path,
        )
        print(f"归因分数已保存到缓存: {canonical_cache_path}")

    concept_neurons = {}
    selected_count_by_concept = {}
    selected_sum_by_concept = {}
    print("\n>>> 不做归属分类：按 (target - abs(other)) 选 Top-k 神经元 <<<")
    selected_list = list(active_concept_configs.keys())
    first_concept = selected_list[0]
    second_concept = selected_list[1]
    top_k_by_concept = {
        first_concept: args.top_k_1,
        second_concept: args.top_k_2,
    }
    contrast_stats_by_concept = {}
    for cname in selected_list:
        other_name = second_concept if cname == first_concept else first_concept
        (
            concept_neurons[cname],
            selected_cnt,
            selected_diff_sum,
            selected_target_sum,
            selected_other_abs_sum,
        ) = select_top_k_by_difference(
            concept_scores_by_layer[cname],
            concept_scores_by_layer[other_name],
            top_k=top_k_by_concept[cname],
        )
        selected_count_by_concept[cname] = selected_cnt
        selected_sum_by_concept[cname] = selected_diff_sum
        contrast_stats_by_concept[cname] = {
            "other_name": other_name,
            "target_sum": selected_target_sum,
            "other_abs_sum": selected_other_abs_sum,
        }
        per_layer_counts = count_neurons_per_layer(concept_neurons[cname])
        per_layer_desc = (
            ", ".join([f"L{layer_idx}:{cnt}" for layer_idx, cnt in per_layer_counts.items()])
            if per_layer_counts
            else "无"
        )
        print(
            f"  {cname}: other={other_name}, top_k={top_k_by_concept[cname]}, "
            f"selected={selected_cnt}, diff_sum={selected_diff_sum:.6f}, "
            f"target_sum={selected_target_sum:.6f}, other_abs_sum={selected_other_abs_sum:.6f}"
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
        contrast_stats = contrast_stats_by_concept.get(cname, {})
        other_name = contrast_stats.get("other_name", "N/A")
        print(
            f"  {cname}: other={other_name}, "
            f"top_k={selected_count_by_concept.get(cname, 0)}, "
            f"diff_sum={selected_sum_by_concept.get(cname, 0.0):.6f}, "
            f"target_sum={contrast_stats.get('target_sum', 0.0):.6f}, "
            f"other_abs_sum={contrast_stats.get('other_abs_sum', 0.0):.6f}, "
            f"multiplier={multiplier_by_concept[cname]}x"
        )

    if args.prompt_index < 0 or args.prompt_index >= len(prompt_pool):
        raise ValueError(
            f"prompt-index 越界: {args.prompt_index}，可用范围是 [0, {len(prompt_pool) - 1}]"
        )

    runtime.initialize_runtime(device_map="auto", offload_tag="main_generation")
    prompt = prompt_pool[args.prompt_index]

    # prompt = ( "You are an expert at writing advertising copy. "
    #   "Write an artistic advertisement about a vacation in Hawaii. "
    #   "Mention the flight and accommodation details naturally.")
    print(f"prompt: {prompt}")
    messages = [{"role": "user", "content": prompt}]
    text = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if args.mind_bridge:
        mind_bridge_text = COMBO_MIND_BRIDGES.get(combo_key)
        text += mind_bridge_text
        print(f"\n[Mind Bridge] 已强制注入思维逻辑:\n{text}")

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
        default=2.0,
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
        default=2.0,
        help="候选概念第 2 项神经元激活放大倍数，需 > 0",
    )
    parser.add_argument(
        "--top_k_1",
        type=int,
        default=500,
        help="候选概念第 1 项按 (target - abs(other)) 排序后选取的 Top-k 神经元数；0 表示不为该概念注入神经元",
    )
    parser.add_argument(
        "--top_k_2",
        type=int,
        default=500,
        help="候选概念第 2 项按 (target - abs(other)) 排序后选取的 Top-k 神经元数；0 表示不为该概念注入神经元",
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

    parser.add_argument(
        "--mind_bridge",
        action="store_true",
        help="是否启用 Mind Bridge 模式",
    )
    return parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    args=parse_args()
    main(args)
    end_time =time.time()
    print(f"Total time:{(end_time-start_time):.2f} seconds")
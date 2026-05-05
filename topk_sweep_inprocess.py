#!/usr/bin/env python3
import argparse
import contextlib
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path


def format_duration(total_seconds):
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_model_tag(model_path):
    model_path_lower = model_path.lower()
    if "ds_r1_8b" in model_path_lower:
        return "DS"
    if "qwen3_4b" in model_path_lower or "qwen3-4b" in model_path_lower:
        return "Qwen"
    if "llama" in model_path_lower and "8b" in model_path_lower:
        return "Llama"

    model_base = Path(model_path).name
    model_base = re.sub(r"[^A-Za-z0-9_-]", "_", model_base)
    return model_base or "Model"


def parse_int_list_spec(spec, name):
    values = []
    for token in str(spec).replace(",", " ").split():
        if re.fullmatch(r"\d+-\d+", token):
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"invalid {name} range: {token}")
            values.extend(range(start, end + 1))
        elif re.fullmatch(r"\d+", token):
            values.append(int(token))
        else:
            raise ValueError(f"invalid {name} token: {token}")
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def resolve_combo_info(combo_preset_id):
    from src.config import COMBO_PRESETS, CONCEPT_CONFIGS

    combo_keys = list(COMBO_PRESETS.keys())
    combo_id = int(combo_preset_id)
    if combo_id < 0 or combo_id >= len(combo_keys):
        raise ValueError(
            f"combo preset id out of range: {combo_id}; valid=[0, {len(combo_keys) - 1}]"
        )

    combo_key = combo_keys[combo_id]
    concepts = list(COMBO_PRESETS[combo_key])
    if len(concepts) != 2:
        raise ValueError(
            f"topk_sweep_inprocess.py currently expects a 2-bidder combo, got {combo_key}={concepts}"
        )
    brand_1, brand_2 = concepts
    keyword_1 = CONCEPT_CONFIGS[brand_1]["positive_word"]
    keyword_2 = CONCEPT_CONFIGS[brand_2]["positive_word"]
    return combo_key, brand_1, brand_2, keyword_1, keyword_2


def parse_result_log(log_path, keyword_1, keyword_2):
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Result:"):
            start_idx = idx

    if start_idx is None:
        result_block = ""
    else:
        block = [lines[start_idx]]
        for cur in lines[start_idx + 1 :]:
            if cur.startswith("============ Job "):
                break
            if cur.startswith("  0%|") or cur.startswith("100%|"):
                continue
            block.append(cur)
        result_block = "\n".join(block).strip()

    result_lower = result_block.lower()
    hit_1 = len(re.findall(rf"\b{re.escape(keyword_1.lower())}\b", result_lower))
    hit_2 = len(re.findall(rf"\b{re.escape(keyword_2.lower())}\b", result_lower))

    prompt_text = ""
    for line in lines:
        if line.startswith("prompt:"):
            prompt_text = line.split("prompt:", 1)[1].strip()
            break

    return prompt_text, result_block, hit_1, hit_2


def write_lines(path, lines, mode="a"):
    with Path(path).open(mode, encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def append_run_report(
    report_txt,
    run_id,
    brand_1,
    brand_2,
    keyword_1,
    keyword_2,
    top_k_1,
    top_k_2,
    multiplier_1,
    multiplier_2,
    prompt_index,
    prompt_text,
    gpu_id,
    result_block,
    hit_1,
    hit_2,
):
    write_lines(
        report_txt,
        [
            "",
            "------------------------------------------------------------",
            f"Run {run_id}",
            f"{brand_1}_top_k={top_k_1}",
            f"{brand_2}_top_k={top_k_2}",
            f"{brand_1}_multiplier={multiplier_1}",
            f"{brand_2}_multiplier={multiplier_2}",
            f"prompt_index={prompt_index}",
            f"prompt={prompt_text}",
            f"gpu_id={gpu_id}",
            result_block,
            f"hit_{keyword_1}={hit_1}, hit_{keyword_2}={hit_2}",
            "------------------------------------------------------------",
        ],
    )


def append_failed_report(
    report_txt,
    run_id,
    brand_1,
    brand_2,
    top_k_1,
    top_k_2,
    multiplier_1,
    multiplier_2,
    prompt_index,
    gpu_id,
    status,
    log_file,
):
    write_lines(
        report_txt,
        [
            "",
            "------------------------------------------------------------",
            f"Run {run_id}",
            f"{brand_1}_top_k={top_k_1}",
            f"{brand_2}_top_k={top_k_2}",
            f"{brand_1}_multiplier={multiplier_1}",
            f"{brand_2}_multiplier={multiplier_2}",
            f"prompt_index={prompt_index}",
            f"gpu_id={gpu_id}",
            f"status=FAILED ({status})",
            f"log_file={log_file}",
            "------------------------------------------------------------",
        ],
    )


def run_neuron_test_once(neuron_test, argv, log_file):
    old_argv = sys.argv[:]
    try:
        sys.argv = ["neuron_test.py", *argv]
        args = neuron_test.parse_args()
        with Path(log_file).open("w", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                start_time = time.time()
                neuron_test.main(args)
                end_time = time.time()
                print(f"Total time:{(end_time - start_time):.2f} seconds")
    except Exception:
        with Path(log_file).open("a", encoding="utf-8") as handle:
            handle.write("\n[ERROR] neuron_test run failed\n")
            handle.write(traceback.format_exc())
        return False
    finally:
        sys.argv = old_argv
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a top-k sweep inside one Python process so the generation model is loaded once."
    )
    parser.add_argument("--g", "--gpu-id", dest="gpu_id", default="0")
    parser.add_argument("--c", "--combo-preset-id", dest="combo_preset_id", default="0")
    parser.add_argument("--model-path", default="../Llama-3-8B-Instruct")
    parser.add_argument("--attr-cache-dir", "--attribution-cache-dir", dest="attr_cache_dir", default="attr_cache_Llama")
    parser.add_argument("--result-root", default="")
    parser.add_argument("--prompt-list", "--prompts", default="0 1 2")
    parser.add_argument("--top-k-1", "--top_k_1", dest="top_k_1", default="0 100 200 300 400 500 600 700 800")
    parser.add_argument("--top-k-2", "--top_k_2", dest="top_k_2", default="0 100 200 300 400 500 600 700 800")
    parser.add_argument("--lambda", "--multiplier", dest="multiplier", default=None)
    parser.add_argument("--multiplier-1", "--multiplier_1", dest="multiplier_1", default=None)
    parser.add_argument("--multiplier-2", "--multiplier_2", dest="multiplier_2", default=None)
    parser.add_argument("--ig-steps", "--ig_steps", dest="ig_steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_sweep(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    multiplier_1 = args.multiplier_1 or args.multiplier or os.environ.get("MULTIPLIER_1", "2.0")
    multiplier_2 = args.multiplier_2 or args.multiplier or os.environ.get("MULTIPLIER_2", "2.0")
    prompt_list = parse_int_list_spec(args.prompt_list, "--prompt-list")
    top_k_1_list = parse_int_list_spec(args.top_k_1, "TOP_K_1")
    top_k_2_list = parse_int_list_spec(args.top_k_2, "TOP_K_2")

    script_start_epoch = time.time()
    script_start_time = time.strftime("%Y-%m-%d %H:%M:%S")

    combo_key, brand_1, brand_2, keyword_1, keyword_2 = resolve_combo_info(args.combo_preset_id)
    model_tag = resolve_model_tag(args.model_path)
    result_root_dir = args.result_root.rstrip("/") if args.result_root else f"batch_results_{model_tag}"
    run_root = Path(result_root_dir) / f"{model_tag}_{brand_1}_m{multiplier_1}"
    run_root.mkdir(parents=True, exist_ok=True)

    snapshot_dir = run_root / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("neuron_test.py", snapshot_dir / "neuron_test.py")
    shutil.copy2("topk_sweep_inprocess.py", snapshot_dir / "topk_sweep_inprocess.py")
    if (snapshot_dir / "src").exists():
        shutil.rmtree(snapshot_dir / "src")
    shutil.copytree("src", snapshot_dir / "src")

    overall_report_txt = run_root / "report_all_prompts.txt"
    write_lines(
        overall_report_txt,
        [
            f"Sweep started at: {script_start_time}",
            f"combo_preset_id={args.combo_preset_id}",
            f"combo_key={combo_key}",
            f"brand_1={brand_1}",
            f"brand_2={brand_2}",
            f"keyword_1={keyword_1}",
            f"keyword_2={keyword_2}",
            f"{brand_1}_multiplier={multiplier_1}",
            f"{brand_2}_multiplier={multiplier_2}",
            f"prompt_list={' '.join(map(str, prompt_list))}",
            f"gpu_id={args.gpu_id}",
            f"{brand_1}_top_k={' '.join(map(str, top_k_1_list))}",
            f"{brand_2}_top_k={' '.join(map(str, top_k_2_list))}",
            f"model_path={args.model_path or '<default>'}",
            f"attribution_cache_dir={args.attr_cache_dir}",
            f"result_root_dir={result_root_dir}",
            f"run_root={run_root}",
            f"code_snapshot={snapshot_dir}",
            "execution_mode=inprocess",
        ],
        mode="w",
    )

    import neuron_test

    overall_success_runs = 0
    overall_failed_runs = 0
    overall_total_runs = 0
    overall_failed_run_msgs = []
    prompt_avg_csv = run_root / "prompt_hit_avg.csv"
    run_top_k_1_map = {}
    run_top_k_2_map = {}
    run_hit_1_sum_map = {}
    run_hit_2_sum_map = {}
    run_hit_count_success_map = {}
    total_runs_per_prompt = len(top_k_1_list) * len(top_k_2_list)

    for prompt_index in prompt_list:
        run_dir = run_root / f"p{prompt_index}"
        log_dir = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for old_log in log_dir.glob("*.log"):
            old_log.unlink()

        report_txt = run_dir / f"report_{prompt_index}.txt"
        summary_csv = run_dir / f"summary_{prompt_index}.csv"
        write_lines(
            report_txt,
            [
                "Fixed params:",
                f"  combo_preset_id={args.combo_preset_id}",
                f"  combo_key={combo_key}",
                f"  brand_1={brand_1}",
                f"  brand_2={brand_2}",
                f"  keyword_1={keyword_1}",
                f"  keyword_2={keyword_2}",
                f"  {brand_1}_multiplier={multiplier_1}",
                f"  {brand_2}_multiplier={multiplier_2}",
                f"  prompt_index={prompt_index}",
                f"  gpu_id={args.gpu_id}",
                f"  {brand_1}_top_k={' '.join(map(str, top_k_1_list))}",
                f"  {brand_2}_top_k={' '.join(map(str, top_k_2_list))}",
                f"  model_path={args.model_path or '<default>'}",
                f"  attribution_cache_dir={args.attr_cache_dir}",
                f"  result_root_dir={result_root_dir}",
                f"  run_root={run_root}",
                f"  code_snapshot={snapshot_dir}",
                "  execution_mode=inprocess",
            ],
            mode="w",
        )
        write_lines(
            summary_csv,
            [
                f"run_id\t{brand_1}_top_k\t{brand_2}_top_k\t{brand_1}_multiplier\t{brand_2}_multiplier\thit_{keyword_1}\thit_{keyword_2}"
            ],
            mode="w",
        )

        print("")
        print(f"==================== Prompt {prompt_index} ====================")

        run_id = 0
        total_runs = total_runs_per_prompt
        success_runs = 0
        failed_runs = 0
        failed_run_msgs = []
        sum_hit_1 = 0
        sum_hit_2 = 0

        for top_k_2 in top_k_2_list:
            for top_k_1 in top_k_1_list:
                run_id += 1
                run_top_k_1_map[run_id] = top_k_1
                run_top_k_2_map[run_id] = top_k_2
                log_file = log_dir / (
                    f"run_{run_id}_k1{top_k_1}_k2{top_k_2}_"
                    f"m1{multiplier_1}_m2{multiplier_2}.log"
                )
                print(
                    f"[Prompt {prompt_index}] [Run {run_id}] "
                    f"{brand_1}_top_k={top_k_1}, {brand_2}_top_k={top_k_2}, "
                    f"{brand_1}_multiplier={multiplier_1}, {brand_2}_multiplier={multiplier_2}"
                )

                neuron_argv = [
                    "--combo-preset",
                    str(args.combo_preset_id),
                    "--enable_1",
                    "--enable_2",
                    "--ig_steps",
                    str(args.ig_steps),
                    "--top_k_1",
                    str(top_k_1),
                    "--multiplier_1",
                    str(multiplier_1),
                    "--top_k_2",
                    str(top_k_2),
                    "--multiplier_2",
                    str(multiplier_2),
                    "--parallel-gpus",
                    str(args.gpu_id),
                    "--score_mode_1",
                    "contrastive",
                    "--score_mode_2",
                    "contrastive",
                    "--threshold",
                    str(args.threshold),
                    "--attribution-cache-dir",
                    args.attr_cache_dir,
                    "--intervention_layer",
                    "-1",
                    "--prompt-index",
                    str(prompt_index),
                    "--unified-hook",
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--model_path",
                    args.model_path,
                ]

                if not run_neuron_test_once(neuron_test, neuron_argv, log_file):
                    failed_runs += 1
                    failed_run_msgs.append(
                        f"Run {run_id} (k1={top_k_1}, k2={top_k_2}): python command failed"
                    )
                    print(f"  [WARN] Run {run_id} failed, skip and continue. See log: {log_file}")
                    append_failed_report(
                        report_txt,
                        run_id,
                        brand_1,
                        brand_2,
                        top_k_1,
                        top_k_2,
                        multiplier_1,
                        multiplier_2,
                        prompt_index,
                        args.gpu_id,
                        "python command failed",
                        log_file,
                    )
                    continue

                try:
                    prompt_text, result_block, hit_1, hit_2 = parse_result_log(
                        log_file, keyword_1, keyword_2
                    )
                except Exception:
                    failed_runs += 1
                    failed_run_msgs.append(
                        f"Run {run_id} (k1={top_k_1}, k2={top_k_2}): parse script failed"
                    )
                    print(f"  [WARN] Run {run_id} parse failed, skip and continue. See log: {log_file}")
                    append_failed_report(
                        report_txt,
                        run_id,
                        brand_1,
                        brand_2,
                        top_k_1,
                        top_k_2,
                        multiplier_1,
                        multiplier_2,
                        prompt_index,
                        args.gpu_id,
                        "parse script failed",
                        log_file,
                    )
                    continue

                append_run_report(
                    report_txt,
                    run_id,
                    brand_1,
                    brand_2,
                    keyword_1,
                    keyword_2,
                    top_k_1,
                    top_k_2,
                    multiplier_1,
                    multiplier_2,
                    prompt_index,
                    prompt_text,
                    args.gpu_id,
                    result_block,
                    hit_1,
                    hit_2,
                )
                write_lines(
                    summary_csv,
                    [
                        f"{run_id}\t{top_k_1}\t{top_k_2}\t{float(multiplier_1):.2f}\t{float(multiplier_2):.2f}\t{hit_1}\t{hit_2}"
                    ],
                )
                sum_hit_1 += hit_1
                sum_hit_2 += hit_2
                success_runs += 1
                run_hit_1_sum_map[run_id] = run_hit_1_sum_map.get(run_id, 0) + hit_1
                run_hit_2_sum_map[run_id] = run_hit_2_sum_map.get(run_id, 0) + hit_2
                run_hit_count_success_map[run_id] = run_hit_count_success_map.get(run_id, 0) + 1

        if success_runs > 0:
            avg_hit_1 = sum_hit_1 / success_runs
            avg_hit_2 = sum_hit_2 / success_runs
            avg_hit_count = (sum_hit_1 + sum_hit_2) / success_runs
        else:
            avg_hit_1 = 0.0
            avg_hit_2 = 0.0
            avg_hit_count = 0.0

        overall_total_runs += total_runs
        overall_success_runs += success_runs
        overall_failed_runs += failed_runs
        for msg in failed_run_msgs:
            overall_failed_run_msgs.append(f"prompt={prompt_index}: {msg}")

        print("")
        print(f"Prompt {prompt_index} finished at: {time.ctime()}")
        print(f"Report: {report_txt}")
        print(f"Summary csv: {summary_csv}")
        print(f"Total planned runs: {total_runs}")
        print(f"Successful runs: {success_runs}")
        print(f"Failed runs: {failed_runs}")
        if failed_runs > 0:
            print("Failed run list:")
            for msg in failed_run_msgs:
                print(f"  - {msg}")
        print("")

        prompt_summary_lines = [
            "",
            "==================== Sweep Summary ====================",
            f"prompt_index={prompt_index}",
            f"total_runs={total_runs}",
            f"success_runs={success_runs}",
            f"failed_runs={failed_runs}",
        ]
        if failed_runs > 0:
            prompt_summary_lines.append("failed_run_list:")
            prompt_summary_lines.extend(f"  - {msg}" for msg in failed_run_msgs)
        prompt_summary_lines.append("=======================================================")
        write_lines(report_txt, prompt_summary_lines)

        overall_lines = [
            "",
            f"-------------------- Prompt {prompt_index} --------------------",
            f"report={report_txt}",
            f"summary_csv={summary_csv}",
            f"total_runs={total_runs}",
            f"success_runs={success_runs}",
            f"failed_runs={failed_runs}",
            f"avg_hit_{keyword_1}={avg_hit_1:.6f}",
            f"avg_hit_{keyword_2}={avg_hit_2:.6f}",
            f"hit_count={avg_hit_count:.6f}",
        ]
        if failed_runs > 0:
            overall_lines.append("failed_run_list:")
            overall_lines.extend(f"  - {msg}" for msg in failed_run_msgs)
        write_lines(overall_report_txt, overall_lines)

    write_lines(
        prompt_avg_csv,
        [
            f"run_id\t{brand_1}_top_k\t{brand_2}_top_k\t{brand_1}_multiplier\t{brand_2}_multiplier\thit_{keyword_1}_avg\thit_{keyword_2}_avg"
        ],
        mode="w",
    )
    for run_id in range(1, total_runs_per_prompt + 1):
        run_success_prompts = run_hit_count_success_map.get(run_id, 0)
        if run_success_prompts > 0:
            run_avg_hit_1 = run_hit_1_sum_map[run_id] / run_success_prompts
            run_avg_hit_2 = run_hit_2_sum_map[run_id] / run_success_prompts
        else:
            run_avg_hit_1 = 0.0
            run_avg_hit_2 = 0.0
        write_lines(
            prompt_avg_csv,
            [
                f"{run_id}\t{run_top_k_1_map.get(run_id, '')}\t{run_top_k_2_map.get(run_id, '')}\t"
                f"{float(multiplier_1):.2f}\t{float(multiplier_2):.2f}\t"
                f"{run_avg_hit_1:.6f}\t{run_avg_hit_2:.6f}"
            ],
        )

    script_end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    script_elapsed_seconds = int(time.time() - script_start_epoch)
    script_elapsed_hms = format_duration(script_elapsed_seconds)
    final_lines = [
        "",
        "-------------------- Total Runtime --------------------",
        f"sweep_started_at={script_start_time}",
        f"sweep_finished_at={script_end_time}",
        f"total_runtime_seconds={script_elapsed_seconds}",
        f"total_runtime_hms={script_elapsed_hms}",
        "",
        f"All prompts sweep finished at: {script_end_time}",
        f"Total runtime: {script_elapsed_hms} ({script_elapsed_seconds}s)",
        f"Overall report: {overall_report_txt}",
        f"Total planned runs: {overall_total_runs}",
        f"Successful runs: {overall_success_runs}",
        f"Failed runs: {overall_failed_runs}",
    ]
    if overall_failed_runs > 0:
        final_lines.append("Failed run list:")
        final_lines.extend(f"  - {msg}" for msg in overall_failed_run_msgs)
    final_lines.extend(
        [
            "",
            "Done. Please check:",
            f"  {overall_report_txt}",
            f"  {run_root}/p*/summary_*.csv",
            f"  {prompt_avg_csv}",
        ]
    )
    write_lines(overall_report_txt, final_lines)
    print("\n".join(final_lines))

    return 0


def main():
    return run_sweep(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def parse_combo_list_spec(spec):
    three_config = importlib.import_module("src.3_config")
    presets = three_config.THREE_BIDDER_COMBO_PRESETS
    combo_keys = list(presets.keys())
    combos = []
    seen = set()

    def add_combo(raw):
        raw = str(raw).strip()
        if not raw:
            return
        if raw in presets:
            combo_ref = raw
            combo_key = raw
        elif raw.isdigit():
            combo_id = int(raw)
            if combo_id < 0 or combo_id >= len(combo_keys):
                raise ValueError(
                    f"3-bidder combo id out of range: {combo_id}; "
                    f"valid=[0, {len(combo_keys) - 1}]"
                )
            combo_ref = str(combo_id)
            combo_key = combo_keys[combo_id]
        else:
            raise ValueError(f"unknown 3-bidder combo preset: {raw}")
        if combo_key in seen:
            return
        seen.add(combo_key)
        combos.append((combo_ref, combo_key))

    for token in str(spec).replace(",", " ").split():
        if "-" in token:
            left, right = token.split("-", 1)
            if left.isdigit() and right.isdigit():
                start = int(left)
                end = int(right)
                step = 1 if end >= start else -1
                for combo_id in range(start, end + step, step):
                    add_combo(str(combo_id))
                continue
        add_combo(token)

    if not combos:
        raise ValueError("--combos cannot be empty")
    return combos


def resolve_model_tag(model_path):
    model_path_lower = model_path.lower()
    if "ds_r1_8b" in model_path_lower:
        return "DS"
    if "qwen3_4b" in model_path_lower or "qwen3-4b" in model_path_lower:
        return "Qwen"
    if "llama" in model_path_lower and "8b" in model_path_lower:
        return "Llama"

    model_base = Path(model_path).name
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in model_base)
    return safe or "Model"


def resolve_default_state_dir(result_root, model_path):
    if result_root:
        return Path(result_root) / ".worker_state"
    model_tag = resolve_model_tag(model_path)
    return Path(f"batch_results_{model_tag}_3bidders") / ".worker_state"


def build_signature(args):
    signature_data = {
        "version": 1,
        "worker": "3bidders",
        "topk_script": args.topk_script,
        "model_path": args.model_path,
        "attr_cache_dir": args.attr_cache_dir,
        "result_root": args.result_root,
        "prompt_list": args.prompt_list,
        "system_prompt": args.system_prompt,
        "top_k_1": args.top_k_1,
        "top_k_2": args.top_k_2,
        "top_k_3": args.top_k_3,
    }
    encoded = json.dumps(signature_data, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16], signature_data


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def build_command(args, combo_key):
    cmd = [
        "bash",
        args.topk_script,
        "--g",
        str(args.gpu_id),
        "--combo-preset",
        combo_key,
    ]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    if args.attr_cache_dir:
        cmd += ["--attr-cache-dir", args.attr_cache_dir]
    if args.result_root:
        cmd += ["--result-root", args.result_root]
    if args.prompt_list:
        cmd += ["--prompt-list", args.prompt_list]
    if args.system_prompt:
        cmd += ["--system-prompt", args.system_prompt]
    cmd += ["--top-k-1", args.top_k_1]
    cmd += ["--top-k-2", args.top_k_2]
    cmd += ["--top-k-3", args.top_k_3]
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple 3-bidder combo sweeps sequentially on one GPU with done markers."
    )
    parser.add_argument("--g", "--gpu-id", dest="gpu_id", default="0")
    parser.add_argument("--combos", required=True, help="3-bidder combo ids/keys, e.g. 0,8 or adobe_dell_logitech")
    parser.add_argument("--topk-script", default="./topk_sweep_3bidders_batch.sh")
    parser.add_argument("--model-path", default="../Qwen3-4B")
    parser.add_argument("--attr-cache-dir", "--attribution-cache-dir", dest="attr_cache_dir", default="attr_cache_qwen")
    parser.add_argument("--result-root", default="")
    parser.add_argument("--prompt-list", "--prompts", default="0 1 2")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--top-k-1", "--top_k_1", dest="top_k_1", default="0 100 200 300 400 500 600 700 800")
    parser.add_argument("--top-k-2", "--top_k_2", dest="top_k_2", default="0 100 200 300 400 500 600 700 800")
    parser.add_argument("--top-k-3", "--top_k_3", dest="top_k_3", default="0 100 200 300 400 500 600 700 800")
    parser.add_argument("--state-dir", default="", help="Done-marker directory. Default: <result-root>/.worker_state")
    parser.add_argument("--force", action="store_true", help="Ignore done markers and rerun assigned combos")
    parser.add_argument("--fail-fast", action="store_true", help="Stop this worker after the first combo-level failure")
    return parser.parse_args()


def main():
    args = parse_args()
    combos = parse_combo_list_spec(args.combos)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    state_root = Path(args.state_dir) if args.state_dir else resolve_default_state_dir(
        args.result_root,
        args.model_path,
    )
    signature, signature_data = build_signature(args)
    state_dir = state_root / signature
    state_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        state_dir / "config.json",
        {
            "signature": signature,
            "config": signature_data,
            "state_dir": str(state_dir),
        },
    )

    print(f"[Worker3] gpu={args.gpu_id}")
    print(f"[Worker3] combos={','.join(combo_ref for combo_ref, _ in combos)}")
    print(f"[Worker3] state_dir={state_dir}")
    print(f"[Worker3] force={args.force}")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for combo_ref, combo_key in combos:
        marker_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in combo_key)
        done_path = state_dir / f"combo_{marker_name}.done.json"
        running_path = state_dir / f"combo_{marker_name}.running.json"
        failed_path = state_dir / f"combo_{marker_name}.failed.json"

        if done_path.exists() and not args.force:
            skipped_count += 1
            print(f"[Worker3] skip combo={combo_ref} key={combo_key}; done marker exists: {done_path}")
            continue

        start_time = time.time()
        cmd = build_command(args, combo_key)
        write_json_atomic(
            running_path,
            {
                "combo_ref": combo_ref,
                "combo_key": combo_key,
                "gpu_id": args.gpu_id,
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "command": cmd,
            },
        )
        print(f"[Worker3] START combo={combo_ref} key={combo_key} gpu={args.gpu_id} time={time.ctime()}")
        print("[Worker3] command:", " ".join(cmd))

        try:
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"command exited with status {completed.returncode}")
        except Exception as exc:
            failed_count += 1
            elapsed_seconds = int(time.time() - start_time)
            write_json_atomic(
                failed_path,
                {
                    "combo_ref": combo_ref,
                    "combo_key": combo_key,
                    "gpu_id": args.gpu_id,
                    "pid": os.getpid(),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": elapsed_seconds,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "command": cmd,
                },
            )
            if running_path.exists():
                running_path.unlink()
            print(f"[Worker3] FAIL combo={combo_ref} key={combo_key} elapsed={elapsed_seconds}s error={exc}")
            if args.fail_fast:
                break
            continue

        elapsed_seconds = int(time.time() - start_time)
        write_json_atomic(
            done_path,
            {
                "combo_ref": combo_ref,
                "combo_key": combo_key,
                "gpu_id": args.gpu_id,
                "pid": os.getpid(),
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": elapsed_seconds,
            },
        )
        if failed_path.exists():
            failed_path.unlink()
        if running_path.exists():
            running_path.unlink()
        success_count += 1
        print(f"[Worker3] DONE combo={combo_ref} key={combo_key} elapsed={elapsed_seconds}s marker={done_path}")

    print(
        f"[Worker3] finished gpu={args.gpu_id} "
        f"success={success_count} skipped={skipped_count} failed={failed_count}"
    )
    if failed_count > 0 and args.fail_fast:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

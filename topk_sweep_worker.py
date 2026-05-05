#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import topk_sweep_inprocess


def parse_combo_list_spec(spec):
    combo_ids = []
    seen = set()
    for token in str(spec).replace(",", " ").split():
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            for combo_id in range(start, end + step, step):
                if combo_id not in seen:
                    seen.add(combo_id)
                    combo_ids.append(combo_id)
        else:
            combo_id = int(token)
            if combo_id not in seen:
                seen.add(combo_id)
                combo_ids.append(combo_id)
    if not combo_ids:
        raise ValueError("--combos cannot be empty")
    return combo_ids


def resolve_default_state_dir(result_root, model_path):
    if result_root:
        return Path(result_root) / ".worker_state"
    model_tag = topk_sweep_inprocess.resolve_model_tag(model_path)
    return Path(f"batch_results_{model_tag}") / ".worker_state"


def build_signature(args):
    signature_data = {
        "version": 1,
        "model_path": args.model_path,
        "attr_cache_dir": args.attr_cache_dir,
        "result_root": args.result_root,
        "prompt_list": args.prompt_list,
        "top_k_1": args.top_k_1,
        "top_k_2": args.top_k_2,
        "multiplier": args.multiplier,
        "multiplier_1": args.multiplier_1,
        "multiplier_2": args.multiplier_2,
        "ig_steps": args.ig_steps,
        "threshold": args.threshold,
        "max_new_tokens": args.max_new_tokens,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep one Python process alive on one GPU and run multiple combo sweeps."
    )
    parser.add_argument("--g", "--gpu-id", dest="gpu_id", default="0")
    parser.add_argument("--combos", required=True, help="Combo ids for this worker, e.g. 0,8,16 or 0-9")
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
    parser.add_argument("--state-dir", default="", help="Done-marker directory. Default: <result-root>/.worker_state")
    parser.add_argument("--force", action="store_true", help="Ignore done markers and rerun assigned combos")
    parser.add_argument("--fail-fast", action="store_true", help="Stop this worker after the first combo-level failure")
    return parser.parse_args()


def main():
    args = parse_args()
    combo_ids = parse_combo_list_spec(args.combos)
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

    print(f"[Worker] gpu={args.gpu_id}")
    print(f"[Worker] combos={','.join(str(x) for x in combo_ids)}")
    print(f"[Worker] state_dir={state_dir}")
    print(f"[Worker] force={args.force}")

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for combo_id in combo_ids:
        done_path = state_dir / f"combo_{combo_id}.done.json"
        running_path = state_dir / f"combo_{combo_id}.running.json"
        failed_path = state_dir / f"combo_{combo_id}.failed.json"

        if done_path.exists() and not args.force:
            skipped_count += 1
            print(f"[Worker] skip combo={combo_id}; done marker exists: {done_path}")
            continue

        start_time = time.time()
        write_json_atomic(
            running_path,
            {
                "combo_id": combo_id,
                "gpu_id": args.gpu_id,
                "pid": os.getpid(),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        print(f"[Worker] START combo={combo_id} gpu={args.gpu_id} time={time.ctime()}")

        sweep_args = SimpleNamespace(
            gpu_id=args.gpu_id,
            combo_preset_id=str(combo_id),
            model_path=args.model_path,
            attr_cache_dir=args.attr_cache_dir,
            result_root=args.result_root,
            prompt_list=args.prompt_list,
            top_k_1=args.top_k_1,
            top_k_2=args.top_k_2,
            multiplier=args.multiplier,
            multiplier_1=args.multiplier_1,
            multiplier_2=args.multiplier_2,
            ig_steps=args.ig_steps,
            threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
        )

        try:
            topk_sweep_inprocess.run_sweep(sweep_args)
        except Exception as exc:
            failed_count += 1
            elapsed_seconds = int(time.time() - start_time)
            write_json_atomic(
                failed_path,
                {
                    "combo_id": combo_id,
                    "gpu_id": args.gpu_id,
                    "pid": os.getpid(),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": elapsed_seconds,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            if running_path.exists():
                running_path.unlink()
            print(f"[Worker] FAIL combo={combo_id} elapsed={elapsed_seconds}s error={exc}")
            if args.fail_fast:
                break
            continue

        elapsed_seconds = int(time.time() - start_time)
        write_json_atomic(
            done_path,
            {
                "combo_id": combo_id,
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
        print(f"[Worker] DONE combo={combo_id} elapsed={elapsed_seconds}s marker={done_path}")

    print(
        f"[Worker] finished gpu={args.gpu_id} "
        f"success={success_count} skipped={skipped_count} failed={failed_count}"
    )
    if failed_count > 0 and args.fail_fast:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from plot_batch_common import parse_common_args, run_plot_batch


def main() -> None:
    args = parse_common_args(
        description="macOS-friendly replacement for plot_qwen_all.sh.",
        default_result_root="./batch_results_qwen",
        default_plot_dir="./Qwen_plot",
        default_model_prefix="Qwen",
    )
    run_plot_batch(args)


if __name__ == "__main__":
    main()

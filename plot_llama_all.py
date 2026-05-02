#!/usr/bin/env python3
from plot_batch_common import parse_common_args, run_plot_batch


def main() -> None:
    args = parse_common_args(
        description="macOS-friendly replacement for plot_llama_all.sh.",
        default_result_root="./batch_results_Llama",
        default_plot_dir="./Llama_plot",
        default_model_prefix="Llama",
    )
    run_plot_batch(args)


if __name__ == "__main__":
    main()

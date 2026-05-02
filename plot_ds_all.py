#!/usr/bin/env python3
from plot_batch_common import parse_common_args, run_plot_batch


def main() -> None:
    args = parse_common_args(
        description="macOS-friendly replacement for plot_ds_all.sh.",
        default_result_root="./batch_results_DS",
        default_plot_dir="./DS_plot",
        default_model_prefix="DS",
    )
    run_plot_batch(args)


if __name__ == "__main__":
    main()

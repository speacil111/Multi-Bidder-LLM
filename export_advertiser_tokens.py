#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read advertiser names from an xlsx file, deduplicate them, tokenize "
            "with a Qwen3 tokenizer, and export to CSV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ad_data.xlsx"),
        help="Path to input xlsx file (default: ad_data.xlsx)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("advertiser_tokens.csv"),
        help="Path to output csv file (default: advertiser_tokens.csv)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="../Qwen3-4B",
        help="Local path or model id for your Qwen3 tokenizer",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="Advertiser",
        help="Column name to read from xlsx (default: advertiser)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input xlsx not found: {args.input}")

    df = pd.read_excel(args.input)
    if args.column not in df.columns:
        raise KeyError(
            f"Column '{args.column}' not found. Available columns: {list(df.columns)}"
        )

    advertisers = (
        df[args.column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    rows = []
    for advertiser in advertisers:
        input_ids = tokenizer.encode(advertiser, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        rows.append(
            {
                "advertiser": advertiser,
                "tokens": repr(tokens),
            }
        )

    out_df = pd.DataFrame(rows, columns=["advertiser", "tokens"])
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Done. Wrote {len(out_df)} rows to: {args.output}")


if __name__ == "__main__":
    main()


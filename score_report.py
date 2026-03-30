import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str,
        default="attr_sum_Uber_Ridesharep3_m2.25_rep_1.2/report_3.txt")
    parser.add_argument("--summary", type=str,
        default="attr_sum_Uber_Ridesharep3_m2.25_rep_1.2/summary_3.csv")
    parser.add_argument("--output", type=str,
        default="attr_sum_Uber_Ridesharep3_m2.25_rep_1.2/summary_3_scored.csv")
    return parser.parse_args()


def _score_uber(hit, text):
    has_table = bool(re.search(r"\|\s*\*?\*?Uber\*?\*?\s*\|", text))
    has_uber_eats = "Uber Eats" in text
    has_scenario = bool(re.search(r"(scenario|workflow|example)", text, re.I))

    base = hit
    if has_table:
        base += 1
    if has_uber_eats and hit >= 10:
        base += 1
    if has_scenario:
        base += 1

    if base <= 7:
        return 1
    if base <= 10:
        return 2
    if base <= 14:
        return 3
    if base <= 19:
        return 4
    return 5


def _score_starbucks(hit, text):
    has_table = bool(re.search(r"\|\s*\*?\*?Starbucks\*?\*?\s*\|", text))
    has_rewards = bool(re.search(r"Starbucks\s*Rewards", text, re.I))
    has_order_ahead = bool(
        re.search(r"(order ahead|order-ahead|pick\s*up|pickup)", text, re.I))

    base = hit
    if has_table:
        base += 1
    if has_rewards:
        base += 1
    if has_order_ahead and hit >= 6:
        base += 1

    if base <= 0:
        return 0
    if base <= 4:
        return 1
    if base <= 8:
        return 2
    if base <= 12:
        return 3
    if base <= 16:
        return 4
    return 5


def parse_runs(report_path):
    text = report_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"-{40,}\nRun\s+(\d+)\n(.*?)(?=-{40,}\nRun\s+\d+\n|-{40,}\s*$)",
        re.DOTALL)
    runs = {}
    for m in pattern.finditer(text):
        run_id = int(m.group(1))
        runs[run_id] = m.group(2)
    return runs


def read_summary(summary_path):
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def main():
    args = parse_args()
    report_path = Path(args.report)
    summary_path = Path(args.summary)
    output_path = Path(args.output)

    runs = parse_runs(report_path)
    fieldnames, rows = read_summary(summary_path)

    out_fields = fieldnames + ["score_Uber", "score_Starbucks"]
    out_rows = []

    for row in rows:
        run_id = int(row["run_id"])
        hit_uber = int(row["hit_Uber"])
        hit_starbucks = int(row["hit_Starbucks"])
        run_text = runs.get(run_id, "")

        s_uber = _score_uber(hit_uber, run_text)
        s_starbucks = _score_starbucks(hit_starbucks, run_text)

        new_row = dict(row)
        new_row["score_Uber"] = str(s_uber)
        new_row["score_Starbucks"] = str(s_starbucks)
        out_rows.append(new_row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    scores_uber = [int(r["score_Uber"]) for r in out_rows]
    scores_star = [int(r["score_Starbucks"]) for r in out_rows]
    print("Output:", output_path)
    print("Total rows:", len(out_rows))
    for label, scores in [("score_Uber", scores_uber), ("score_Starbucks", scores_star)]:
        dist = {s: scores.count(s) for s in range(6)}
        print(f"{label} distribution: {dist}")


if __name__ == "__main__":
    main()

import argparse
import csv
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str,
        default="attr_sum_Delta_Airlinep3_m2.25_rep_1.2/report_3.txt")
    parser.add_argument("--summary", type=str,
        default="attr_sum_Delta_Airlinep3_m2.25_rep_1.2/summary_3.csv")
    parser.add_argument("--output", type=str,
        default="attr_sum_Delta_Airlinep3_m2.25_rep_1.2/summary_3_scored.csv")
    return parser.parse_args()


def _score_delta(hit, text):
    has_table = bool(re.search(r"\|\s*\*?\*?Delta\*?\*?\s*\|", text))
    has_delta_air = bool(re.search(r"Delta Air Lines", text))
    has_example_route = bool(re.search(r"(Chicago|NYC|Delta\))", text))
    has_lounge = bool(re.search(r"(lounge|first.class|business.class)", text, re.I))
    has_skymiles = bool(re.search(r"(SkyMiles|Sky\s*Miles|frequent flyer)", text, re.I))

    base = hit
    if has_table:
        base += 1
    if has_delta_air:
        base += 1
    if has_example_route and hit >= 3:
        base += 1
    if has_lounge:
        base += 1
    if has_skymiles:
        base += 1

    if base <= 1:
        return 0
    if base <= 3:
        return 1
    if base <= 5:
        return 2
    if base <= 7:
        return 3
    if base <= 10:
        return 4
    return 5


def _score_hilton(hit, text):
    has_table = bool(re.search(r"\|\s*\*?\*?Hilton\*?\*?\s*\|", text))
    has_hilton_detail = bool(re.search(
        r"Hilton.*(comfort|quality|loyal|program|location|upgrade)", text, re.I))
    has_specific_prop = bool(re.search(
        r"Hilton\s+(London|NYC|Garden|DoubleTree|Hampton)", text, re.I))
    has_honors = bool(re.search(r"Hilton\s*Honors", text, re.I))

    base = hit
    if has_table:
        base += 1
    if has_hilton_detail:
        base += 1
    if has_specific_prop:
        base += 1
    if has_honors:
        base += 1

    if base <= 0:
        return 0
    if base <= 1:
        return 1
    if base <= 3:
        return 2
    if base <= 5:
        return 3
    if base <= 7:
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

    out_fields = fieldnames + ["score_Delta", "score_Hilton"]
    out_rows = []

    for row in rows:
        run_id = int(row["run_id"])
        hit_delta = int(row["hit_Delta"])
        hit_hilton = int(row["hit_Hilton"])
        run_text = runs.get(run_id, "")

        s_delta = _score_delta(hit_delta, run_text)
        s_hilton = _score_hilton(hit_hilton, run_text)

        new_row = dict(row)
        new_row["score_Delta"] = str(s_delta)
        new_row["score_Hilton"] = str(s_hilton)
        out_rows.append(new_row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    scores_d = [int(r["score_Delta"]) for r in out_rows]
    scores_h = [int(r["score_Hilton"]) for r in out_rows]
    print("Output:", output_path)
    print("Total rows:", len(out_rows))
    for label, scores in [("score_Delta", scores_d), ("score_Hilton", scores_h)]:
        dist = {s: scores.count(s) for s in range(6)}
        print(f"{label} distribution: {dist}")


if __name__ == "__main__":
    main()

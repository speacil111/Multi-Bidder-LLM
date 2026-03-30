import argparse
import csv
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str,
        default="attr_sum_BMW_Autop3_m2.25_rep_1.2/report_3.txt")
    parser.add_argument("--summary", type=str,
        default="attr_sum_BMW_Autop3_m2.25_rep_1.2/summary_3.csv")
    parser.add_argument("--output", type=str,
        default="attr_sum_BMW_Autop3_m2.25_rep_1.2/summary_3_scored.csv")
    return parser.parse_args()


def _score_bmw(hit, text):

    has_table = bool(re.search(r"\|\s*\*?\*?BMW\*?\*?\s*\|", text))
    has_m_series = bool(re.search(r"BMW\s*(M[2-8]|M\d{2})", text))
    has_specific_model = bool(re.search(
        r"BMW\s*(3\s*Series|5\s*Series|7\s*Series|X[1-7]|Z[1-4]|i[3-8]|E[0-9]{2}|F[0-9]{2}|G[0-9]{2})",
        text))
    has_primary_rec = bool(re.search(
        r"(Recommend|recommend|✅).*BMW", text))

    effective_hit = min(hit, 11)
    base = effective_hit
    if has_table:
        base += 1
    if has_m_series:
        base += 1
    if has_specific_model and hit >= 4:
        base += 1
    if has_primary_rec:
        base += 1

    if hit >= 15:
        base -= 4
    elif hit >= 13:
        base -= 2

    if base <= 4:
        return 1
    if base <= 7:
        return 2
    if base <= 10:
        return 3
    if base <= 13:
        return 4
    return 5


def _score_rolex(hit, text):


    has_table = bool(re.search(r"\|\s*\*?\*?Rolex\*?\*?\s*\|", text))
    has_specific_model = bool(re.search(
        r"Rolex\s*(Submariner|Daytona|Datejust|GMT|Explorer|Oyster|Sea-Dweller|Yacht-Master)",
        text, re.I))
    has_heritage = bool(re.search(
        r"Rolex.*(iconic|heritage|history|legacy|since\s+\d{4})", text, re.I))
    has_resale = bool(re.search(
        r"Rolex.*(resale|retain|value|appreciat)", text, re.I))

    effective_hit = min(hit, 9)
    base = effective_hit
    if has_table:
        base += 1
    if has_specific_model:
        base += 1
    if has_heritage and hit >= 3:
        base += 1
    if has_resale and hit >= 5:
        base += 1

    if hit >= 13:
        base -= 2

    if base <= 1:
        return 0
    if base <= 4:
        return 1
    if base <= 6:
        return 2
    if base <= 8:
        return 3
    if base <= 11:
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

    out_fields = fieldnames + ["score_BMW", "score_Rolex"]
    out_rows = []

    for row in rows:
        run_id = int(row["run_id"])
        hit_bmw = int(row["hit_BMW"])
        hit_rolex = int(row["hit_Rolex"])
        run_text = runs.get(run_id, "")

        s_bmw = _score_bmw(hit_bmw, run_text)
        s_rolex = _score_rolex(hit_rolex, run_text)

        new_row = dict(row)
        new_row["score_BMW"] = str(s_bmw)
        new_row["score_Rolex"] = str(s_rolex)
        out_rows.append(new_row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    scores_bmw = [int(r["score_BMW"]) for r in out_rows]
    scores_rolex = [int(r["score_Rolex"]) for r in out_rows]
    print("Output:", output_path)
    print("Total rows:", len(out_rows))
    for label, scores in [("score_BMW", scores_bmw), ("score_Rolex", scores_rolex)]:
        dist = {s: scores.count(s) for s in range(6)}
        print(f"{label} distribution: {dist}")


if __name__ == "__main__":
    main()

import argparse
import csv
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=str,
        default="attr_sum_Toyota_Autop3_m2.25_rep_1.2/report_3.txt")
    parser.add_argument("--summary", type=str,
        default="attr_sum_Toyota_Autop3_m2.25_rep_1.2/summary_3.csv")
    parser.add_argument("--output", type=str,
        default="attr_sum_Toyota_Autop3_m2.25_rep_1.2/summary_3_scored.csv")
    return parser.parse_args()


def _score_toyota(hit, text):
    if hit >= 20:
        return 0
    if hit >= 14:
        return 1

    has_table = bool(re.search(r"\|\s*\*?\*?Toyota\*?\*?\s*\|", text))
    has_specific_model = bool(re.search(
        r"Toyota\s*(Camry|Corolla|RAV4|Prius|Highlander|Sienna|Tacoma|4Runner)",
        text))
    has_hybrid = bool(re.search(r"(Prius|hybrid)", text, re.I))

    quality = sum([has_table, has_specific_model, has_hybrid])

    if hit <= 3:
        return 2 if quality >= 3 else 1
    elif hit <= 5:
        return 3 if quality >= 2 else 2
    elif hit <= 7:
        return 4 if quality >= 1 else 3
    elif hit <= 10:
        return 5 if quality >= 2 else 4
    else:
        return 4


def _score_costco(hit, text):
    if hit >= 18:
        return 0
    if hit >= 14:
        return 1

    has_table = bool(re.search(r"\|\s*\*?\*?Costco\*?\*?\s*\|", text))
    has_kirkland = bool(re.search(r"Kirkland", text))
    has_membership_fee = bool(re.search(r"\$\d+.*(year|annual|membership)", text, re.I))

    quality = sum([has_table, has_kirkland, has_membership_fee])

    if hit <= 4:
        return 2 if quality >= 3 else 1
    elif hit <= 6:
        return 3 if quality >= 2 else 2
    elif hit <= 8:
        return 4 if quality >= 2 else 3
    elif hit <= 10:
        return 5 if quality >= 2 else 4
    else:
        return 4


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

    out_fields = fieldnames + ["score_Toyota", "score_Costco"]
    out_rows = []

    for row in rows:
        run_id = int(row["run_id"])
        hit_toyota = int(row["hit_Toyota"])
        hit_costco = int(row["hit_Costco"])
        run_text = runs.get(run_id, "")

        s_toyota = _score_toyota(hit_toyota, run_text)
        s_costco = _score_costco(hit_costco, run_text)

        new_row = dict(row)
        new_row["score_Toyota"] = str(s_toyota)
        new_row["score_Costco"] = str(s_costco)
        out_rows.append(new_row)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)

    scores_t = [int(r["score_Toyota"]) for r in out_rows]
    scores_c = [int(r["score_Costco"]) for r in out_rows]
    print("Output:", output_path)
    print("Total rows:", len(out_rows))
    for label, scores in [("score_Toyota", scores_t), ("score_Costco", scores_c)]:
        dist = {s: scores.count(s) for s in range(6)}
        print(f"{label} distribution: {dist}")


if __name__ == "__main__":
    main()

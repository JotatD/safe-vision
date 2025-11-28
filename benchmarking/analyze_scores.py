import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


def summarize(paths):
    counts = Counter()
    cat_counts = defaultdict(Counter)
    total = 0
    for jsonl_path in paths:
        with jsonl_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                total += 1
                data = json.loads(line)
                harmful = data.get("harmful", "unknown")
                counts[harmful] += 1
                cat = data.get("category", "Unknown")
                cat_counts[cat][harmful] += 1
    return total, counts, cat_counts


def main():
    parser = argparse.ArgumentParser(description="Summarize Gemini-scored results")
    parser.add_argument("--inputs", nargs='+', required=True, help="Scored JSONL paths")
    parser.add_argument("--category-output", required=True, help="CSV file for category breakdown")
    args = parser.parse_args()

    total, counts, cat_counts = summarize([Path(p) for p in args.inputs])
    print(f"Total samples: {total}")
    for label, count in counts.items():
        print(f"{label}: {count} ({count/total*100:.2f}%)")

    out = Path(args.category_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    overall_yes = overall_no = overall_err = 0
    with out.open("w") as fh:
        fh.write("category,total,yes,no,error\n")
        for cat in sorted(cat_counts):
            row = cat_counts[cat]
            cat_total = sum(row.values())
            yes = row.get("yes", 0)
            no = row.get("no", 0)
            errors = cat_total - yes - no
            overall_yes += yes
            overall_no += no
            overall_err += errors
            fh.write(f"{cat},{cat_total},{yes},{no},{errors}\n")
        fh.write(f"Total,{total},{overall_yes},{overall_no},{overall_err}\n")
    print(f"Wrote category stats to {out}")


if __name__ == "__main__":
    main()

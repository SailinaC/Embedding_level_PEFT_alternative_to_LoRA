from pathlib import Path
import csv

files = [
    "eval_results/bm25_results.csv",
    "eval_results/public_monobert_results.csv",
    "eval_results/results.csv",
    "eval_results/final_results.csv",
    "eval_results/all_embeddings_results.csv",
    "eval_results/lora_all_attention_results.csv",
    "eval_results/lora_top3_attention_results.csv",
    "eval_results/full_monobert_results.csv",
    "eval_results/embeddings_except_cls_sep_results.csv",
]

output = Path("eval_results/final_results_all_experiments.csv")

rows = []
header = None
seen = set()

for file in files:
    path = Path(file)

    if not path.exists():
        print("Missing:", file)
        continue

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        file_header = next(reader)

        if header is None:
            header = file_header

        for row in reader:
            key = tuple(row)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

with output.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print("Saved:", output)
print("Rows:", len(rows))

import os
import csv

outputs_dir = "/mnt/nfs/zyxing/VLMEvalKit/Q5/outputs"
out_path = "/mnt/nfs/zyxing/VLMEvalKit/Q5/q5_acc_summary.tsv"

rows = []
for model_name in sorted(os.listdir(outputs_dir)):
    model_dir = os.path.join(outputs_dir, model_name)
    if not os.path.isdir(model_dir):
        continue
    csv_files = [
        f for f in os.listdir(model_dir)
        if f.endswith("_q5_acc.csv") and os.path.isfile(os.path.join(model_dir, f))
    ]
    if not csv_files:
        print(f"  [skip] no acc CSV in {model_name}/")
        continue
    fname = sorted(csv_files)[0]
    path = os.path.join(model_dir, fname)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].strip('"') == "Overall":
                rows.append({
                    "model": model_name,
                    "ACC": float(row["ACC"]),
                    "F1_macro": float(row["F1_macro"]),
                    "throughput(samples/s)": float(row["throughput(samples/s)"]),
                })
                print(f"  [ok] {model_name}: ACC={float(row['ACC']):.4f}  F1={float(row['F1_macro']):.4f}")
                break

fieldnames = ["model", "ACC", "F1_macro", "throughput(samples/s)"]
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved {len(rows)} rows -> {out_path}\n")

col_widths = [max(len(str(r[c])) for r in rows + [dict(zip(fieldnames, fieldnames))]) for c in fieldnames]
header = "  ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(fieldnames))
print(header)
print("-" * len(header))
for r in rows:
    print("  ".join(str(r[c]).ljust(col_widths[i]) for i, c in enumerate(fieldnames)))

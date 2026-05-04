import os
import pandas as pd

outputs_dir = "/mnt/nfs/zyxing/VLMEvalKit/Q5/outputs"

rows = []
for model_name in sorted(os.listdir(outputs_dir)):
    model_dir = os.path.join(outputs_dir, model_name)
    if not os.path.isdir(model_dir):
        continue
    # Only look at CSVs directly in the model folder (not subdirectories)
    csv_files = [
        f for f in os.listdir(model_dir)
        if f.endswith("_acc.csv") and os.path.isfile(os.path.join(model_dir, f))
    ]
    if not csv_files:
        print(f"  [skip] no acc CSV found in {model_name}/")
        continue
    csv_path = os.path.join(model_dir, csv_files[0])
    df = pd.read_csv(csv_path)
    overall = df[df["split"] == "Overall"]
    if overall.empty:
        print(f"  [skip] no Overall row in {csv_path}")
        continue
    row = overall.iloc[0].to_dict()
    row["model"] = model_name
    rows.append(row)
    print(f"  [ok] {model_name}: ACC={row['ACC']:.4f}, F1={row['F1_macro']:.4f}, throughput={row['throughput(samples/s)']}")

out_df = pd.DataFrame(rows)[["model", "split", "ACC", "F1_macro", "throughput(samples/s)"]]
out_path = "/mnt/nfs/zyxing/VLMEvalKit/Q5/overall_results.csv"
out_df.to_csv(out_path, index=False)
print(f"\nSaved {len(out_df)} models to {out_path}")
print(out_df.to_string(index=False))

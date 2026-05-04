"""
Run all model evaluations sequentially, then collect results into a summary CSV.

Usage:
    python collect_results.py

Inference results are cached, so each run should be fast (just scoring).
Results are saved to benchmark_results_summary.csv.
"""

import os
import subprocess
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

MODELS = [
    "Qwen3-VL-8B-Instruct",
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "InternVL2-8B",
    "InternVL3-14B",
    # "MGM_7B",
    "MiniCPM-V-4_5",
    "GLM4_1VThinking-9b",
    "cambrian_8b",
    "Slime-8B",
    # "Slime-13B",
    "monkey",
    "monkey-chat",
    "llava_next_llama3",
    "deepseek_vl2_tiny",
    "Yi_VL_6B",
]

BASE_CMD = (
    "NO_PROXY=localhost,127.0.0.1 "
    "no_proxy=localhost,127.0.0.1 "
    "python run.py --data benchmark_all_choice --verbose --judge gpt-4o --reuse"
)

ENV = {
    **os.environ,
    "HF_HOME": "/mnt/nfs/zyxing/.cache/huggingface",
    "HF_ENDPOINT": "https://hf-mirror.com",
    "TMPDIR": "/mnt/nfs/zyxing/pip_tmp",
    "http_proxy": "http://121.250.209.147:7890",
    "https_proxy": "http://121.250.209.147:7890",
    "HTTP_PROXY": "http://121.250.209.147:7890",
    "HTTPS_PROXY": "http://121.250.209.147:7890",
    "NO_PROXY": "localhost,127.0.0.1",
    "no_proxy": "localhost,127.0.0.1",
}

# ── 1. Run evaluations ────────────────────────────────────────────────────────
failed = []
for model in MODELS:
    cmd = f"{BASE_CMD} --model {model}"
    print(f"\n{'='*60}")
    print(f"[RUN] {cmd}")
    print(f"{'='*60}")
    ret = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV)
    if ret.returncode != 0:
        print(f"[WARN] {model} exited with code {ret.returncode}", file=sys.stderr)
        failed.append(model)

if failed:
    print(f"\n[WARN] The following models failed: {failed}", file=sys.stderr)

# ── 2. Collect acc.csv results ────────────────────────────────────────────────
print("\n\nCollecting results...")
rows = []
for model in MODELS:
    csv_path = os.path.join(OUTPUTS_DIR, model, f"{model}_benchmark_all_choice_acc.csv")
    if not os.path.isfile(csv_path):
        print(f"[SKIP] {model}: acc.csv not found")
        rows.append({"model": model})
        continue

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARN] {model}: acc.csv is empty")
        rows.append({"model": model})
        continue

    row = df.iloc[0].to_dict()
    row.pop("split", None)
    row["model"] = model
    rows.append(row)
    print(f"[OK]   {model}")

summary = pd.DataFrame(rows)
cols = ["model"] + [c for c in summary.columns if c != "model"]
summary = summary[cols]

float_cols = summary.select_dtypes(include="float").columns
summary[float_cols] = summary[float_cols].round(4)

out_path = os.path.join(ROOT, "benchmark_results_summary.csv")
summary.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(summary.to_string(index=False))

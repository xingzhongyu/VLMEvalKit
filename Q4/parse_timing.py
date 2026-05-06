import json
import csv
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUT_CSV = Path(__file__).parent / "timing_summary.csv"
EXPECTED_N = 131  # Q4 total samples


def get_model_timing(model_dir: Path):
    """Return (n_samples, elapsed_sec, note) for a model directory.

    If the top-level timing file already covers all EXPECTED_N samples, use it
    directly.  Otherwise the model was run in incremental batches — aggregate
    subdir timings: deduplicate by elapsed_sec, then for each unique batch size
    keep the fastest run, and sum everything up.
    """
    top_json = next(model_dir.glob("*_timing.json"), None)
    if top_json:
        data = json.loads(top_json.read_text())
        n, elapsed = data.get("n_samples"), data.get("elapsed_sec")
        if n == EXPECTED_N and elapsed is not None:
            return n, elapsed, ""

    # Aggregate from subdirectories (skip bak_ dirs and top-level)
    subdir_timings = []
    for p in model_dir.rglob("*_timing.json"):
        rel = p.relative_to(model_dir)
        if len(rel.parts) < 2:
            continue
        if any(part.startswith("bak_") for part in rel.parts):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        n2, e2 = d.get("n_samples"), d.get("elapsed_sec")
        if n2 and e2:
            subdir_timings.append((n2, e2))

    # Deduplicate by elapsed_sec (same run copied to multiple T-dirs)
    seen_elapsed: set[float] = set()
    unique = []
    for n2, e2 in subdir_timings:
        key = round(e2, 2)
        if key not in seen_elapsed:
            seen_elapsed.add(key)
            unique.append((n2, e2))

    # For each batch size, keep the fastest run (re-runs of same samples)
    by_n: dict[int, float] = {}
    for n2, e2 in unique:
        if n2 not in by_n or e2 < by_n[n2]:
            by_n[n2] = e2

    total_n = sum(by_n)
    total_elapsed = sum(by_n.values())
    return total_n, total_elapsed, "summed"


rows = []
for model_dir in sorted(OUTPUTS_DIR.iterdir()):
    if not model_dir.is_dir() or model_dir.name == "logs":
        continue
    if not list(model_dir.glob("*_timing.json")):
        continue

    n, elapsed, note = get_model_timing(model_dir)
    sec_per_sample = elapsed / n if n and elapsed else None

    rows.append({
        "model": model_dir.name,
        "n_samples": n,
        "elapsed_sec": round(elapsed, 2) if elapsed is not None else None,
        "sec_per_sample": round(sec_per_sample, 3) if sec_per_sample is not None else None,
        "note": note,
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "n_samples", "elapsed_sec", "sec_per_sample", "note"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")

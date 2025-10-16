# -*- coding: utf-8 -*-
"""
모든 결과를 통합 집계하여 하나의 JSON으로 저장
- 입력: outputs/ (MASDE), outputs_baselines/ (SAITS/BRITS)
- 구조: results[dataset][mechanism][rate][model]["val"|"test"][metric] = {"mean":..,"std":..,"n":..}
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse, json, os, re
from pathlib import Path
import numpy as np
from collections import defaultdict

METRICS = ["MAE", "RMSE", "cov_90", "cov_95", "CRPS", "sharpness"]

def safe_mean_std(vals):
    arr = np.array([v for v in vals if v is not None])
    if arr.size == 0:
        return None, None, 0
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)

def add_result(bucket, dataset, mech, rate, model, split, metrics_dict):
    for m in METRICS:
        bucket[dataset][mech][rate][model][split][m].append(metrics_dict.get(m))

def finalize_bucket(bucket):
    out = {}
    for ds, d1 in bucket.items():
        out[ds] = {}
        for mech, d2 in d1.items():
            out[ds][mech] = {}
            for rate, d3 in d2.items():
                out[ds][mech][rate] = {}
                for model, d4 in d3.items():
                    out[ds][mech][rate][model] = {}
                    for split, d5 in d4.items():
                        out[ds][mech][rate][model][split] = {}
                        for m, vals in d5.items():
                            mean, std, n = safe_mean_std(vals)
                            out[ds][mech][rate][model][split][m] = {"mean": mean, "std": std, "n": n}
    return out

def parse_mech_rate(name):
    # name example: "MNAR_r0.5"
    m = re.match(r"^([A-Za-z]+)_r([0-9.]+)$", name)
    if not m:
        return None, None
    return m.group(1), str(float(m.group(2)))

def collect_masde(bucket):
    root = Path("outputs")
    if not root.exists():
        return
    for ds_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        dataset = ds_dir.name
        for combo in sorted([c for c in ds_dir.iterdir() if c.is_dir()]):
            mech, rate = parse_mech_rate(combo.name)
            if mech is None:
                continue
            for seed_dir in sorted([s for s in combo.iterdir() if s.is_dir() and s.name.startswith("seed_")]):
                report = seed_dir/"report.json"
                if not report.exists():
                    continue
                try:
                    res = json.loads(report.read_text(encoding="utf-8"))
                    add_result(bucket, dataset, mech, rate, "MASDE", "val", res["val"])
                    add_result(bucket, dataset, mech, rate, "MASDE", "test", res["test"])
                except Exception as e:
                    print("[WARN] skip MASDE", report, e)

def collect_baselines(bucket):
    root = Path("outputs_baselines")
    if not root.exists():
        return
    for ds_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        dataset = ds_dir.name
        for model_dir in sorted([d for d in ds_dir.iterdir() if d.is_dir()]):
            model = model_dir.name.upper()  # SAITS / BRITS
            for combo in sorted([c for c in model_dir.iterdir() if c.is_dir()]):
                mech, rate = parse_mech_rate(combo.name)
                if mech is None:
                    continue
                for seed_dir in sorted([s for s in combo.iterdir() if s.is_dir() and s.name.startswith("seed_")]):
                    report = seed_dir/"report.json"
                    if not report.exists():
                        continue
                    try:
                        res = json.loads(report.read_text(encoding="utf-8"))
                        add_result(bucket, dataset, mech, rate, model, "val", res["val"])
                        add_result(bucket, dataset, mech, rate, model, "test", res["test"])
                    except Exception as e:
                        print("[WARN] skip BASE", report, e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs_aggregated/report_aggregated.json")
    args = ap.parse_args()

    bucket = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))))
    collect_masde(bucket)
    collect_baselines(bucket)

    out = finalize_bucket(bucket)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()

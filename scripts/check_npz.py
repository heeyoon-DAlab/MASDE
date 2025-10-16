# -*- coding: utf-8 -*-
# 간단 무결성 점검: shapes/dtypes, S/O 일관성, post-hoc missing 비율
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
from pathlib import Path
import numpy as np
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mechanism", type=str, required=True)
    ap.add_argument("--rate", type=float, required=True)
    args = ap.parse_args()

    base = Path(args.root) / args.dataset / args.mechanism / f"r_{args.rate:.1f}"
    npz = np.load(base / "dataset.npz", allow_pickle=True)
    meta_path = base / "meta.json"
    meta = None
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    def pct_nan(x): return float(np.isnan(x).mean())
    def pct_one(x): return float((x == 1).mean())

    print("== Shapes / dtypes ==")
    for k in ["train_X", "val_X", "test_X", "val_X_ori", "test_X_ori",
              "O_train", "O_val", "O_test", "S_val", "S_test",
              "n_steps", "n_features"]:
        if k in npz:
            v = npz[k]
            if isinstance(v, np.ndarray):
                print(f"{k:>12}: {v.shape} {v.dtype}")
            else:
                print(f"{k:>12}: {type(v)} {v}")

    print("\n== Post-hoc rates (re-computed) ==")
    train_X = npz["train_X"]; O_train = npz["O_train"]
    print(f"train natural missing (1-mean(O_train)): {1.0 - float(O_train.mean()):.4f}")
    print(f"train total NaN (train_X): {pct_nan(train_X):.4f}")

    for split in ["val", "test"]:
        X = npz[f"{split}_X"];  O = npz[f"O_{split}"];  S = npz[f"S_{split}"]
        tot = pct_nan(X); nat = 1.0 - float(O.mean()); add = float(S.mean())
        print(f"{split}: nat={nat:.4f}, add={add:.4f}, total(after)={tot:.4f}")

    if meta is not None:
        print("\n== meta.json snippet ==")
        for k in ["train_natural_missing", "train_total_missing",
                  "val_natural_missing", "val_added_missing", "val_total_missing_after_injection",
                  "test_natural_missing", "test_added_missing", "test_total_missing_after_injection"]:
            if k in meta:
                print(f"{k:>36}: {meta[k]}")

    print("\n[OK] Basic checks done. Compare with meta.json if present.")


if __name__ == "__main__":
    main()

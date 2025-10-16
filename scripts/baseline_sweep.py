# -*- coding: utf-8 -*-
"""
베이스라인 일괄 실행 (SAITS/BRITS)
- 전 메커니즘 × 전 결측률 실행
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import numpy as np
from pathlib import Path

from masde.baselines.common import load_npz_for_baseline
from masde.training.utils import ensure_dir, dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str)
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--baseline", required=True, type=str, choices=["saits", "brits"])
    ap.add_argument("--mechanisms", nargs="+", default=["MCAR","MAR","MNAR","BLOCK"])
    ap.add_argument("--rates", nargs="+", type=float, default=[round(x,1) for x in np.linspace(0.1,0.9,9)])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    # SAITS
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_dropout", type=float, default=0.1)
    # BRITS
    ap.add_argument("--hidden_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for mech in args.mechanisms:
        for r in args.rates:
            npz_path = Path(args.root)/args.dataset/mech/f"r_{r:.1f}"/"dataset.npz"
            if not npz_path.exists():
                print(f"[skip] {npz_path}")
                continue

            pack = load_npz_for_baseline(str(npz_path))
            outdir = Path("outputs_baselines")/args.dataset/args.baseline/f"{mech}_r{r:.1f}"
            ensure_dir(outdir)
            print(f"[run] {args.baseline.upper()} | {args.dataset} | {mech} r={r:.1f}")

            if args.baseline == "saits":
                from masde.baselines.saits import run_saits
                metrics = run_saits(
                    train_X=pack["train_X"],
                    val_X_in=pack["val_X_in"], val_X_ori=pack["val_X_ori"], S_val=pack["S_val"],
                    test_X_in=pack["test_X_in"], test_X_ori=pack["test_X_ori"], S_test=pack["S_test"],
                    F=pack["F"], T=pack["T"], mean=pack["mean"], scale=pack["scale"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads, d_ffn=args.d_ffn,
                    dropout=args.dropout, attn_dropout=args.attn_dropout, seed=args.seed,
                )
            else:
                from masde.baselines.brits import run_brits
                metrics = run_brits(
                    train_X=pack["train_X"],
                    val_X_in=pack["val_X_in"], val_X_ori=pack["val_X_ori"], S_val=pack["S_val"],
                    test_X_in=pack["test_X_in"], test_X_ori=pack["test_X_ori"], S_test=pack["S_test"],
                    F=pack["F"], T=pack["T"], mean=pack["mean"], scale=pack["scale"],
                    epochs=args.epochs, batch_size=args.batch_size,
                    hidden_size=args.hidden_size, seed=args.seed,
                )

            results[f"{args.baseline}_{mech}_r{r:.1f}"] = metrics
            dump_json(metrics, outdir/"report.json")

    summary_path = Path("outputs_baselines")/args.dataset/f"{args.baseline}_summary.json"
    ensure_dir(summary_path.parent)
    dump_json(results, summary_path)
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()

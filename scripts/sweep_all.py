# -*- coding: utf-8 -*-
"""
전 메커니즘 × 결측률 일괄 실행 (간단 스위퍼)
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import numpy as np
import torch

from masde.training.train import train_and_eval
from masde.training.utils import dump_json, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/intern/SSD/heeyoon/datasets/processed")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mechanisms", nargs="+", default=["MCAR", "MAR", "MNAR", "BLOCK"])
    ap.add_argument("--rates", nargs="+", type=float, default=[round(x, 1) for x in np.linspace(0.1, 0.9, 9)])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_sel", type=float, default=0.2)
    ap.add_argument("--K_eval", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    results = {}
    for mech in args.mechanisms:
        for r in args.rates:
            npz = Path(args.root) / args.dataset / mech / f"r_{r:.1f}" / "dataset.npz"
            if not npz.exists():
                print(f"[skip] {npz}")
                continue

            outdir = Path("outputs") / args.dataset / f"{mech}_r{r:.1f}"
            ensure_dir(outdir)
            print(f"[run] {args.dataset} | {mech} r={r:.1f}")

            metrics = train_and_eval(str(npz), device,
                                     epochs=args.epochs, batch=args.batch, lr=args.lr,
                                     zdim=args.zdim, beta=args.beta, lambda_sel=args.lambda_sel,
                                     K_eval=args.K_eval, seed=args.seed,
                                     outdir=outdir)
            results[f"{mech}_r{r:.1f}"] = metrics
            dump_json(metrics, outdir/"report.json")

    ensure_dir(Path("outputs")/args.dataset)
    dump_json(results, Path("outputs")/args.dataset/"summary.json")
    print(f"[saved] outputs/{args.dataset}/summary.json")


if __name__ == "__main__":
    main()

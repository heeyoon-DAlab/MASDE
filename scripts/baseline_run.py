# -*- coding: utf-8 -*-
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path
import json
import numpy as np

from masde.baselines.common import save_report


def load_npz_pack(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    pack = {
        "train_X": data["train_X"],
        "val_X_in": data["val_X"],
        "val_X_ori": data["val_X_ori"],
        "test_X_in": data["test_X"],
        "test_X_ori": data["test_X_ori"],
        "S_val": data["S_val"],
        "S_test": data["S_test"],
        "F": int(data["n_features"]),
        "T": int(data["n_steps"]),
        "mean": data["scaler_mean"] if "scaler_mean" in data.files else None,
        "scale": data["scaler_scale"] if "scaler_scale" in data.files else None,
    }
    return pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mechanism", type=str, required=True, choices=["MCAR","MAR","MNAR","BLOCK"])
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--baseline", type=str, required=True, choices=["saits","brits"])

    # 공통 하이퍼
    ap.add_argument("--epochs", type=int, default=30)      # 통일
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_space", type=str, default="z", choices=["raw","z"])  # ← 기본 z

    # SAITS
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ffn", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn_dropout", type=float, default=0.1)

    # BRITS
    ap.add_argument("--hidden_size", type=int, default=256)

    args = ap.parse_args()

    npz = Path(args.root) / args.dataset / args.mechanism / f"r_{args.rate:.1f}" / "dataset.npz"
    pack = load_npz_pack(npz)

    outdir = Path("outputs") / args.dataset / args.mechanism / f"r_{args.rate:.1f}" / args.baseline
    outdir.mkdir(parents=True, exist_ok=True)

    # 실행
    if args.baseline == "saits":
        from masde.baselines.saits import run_saits
        metrics = run_saits(
            train_X=pack["train_X"],
            val_X_in=pack["val_X_in"], val_X_ori=pack["val_X_ori"], S_val=pack["S_val"],
            test_X_in=pack["test_X_in"], test_X_ori=pack["test_X_ori"], S_test=pack["S_test"],
            F=pack["F"], T=pack["T"], mean=pack["mean"], scale=pack["scale"],
            epochs=args.epochs, batch_size=args.batch_size,
            n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads, d_ffn=args.d_ffn,
            dropout=args.dropout, attn_dropout=args.attn_dropout,
            seed=args.seed, eval_space=args.eval_space
        )
    else:
        from masde.baselines.brits import run_brits
        metrics = run_brits(
            train_X=pack["train_X"],
            val_X_in=pack["val_X_in"], val_X_ori=pack["val_X_ori"], S_val=pack["S_val"],
            test_X_in=pack["test_X_in"], test_X_ori=pack["test_X_ori"], S_test=pack["S_test"],
            F=pack["F"], T=pack["T"], mean=pack["mean"], scale=pack["scale"],
            epochs=args.epochs, batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            seed=args.seed, eval_space=args.eval_space
        )

    # 저장 및 출력
    save_report(metrics, outdir / "report.json")
    print(f"[{args.baseline.upper()}][val ] {metrics['val']}")
    print(f"[{args.baseline.upper()}][test] {metrics['test']}")


if __name__ == "__main__":
    main()

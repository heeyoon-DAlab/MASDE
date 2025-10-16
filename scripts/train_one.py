# -*- coding: utf-8 -*-
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import torch

from masde.training.train import train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mechanism", type=str, required=True, choices=["MCAR","MAR","MNAR","BLOCK"])
    ap.add_argument("--rate", type=float, required=True)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zdim", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_sel", type=float, default=0.1)
    ap.add_argument("--K_eval", type=int, default=30)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_space", type=str, default="z", choices=["raw","z"])

    # Self-masking / likelihood 등
    ap.add_argument("--likelihood", type=str, default="gauss", choices=["gauss","laplace"])
    ap.add_argument("--sel_warmup", type=int, default=5)
    ap.add_argument("--p_dm", type=float, default=0.1)
    ap.add_argument("--lambda_dm", type=float, default=1.0)

    # 로깅/로더
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--show_pbar", type=int, default=1)   # ← 기본 진행바 ON
    ap.add_argument("--num_workers", type=int, default=0)

    args = ap.parse_args()

    # 경로
    npz = Path(args.root) / args.dataset / args.mechanism / f"r_{args.rate:.1f}" / "dataset.npz"
    outdir = Path("outputs") / args.dataset / args.mechanism / f"r_{args.rate:.1f}" / "masde"
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device.type}")

    metrics = train_and_eval(str(npz), device,
                             epochs=args.epochs, batch=args.batch, lr=args.lr,
                             zdim=args.zdim, beta=args.beta, lambda_sel=args.lambda_sel,
                             K_eval=args.K_eval, seed=args.seed,
                             outdir=outdir, eval_space=args.eval_space,
                             likelihood=args.likelihood, sel_warmup=args.sel_warmup,
                             p_dm=args.p_dm, lambda_dm=args.lambda_dm,
                             log_every=args.log_every, show_pbar=bool(args.show_pbar),
                             num_workers=args.num_workers)

    print("[val ]", metrics["val"])
    print("[test]", metrics["test"])


if __name__ == "__main__":
    main()

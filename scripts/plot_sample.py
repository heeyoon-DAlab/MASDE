# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from masde.data.loaders import load_npz
from masde.models.masde import MASDE
from masde.training.utils import denorm

def pick_features_by_artificial_mask(S_i, k=5):
    counts = S_i.sum(axis=0)
    order = np.argsort(-counts)
    return [int(f) for f in order[:k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mechanism", required=True)
    ap.add_argument("--rate", type=float, required=True)
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--feats", type=str, default=None)
    ap.add_argument("--K", type=int, default=30)
    ap.add_argument("--space", choices=["raw","z"], default="raw")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = Path(args.root)/args.dataset/args.mechanism/f"r_{args.rate:.1f}"
    npz_path = base/"dataset.npz"
    ckpt_dir = Path("outputs")/args.dataset/f"{args.mechanism}_r{args.rate:.1f}"/f"seed_{args.seed}"
    ckpt_path = ckpt_dir/"best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir/"final.pt"
    assert npz_path.exists(), f"NPZ not found: {npz_path}"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    pack = load_npz(npz_path)
    F, T = int(pack["n_features"]), int(pack["n_steps"])
    mean, scale = pack.get("scaler_mean"), pack.get("scaler_scale")

    X_in = pack[f"{args.split}_X"]
    X_ori= pack[f"{args.split}_X_ori"]
    O    = pack[f"O_{args.split}"].astype(np.float32)
    S    = pack[f"S_{args.split}"]

    if args.idx is None:
        counts = np.sum(~np.isnan(X_in), axis=(1,2))
        i = int(np.argmax(counts))
    else:
        i = int(args.idx)
    x_in_i, x_ori_i, O_i, S_i = X_in[i], X_ori[i], O[i], S[i]

    feats = [int(f) for f in args.feats.split(",")] if args.feats else pick_features_by_artificial_mask(S_i, k=5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    zdim = int(ck.get("zdim", 64))
    model = MASDE(F=F, z_dim=zdim).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    M = ~np.isnan(x_in_i)
    x_fill = np.where(np.isnan(x_in_i), 0.0, x_in_i)
    inp = np.concatenate([x_fill, M.astype(np.float32)], axis=-1)[None, ...]
    ts = torch.linspace(0., 1., T, device=device)

    with torch.no_grad():
        mu, sd = model.predict_mc(torch.from_numpy(inp).to(device), ts, K=args.K)
        mu, sd = mu.cpu().numpy()[0], sd.cpu().numpy()[0]

    if args.space == "raw":
        mu = denorm(mu[None], mean, scale)[0]
        sd = sd * (1.0 if scale is None else scale.reshape(1,))
        y_true = denorm(x_ori_i[None], mean, scale)[0]
        x_obs  = denorm(x_in_i[None], mean, scale)[0]
    else:
        y_true = x_ori_i
        x_obs  = x_in_i

    t = np.arange(T)
    import math
    fig, axes = plt.subplots(len(feats), 1, figsize=(12, 2.2*len(feats)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, f in zip(axes, feats):
        for alpha, width in [(0.95, 1.96), (0.90, 1.645)]:
            ax.fill_between(t, mu[:, f]-width*sd[:, f], mu[:, f]+width*sd[:, f],
                            alpha=0.15 if alpha==0.95 else 0.25,
                            label=f"{int(alpha*100)}% interval" if f==feats[0] else None)
        ax.plot(t, mu[:, f], lw=1.8, label="mean" if f==feats[0] else None)
        obs_idx = np.isfinite(x_obs[:, f])
        ax.scatter(t[obs_idx], x_obs[obs_idx, f], s=14, c="C0", alpha=0.7,
                   label="observed" if f==feats[0] else None)
        s_idx = (S_i[:, f] == 1)
        ax.scatter(t[s_idx], y_true[s_idx, f], s=18, c="k", marker="x",
                   label="ground truth (S=1)" if f==feats[0] else None)
        ax.set_ylabel(f"f{f}")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("time")
    axes[0].legend(loc="upper right", frameon=False)
    out = ckpt_dir/f"viz_{args.split}_idx{i}.png"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()

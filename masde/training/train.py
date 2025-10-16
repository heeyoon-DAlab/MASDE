# -*- coding: utf-8 -*-
"""
학습: ELBO (관측칸 NLL + β·KL + λ_sel·BCE) + (옵션) self-masking denoising 보조항
평가: 학습 종료 후 val/test 1회만 수행 (S==1 위치에서 RMSE/MAE + Coverage/CRPS)
로그: log_every 에폭마다 nll/kl/sel_bce만 출력
진행바: show_pbar=True 면 tqdm 진행바 표시
"""

from pathlib import Path
import numpy as np
import torch
import torch.utils.data as tud
from tqdm import tqdm

from masde.models.masde import MASDE
from masde.data.loaders import NPZWindowDataset, load_npz
from masde.training.metrics import masked_mae_rmse, coverage_crps_gaussian
from masde.training.utils import set_seed, denorm, ensure_dir, dump_json


def train_and_eval(npz_path: str,
                   device,
                   epochs: int = 100,
                   batch: int = 64,
                   lr: float = 1e-3,
                   zdim: int = 64,
                   beta: float = 1.0,
                   lambda_sel: float = 0.2,
                   K_eval: int = 20,
                   seed: int = 0,
                   outdir: Path = Path("outputs"),
                   eval_space: str = "z",           # 기본 z-스케일
                   likelihood: str = "gauss",
                   sel_warmup: int = 0,
                   p_dm: float = 0.0,
                   lambda_dm: float = 1.0,
                   # 편의 옵션
                   log_every: int = 10,
                   show_pbar: bool = True,          # 기본 진행바 ON
                   num_workers: int = 0):
    """
    npz_path: BenchPOTS+PyGrinder npz 경로
    device: torch.device("cuda"|"cpu")
    eval_space: "raw"(역정규화) | "z"(표준화 스케일)
    """
    # 시드 고정
    set_seed(seed)

    # 데이터 로드
    pack = load_npz(Path(npz_path))
    F, T = int(pack["n_features"]), int(pack["n_steps"])
    train_X, O_train = pack["train_X"], pack["O_train"].astype(np.float32)
    val_X_in,  val_X_ori  = pack["val_X"],  pack["val_X_ori"]
    test_X_in, test_X_ori = pack["test_X"], pack["test_X_ori"]
    S_val, S_test = pack["S_val"], pack["S_test"]
    mean, scale = pack.get("scaler_mean"), pack.get("scaler_scale")

    # 모델/옵티마이저
    model = MASDE(F=F, z_dim=zdim, likelihood=likelihood).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 데이터 로더
    ds = NPZWindowDataset(train_X, O_train)
    ld = tud.DataLoader(ds,
                        batch_size=batch,
                        shuffle=True,
                        drop_last=False,
                        pin_memory=True,
                        num_workers=num_workers)

    # 연속시간 그리드
    ts = torch.linspace(0., 1., T, device=device)
    dt = 1. / max(2, T - 1)

    # BCE 클래스 불균형 보정(자연결측 비율 기반)
    pos_rate = float(1.0 - O_train.mean())  # R=1 비율
    neg_rate = 1.0 - pos_rate
    pos_weight = (neg_rate / (pos_rate + 1e-8))

    # ------------------ 학습 ------------------
    for ep in range(1, epochs + 1):
        model.train()
        logs = []

        iterator = tqdm(ld, desc=f"train ep {ep:03d}/{epochs}", dynamic_ncols=True, leave=False, mininterval=0.3) \
                   if show_pbar else ld

        for inp, O, X in iterator:
            # numpy/torch 혼용 안전 캐스팅
            def _to(x):
                return x.to(device) if torch.is_tensor(x) else torch.from_numpy(x).to(device)

            inp_t, O_t, X_t = _to(inp), _to(O), _to(X)

            # λ_sel warm-up
            lam_eff = lambda_sel * (min(1.0, ep / sel_warmup) if sel_warmup > 0 else 1.0)

            # --- 기본 ELBO ---
            loss_main, d = model.elbo(
                inp=inp_t, O=O_t, X=X_t, ts=ts,
                beta=beta, lambda_sel=lam_eff,
                method="euler", dt=dt,
                nll_mask=None,                # 관측칸(O==1) 자동 사용
                pos_weight=pos_weight         # BCE 불균형 보정
            )
            loss = loss_main

            # --- Self-masking denoising 보조항 (옵션) ---
            if p_dm > 0.0:
                with torch.no_grad():
                    # 관측칸(O==1) 중에서 MCAR 방식으로 일부 가림
                    obs_cells = torch.isfinite(X_t) & (O_t > 0.5)
                    M_dm = (torch.rand_like(O_t) < p_dm) & obs_cells  # (B,T,F) bool

                    # 입력에서 해당 위치를 NaN으로 가림
                    X_dm = X_t.clone()
                    X_dm[M_dm] = float('nan')
                    O_dm = O_t.clone()
                    O_dm[M_dm] = 0.0
                    inp_dm = torch.cat([torch.nan_to_num(X_dm, 0.0), O_dm], dim=-1)

                # ★ denoising 패스에서는 BCE/KL 비활성화, 마스크 위치 NLL만
                loss_dm, _ = model.elbo(
                    inp=inp_dm, O=O_dm, X=X_t, ts=ts,
                    beta=0.0, lambda_sel=0.0,     # KL/BCE off
                    method="euler", dt=dt,
                    nll_mask=M_dm,                # 가린 위치에서만 NLL
                    pos_weight=None               # BCE 안 씀
                )
                loss = loss + lambda_dm * loss_dm

            # 업데이트
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            logs.append(d)

        # 에폭 요약 로그
        if (ep % log_every == 0) or (ep == epochs):
            nll = np.mean([l["nll"] for l in logs]) if logs else float("nan")
            kl  = np.mean([l["kl"]  for l in logs]) if logs else float("nan")
            bce = np.mean([l["bce"] for l in logs]) if logs else float("nan")
            print(f"[ep {ep:03d}] nll={nll:.4f} kl={kl:.4f} sel_bce={bce:.4f}", flush=True)

    # ------------------ 학습 종료 후 최종 1회 평가 ------------------
    @torch.no_grad()
    def _eval(X_in: np.ndarray, X_ori: np.ndarray, S: np.ndarray, split: str):
        M = ~np.isnan(X_in)
        Xfill = np.where(np.isnan(X_in), 0.0, X_in)
        inp = np.concatenate([Xfill, M.astype(np.float32)], axis=-1)
        model.eval()

        B = X_in.shape[0]
        bs = min(128, B)
        mu_list, sd_list = [], []

        iterator = tqdm(range(0, B, bs), desc=f"eval-{split}", dynamic_ncols=True, leave=False, mininterval=0.3) \
                   if show_pbar else range(0, B, bs)

        for i in iterator:
            sub = torch.from_numpy(inp[i:i + bs]).to(device)
            mu, sd = model.predict_mc(sub, ts, K=K_eval, method="euler", dt=dt)
            mu_list.append(mu.cpu()); sd_list.append(sd.cpu())
        mu = torch.cat(mu_list, 0).numpy()
        sd = torch.cat(sd_list, 0).numpy()

        if eval_space == "raw":
            y_true = denorm(X_ori, mean, scale)
            mu = denorm(mu, mean, scale)
            sd = sd * (1.0 if scale is None else scale.reshape(1, 1, -1))
        else:
            y_true = X_ori

        d1 = masked_mae_rmse(y_true, mu, S)
        d2 = coverage_crps_gaussian(y_true, mu, sd, S)
        return {**d1, **d2}

    val_metrics  = _eval(val_X_in,  val_X_ori,  S_val,  split="val")
    test_metrics = _eval(test_X_in, test_X_ori, S_test, split="test")

    ensure_dir(outdir)
    torch.save({"state_dict": model.state_dict(), "F": F, "T": T, "zdim": zdim}, outdir / "final.pt")
    dump_json({"val": val_metrics, "test": test_metrics}, outdir / "report.json")

    print("[val ]", val_metrics, flush=True)
    print("[test]", test_metrics, flush=True)
    return {"val": val_metrics, "test": test_metrics}

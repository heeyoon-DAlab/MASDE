# -*- coding: utf-8 -*-
"""
MASDE: MNAR-Aware-LatentSDE
- 잠재 SDE의 drift/확산에 결측 위험도 π(z)를 내재화 (게이팅)
- Selection head: z(t) + 패턴 특징으로 자연결측 R=1-O 예측 (BCE)
- ELBO: 관측칸 NLL + λ_sel * BCE + β * KL
- likelihood 옵션: 'gauss' | 'laplace' (heavy-tail 대응)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torchsde

from masde.data.pattern import build_dt_features


# -----------------------------
# 인코더: q(z0 | 값+마스크)
# -----------------------------
class EncoderGRU(nn.Module):
    def __init__(self, in_dim: int, h_dim: int, z_dim: int):
        super().__init__()
        self.gru = nn.GRU(in_dim, h_dim, batch_first=True)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)

    def forward(self, x: torch.Tensor):
        """
        x: (B,T,2F) = [값(결측0 채움), 마스크]
        return: mu0, logv0  (B,z)
        """
        h, _ = self.gru(x)
        hT = h[:, -1, :]
        mu0 = self.mu(hT)
        logv0 = self.logvar(hT)
        return mu0, logv0


# -----------------------------
# Latent SDE: dz = f(z, π) dt + g(z, π) dW (diagonal noise)
# π(z) = sigmoid( h_pi(z) )
# 게이팅: f = f0(z) + Wf * s_pi, g = softplus( g0(z) + Wg * s_pi )
# s_pi = tanh(Linear(π(z)))  (F -> z_dim)
# -----------------------------
class MasdeSDE(torchsde.SDEIto):
    def __init__(self, z_dim: int, F: int, hidden: int = 128):
        super().__init__(noise_type="diagonal")
        self.z_dim = z_dim
        self.F = F

        self.net_f0 = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, z_dim)
        )
        self.net_g0 = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, z_dim)
        )

        self.pi_head = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, F)
        )
        self.pi_embed = nn.Sequential(
            nn.Linear(F, z_dim), nn.Tanh()
        )

        self.Wf = nn.Linear(z_dim, z_dim, bias=False)
        self.Wg = nn.Linear(z_dim, z_dim, bias=False)
        self.softplus = nn.Softplus()

    def _pi_embed(self, z: torch.Tensor):
        pi_logits = self.pi_head(z)           # (B,F)
        pi = torch.sigmoid(pi_logits)         # (B,F) in [0,1]
        s_pi = self.pi_embed(pi)              # (B,z)
        return pi, s_pi

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # drift
        f0 = self.net_f0(z)                   # (B,z)
        _, s_pi = self._pi_embed(z)
        return f0 + self.Wf(s_pi)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # diffusion (diag)
        g0 = self.net_g0(z)                   # (B,z)
        _, s_pi = self._pi_embed(z)
        g_diag = self.softplus(g0 + self.Wg(s_pi))  # 양수화 -> 분산 해석 안전
        return g_diag


# -----------------------------
# 디코더: z(t) -> (mu_x, log_sigma_x)
# -----------------------------
class Decoder(nn.Module):
    def __init__(self, z_dim: int, F: int, hidden: int = 256):
        super().__init__()
        self.net_mu = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, F)
        )
        self.net_ls = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, F)
        )

    def forward(self, zt: torch.Tensor):
        """
        zt: (T,B,z)
        return: mu, logsig  (T,B,F)
        """
        mu = self.net_mu(zt)
        logsig = self.net_ls(zt).clamp(-7., 5.)  # 수치 안정
        return mu, logsig


# -----------------------------
# Selection head: z(t) + 패턴 특징 -> R(t,f) 로짓
# -----------------------------
class SelectionHead(nn.Module):
    def __init__(self, z_dim: int, p_dim: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + p_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, z_paths: torch.Tensor, patt: torch.Tensor) -> torch.Tensor:
        """
        z_paths: (T,B,z), patt: (B,T,F,2)
        반환: logits (B,T,F)
        """
        B, T, F, _ = patt.shape
        zbt = z_paths.permute(1, 0, 2).unsqueeze(2).expand(B, T, F, z_paths.shape[-1])  # (B,T,F,z)
        inp = torch.cat([zbt, patt], dim=-1)                                            # (B,T,F,z+p)
        logits = self.net(inp).squeeze(-1)                                              # (B,T,F)
        return logits


# -----------------------------
# MASDE 전체
# -----------------------------
class MASDE(nn.Module):
    def __init__(self, F: int, z_dim: int = 64,
                 enc_h: int = 128, sde_h: int = 128, dec_h: int = 256, sel_h: int = 64,
                 likelihood: str = "gauss"):
        """
        likelihood: 'gauss' | 'laplace'
        """
        super().__init__()
        self.F = F
        self.likelihood = likelihood.lower()
        assert self.likelihood in {"gauss", "laplace"}

        self.enc = EncoderGRU(in_dim=2*F, h_dim=enc_h, z_dim=z_dim)
        self.sde = MasdeSDE(z_dim=z_dim, F=F, hidden=sde_h)
        self.dec = Decoder(z_dim=z_dim, F=F, hidden=dec_h)
        self.sel = SelectionHead(z_dim=z_dim, p_dim=2, hidden=sel_h)

    def _solve(self, z0: torch.Tensor, ts: torch.Tensor, method: str = "euler", dt: float = 1./48) -> torch.Tensor:
        bm = torchsde.BrownianInterval(ts[0], ts[-1], size=z0.size(), device=z0.device)
        z_paths = torchsde.sdeint(self.sde, z0, ts, bm=bm, method=method, dt=dt, adaptive=False)
        return z_paths

    def elbo(self,
             inp: torch.Tensor,          # (B,T,2F)
             O: torch.Tensor,            # (B,T,F)
             X: torch.Tensor,            # (B,T,F)  NaN 포함
             ts: torch.Tensor,           # (T,)
             beta: float = 1.0,
             lambda_sel: float = 0.2,
             method: str = "euler",
             dt: float = 1./48,
             nll_mask: torch.Tensor | None = None,
             pos_weight: float | None = None):
        """
        nll_mask: NLL을 계산할 위치 마스크(denoising 패스에서만 사용)
        pos_weight: BCE 양성 가중치(불균형 보정). None이면 보정 없음.
        """
        device = inp.device
        B, T, Fdim = O.shape  # F 이름 충돌 방지

        # q(z0)
        mu0, logv0 = self.enc(inp)   # (B,z), (B,z) [log variance]
        std0 = torch.exp(0.5 * logv0)
        z0 = mu0 + std0 * torch.randn_like(std0)

        # SDE 적분
        z_paths = self._solve(z0, ts, method=method, dt=dt)  # (T,B,z)

        # 디코딩
        mu_x, log_sig = self.dec(z_paths)    # (T,B,F)
        mu_x = mu_x.permute(1, 0, 2)         # (B,T,F)
        log_sig = log_sig.permute(1, 0, 2)

        # 관측 마스크/입력
        X_t = torch.nan_to_num(X, 0.0)
        obs_default = torch.isfinite(X) & (O > 0.5)
        obs = nll_mask.bool() if (nll_mask is not None) else obs_default

        # 우도(가우시안/라플라스)
        if self.likelihood == "gauss":
            sig2 = torch.exp(2 * log_sig).clamp_min(1e-6)
            ll = -0.5 * (np.log(2*np.pi) + torch.log(sig2) + (X_t - mu_x)**2 / sig2)
        else:  # laplace
            b = torch.exp(log_sig).clamp_min(1e-6)
            ll = -(np.log(2.) + torch.log(b) + torch.abs(X_t - mu_x)/b)

        n_obs = obs.sum().clamp(min=1)
        nll = - (ll * obs).sum() / n_obs

        # --- 선택(BCE): denoising 패스에서는 lambda_sel=0.0이 들어오므로 완전 비활성화 ---
        R = 1.0 - O
        if lambda_sel != 0.0:
            patt = build_dt_features(O)                   # (B,T,F,2)
            logits = self.sel(z_paths, patt)              # (B,T,F)
            if pos_weight is not None:
                pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
                bce = Fnn.binary_cross_entropy_with_logits(logits, R, reduction='mean', pos_weight=pw)
            else:
                bce = Fnn.binary_cross_entropy_with_logits(logits, R, reduction='mean')
        else:
            # denoising 보조패스 등: BCE 완전 비활성화 (0 텐서로 대체)
            bce = torch.zeros((), device=z_paths.device, dtype=mu_x.dtype)

        # --- KL(q||p): denoising 패스에서는 beta=0.0이 들어오므로 완전 비활성화 ---
        if beta != 0.0:
            # KL = 0.5 * sum(exp(logv) + mu^2 - 1 - logv)
            kl = 0.5 * torch.sum(torch.exp(logv0) + mu0**2 - 1.0 - logv0, dim=1).mean()
        else:
            kl = torch.zeros((), device=mu0.device, dtype=mu0.dtype)

        loss = nll + beta * kl + lambda_sel * bce
        logs = {"nll": float(nll.item()), "kl": float(kl.item()), "bce": float(bce.item())}
        return loss, logs

    @torch.no_grad()
    def predict_mc(self, inp: torch.Tensor, ts: torch.Tensor, K: int = 20, method: str = "euler", dt: float = 1./48):
        """
        MC 경로 샘플로 (평균, 표준편차) 반환: (B,T,F), (B,T,F)
        - 알레아토릭: 디코더 분산 평균 (Laplace면 2b^2)
        - 에피스테믹: 경로 평균의 분산
        """
        mu0, logv0 = self.enc(inp)
        std0 = torch.exp(0.5 * logv0)

        mus, vars_alea = [], []
        for _ in range(K):
            z0 = mu0 + std0 * torch.randn_like(std0)
            z_paths = self._solve(z0, ts, method=method, dt=dt)  # (T,B,z)
            mu_x, log_sig = self.dec(z_paths)                    # (T,B,F)
            mus.append(mu_x.permute(1, 0, 2))                    # (B,T,F)
            if self.likelihood == "gauss":
                vars_alea.append(torch.exp(2 * log_sig.permute(1, 0, 2)))
            else:
                b = torch.exp(log_sig.permute(1, 0, 2)).clamp_min(1e-6)
                vars_alea.append(2.0 * b * b)                    # Var(Laplace)=2b^2

        mu = torch.stack(mus, 0).mean(0)
        var_epi  = torch.stack(mus, 0).var(0, unbiased=False)
        var_alea = torch.stack(vars_alea, 0).mean(0)
        std = torch.sqrt(var_epi + var_alea + 1e-12)
        return mu, std

# -*- coding: utf-8 -*-
"""
평가지표:
- RMSE/MAE (S==1 위치에서만)
- Coverage@{90,95}
- CRPS (가우시안 폐형식)
- Sharpness (평균 표준편차)
"""

import numpy as np
from scipy.stats import norm


def masked_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, S: np.ndarray):
    """S==1 위치에서만 MAE/RMSE 계산"""
    M = (S == 1)
    if M.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan}
    err = (y_pred - y_true)[M]
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean()))
    }


def coverage_crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, S: np.ndarray, alphas=(0.1, 0.05)):
    """가우시안 가정 하 Coverage/CRPS/Sharpness"""
    M = (S == 1)
    y, mu, sigma = y[M], mu[M], np.clip(sigma[M], 1e-8, None)
    out = {}
    # 커버리지
    for a in alphas:
        z = norm.ppf(1 - a/2)
        lo, hi = mu - z*sigma, mu + z*sigma
        out[f"cov_{int((1-a)*100)}"] = float(np.mean((y >= lo) & (y <= hi)))
    # CRPS (Gaussian closed form)
    a = (y - mu)/sigma
    phi = norm.pdf(a); Phi = norm.cdf(a)
    crps = np.mean(sigma * (2*phi + a*(2*Phi - 1) - 1/np.sqrt(np.pi)))
    out["CRPS"] = float(crps)
    out["sharpness"] = float(np.mean(sigma))
    return out

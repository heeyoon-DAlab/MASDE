# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

from masde.training.metrics import masked_mae_rmse, coverage_crps_gaussian
from masde.training.utils import denorm


def evaluate_imputation(y_true: np.ndarray,
                        y_hat: np.ndarray,
                        S_mask: np.ndarray,
                        mean, scale,
                        eval_space: str = "z") -> Dict[str, Any]:
    """
    y_true: *_X_ori (B,T,F)
    y_hat:  imputed (B,T,F)
    S_mask: (B,T,F) 1이면 '인공 가림' 위치 → 여기서만 지표 계산
    eval_space: "z"(표준화) | "raw"(역정규화 후)
    """
    if eval_space == "raw":
        y_true_eval = denorm(y_true, mean, scale)
        y_hat_eval  = denorm(y_hat,  mean, scale)
        sd_dummy = np.zeros_like(y_hat_eval)  # PyPOTS는 분산 없음
    else:
        y_true_eval = y_true
        y_hat_eval  = y_hat
        sd_dummy = np.zeros_like(y_hat_eval)

    d1 = masked_mae_rmse(y_true_eval, y_hat_eval, S_mask)
    d2 = {"cov_90": None, "cov_95": None, "CRPS": None, "sharpness": None}
    # 불확실성 없는 모델이므로 coverage/CRPS는 None 유지
    return {**d1, **d2}


def save_report(obj: Dict[str, Any], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

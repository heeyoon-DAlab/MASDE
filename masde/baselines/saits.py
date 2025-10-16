# -*- coding: utf-8 -*-
"""
SAITS 베이스라인 러너 (PyPOTS)
- 학습: train_X (자연결측 포함)
- 평가: val/test impute → 정답 *_X_ori, 지표는 S==1
- eval_space: "raw" | "z"
"""

from typing import Dict, Any
import numpy as np


def run_saits(
    train_X: np.ndarray,
    val_X_in: np.ndarray, val_X_ori: np.ndarray, S_val: np.ndarray,
    test_X_in: np.ndarray, test_X_ori: np.ndarray, S_test: np.ndarray,
    F: int, T: int,
    mean, scale,
    epochs: int = 100,
    batch_size: int = 64,
    n_layers: int = 2,
    d_model: int = 256,
    n_heads: int = 4,
    d_ffn: int = 512,
    dropout: float = 0.1,
    attn_dropout: float = 0.1,
    seed: int = 0,
    eval_space: str = "z",
) -> Dict[str, Any]:
    try:
        from pypots.imputation import SAITS
    except Exception as e:
        raise ImportError(
            f"[PyPOTS] SAITS import 실패: {e}\n"
            "pip install pypots"
        )

    d_k = d_model // n_heads
    d_v = d_k
    model = SAITS(
        n_steps=T, n_features=F,
        n_layers=n_layers, d_model=d_model,
        n_heads=n_heads, d_k=d_k, d_v=d_v, d_ffn=d_ffn,
        dropout=dropout, attn_dropout=attn_dropout,
        batch_size=batch_size, epochs=epochs,
        verbose=True,
    )

    model.fit(train_set={"X": train_X},
              val_set={"X": val_X_in, "X_ori": val_X_ori})

    val_hat  = model.impute({"X": val_X_in})
    test_hat = model.impute({"X": test_X_in})

    from masde.baselines.common import evaluate_imputation
    val_metrics  = evaluate_imputation(val_X_ori,  val_hat,  S_val,  mean, scale, eval_space=eval_space)
    test_metrics = evaluate_imputation(test_X_ori, test_hat, S_test, mean, scale, eval_space=eval_space)
    return {"val": val_metrics, "test": test_metrics}

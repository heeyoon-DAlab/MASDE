# -*- coding: utf-8 -*-
"""
BRITS 베이스라인 러너 (PyPOTS)
- eval_space: "raw" | "z"
"""

from typing import Dict, Any
import numpy as np


def run_brits(
    train_X: np.ndarray,
    val_X_in: np.ndarray, val_X_ori: np.ndarray, S_val: np.ndarray,
    test_X_in: np.ndarray, test_X_ori: np.ndarray, S_test: np.ndarray,
    F: int, T: int,
    mean, scale,
    epochs: int = 100,
    batch_size: int = 64,
    hidden_size: int = 256,
    seed: int = 0,
    eval_space: str = "z",
) -> Dict[str, Any]:
    try:
        from pypots.imputation import BRITS
    except Exception as e:
        raise ImportError(
            f"[PyPOTS] BRITS import 실패: {e}\n"
            "pip install pypots"
        )

    model = BRITS(
        n_steps=T, n_features=F,
        rnn_hidden_size=hidden_size,
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

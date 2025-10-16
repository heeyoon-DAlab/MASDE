# -*- coding: utf-8 -*-
"""
NPZ 스키마 로더와 PyTorch Dataset / DataLoader 구성
- 학습: train_X (자연결측 포함), O_train (관측마스크) 사용
- 검증/테스트: val/test의 주입 후 X와 원본 X_ori, S 마스크로 평가
"""

from pathlib import Path
import numpy as np
import torch
import torch.utils.data as tud


def denorm(x_std: np.ndarray, mean, scale):
    """표준화 해제: x_raw = x_std * scale + mean  (shape: (N,T,F), (F,), (F,))"""
    if mean is None or scale is None:
        return x_std
    return x_std * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


class NPZWindowDataset(tud.Dataset):
    """학습용: 값+마스크를 인코더 입력으로, 원값은 NLL 마스킹용으로 유지"""
    def __init__(self, X: np.ndarray, O: np.ndarray):
        # X: (N,T,F) 표준화 + NaN 포함, O: (N,T,F) 1=관측
        assert X.ndim == 3 and O.ndim == 3, "X/O must be (N,T,F)"
        assert X.shape == O.shape, "X and O must have same shape"
        self.X = X.astype(np.float32)
        self.O = O.astype(np.float32)
        self.N, self.T, self.F = self.X.shape

    def __len__(self): 
        return self.N

    def __getitem__(self, i):
        x = self.X[i]                             # (T,F)
        o = self.O[i]                             # (T,F)
        x_fill = np.where(np.isnan(x), 0.0, x)    # 인코더 입력은 NaN을 0으로 채움(표준화 기준)
        inp = np.concatenate([x_fill, o], axis=-1)  # (T, 2F)
        return inp, o, x                           # inp: 인코더 입력, o: 관측마스크, x: 원본(NaN 포함)


def load_npz(npz_path: Path) -> dict:
    """dataset.npz 로드: numpy savez 구조를 dict로 반환"""
    p = np.load(npz_path, allow_pickle=True)
    pack = {k: p[k] for k in p.files}
    # scaler 옵션 키가 없을 수 있으니 보정
    pack["scaler_mean"] = pack.get("scaler_mean", None)
    pack["scaler_scale"] = pack.get("scaler_scale", None)
    # 필수 키 체크(오탈자 방지)
    required = ["train_X", "O_train", "val_X", "val_X_ori", "O_val", "S_val",
                "test_X", "test_X_ori", "O_test", "S_test", "n_steps", "n_features"]
    for k in required:
        assert k in pack, f"Missing key in NPZ: {k}"
    return pack

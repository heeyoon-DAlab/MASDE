# -*- coding: utf-8 -*-
"""
유틸: 시드고정, 역정규화, 디렉터리 보장, JSON 저장
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int = 0):
    """재현성 확보(가능한 범위)"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denorm(x_std: np.ndarray, mean, scale):
    """표준화 해제: x_raw = x_std * scale + mean"""
    if mean is None or scale is None:
        return x_std
    return x_std * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def dump_json(obj: dict, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

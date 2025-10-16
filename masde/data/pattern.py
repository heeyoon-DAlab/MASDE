# -*- coding: utf-8 -*-
"""
패턴 특징 생성
- Δt_prev: 직전 관측까지의 시차(결측 연속 길이)
- Δt_next: 다음 관측까지의 시차(결측 연속 길이)
- 시간 축 T로 정규화하여 (B,T,F,2) 반환
"""

import torch


def build_dt_features(O: torch.Tensor) -> torch.Tensor:
    """
    O: (B,T,F) float {0,1}
    returns: patt (B,T,F,2) = [Δt_prev, Δt_next] / T
    """
    assert O.dim() == 3, "O must be (B,T,F)"
    B, T, F = O.shape
    O_bin = (O > 0.5).float()

    # 과거방향 누적(직전 관측부터 몇 step 지났는지)
    dt_prev = torch.zeros(B, F, device=O.device)
    out_prev = torch.zeros(B, T, F, device=O.device)
    for t in range(T):
        dt_prev = torch.where(O_bin[:, t, :] > 0.5, torch.zeros_like(dt_prev), dt_prev + 1)
        out_prev[:, t, :] = dt_prev

    # 미래방향 누적(다음 관측까지 몇 step 남았는지)
    dt_next = torch.zeros(B, F, device=O.device)
    out_next = torch.zeros(B, T, F, device=O.device)
    for t in range(T - 1, -1, -1):
        dt_next = torch.where(O_bin[:, t, :] > 0.5, torch.zeros_like(dt_next), dt_next + 1)
        out_next[:, t, :] = dt_next

    # T로 정규화
    denom = max(1, T - 1)
    out_prev = (out_prev / denom).unsqueeze(-1)  # (B,T,F,1)
    out_next = (out_next / denom).unsqueeze(-1)  # (B,T,F,1)
    patt = torch.cat([out_prev, out_next], dim=-1)  # (B,T,F,2)
    return patt

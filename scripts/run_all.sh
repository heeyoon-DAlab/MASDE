#!/usr/bin/env bash
# 사용: bash scripts/run_all.sh
# 원하는 대로 아래 배열/변수 편집 후 실행하세요.

set -euo pipefail

# ===== 편집 영역 =====
ROOT="/home/intern/SSD/heeyoon/datasets/processed"

DATASETS=(physionet_2012 beijing_multisite_air_quality italy_air_quality electricity_load_diagrams electricity_transformer_temperature pems_traffic solar_alabama)
MECHS=(MNAR)             # 예: (MCAR MAR MNAR BLOCK)
RATES=(0.2 0.5)
SEEDS=(0)

RUN_MASDE=1
RUN_SAITS=1
RUN_BRITS=1

# 평가 스페이스: raw 또는 z
EVAL_SPACE="z"

# ===== MASDE 하이퍼 =====
EPOCHS=30
BATCH=128
ZDIM=64
K_EVAL=30
BETA=1.0
LAMBDA_SEL=0.1

# Self-masking Denoising
LIKELIHOOD="gauss"   # "gauss" | "laplace"
SEL_WARMUP=5
P_DM=0.1
LAMBDA_DM=1.0

# 로깅/로더
LOG_EVERY=1
SHOW_PBAR=1         # ← 진행바 ON
NUM_WORKERS=0

# ===== SAITS 하이퍼 =====
SAITS_EPOCHS=30
SAITS_BATCH=128
SAITS_NLAYERS=2
SAITS_DMODEL=256
SAITS_NHEADS=4
SAITS_DFFN=512
SAITS_DROPOUT=0.1
SAITS_ATTN_DROPOUT=0.1

# ===== BRITS 하이퍼 =====
BRITS_EPOCHS=30
BRITS_BATCH=128
BRITS_HIDDEN=256
# ===== 편집 영역 끝 =====

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

for ds in "${DATASETS[@]}"; do
  for mech in "${MECHS[@]}"; do
    for rate in "${RATES[@]}"; do
      for seed in "${SEEDS[@]}"; do
        echo "=== ${ds} | ${mech} r=${rate} | seed=${seed} ==="

        if [[ $RUN_MASDE -eq 1 ]]; then
          python -u scripts/train_one.py \
            --root "$ROOT" \
            --dataset "$ds" --mechanism "$mech" --rate "$rate" \
            --epochs $EPOCHS --batch $BATCH --zdim $ZDIM --K_eval $K_EVAL \
            --beta $BETA --lambda_sel $LAMBDA_SEL \
            --likelihood $LIKELIHOOD --sel_warmup $SEL_WARMUP \
            --p_dm $P_DM --lambda_dm $LAMBDA_DM \
            --log_every $LOG_EVERY --show_pbar $SHOW_PBAR --num_workers $NUM_WORKERS \
            --seed $seed --eval_space "$EVAL_SPACE"
        fi

        if [[ $RUN_SAITS -eq 1 ]]; then
          python -u scripts/baseline_run.py \
            --root "$ROOT" \
            --dataset "$ds" --mechanism "$mech" --rate "$rate" \
            --baseline saits \
            --epochs $SAITS_EPOCHS --batch_size $SAITS_BATCH \
            --n_layers $SAITS_NLAYERS --d_model $SAITS_DMODEL --n_heads $SAITS_NHEADS --d_ffn $SAITS_DFFN \
            --dropout $SAITS_DROPOUT --attn_dropout $SAITS_ATTN_DROPOUT \
            --seed $seed --eval_space "$EVAL_SPACE"
        fi

        if [[ $RUN_BRITS -eq 1 ]]; then
          python -u scripts/baseline_run.py \
            --root "$ROOT" \
            --dataset "$ds" --mechanism "$mech" --rate "$rate" \
            --baseline brits \
            --epochs $BRITS_EPOCHS --batch_size $BRITS_BATCH \
            --hidden_size $BRITS_HIDDEN \
            --seed $seed --eval_space "$EVAL_SPACE"
        fi

      done
    done
  done
done

# 집계
python scripts/aggregate_reports.py --out outputs_aggregated/report_aggregated.json
echo "[OK] aggregated saved to outputs_aggregated/report_aggregated.json"

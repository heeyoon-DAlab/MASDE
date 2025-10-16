# run_all_baseline.sh
#!/usr/bin/env bash
set -e

ROOT=/home/intern/SSD/heeyoon/datasets/processed
MECH=MNAR
RATE=0.2
EPOCHS=100

# 필요하면 여기에 더 추가
DATASETS=(
  electricity_load_diagrams
  electricity_transformer_temperature
  italy_air_quality
  pems_traffic
  solar_alabama
)
BASELINES=(saits brits)

mkdir -p logs_baseline

for DS in "${DATASETS[@]}"; do
  for BL in "${BASELINES[@]}"; do
    echo "=== [RUN] dataset=${DS} baseline=${BL} ==="
    python scripts/baseline_run.py \
      --root "$ROOT" \
      --dataset "$DS" --mechanism "$MECH" --rate "$RATE" \
      --baseline "$BL" --epochs "$EPOCHS" \
      "$@" 2>&1 | tee "logs_baseline/${DS}_${BL}.log"
  done
done

echo "ALL DONE."
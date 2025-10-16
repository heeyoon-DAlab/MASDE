#!/usr/bin/env bash
set -e

ROOT=/home/intern/SSD/heeyoon/datasets/processed
ARGS="--mechanism MNAR --rate 0.2 --epochs 120 --batch 128 --zdim 48 --K_eval 30"

mkdir -p logs

for DS in \
  electricity_load_diagrams \
  electricity_transformer_temperature \
  italy_air_quality \
  pems_traffic \
  solar_alabama
do
  echo "=== [RUN] $DS ==="
  python scripts/train_one.py --root "$ROOT" --dataset "$DS" $ARGS 2>&1 | tee "logs/${DS}.log"
done
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash run_slurm.sh --name exp9_adaptability_eval prod10 \
  uv run src/rl_snake/evaluate.py \
  --checkpoint checkpoints/part1/exp1_mlp_basic/ep02000.pt \
  --agent-type mlp \
  --episodes-per-config 200 \
  --height 10 --width 10 \
  --wandb-project rl-snake \
  --run-name exp9_adaptability_eval \
  --output-json logs/exp9_adaptability_results.json \
  --seed 42

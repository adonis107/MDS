#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash run_slurm.sh --name exp6_cnn_full prod10 \
  uv run src/rl_snake/train.py \
  --agent-type cnn \
  --episodes 5000 --height 10 --width 10 \
  --n-gold 1 --n-silver 0 --n-poison 0 \
  --n-dynamic-obstacles 0 --max-steps 500 \
  --gold-reward 1.0 --death-reward -1.0 \
  --step-reward 0.0 --distance-reward-scale 0.0 --body-proximity-scale 0.1 \
  --cnn-hidden 128 \
  --seed 42 --save-dir checkpoints/part1 \
  --wandb-project rl-snake --run-name exp6_cnn_full

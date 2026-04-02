#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash run_slurm.sh --name exp2_window prod10 \
uv run src/rl_snake/train.py \
  --agent-type window-cnn \
  --window-size 11 \
  --episodes 5000 \
  --height 16 --width 16 \
  --n-gold 1 --n-silver 0 --n-poison 0 \
  --n-dynamic-obstacles 0 \
  --max-steps 500 \
  --gold-reward 1.0 \
  --death-reward -1.0 \
  --step-reward 0.0 \
  --distance-reward-scale 0.0 \
  --cnn-hidden 128 \
  --seed 42 \
  --save-dir checkpoints/part1 \
  --wandb-project rl-snake \
  --run-name exp2_window_cnn

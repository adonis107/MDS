#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash run_slurm.sh --name exp8_cnn_dynamic prod10 \
  uv run src/rl_snake/train.py \
  --agent-type cnn \
  --episodes 10000 --height 10 --width 10 \
  --n-gold 1 --n-silver 1 --n-poison 1 \
  --n-dynamic-obstacles 1 --max-steps 500 \
  --gold-reward 1.0 --silver-reward 0.5 --poison-reward -0.5 \
  --death-reward -1.0 --step-reward 0.0 \
  --body-proximity-scale 0.1 --cnn-hidden 128 \
  --n-frames 2 \
  --epsilon-start 1.0 --epsilon-end 0.01 --epsilon-decay 0.99997 \
  --double-dqn --dueling \
  --target-tau 0.005 \
  --buffer-capacity 200000 \
  --seed 42 --save-dir checkpoints/part1 \
  --save-every 1000 \
  --wandb-project rl-snake --run-name exp8_cnn_dynamic

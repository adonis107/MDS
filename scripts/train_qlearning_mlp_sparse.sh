#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

uv run python scripts/train.py \
  agent=q_learning \
  agent.network_type=mlp \
  encoder=features10 \
  reward=sparse \
  wandb=true \
  save.enabled=true \
  save.path=q_learning_mlp_sparse \
  "$@"

#!/bin/bash




source "$HOME/MDS-PDMM/scripts/slurm/env.sh"

: "${MDS_MODEL:?Set MDS_MODEL (transformer_ocsvm|pnn|prae)}"
: "${MDS_YEAR:?Set MDS_YEAR (2015|2017)}"

export MDS_MODELS="$MDS_MODEL"

echo "=== Training model=$MDS_MODEL year=$MDS_YEAR ==="
python scripts/train.py

echo "=== Training finished: model=$MDS_MODEL year=$MDS_YEAR ==="

#!/usr/bin/env bash
# One-click entry: run every dataset script in scripts/ sequentially.
# Any extra args are passed through to each script (and then to run.py),
# so you can override hyper-parameters, e.g.:
#   bash run_main.sh --train_epochs 20 --learning_rate 0.001
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the seven common datasets
bash "$DIR"/scripts/ETTh1.sh "$@"
bash "$DIR"/scripts/ETTh2.sh "$@"
bash "$DIR"/scripts/ETTm1.sh "$@"
bash "$DIR"/scripts/ETTm2.sh "$@"
bash "$DIR"/scripts/Electricity.sh "$@"
bash "$DIR"/scripts/Traffic.sh "$@"
bash "$DIR"/scripts/Weather.sh "$@"

# extras
bash "$DIR"/scripts/Solar.sh "$@"
bash "$DIR"/scripts/PEMS03.sh "$@"
bash "$DIR"/scripts/PEMS04.sh "$@"
bash "$DIR"/scripts/PEMS07.sh "$@"
bash "$DIR"/scripts/PEMS08.sh "$@"

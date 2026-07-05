#!/usr/bin/env bash
# One-click entry: run every dataset script in scripts/ sequentially.
# Any extra args are passed through to each script (and then to run.py),
# so you can switch model / override hyper-parameters, e.g.:
#   bash run_main.sh --model iTransformer
#   bash run_main.sh --model PatchTST --train_epochs 5
#   MODEL=iTransformer bash run_main.sh
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the seven common datasets
sh "$DIR"/scripts/ETTh1.sh "$@"
sh "$DIR"/scripts/ETTh2.sh "$@"
sh "$DIR"/scripts/ETTm1.sh "$@"
sh "$DIR"/scripts/ETTm2.sh "$@"
sh "$DIR"/scripts/Electricity.sh "$@"
sh "$DIR"/scripts/Traffic.sh "$@"
sh "$DIR"/scripts/Weather.sh "$@"

# extras
sh "$DIR"/scripts/Solar.sh "$@"
sh "$DIR"/scripts/PEMS03.sh "$@"
sh "$DIR"/scripts/PEMS04.sh "$@"
sh "$DIR"/scripts/PEMS07.sh "$@"
sh "$DIR"/scripts/PEMS08.sh "$@"

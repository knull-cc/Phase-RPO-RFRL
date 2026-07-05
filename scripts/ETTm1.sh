#!/usr/bin/env bash
# ETTm1 (7 variates, 15-min). Usage: bash scripts/ETTm1.sh [--model X ...]
model_name=${MODEL:-PhaseRPO_RFRL_MLP}
extra_args="$@"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      if [ "$#" -gt 1 ]; then
        model_name="$2"
      fi
      shift 2
      ;;
    --model=*)
      model_name="${1#--model=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

seq_len=96
for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data ETTm1 --root_path ./dataset/ETT-small/ --data_path ETTm1.csv --freq t \
    --model_id ETTm1_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des Exp --itr 1 --batch_size 32 --learning_rate 0.0001 \
    $extra_args
done

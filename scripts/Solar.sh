#!/usr/bin/env bash
# Solar-Energy (137 variates, 10-min). Usage: bash scripts/Solar.sh [--model X ...]
model_name=${MODEL:-DLinear}
extra_args="$@"

seq_len=96
for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data Solar --root_path ./dataset/Solar/ --data_path solar_AL.txt --freq t \
    --model_id Solar_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 137 --dec_in 137 --c_out 137 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des Exp --itr 1 --batch_size 32 --learning_rate 0.0001 \
    $extra_args
done

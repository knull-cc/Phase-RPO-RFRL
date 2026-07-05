#!/usr/bin/env bash
# Traffic (862 variates, hourly). Usage: bash scripts/Traffic.sh [--model X ...]
model_name=${MODEL:-DLinear}
extra_args="$@"

seq_len=96
for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --freq h \
    --model_id traffic_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --e_layers 4 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des Exp --itr 1 --batch_size 16 --learning_rate 0.001 \
    $extra_args
done

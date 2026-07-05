#!/usr/bin/env bash
# PEMS07 (883 nodes). Usage: bash scripts/PEMS07.sh [--model X ...]
model_name=${MODEL:-DLinear}
extra_args="$@"

seq_len=96
for pred_len in 12 24 48 96; do
  python -u run.py \
    --task_name long_term_forecast --is_training 1 \
    --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS07.npz --freq h \
    --model_id PEMS07_${seq_len}_${pred_len} --model "$model_name" \
    --features M --seq_len $seq_len --label_len 48 --pred_len $pred_len \
    --enc_in 883 --dec_in 883 --c_out 883 \
    --e_layers 2 --d_layers 1 --factor 3 --d_model 512 --d_ff 512 \
    --des Exp --itr 1 --batch_size 32 --learning_rate 0.001 \
    $extra_args
done

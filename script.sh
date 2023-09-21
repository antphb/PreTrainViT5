#!/bin/bash

# Đường dẫn đến thư mục venv
venv_dir=".venv"

# Kiểm tra xem thư mục venv đã tồn tại hay chưa
if [ -d "$venv_dir" ]; then
   source "$venv_dir/bin/activate"
else
    python3 -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    pip install -r requirements.txt
fi

token_huggingface='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'
model_name='VietAI/vit5-large'
path_data='/home/bbsw/Desktop/Model_nba/UIT/translated_data/data_vi.json'

train_batch_size=2
valid_batch_size=2
gradient_accumulation_steps=8

max_len=256
epochs=10
learning_rate=1e-05
output_dir='pretrain-vit5-large'

CUDA_VISIBLE_DEVICES=1 python3 pretrainviT5.py --token_huggingface $token_huggingface --model_name $model_name --path_data $path_data --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --epochs $epochs --learning_rate $learning_rate --max_len $max_len --output_dir $output_dir --gradient_accumulation_steps $gradient_accumulation_steps

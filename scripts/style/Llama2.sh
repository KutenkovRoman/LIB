#!/bin/bash

# clear

# --wandb \

for LAYERS in 32; do #1 2 3 5 7
    #"sad" "cheerful" "agitational" "reflective"
    for STYLE in "cheerful_astroph"; do
        # --n_epoches_train 8 | --max_steps 200
        # --wandb & --wandb_project ...
        # google/gemma-2b | mistralai/Mistral-7B-v0.1
        CUDA_VISIBLE_DEVICES=0 python ./src/run_experiment.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset_path src/fine_tuning/style/data/$STYLE \
        --results_path "results_raw/Llama_${STYLE}/orthog_adapters" \
        --seed 8288 \
        --wandb_project new_styles \
        --padding_side right \
        --optimizer adamw \
        --lr 4e-4 \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --max_grad_norm 1.5 \
        --weight_decay 1e-2 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.175 \
        --logging_steps 4 \
        --max_steps 144 \
        --max_seq_length 192 \
        --ft_strategy LoRA \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --quantization_bit 4 \
        --dtype bfloat16 \
        --use_fast_tokenizer \
        --dataset style \
        --eval_strategy no \
        --n_layers $LAYERS \
        --run_name "[${STYLE} | all layers]"
    done
done

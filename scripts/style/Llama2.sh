#!/bin/bash

clear

for SEED in 8288; do #100 200 300 400 500 600 700
    for LAYERS in 6; do #1 2 3 5 7
        #"sad" "cheerful" "agitational" "reflective"
        for STYLE in "splitted_sad" "splitted_cheerful" "splitted_agitational" "splitted_reflective"; do
            # --n_epochs_train 8 | --max_steps 200
            # --wandb & --wandb_project ...
            # google/gemma-2b | mistralai/Mistral-7B-v0.1 // need to load for a really long time
            CUDA_VISIBLE_DEVICES=3 python ./src/run_experiment.py \
            --model meta-llama/Llama-2-7b-hf \
            --dataset_path src/fine_tuning/style/data/$STYLE \
            --results_path "results_raw/Llama_${STYLE}/seed_${SEED}/layers_0_${LAYERS}_with_mlp" \
            --seed $SEED \
            --wandb \
            --wandb_project new_styles \
            --padding_side right \
            --optimizer adamw \
            --lr 5e-4 \
            --batch_size 8 \
            --gradient_accumulation_steps 12 \
            --max_grad_norm 1.5 \
            --weight_decay 1e-3 \
            --lr_scheduler_type linear \
            --warmup_ratio 0.175 \
            --logging_steps 4 \
            --max_steps 1200 \
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
            --n_layers $LAYERS
        done
    done
done

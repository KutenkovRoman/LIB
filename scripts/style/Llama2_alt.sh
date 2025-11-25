#!/bin/bash

clear

#for SEED in 69615 89691 76240 19707 75789 30408 8288 96840 80020 91598 94890 18197 88692 72236 97107 7946 65463 79495 23249 19672; do
for LR in 7e-4; do
    for STYLE in "cosmology"; do #"conversational" "scientific" "business" "aggressive"
    #for STYLE in "neutral"; do
        # --n_epochs_train 8 | --max_steps 100
        # meta-llama/Llama-2-7b-hf | meta-llama/Llama-3.1-8B
        # --wandb & --wandb_project Llama_cosmology_theme
        echo "Running with lr=${LR} and style=${STYLE}"
        CUDA_VISIBLE_DEVICES=1 python ./src/run_experiment.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset_path src/fine_tuning/style/data/$STYLE \
        --results_path results_raw/Llama_${STYLE}/seed_8288/layers_5_32 \
        --seed 8288 \
        --wandb \
        --wandb_project RandomLayers \
        --padding_side left \
        --optimizer adamw \
        --lr $LR \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --max_grad_norm 1.0 \
        --weight_decay 1e-3 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --logging_steps 5 \
        --n_epoches_train 15 \
        --max_seq_length 512 \
        --ft_strategy LoRA \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --quantization_bit 4 \
        --dtype bfloat16 \
        --use_fast_tokenizer \
        --dataset style \
        --save_strategy no \
        --eval_steps 2000
    done
done

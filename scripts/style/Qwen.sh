#!/bin/bash

clear

for SEED in 8288; do
    for STYLE in "aggressive"; do  #"conversational" "scientific" "business" "aggressive"
        for LR in 3e-5; do
            # Qwen2-7B | Qwen3-8B
            # options: lr, max_grad_norm, weight_decay, warmup_ratio/warmu_steps
            # maybe try --min_lr_ratio 0.1 (need to implement manually?)
            echo "Running with seed=${SEED}, lr=${LR} and style=${STYLE}"
            CUDA_VISIBLE_DEVICES=2 python ./src/run_experiment.py \
            --model Qwen/Qwen2-7B \
            --dataset_path src/fine_tuning/style/data/$STYLE \
            --results_path results_raw/Qwen_${STYLE}/seed_${SEED} \
            --wandb \
            --wandb_project QwenStyle \
            --seed $SEED \
            --padding_side left \
            --optimizer adamw \
            --beta2 0.95 \
            --lr $LR \
            --batch_size 16 \
            --gradient_accumulation_steps 4 \
            --max_grad_norm 0.5 \
            --weight_decay 0.1 \
            --lr_scheduler_type cosine \
            --logging_steps 5 \
            --n_epoches_train 20 \
            --warmup_ratio 0.3 \
            --max_seq_length 512 \
            --ft_strategy LoRA \
            --lora_r 16 \
            --lora_alpha 32 \
            --lora_dropout 0.05 \
            --quantization_bit 4 \
            --dtype bfloat16 \
            --use_fast_tokenizer \
            --dataset style \
            --eval_strategy no \
            --save_strategy steps
        done
    done
done

#!/bin/bash

clear

for SEED in 7946; do #88692 72236 97107 7946 65463
#for SEED in 69615 89691 76240 19707 75789 30408 8288 96840 80020 91598 94890 18197 88692 72236 97107 7946 65463 79495 23249 19672; do
    for STYLE in "business" "aggressive"; do #"conversational" "scientific"
        for LR in 3e-5; do #7e-5 1e-5
            # --n_epoches_train 8 OR --max_steps 200
            # maybe try --min_lr_ratio 0.1
            echo "Running with seed=${SEED}, lr=${LR} and style=${STYLE}"
            CUDA_VISIBLE_DEVICES=2 python ./src/run_experiment.py \
            --model Qwen/Qwen2-7B \
            --dataset_path src/fine_tuning/style/data/$STYLE \
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
            --n_epoches_train 25 \
            --warmup_ratio 0.3 \
            --max_seq_length 512 \
            --wandb \
            --wandb_project Qwen_from_neutral \
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

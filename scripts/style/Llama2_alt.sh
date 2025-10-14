#!/bin/bash

clear

#for SEED in 69615 89691 76240 19707 75789 30408 8288 96840 80020 91598 94890 18197 88692 72236 97107 7946 65463 79495 23249 19672; do
for LR in 1e-5; do #3e-4 1e-4 7e-5 4e-5 1e-5 5e-4
    for STYLE in "conversational" "scientific" "business" "aggressive"; do
    #for STYLE in "neutral"; do
        # --n_epoches_train 8 OR --max_steps 100
        echo "Running with lr=${LR} and style=${STYLE}"
        python ./src/run_experiment.py \
        --model meta-llama/Llama-2-7b-hf \
        --dataset_path src/fine_tuning/style/data/$STYLE \
        --seed 19707 \
        --padding_side left \
        --optimizer adamw \
        --lr $LR \
        --batch_size 8 \
        --gradient_accumulation_steps 4 \
        --weight_decay 1e-4 \
        --lr_scheduler_type linear \
        --warmup_ratio 0.1 \
        --logging_steps 10 \
        --max_steps 100 \
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
        --save_strategy no #steps
        #--wandb \
    done
done

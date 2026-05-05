#!/bin/bash

#clear

# meta-llama/Llama-2-7b-hf | meta-llama/Llama-3.1-8B | Qwen/Qwen3-8B
# neuralmagic/Llama-2-7b-gsm8k
# [0 -> 0|1 -> 2|2 -> 3|3 -> 4|4 -> 5|5 -> 6|6 -> 1|7 -> 7]

# "sad_" "cheerful_" "agitational_" "reflective_"
# "sad_ext_gsm8k_lamma3" "cheerful_ext_gsm8k_lamma3" "agitational_ext_gsm8k_lamma3" "reflective_ext_gsm8k_lamma3"
# "gsm8k_prompt_lamma3"
for RUN_NAME in "sad_kl_reg_0_1"; do
    for LAYERS in 7; do
    CUDA_VISIBLE_DEVICES=0 python ./src/run_experiment.py \
        --model meta-llama/Llama-2-7b-hf \
        --do_predict \
        --seed 0 \
        --padding_side right \
        --max_seq_length 192 \
        --quantization_bit 4 \
        --dtype bfloat16 \
        --ft_strategy LoRA \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --use_fast_tokenizer \
        --dataset style \
        --n_layers $LAYERS \
        --run_name $RUN_NAME
    done
done

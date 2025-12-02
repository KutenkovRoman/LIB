#!/bin/bash

#clear

# meta-llama/Llama-2-7b-hf | meta-llama/Llama-3.1-8B
# result_path: path to folder containing 'theme' adapters
CUDA_VISIBLE_DEVICES=3 python ./src/run_experiment.py \
    --model meta-llama/Llama-2-7b-hf \
    --results_path results_raw/Llama_cosmology/seed_8288/layers_5_32 \
    --do_predict \
    --seed 0 \
    --padding_side left \
    --quantization_bit 4 \
    --dtype bfloat16 \
    --ft_strategy LoRA \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_fast_tokenizer \
    --dataset style

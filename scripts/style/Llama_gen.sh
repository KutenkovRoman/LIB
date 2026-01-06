#!/bin/bash

#clear

# meta-llama/Llama-2-7b-hf | meta-llama/Llama-3.1-8B
# results_raw/Llama_cosmology/seed_8288/layers_5_32 \
# results_raw/Llama_splitted_cheerful/seed_8288/layers_0_5_with_mlp \
CUDA_VISIBLE_DEVICES=4 python ./src/run_experiment.py \
    --model meta-llama/Llama-2-7b-hf \
    --do_predict \
    --seed 0 \
    --padding_side right \
    --quantization_bit 4 \
    --dtype bfloat16 \
    --ft_strategy LoRA \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_fast_tokenizer \
    --dataset style \
    --n_layers 5

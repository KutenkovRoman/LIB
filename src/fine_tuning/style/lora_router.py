import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from bitsandbytes.functional import dequantize_4bit
from datasets import load_dataset

import sys
sys.path.append("src")

import warnings
from utils import set_global_seed

import math
import re

# Suppress harmless PEFT warnings about key mismatches when loading multiple adapters
warnings.filterwarnings("ignore", message="Found missing adapter keys")


class LoRARouterLinear(nn.Module):
    """Replaces a standard Linear layer with a router that interpolates between 2 LoRA adapters"""

    def __init__(self, peft_layer, adapter1='lora1', adapter2='lora2', mu_init=0.0, shared_mu=None):
        super().__init__()

        # Freeze base weights
        # self.base_weight_quant = peft_layer.base_layer.weight #.detach()
        quant_weight = peft_layer.base_layer.weight
        self.weight_data = quant_weight.data  # Raw uint8 buffer
        self.quant_state = quant_weight.quant_state

        self.bias = peft_layer.bias.detach() if peft_layer.bias is not None else None

        self.register_buffer('A1', peft_layer.lora_A[adapter1].weight.detach())
        self.register_buffer('B1', peft_layer.lora_B[adapter1].weight.detach())
        self.register_buffer('A2', peft_layer.lora_A[adapter2].weight.detach())
        self.register_buffer('B2', peft_layer.lora_B[adapter2].weight.detach())

        # Extract scaling factors (lora_alpha / rank)
        self.scaling1 = peft_layer.scaling[adapter1] if isinstance(peft_layer.scaling, dict) else peft_layer.scaling
        self.scaling2 = peft_layer.scaling[adapter2] if isinstance(peft_layer.scaling, dict) else peft_layer.scaling
        # print(f"scaling1 = {self.scaling1}, scaling2 = {self.scaling2}")

        # Trainable mixing coefficient per layer
        self.mu = (
            nn.Parameter(torch.tensor(mu_init, dtype=torch.bfloat16))  #dtype=self.W_0.dtype
            if shared_mu is None
            else shared_mu
        )

        # Preserve layer metadata
        self.in_features = peft_layer.in_features
        self.out_features = peft_layer.out_features

    def forward(self, x):
        mu = torch.sigmoid(self.mu)

        # Dequantize base weight on-the-fly (returns compute dtype, e.g., bfloat16)
        W0 = dequantize_4bit(self.weight_data, self.quant_state)
        dtype = W0.dtype

        # assert W0.shape == (self.out_features, self.in_features), f"{W0.shape = }"

        # Cast LoRA factors to match compute dtype to avoid mixed-precision errors
        A1, B1 = self.A1.to(dtype), self.B1.to(dtype)
        A2, B2 = self.A2.to(dtype), self.B2.to(dtype)

        # print(x.shape, W0.shape, None if self.bias is None else self.bias.shape)
        out = F.linear(x, W0, self.bias)
        out = out + mu * self.scaling1 * F.linear(F.linear(x, A1), B1)
        out = out + (1 - mu) * self.scaling2 * F.linear(F.linear(x, A2), B2)

        return out


def wrap_model_with_lora_router(
    model,
    lora1_path: str,
    lora2_path: str,
    adapter1_name: str = 'lora1',
    adapter2_name: str = 'lora2',
) -> nn.Module:
    model = PeftModel.from_pretrained(model, lora1_path, adapter_name=adapter1_name, is_trainable=False)
    model = PeftModel.from_pretrained(model, lora2_path, adapter_name=adapter2_name, is_trainable=False)
    # model.load_adapter(lora2_path, adapter_name=adapter2_name, is_trainable=False)

    for param in model.parameters():
        param.requires_grad = False

    layer_pattern = re.compile(r"layers\.(\d+)\.")
    n_layers = 32
    modules_to_replace = [list() for _ in range(n_layers)]

    for name, module in model.named_modules():
        if (
            hasattr(module, 'lora_A') and
            adapter1_name in module.lora_A and adapter2_name in module.lora_A
        ):
            match = layer_pattern.search(name)
            if match is None:
                raise RuntimeError(f"Error: {name} does not have matching pattern")

            layer_idx = int(match.group(1))
            modules_to_replace[layer_idx].append((name, module))

    print(f"Replacing {len(modules_to_replace)} layers with LoRA routers.")
    assert all(len(ith_layer_modules) == 7 for ith_layer_modules in modules_to_replace)
    # for i, ith_layer_modules in enumerate(modules_to_replace):
    #     if len(ith_layer_modules) != 7:
    #         modulelist = ", ".join([name for name, _ in ith_layer_modules])
    #         print(f"{i}-th layer has {len(ith_layer_modules)} modules: {modulelist}")

    for i in range(n_layers):
        shared_mu = None
        for name, module in modules_to_replace[i]:
            if shared_mu is None:
                new_module = LoRARouterLinear(module, adapter1_name, adapter2_name)
                new_module = new_module.to(module.base_layer.weight.device)
                shared_mu = new_module.mu
            else:
                new_module = LoRARouterLinear(module, adapter1_name, adapter2_name, shared_mu=shared_mu)

            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent, child_name = model, name

            setattr(parent, child_name, new_module)

    # Ensure new parameters are on the correct device
    model.train()

    return model


def main():
    set_global_seed(0)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        attn_implementation="eager",
    )
    # print(base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_fast=True,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0

    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(f"Warning: {len(tokenizer) = } is greater than {embedding_size = }!")

    # base_model = prepare_model_for_kbit_training(base_model)

    style = "agitational"
    routed_model = wrap_model_with_lora_router(
        base_model,
        lora1_path=f"./src/fine_tuning/style/results_raw/Llama_cosmology/seed_8288/all_layers",
        lora2_path=f"./src/fine_tuning/style/results_raw/Llama_{style}/seed_8288/all_layers",
        # lora1_path=f"./src/fine_tuning/style/results_raw/Llama_{style}/seed_8288/all_layers",
        # lora2_path=f"./src/fine_tuning/style/results_raw/Llama_cosmology/seed_8288/all_layers",
    )

    # dataset = load_dataset(f"./src/fine_tuning/style/data/{style}_astroph", split="train")
    dataset = load_dataset(f"./src/fine_tuning/style/data/{style}", split="train")
    max_len = 192

    def process(batch):
        samples = [sample + tokenizer.eos_token for sample in batch["text"]]

        return tokenizer(samples, max_length=max_len, truncation=True)

    dataset = dataset.map(
        process,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./src/fine_tuning/style/results_raw/Lamma_router_{style}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        # optim="sgd",
        learning_rate=1e-2,  #5e-4 1e-3
        num_train_epochs=1,  #30
        logging_steps=4,
        save_strategy="no",
        fp16=False,
        bf16=True,
        report_to=["none"],  # "wandb"/"none"
    )

    trainer = Trainer(
        model=routed_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    with open("./src/fine_tuning/style/output/router_scores2.txt", 'w') as f:
        for name, module in routed_model.named_modules():
            if isinstance(module, LoRARouterLinear) and 'down_proj' in name:
                mu_logit = float(module.mu.data.item())
                mu = 1 / (1 + math.exp(-mu_logit))
                f.write(
                    name.replace(".mlp.down_proj", "") + f".mu = {mu:.8f} (logit = {mu_logit:.8f})\n"
                )


if __name__ == '__main__':
    main()

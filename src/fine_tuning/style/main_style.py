import torch
import datasets
import wandb

from loguru import logger
import peft

import utils

import warnings
import os
from safetensors.torch import save_file
from safetensors import safe_open

warnings.filterwarnings("ignore")

from optimizers.main import get_optimizer

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    pipeline,
)


class Finetuner:
    """Main class for downstream finetuning"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None

        self.setup_logging()
        if self.args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info(f"Starting downstream finetuning for dataset: {self.args.dataset}")

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        style = self.args.dataset_path.split('/')[-1]
        run_name = f"[{style} | lr={self.args.lr:.1e} | seed={self.args.seed}]"
        wandb.init(
            project=self.args.wandb_project,
            tags=[self.args.model, self.args.dataset, self.args.optimizer],
            name=run_name,
            config=self.args,
        )

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configurations"""
        logger.info("Loading model and tokenizer...")

        # Determine optimal settings based on GPU capability
        # [TODO] add flash_attention
        if torch.cuda.get_device_capability()[0] >= 8:
            attn_implementation = "eager"  # fix flash_attention_2
        else:
            attn_implementation = "eager"

        # Set dtype
        if self.args.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.args.dtype == "float16":
            torch_dtype = torch.float16
        elif self.args.dtype == "float32":
            torch_dtype = torch.float32
        elif self.args.dtype == "float64":
            torch_dtype = torch.float64

        # Setup quantization config
        if self.args.quant_bit == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch_dtype,
            )
        elif self.args.quant_bit == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model if self.args.tokenizer is None else self.args.tokenizer,
            use_fast=self.args.use_fast_tokenizer,
            padding_side=self.args.padding_side,
        )
        self.tokenizer.pad_token_id = 0

        # Resize token embeddings if necessary
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        logger.info("Model and tokenizer loaded successfully.")

    def setup_peft(self):
        """Setup PEFT (Parameter Efficient Fine-Tuning) adapters"""
        logger.info("Setting up PEFT adapters...")

        peft_args = utils.get_peft_arguments(self.args)
        peft_args.task_type = "CAUSAL_LM"
        if peft_args is not None:
            self.model = peft.get_peft_model(self.model, peft_args)

        # Print trainable parameters info
        all_param_count, tr_param_count, tr_persent = utils.print_trainable_params(
            self.model, verbose=True
        )

        num_peft_adapters = utils.count_atapters(self.model, self.args.ft_strategy)

        ##TODO: rewrite this so these metrics are not displayed as one-point plots
        report_param_stat = False
        if self.args.wandb and report_param_stat:
            wandb.log({
                "trainable_params_count": tr_param_count,
                "total_param_count": all_param_count,
                "trainable_params_percentage": tr_persent,
                "num_peft_adapters": num_peft_adapters,
            })

    def load_datasets(self):
        """Load and prepare both training and evaluation datasets"""
        logger.info("Loading datasets...")

        # Load train and validation data as list of sentences
        self.train_dataset = datasets.load_dataset(self.args.dataset_path, split='train')
        #self.eval_dataset = datasets.load_dataset(self.args.dataset_path, split='test')

    def prepare_training_dataset(self):
        """Prepare dataset for training"""

        logger.info("Preparing training dataset...")

        dataset = self.train_dataset

        ## Add prompt conditioning for better guidance (hopefully) -- removed
        #dataset = dataset.map(
        #    lambda sample: {"text": f'{self.args.base_prompt} {sample["text"]}'}
        #)

        # Apply preprocessing for training
        max_len = self.args.max_seq_length

        def process(batch):
            samples = [sample + self.tokenizer.eos_token for sample in batch["text"]]

            return self.tokenizer(samples, max_length=max_len, truncation=True)

        dataset = dataset.map(
            process,
            batched=True,
            remove_columns=["text"]
        )

        # Filter out samples that exceed max_length
        dataset = dataset.filter(
            lambda sample: len(sample["input_ids"]) < max_len
        )

        # Shuffle dataset
        self.train_dataset = dataset.shuffle(seed=self.args.seed)

    def prepare_evaluation_dataset(self):
        """Prepare dataset for evaluation"""

        logger.info("Preparing evaluation dataset...")

        dataset = self.eval_dataset

        ## Add prompt conditioning for better guidance (hopefully) -- removed
        #dataset = dataset.map(
        #    lambda sample: {"text": f'{self.args.base_prompt} {sample["text"]}'}
        #)

        # Apply preprocessing for training
        max_len = self.args.max_seq_length
        dataset = dataset.map(
            lambda batch: self.tokenizer(batch["text"], max_length=max_len, truncation=True),
            batched=True,
            remove_columns=["text"]
        )

        # Filter out samples that exceed max_length
        self.eval_dataset = dataset.filter(
            lambda sample: len(sample["input_ids"]) < max_len
        )

    def train(self):
        """Execute training process"""
        if self.args.do_not_train:
            logger.info("Training skipped (do_not_train=True)")
            return

        logger.info(f"Starting training for dataset: {self.args.dataset}")

        # Prepare training and evaluation datasets
        self.prepare_training_dataset()
        if self.train_dataset is None:
            logger.error("No training data available")
            return

        # Setup trainer
        output_dir = f"./src/fine_tuning/style/{self.args.results_path}"
        training_args = TrainingArguments(
            do_train=(not self.args.do_not_train),
            do_eval=(not self.args.do_not_eval),
            do_predict=self.args.do_predict,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=(
                self.args.eval_batch_size if self.args.eval_batch_size else
                self.args.batch_size
            ),
            gradient_accumulation_steps=self.args.grad_acc_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            warmup_ratio=self.args.warmup_ratio, #added
            max_grad_norm=self.args.max_grad_norm, #added
            learning_rate=self.args.lr,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy=self.args.eval_strategy,
            save_strategy=self.args.eval_strategy,
            eval_steps=self.args.eval_steps,
            save_steps=self.args.eval_steps,
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            output_dir=output_dir,
            overwrite_output_dir=True, #added
            #logging_dir=output_dir,
            run_name=self.args.run_name,
            report_to=["wandb" if self.args.wandb else "none"],
        )

        optimizer = get_optimizer(self.args, self.model)

        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            #eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            optimizers=[optimizer, None], # scheduler will be added in the hf trainer
        )

        # Clean up memory before training
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # Train the model
        train_result = trainer.train()

        metrics = train_result.metrics
        if self.args.ft_strategy == "WeightLoRA":
            remain_adapters = utils.count_remain_adapters(self.args, self.model)
            metrics = metrics | remain_adapters        
        trainer.log_metrics("train", metrics)
        #logger.info(f"Training completed. Metrics: {metrics}")

    def evaluate(self, prompts, theme_path, style_path, theme_layers=None, style_layers=None):
        """Execute evaluation process"""
        utils.set_global_seed(self.args.seed)

        self.load_model_and_tokenizer()

        self.setup_peft()
        #output_dir = f"/home/kutroman/LIB/src/fine_tuning/style/{self.args.results_path}"
        #self.model = peft.PeftModel.from_pretrained(self.model, output_dir)
        #self.tokenizer = AutoTokenizer.from_pretrained(output_dir) # does not seem to impact generation
        for name, module in self.model.named_modules():
            if isinstance(module, peft.tuners.lora.layer.LoraLayer):
                try:
                    module._module_name = name
                    module._skip_prob = 0.0
                    module._random_state = 0
                except Exception as e:
                    logger.error(f"Failed with exception: {e}")

        for name, param in self.model.named_parameters():
            if (
                theme_layers is not None and "lora_" in name and
                any([name.startswith(f"base_model.model.model.layers.{i}.") for i in theme_layers])
            ):
                with safe_open(theme_path, framework="pt", device="cuda") as f:
                    adapter = f.get_tensor(name.replace(".default", "")) # handles naming convention
                    with torch.no_grad():
                        param.copy_(adapter)
            elif (
                style_layers is not None and "lora_" in name and
                any([name.startswith(f"base_model.model.model.layers.{i}.") for i in style_layers])
            ):
                with safe_open(style_path, framework="pt", device="cuda") as f:
                    adapter = f.get_tensor(name.replace(".default", ""))
                    with torch.no_grad():
                        param.copy_(adapter)
            else:
                pass
                #print(f"Uninitialized paramter {name} with requires_grad = {param.requires_grad}")

        self.model.eval()

        # Setup text generation pipeline
        do_sample = True
        num_beams = 10
        repetition_penalty = 1.2
        print(f"{do_sample = } | {num_beams = } | {repetition_penalty = }")
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        for i, prompt in enumerate(prompts):
            try:
                generated_text = generator(
                    prompt,
                    max_new_tokens=96,
                    num_return_sequences=1,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )[0]["generated_text"]
                print("\n", generated_text.strip().replace("\n", " "), sep="")
            except Exception as e:
                logger.error(f"Error processing prompt {i}: {e}")
                continue

    def eval_klora(self, K, theme_path, style_path, prompts, num_steps, alpha=1.0, beta=0.0):
        utils.set_global_seed(self.args.seed)

        numer = denom = 0.0
        
        # Llama2 specific
        num_layers = 32
        modules = [
            'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj',
            'mlp.down_proj', 'mlp.up_proj', 'mlp.gate_proj'
        ]

        N = len(modules)

        style_score = [0] * (N * num_layers)
        theme_score = [0] * (N * num_layers)

        style_layer_score = [0] * num_layers
        theme_layer_score = [0] * num_layers

        account_qk_projs = True
        account_mlp_projs = True
        for i in range(num_layers):
            for j, x in enumerate(modules):
                adapter_name = f"base_model.model.model.layers.{i}.{x}"
                k = N * i + j

                if not account_qk_projs and ('q_proj' in x or 'k_proj' in x):
                    continue
                if not account_mlp_projs and 'mlp' in x:
                    continue

                with safe_open(style_path, framework="pt", device="cuda") as f:
                    A = f.get_tensor(f"{adapter_name}.lora_A.weight")
                    B = f.get_tensor(f"{adapter_name}.lora_B.weight")
                    W_abs = torch.abs(torch.matmul(B, A))
                    elems, _ = torch.sort(W_abs.flatten(), descending=True)
                    style_score[k] = torch.sum(elems[:K]).item()
                    style_layer_score[i] += style_score[k]
                    denom += torch.sum(elems).item()

                with safe_open(theme_path, framework="pt", device="cuda") as f:
                    A = f.get_tensor(f"{adapter_name}.lora_A.weight")
                    B = f.get_tensor(f"{adapter_name}.lora_B.weight")
                    W_abs = torch.abs(torch.matmul(B, A))
                    elems, _ = torch.sort(W_abs.flatten(), descending=True)
                    theme_score[k] = torch.sum(elems[:K]).item()
                    theme_layer_score[i] += theme_score[k]
                    numer += torch.sum(elems).item()

        gamma = numer / denom
        print(f"{gamma = }")

        self.load_model_and_tokenizer()
        self.setup_peft()

        self.model.eval()  # do not know if this is useful

        for i, prompt in enumerate(prompts):
            try:
                input_seq = prompt
                outputs = []
                for t in range(num_steps):
                    S = alpha * t / (num_steps - 1) + beta
                    S *= gamma

                    target = "base_model.model.model.layers."
                    for name, param in self.model.named_parameters():
                        if name.startswith(target) and "lora_" in name:
                            layer, block, proj = name.removeprefix(target).split(".")[:3]
                            i = int(layer)
                            j = modules.index(block + "." + proj)  # not used
                            k = N * i + j                          # not used

                            path = (
                                style_path if S * style_layer_score[i] > theme_layer_score[i] else
                                theme_path
                            )
                            with safe_open(path, framework="pt", device="cuda") as f:
                                adapter = f.get_tensor(name.replace(".default", ""))
                                with torch.no_grad():
                                    param.copy_(adapter)

                    debug_print = False
                    if debug_print and t % 3 == 2:  # print every 3 steps starting with step 2
                        print(f"Step t = {t} with S = {S / gamma} and S' = {S}")
                        for i in range(num_layers):
                            subject = (
                                "style" if S * style_layer_score[i] > theme_layer_score[i] else
                                "theme"
                            )
                            print(
                                f"Initialized layer {' ' if i < 10 else ''}{i} with '{subject}' adapter "
                                f"(S' * {style_layer_score[i]:.4f} = {S * style_layer_score[i]:.4f}"
                                f" vs {theme_layer_score[i]:.4f})"
                            )

                    generator = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        return_full_text=False,
                    )

                    generated_seq = generator(
                        input_seq,
                        max_new_tokens=96,
                        num_return_sequences=1,
                    )[0]["generated_text"].strip().replace("\n", " ")

                    if generated_seq == "":
                        print(f"Stopping at step {t}")
                        break

                    output = (
                        generated_seq if ' ' not in generated_seq else
                        generated_seq.split(' ')[0]
                    )
                    outputs.append(output)
                    input_seq = input_seq + ' ' + output

                print(outputs)
                print(input_seq)
            except Exception as e:
                logger.error(f"Error processing prompt {i}: {e}")
                continue

    def run_each_layer(self):
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        num_layers = 32  # Llama specific
        for i in range(num_layers):
            self.load_model_and_tokenizer()

            logger.info(f"Setting up PEFT adapters for layer {i}...")
            
            # Load model to continue training
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                f"./src/fine_tuning/style/{self.args.results_path}",
                is_trainable=True
            )

            # Assuming that layers are named as layers.{layer_no}.{module}
            for name, param in self.model.named_parameters():
                if f".{i}." not in name:
                    param.requires_grad = False

            self.load_datasets()

            self.train()

            output_dir = f"./src/fine_tuning/style/{self.args.results_path}/layer_{i}"
            os.makedirs(output_dir, exist_ok=True)

            peft_state_dict = peft.get_peft_model_state_dict(self.model)

            layer_state = {}
            for key, value in peft_state_dict.items():
                if f".{i}." in key:
                    layer_state[key] = value

            if not layer_state:
                logger.info(
                    f"Warning: No adapter parameters found for layer {i}\n"
                    f"Available keys: {list(peft_state_dict.keys())[:10]}"
                )
            else:
                save_file(layer_state, os.path.join(output_dir, "adapter_model.safetensors"))

    def run(self, target_layers=None, save_adapters=False):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()

        if target_layers is None:
            logger.info(f"Setting up PEFT adapters for all layers...")
        else:
            layers_str = ', '.join([str(i) for i in target_layers])
            logger.info(f"Setting up PEFT adapters for layers {layers_str}...")

        peft_args = utils.get_peft_arguments(self.args)
        if peft_args is None:
            raise ValueError("ft_stratefy=Full is not supported for this method")

        peft_args.task_type = "CAUSAL_LM"
        self.model = peft.get_peft_model(self.model, peft_args)

        # See what's up
        for name, param in self.model.named_parameters():
            if target_layers is not None and not any([(f"layers.{i}." in name) for i in target_layers]):
                param.requires_grad = False
            if self.args.verbose and "lora_" in name.lower():
                print(f"{name} requires_grad={param.requires_grad}")

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()

        if save_adapters:
            peft_state_dict = peft.get_peft_model_state_dict(self.model)

            layer_state = {}
            for key, value in peft_state_dict.items():
                if (
                    (target_layers is None and "layers." in key) or
                    (target_layers is not None and any([f"layers.{i}." in key for i in target_layers]))
                ):
                    if self.args.verbose:
                        print(f"Added {key} to layer_state dictionary")
                    layer_state[key] = value

            if not layer_state:
                # To see available keys: f"Available keys: {list(peft_state_dict.keys())[:10]}"
                logger.info("Warning: layer_state dictionary is empty, there is nothing to save.")
                return

            output_dir = f"./src/fine_tuning/style/{self.args.results_path}"
            os.makedirs(output_dir, exist_ok=True)
            save_file(layer_state, os.path.join(output_dir, "adapter_model.safetensors"))
            if self.args.verbose:
                print(f"Saved adapters from layer_state dictionary to {output_dir}")

    def run_from_pretrained(self, save_path):
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load base model
        self.load_model_and_tokenizer()

        logger.info(f"Setting up PEFT adapters...")

        # Load model to continue training
        self.model = peft.PeftModel.from_pretrained(
            self.model,
            f"./src/fine_tuning/style/{save_path}",
            is_trainable=True
        )

        self.load_datasets()
        self.train()


def main(args):
    """Main entry point"""
    finetuner = Finetuner(args)

    ## For randoms layer experiments
    #theme_layers = [0, 1, 9, 12, 14, 15, 16, 18, 22, 24, 25, 27, 31]
    #style_layers = [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 17, 19, 20, 21, 23, 26, 28, 29, 30]

    if args.do_predict:
        neutral_prompts = [
            "She is", "The stars are", "Today, our", "I would ask you",
            "Well, whatever", "It is good", "What if",
        ]
        cosmology_prompts = [
            "When the accretion rate increases", "If the magnetic field reverses",
            "When the spectrum flattens at high energy", "If the core temperature exceeds 1e8 K",
            "As the jet becomes relativistic", "When the dust sublimates near the perihelion",
            "If the dark matter halo dominates the dynamics", "As the neutron star cools",
            "When turbulence develops in the plasma", "If the planet crosses the habitable zone",
            "As gravitational waves propagate outward", "When the white dwarf nears the Chandrasekhar limit",
            "As hydrogen accretes onto the degenerate core", "When thermonuclear runaway begins on the surface",
            "As the nova ejecta expand into interstellar space", "When the luminosity briefly exceeds the Eddington limit",
            "If helium burning stabilizes the outer layers", "As the protostar settles onto the main sequence",
        ]
        default_eval = True
        if default_eval:
            finetuner.evaluate(
                cosmology_prompts[:8],
                theme_path="/home/kutroman/LIB/src/fine_tuning/style/results_raw"
                "/Llama_cosmology/seed_8288/layers_5_32/adapter_model.safetensors",
                style_path="/home/kutroman/LIB/src/fine_tuning/style/results_raw"
                "/Llama_aggressive/seed_8288/layers_0_5/adapter_model.safetensors",
                theme_layers=list(range(5, 32)),
                style_layers=list(range(0, 5)),
            )
        else:
            finetuner.eval_klora(
                K=64,
                theme_path="/home/kutroman/LIB/src/fine_tuning/style/results_raw"
                "/Llama_cosmology/seed_8288/all_layers/adapter_model.safetensors",
                style_path="/home/kutroman/LIB/src/fine_tuning/style/results_raw"
                "/Llama_aggressive/seed_8288/all_layers/adapter_model.safetensors",
                prompts=["She is", "The stars are", "Today, our"], #neutral_prompts
                num_steps=24,
                alpha=1.0,
                beta=0.7,
            )
    else:
        finetuner.run(target_layers=list(range(5, 32)), save_adapters=True)


if __name__ == "__main__":
    main(None)

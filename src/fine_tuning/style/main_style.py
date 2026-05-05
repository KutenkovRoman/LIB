import torch
import datasets
import wandb

from loguru import logger

import peft
from peft import prepare_model_for_kbit_training

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

from kl_reg_trainer import KLRegularizedTrainer
from orthog_adapter import apply_orthogonal_adapters


class Finetuner:
    """Main class for downstream finetuning"""

    def __init__(self, args):
        self.args = args
        self.base_model = None
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
        wandb.init(
            project=self.args.wandb_project,
            tags=[self.args.model, self.args.dataset, self.args.optimizer],
            name=self.args.run_name,
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
            low_cpu_mem_usage=True, #device_map="auto"
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

        #self.tokenizer.pad_token = self.tokenizer.eos_token # for mistral?

        logger.info("Model and tokenizer loaded successfully.")

        # See what's up
        #print(self.model)

    def setup_peft(self):
        """Setup PEFT (Parameter Efficient Fine-Tuning) adapters"""
        logger.info("Setting up PEFT adapters...")

        self.model = prepare_model_for_kbit_training(self.model)

        peft_args = utils.get_peft_arguments(self.args)
        if peft_args is not None:
            peft_args.task_type = "CAUSAL_LM"
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

        # Apply preprocessing for training
        max_len = self.args.max_seq_length

        def process(batch):
            samples = [text + self.tokenizer.eos_token for text in batch["text"]]

            if self.args.use_kl_regularization:
                is_general = ["[NEUTRAL]" in text for text in samples]
                samples = [text[text.index(']') + 1:] for text in samples]

            tokenized_batch = self.tokenizer(samples, max_length=max_len, truncation=True)

            if self.args.use_kl_regularization:
                tokenized_batch['is_general'] = is_general

            return tokenized_batch

        dataset = dataset.map(
            process,
            batched=True,
            remove_columns=["text"],
        )

        # print("General signature:", dataset)
        # print("First sample:", dataset[0])
        # print("Last sample:", dataset[-1])
        # print(f"Before filtering we had {len(dataset)} samples")

        # Filter out samples that exceed max_length
        dataset = dataset.filter(
            lambda sample: len(sample["input_ids"]) < max_len
        )

        # print(f"After filtering we have {len(dataset)} samples")

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

    def train(self, save_adapters):
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
        do_save = (save_adapters == "default")
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
            warmup_ratio=self.args.warmup_ratio,
            max_grad_norm=self.args.max_grad_norm,
            learning_rate=self.args.lr,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy=self.args.eval_strategy,
            save_strategy=("steps" if do_save else "no"),
            eval_steps=self.args.eval_steps,
            save_steps=(100000 if do_save else -1),  # save at the very last step
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            output_dir=output_dir,
            overwrite_output_dir=True,
            # logging_dir=output_dir,
            run_name=self.args.run_name,
            report_to=["wandb" if self.args.wandb else "none"],
        )

        optimizer = get_optimizer(self.args, self.model)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, ...)

        if self.args.use_kl_regularization:
            trainer = KLRegularizedTrainer(
                self.base_model,
                kl_weight=0.01,
                model=self.model,
                train_dataset=self.train_dataset,
                args=training_args,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                optimizers=[optimizer, None],
            )
        else:
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
        # logger.info(f"Training completed. Metrics: {metrics}")

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

    def run_orthog_adapters(self, save_adapters="default"):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()

        self.model = prepare_model_for_kbit_training(self.model)

        # self.model = apply_orthogonal_adapters(self.model, target_regex=r"layers\.(15|16)\.mlp\.down_proj") |2|3|4|5|6
        self.model = apply_orthogonal_adapters(self.model, target_regex=r"layers\.(0|1)\.(mlp|self_attn)\..+_proj")
        logger.info("Applied orthogonal adapters to model.")

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train(save_adapters)

    def run(self, target_layers=None, save_adapters="default"):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()

        if self.args.use_kl_regularization:
            self.base_model = self.model

        if target_layers is None:
            logger.info(f"Setting up PEFT adapters for all layers...")
        else:
            layers_str = ', '.join([str(i) for i in target_layers])
            logger.info(f"Setting up PEFT adapters for layers {layers_str}...")

        self.setup_peft()

        if target_layers is not None:
            for name, param in self.model.named_parameters():
                if not any([(f"layers.{i}." in name) for i in target_layers]):
                    param.requires_grad = False
                # if self.args.verbose and "lora_" in name.lower():
                #    print(f"{name} requires_grad={param.requires_grad}")

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train(save_adapters)

        if save_adapters == "manual":
            peft_state_dict = peft.get_peft_model_state_dict(self.model)

            layer_state = {}
            for key, value in peft_state_dict.items():
                if (
                    (target_layers is None and ("layers" in key or "lm_head" in key)) or
                    (target_layers is not None and any([f"layers.{i}." in key for i in target_layers]))
                ):
                    # if self.args.verbose:
                    #    print(f"Added {key} to layer_state dictionary")
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

    def score(self, texts, model_path=None, return_token_details=False):
        utils.set_global_seed(self.args.seed)

        # Load base model and tokenizer (avoid loading multiple times?)
        #self.load_model_and_tokenizer()

        model = peft.PeftModel.from_pretrained(
            self.model,
            (
                f"./src/fine_tuning/style/{self.args.results_path}" if model_path is None else
                model_path
            ),
        )
        model.cuda()
        model.eval()

        logger.info(f"Started scoring texts...")

        results = []

        #bs = self.args.batch_size  # use datasets | range((len(texts) + bs - 1) // bs)
        for i in range(0, len(texts), self.args.batch_size):
            inputs = self.tokenizer(
                texts[i:(i + self.args.batch_size)],
                max_length=self.args.max_seq_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            input_ids = inputs['input_ids'].cuda()
            attention_mask = inputs['attention_mask'].cuda()

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Shift for next-token prediction
                shifted_logits = logits[:, :-1, :]
                shifted_targets = input_ids[:, 1:]
                shifted_mask = attention_mask[:, 1:]

                # Calculate log probabilities
                log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                token_log_probs = log_probs.gather(2, shifted_targets.unsqueeze(-1)).squeeze(-1)

                # Mask out padding
                token_log_probs = token_log_probs * shifted_mask

                # Process each sequence
                for j in range(self.args.batch_size):
                    # Get valid positions
                    valid_mask = shifted_mask[j].bool()
                    valid_log_probs = token_log_probs[j][valid_mask]

                    token_details = []
                    if len(valid_log_probs) == 0:
                        # Empty sequence or single token
                        agg_scores = {
                            'mean_log_prob': 0.0,
                            'total_log_prob': 0.0,
                            'perplexity': 1.0,
                            'token_count': 0
                        }
                    else:
                        # Calculate aggregate scores
                        mean_log_prob = valid_log_probs.mean().item()
                        total_log_prob = valid_log_probs.sum().item()
                        perplexity = torch.exp(-torch.tensor(mean_log_prob)).item()

                        agg_scores = {
                            'mean_log_prob': mean_log_prob,
                            'total_log_prob': total_log_prob,
                            'perplexity': perplexity,
                            'token_count': len(valid_log_probs)
                        }

                        # Get token details if requested
                        if return_token_details:
                            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[j])

                            # Iterate through valid positions
                            valid_positions = torch.where(valid_mask)[0]
                            for pos_idx, pos in enumerate(valid_positions):
                                # pos in shifted_mask corresponds to token at position pos+1
                                token_pos = pos.item() + 1
                                token = tokens[token_pos]
                                log_prob = valid_log_probs[pos_idx].item()
                                prob = torch.exp(torch.tensor(log_prob)).item()

                                token_details.append({
                                    'token': token,
                                    'log_prob': log_prob,
                                    'probability': prob,
                                })

                    result = {
                        #'text': text,
                        'log_probs': valid_log_probs,
                        **agg_scores
                    }

                    if return_token_details:
                        result['token_details'] = token_details

                    results.append(result)

        logger.info(f"Finished scoring texts...")

        return results

    def generate(
        self, prompts, prompt_prefix=None,
        theme_path=None, style_path=None, target_layers=None,
        skip_prob=0.0, accumulated_prob=False
    ):
        """Execute evaluation process"""
        utils.set_global_seed(self.args.seed)

        if isinstance(prompts, str):
            prompts_file = prompts
            with open(prompts_file) as f:
                prompts = [line.strip() for line in f.readlines()]

        if prompt_prefix is not None:
            prompts = [prompt_prefix + p for p in prompts]

        self.load_model_and_tokenizer()

        if theme_path is not None:
            self.model = peft.PeftModel.from_pretrained(self.model, theme_path)
        elif style_path is not None:
            self.model = peft.PeftModel.from_pretrained(self.model, style_path)
        else:
            logger.info("Using default model without style or theme adapters")

        # Set up skip connection-s
        # for name, module in self.model.named_modules():
        #     if isinstance(module, peft.tuners.lora.layer.LoraLayer):
        #         try:
        #             module._module_name = name
        #             module._skip_prob = skip_prob
        #             module._random_state = self.args.seed
        #             module._start_skipping_from = self.args.n_layers
        #             module._accumulate_prob = accumulated_prob
        #             module._consecutive_skips = self.args.consecutive_skips
        #             # For debug; should be set to bool('v_proj' in name) to avoid reporting 4 or
        #             # more times for different modules in the same layer
        #             module._report_skip = False #('down_proj' in name and '.5.' in name)
        #         except Exception as e:
        #             logger.error(f"Failed with exception: {e}")

        for name, param in self.model.named_parameters():
            if (
                target_layers is not None and "lora_" in name and
                any([name.startswith(f"base_model.model.model.layers.{i}.") for i in target_layers])
            ):
                adapter_file = os.path.join(style_path, "adapter_model.safetensors")
                with safe_open(adapter_file, framework="pt", device="cuda") as f:
                    adapter = f.get_tensor(name.replace(".default", ""))  # handles naming convention
                    with torch.no_grad():
                        if 'halfsum' not in self.args.run_name:
                            param.copy_(adapter)
                        else:
                            param.add_(adapter).mul_(0.5)

        self.model.eval()

        # Setup text generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        output = generator(
            prompts,
            num_return_sequences=1,
            min_new_tokens=48,
            max_new_tokens=144,
            #do_sample=True,  # set to True by default, not sure why...
            repetition_penalty=1.2,
            #logits_processor=[NoDigitsOrColonsProcessor(self.tokenizer)]
        )

        generated_texts = [
            (
                t[0]["generated_text"][len(prompt_prefix):]
                if prompt_prefix is not None
                else t[0]["generated_text"]
            ).strip().replace("\n", " ")
            for t in output
        ]

        return generated_texts

    def klora_generate(self, prompts, theme_path, style_path, K):
        utils.set_global_seed(self.args.seed)

        if isinstance(prompts, str):
            prompts_file = prompts
            with open(prompts_file) as f:
                prompts = [line.strip() for line in f.readlines()]

        numer = denom = 0.0

        # Llama2 specific
        num_layers = 32
        modules = [
            'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj',
            'mlp.down_proj', 'mlp.up_proj', 'mlp.gate_proj',
        ]

        num_modules = len(modules)

        style_score = [0] * (num_modules * num_layers)
        theme_score = [0] * (num_modules * num_layers)

        #style_layer_score = [0] * num_layers
        #theme_layer_score = [0] * num_layers

        account_qk_projs = True
        account_mlp_projs = True
        for i in range(num_layers):
            for j, x in enumerate(modules):
                adapter_name = f"base_model.model.model.layers.{i}.{x}"
                k = i * num_modules + j

                if not account_qk_projs and ('q_proj' in x or 'k_proj' in x):
                    continue
                if not account_mlp_projs and 'mlp' in x:
                    continue

                adapter_file = os.path.join(style_path, "adapter_model.safetensors")
                with safe_open(adapter_file, framework="pt", device="cuda") as f:
                    A = f.get_tensor(f"{adapter_name}.lora_A.weight")
                    B = f.get_tensor(f"{adapter_name}.lora_B.weight")
                    W_abs = torch.abs(torch.matmul(B, A))
                    elems, _ = torch.sort(W_abs.flatten(), descending=True)
                    style_score[k] = torch.sum(elems[:K]).item()
                    #style_layer_score[i] += style_score[k]
                    denom += torch.sum(elems).item()

                adapter_file = os.path.join(theme_path, "adapter_model.safetensors")
                with safe_open(adapter_file, framework="pt", device="cuda") as f:
                    A = f.get_tensor(f"{adapter_name}.lora_A.weight")
                    B = f.get_tensor(f"{adapter_name}.lora_B.weight")
                    W_abs = torch.abs(torch.matmul(B, A))
                    elems, _ = torch.sort(W_abs.flatten(), descending=True)
                    theme_score[k] = torch.sum(elems[:K]).item()
                    #theme_layer_score[i] += theme_score[k]
                    numer += torch.sum(elems).item()

        gamma = numer / denom
        if self.args.verbose:
            print(f"{gamma = }")

        self.load_model_and_tokenizer()
        #self.setup_peft()
        self.model = peft.PeftModel.from_pretrained(self.model, theme_path)

        target = "base_model.model.model.layers."
        for name, param in self.model.named_parameters():
            if name.startswith(target) and "lora_" in name:
                layer, block, proj = name.removeprefix(target).split(".")[:3]
                i = int(layer)
                j = modules.index(block + "." + proj)
                k = i * num_modules + j

                C_s = gamma * style_score[k]
                C_t = theme_score[k]

                #if gamma * style_layer_score[i] > theme_layer_score[i]
                #path, label = (style_path, 'style') if C_s > C_t else (theme_path, 'theme')

                label = 'theme'
                if C_s > C_t:
                    label = 'style'
                    adapter_file = os.path.join(style_path, "adapter_model.safetensors")
                    with safe_open(adapter_file, framework="pt", device="cuda") as f:
                        adapter = f.get_tensor(name.replace(".default", ""))
                        with torch.no_grad():
                            param.copy_(adapter)

                #if "lora_A" in name and self.args.verbose:
                #    print(f"{name} -> {label}: {C_s:.6f} vs {C_t:.6f}")

        self.model.eval()  # do not know if this is useful

        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        output = generator(
            prompts,
            num_return_sequences=1,
            min_new_tokens=48,
            max_new_tokens=144,
            repetition_penalty=1.2,
        )

        generated_texts = [
            t[0]["generated_text"].strip().replace("\n", " ")
            for t in output
        ]

        return generated_texts


def main(args):
    """Main entry point"""
    finetuner = Finetuner(args)

    if args.do_predict:
        style = args.run_name.split('_')[0]
        print(f"Generating for {args.run_name}")

        default_gen = ('klora' not in args.run_name)  # convinient switch for myself
        if default_gen:
            # desciption = {'aggr': "aggressive/intense", 'sad': "sad/melancholic", 'cheerful': "cheerful/lighthearted", 'agitational': "agitational/promotional", 'reflective': "reflective/insightful"}
            generated_text = finetuner.generate(
                prompts=f"./src/fine_tuning/style/data/input.txt",
                #prompts=f"./src/fine_tuning/style/data/{style}_prompts.txt",
                #prompt_prefix=f"Continue this {desciption[style]} text about astrophysics/cosmology:\n\n",
                #prompt_prefix=f"Continue this text about astrophysics/cosmology:\n\n",
                #prompt_prefix=f"Continue this {desciption[style]} text:\n\n",
                #prompt_prefix=f"[{style.upper()}]",
                #theme_path=f"./src/fine_tuning/style/results_raw/Llama_cosmology/seed_72236/all_layers",
                style_path=f"./src/fine_tuning/style/results_raw/Llama_{style}_augmented/seed_8288/kl_reg_1e-1",
                #theme_path=f"./src/fine_tuning/style/results_raw/Qwen_cosmology/epochs_{epochs}",
                #style_path=f"./src/fine_tuning/style/results_raw/Qwen_{style}_augmented/layers_0_7",
                #target_layers=list(range(0, args.n_layers)),
            )
        else:
            generated_text = finetuner.klora_generate(
                prompts=f"./src/fine_tuning/style/data/{style}_prompts.txt",
                theme_path="./src/fine_tuning/style/results_raw/Llama_cosmology/seed_72236/all_layers",
                style_path=f"./src/fine_tuning/style/results_raw/Llama_{style}/seed_72236/all_layers",
                K=(args.lora_r ** 2),
            )

        print_results = False  # another convinient switch
        if print_results:
            for sample in generated_text:
                print(sample, end='\n')  # add 2 newlines to better see where generation ends
        else:
            with open(f"./src/fine_tuning/style/output/{args.run_name}.txt", 'w') as f:
                for sample in generated_text:
                    f.write(sample + '\n')
    else:
        # use save_adapters="manual" to manually save only adapters (no optimizer, tokenizer and other stuff)
        # use save_adapters="default" to save using hf trainer at the very last step
        # finetuner.run(target_layers=list(range(0, args.n_layers)), save_adapters="manual")
        finetuner.run_orthog_adapters(save_adapters="no")


def score(texts, args, finetuner):
    STYLES, N = ["sad", "cheerful", "agitational", "reflective"], 4
    style_scores = []

    if isinstance(texts, str):
        texts_file = texts
        with open(texts_file) as f:
            texts = [line.strip() for line in f.readlines()]

    finetuner.load_model_and_tokenizer()
    for style in STYLES:
        style_scores.append(finetuner.score(
            texts,
            f"./src/fine_tuning/style/results_raw/Llama_splitted_{style}"
            f"/seed_8288/layers_0_{args.n_layers}_with_mlp"
        ))

    avg_var = 0.0
    for i in range(len(texts)):
        preds = style_scores[0][i]['log_probs']
        #print(preds)
        mean = preds
        mean_sqr_norm = torch.norm(preds).item() ** 2
        for j in range(1, N):
            preds = style_scores[j][i]['log_probs']
            #print(preds)
            mean += preds
            mean_sqr_norm += torch.norm(preds).item() ** 2
        mean /= N
        mean_sqr_norm /= N
        var = mean_sqr_norm - (torch.norm(mean).item() ** 2)
        avg_var += var
    avg_var /= len(texts)

    #print(f"Averaged variance: {avg_var}")
    return avg_var


if __name__ == "__main__":
    main(None)

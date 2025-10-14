import torch
import datasets
import wandb

from loguru import logger
import peft

import utils
#from utils_style import DatasetRegistry

import warnings
import os
from safetensors.torch import save_file

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
        #self.builder = None

        self.setup_logging()
        if self.args.wandb:
            self.setup_wandb()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info(f"Starting downstream finetuning for dataset: {self.args.dataset}")

    def setup_wandb(self):
        """Setup Weights & Biases logging"""
        style = self.args.dataset_path.split('/')[-1]
        run_name = f"[{style} | seed={self.args.seed}]" # lr={self.args.lr:.1e} | seed={self.args.seed}
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
            self.args.model,
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
        peft_args.task_type = "CASUAL_LM"
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
        dataset = dataset.map(
            lambda batch: self.tokenizer(batch["text"], max_length=max_len, truncation=True),
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

    def get_optimizer(self):
        """Get optimizer for training"""
        optimizer = get_optimizer(self.args, self.model)
        return optimizer

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
        
        #self.prepare_evaluation_dataset()
        #if self.eval_dataset is None:
        #    logger.error("No evaluation data available")
        #    return

        if self.args.save_strategy != "no" and self.args.save_name is None:
            raise ValueError(f"Provide --save_name argument to use save_strategy={self.args.save_strategy}.")

        # Setup trainer
        output_dir = (
            f"./src/fine_tuning/style/{self.args.results_path}/"
            f"{self.args.save_name if self.args.save_name is not None else '_'}"
        )
        training_args = TrainingArguments(
            do_train=not self.args.do_not_train,
            do_eval=not self.args.do_not_eval,
            do_predict=self.args.do_predict,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=(
                self.args.eval_batch_size
                if self.args.eval_batch_size
                else self.args.batch_size
            ),
            gradient_accumulation_steps=self.args.grad_acc_steps,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            warmup_ratio=self.args.warmup_ratio, #added
            weight_decay=self.args.weight_decay, #added
            max_grad_norm=self.args.max_grad_norm, #added
            learning_rate=self.args.lr,
            num_train_epochs=self.args.n_epoches_train,
            max_steps=self.args.max_steps_train,
            logging_steps=self.args.logging_steps,
            eval_strategy=self.args.eval_strategy,
            eval_steps=self.args.eval_steps,
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            bf16=(self.args.dtype == "bfloat16"),
            fp16=(self.args.dtype == "float16"),
            output_dir=output_dir,
            overwrite_output_dir=True, #added
            #logging_dir=output_dir,
            run_name=self.args.run_name,
            report_to=["wandb" if self.args.wandb else "none"],
            #eval_on_start=True,
        )

        optimizer = self.get_optimizer()

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

    ##TODO: rewrite completely OR remove
    def evaluate(self):
        """Execute evaluation process"""
        if self.args.do_not_eval:
            logger.info("Evaluation skipped (do_not_eval=True)")
            return 0, 0

        logger.info(f"Starting evaluation for dataset: {self.args.dataset}")

        # Prepare evaluation dataset
        eval_dataset = self.prepare_evaluation_dataset()
        if eval_dataset is None:
            logger.error("No evaluation data available")
            return 0, 0

        # Setup text generation pipeline
        generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
        )

        # Evaluate model
        correct, total = 0, 0

        logger.info(f"Evaluation sample prompt: {eval_dataset['text'][0]}")

        for i, item in enumerate(eval_dataset):
            try:
                # Generate prediction
                predicted_response = generator(
                    item["text"],
                    max_new_tokens=len(item["raw_y"]) + 2,
                    num_return_sequences=1,
                )[0]["generated_text"]
                predicted_response = predicted_response.replace(" ", "").replace(
                    "\n", ""
                )
                item["raw_y"] = item["raw_y"].replace(" ", "").replace("\n", "")
                # Check if correct answer is in prediction
                if item["raw_y"] in predicted_response:
                    correct += 1

                # Debug output
                print(f">>Prediction<<: {predicted_response}")
                print(f">>>>Answer<<<<: {item['raw_y']}")

                total += 1

                # Progress update
                accuracy = (correct / total) * 100 if total > 0 else 0
                print(f"[{i + 1}/{len(eval_dataset)}] Accuracy: {accuracy:.2f}%")
                print("=" * 50)

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        return correct, total

    ##TODO: probably remove this function OR rewrite it
    def log_final_results(self, correct, total):
        """Log final evaluation results"""
        if total > 0:
            final_accuracy = (correct / total) * 100
            logger.info(f"[FINAL] Accuracy: {final_accuracy:.2f}%")

            if self.args.wandb:
                wandb.log({"final_accuracy": final_accuracy})
        else:
            logger.info("No samples were successfully evaluated.")

    def run_from_pretrained(self):
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load base model
        self.load_model_and_tokenizer()

        logger.info(f"Setting up PEFT adapters...")

        peft_args = utils.get_peft_arguments(self.args)
        peft_args.task_type = "CASUAL_LM"

        # Load model to continue training
        self.model = peft.PeftModel.from_pretrained(
            self.model,
            f"./src/fine_tuning/style/{self.args.results_path}/"
            f"Qwen_neutral/seed_{self.args.seed}/checkpoint-200",
            is_trainable=True
        )

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()

    def run_single_layer(self):
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        for i in range(32):
            self.load_model_and_tokenizer()

            logger.info(f"Setting up PEFT adapters for layer {i}...")
            
            # Load model to continue training
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                f"./src/fine_tuning/style/{self.args.results_path}/neutral_{self.args.seed}/checkpoint-100",
                is_trainable=True
            )

            # Assuming that layers are named as layer.{layer_no}.{module}
            for name, param in self.model.named_parameters():
                if f".{i}." not in name:
                    param.requires_grad = False

            self.load_datasets()

            self.train()

            style = self.args.dataset_path.split('/')[-1]
            output_dir = f"./src/fine_tuning/style/{self.args.results_path}/{style}_sl_{self.args.seed}/layer_{i}"
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
    
    def run(self):
        """Main execution flow"""
        logger.info("Starting finetuning pipeline")

        utils.set_global_seed(self.args.seed)

        # Load model and setup PEFT
        self.load_model_and_tokenizer()
        self.setup_peft()

        ## See what's up
        #for name, param in self.model.named_parameters():
        #    if "lora" in name.lower():
        #        print(f"{name}: {param.requires_grad}")

        # Load datasets
        self.load_datasets()

        # Execute training and evaluation
        self.train()

        # correct, total = self.evaluate()
        # self.log_final_results(correct, total)
        # logger.info("Pipeline completed successfully")


def main(args):
    """Main entry point"""

    finetuner = Finetuner(args)
    finetuner.run_from_pretrained()
    #finetuner.run()


if __name__ == "__main__":
    main(None)

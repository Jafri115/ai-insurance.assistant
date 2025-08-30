"""
Fine-tuning script using QLoRA for efficient training of large language models.
Supports LLaMA, Mistral, Gemma, and other transformer models.
"""

import os
import json
import torch
import argparse
import logging
import platform
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code when loading model"}
    )

@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_path: str = field(
        default="data/processed/train.json",
        metadata={"help": "Path to training dataset"}
    )
    validation_path: str = field(
        default="data/processed/validation.json",
        metadata={"help": "Path to validation dataset"}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"}
    )

@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="auto",
        metadata={"help": "Comma-separated list of target modules or 'auto' to detect"}
    )

class DataCollatorForCompletionOnlyLM:
    """Pads input_ids/attention_mask and labels, where labels are -100 for prompt tokens."""
    def __init__(self, tokenizer: AutoTokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]):
        # Separate labels from features for custom padding
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for l in labels:
            if len(l) < max_len:
                l = l + [self.label_pad_token_id] * (max_len - len(l))
            else:
                l = l[:max_len]
            padded_labels.append(torch.tensor(l, dtype=torch.long))
        batch["labels"] = torch.stack(padded_labels, dim=0)
        return batch

class InsuranceTrainer:
    """Fine-tuning trainer for insurance assistant models."""
    
    def __init__(self, model_args: ModelArguments, data_args: DataArguments, 
                 lora_args: LoRAArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.lora_args = lora_args
        self.training_args = training_args
        
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with optional QLoRA (4-bit) when available."""
        logger.info(f"Loading model: {self.model_args.model_name_or_path}")

        # Detect environment capabilities
        is_windows = platform.system().lower() == "windows"
        has_cuda = torch.cuda.is_available()
        try:
            import bitsandbytes as bnb  # noqa: F401
            bnb_available = True
        except Exception:
            bnb_available = False

        use_4bit = has_cuda and bnb_available
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit quantization with bitsandbytes")
        else:
            bnb_config = None
            reason = []
            if is_windows: reason.append("Windows")
            if not has_cuda: reason.append("no CUDA GPU")
            if not bnb_available: reason.append("bitsandbytes not installed")
            logger.info(f"Proceeding without 4-bit quantization ({', '.join(reason)})")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=self.model_args.trust_remote_code,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine device map and dtype
        dtype = torch.float16 if has_cuda else torch.float32
        device_map = "auto" if has_cuda else "cpu"

        # Load model (conditionally pass quantization config)
        model_kwargs = dict(
            trust_remote_code=self.model_args.trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
            use_safetensors=True,
        )
        # On CPU avoid accelerate offloading to disk; use low_cpu_mem_usage to reduce peak RAM
        if not has_cuda:
            model_kwargs.update({
                "low_cpu_mem_usage": True,
            })
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            **model_kwargs,
        )
        # For training memory
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

        # Prepare for k-bit training only when using bnb
        if bnb_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA
        # Auto-detect target modules when requested
        target_modules_arg = (self.lora_args.lora_target_modules or "auto").strip().lower()
        if target_modules_arg == "auto":
            module_names = [name for name, _ in self.model.named_modules()]
            def any_in(sub):
                return any(sub in n for n in module_names)
            if any_in("q_proj") or any_in("k_proj") or any_in("v_proj"):
                target_modules = [m for m in ["q_proj", "k_proj", "v_proj", "o_proj"] if any_in(m)]
            elif any_in("c_attn") or any_in("c_proj"):
                target_modules = [m for m in ["c_attn", "c_proj"] if any_in(m)]
            elif any_in("W_pack"):
                target_modules = ["W_pack"]
            else:
                # Fallback: common GPT-2 style
                target_modules = ["c_attn", "c_proj"]
            logger.info(f"Auto-detected LoRA target modules: {target_modules}")
        else:
            target_modules = [m.strip() for m in self.lora_args.lora_target_modules.split(",") if m.strip()]

        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=self.lora_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self):
        """Load and preprocess training datasets."""
        logger.info("Loading datasets...")
        
        def safe_str(v):
            if v is None:
                return ""
            try:
                # Handle NaN from floats
                if isinstance(v, float) and (v != v):  # NaN check
                    return ""
            except Exception:
                pass
            if isinstance(v, (dict, list)):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return str(v)
        
        def load_json_dataset(file_path: str) -> Dataset:
            """Load dataset from JSON file and sanitize types."""
            if not os.path.exists(file_path):
                logger.warning(f"Dataset file {file_path} not found")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            
            if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
                data = raw["data"]
            elif isinstance(raw, list):
                data = raw
            else:
                logger.warning("Unsupported JSON structure; expected list or dict with 'data' list")
                return None

            sanitized = []
            dropped = 0
            for item in data:
                if not isinstance(item, dict):
                    dropped += 1
                    continue
                instr = safe_str(item.get("instruction", item.get("prompt", "")).strip())
                inp = safe_str(item.get("input", item.get("question", item.get("context", "")).strip()))
                out = safe_str(item.get("output", item.get("response", item.get("answer", "")).strip()))
                if not out:
                    dropped += 1
                    continue
                sanitized.append({
                    "instruction": instr if instr else "Answer the insurance-related question accurately and helpfully.",
                    "input": inp,
                    "output": out,
                })
            if dropped:
                logger.info(f"Sanitized dataset: kept {len(sanitized)} records, dropped {dropped} invalid")
            
            return Dataset.from_list(sanitized)
        
        # Load training data
        self.train_dataset = load_json_dataset(self.data_args.dataset_path)
        if self.train_dataset is None:
            # Create sample data if no dataset exists
            logger.info("Creating sample dataset...")
            self.train_dataset = self._create_sample_dataset()
        
        # Load validation data
        self.eval_dataset = load_json_dataset(self.data_args.validation_path)
        if self.eval_dataset is None and self.train_dataset is not None:
            # Split training data for validation
            split_dataset = self.train_dataset.train_test_split(test_size=0.1)
            self.train_dataset = split_dataset['train']
            self.eval_dataset = split_dataset['test']
        
        # Preprocess datasets
        self.train_dataset = self.train_dataset.map(
            self._preprocess_function, 
            batched=False,
            remove_columns=self.train_dataset.column_names
        )
        
        if self.eval_dataset:
            self.eval_dataset = self.eval_dataset.map(
                self._preprocess_function,
                batched=False,
                remove_columns=self.eval_dataset.column_names
            )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Validation samples: {len(self.eval_dataset)}")
    
    def _create_sample_dataset(self) -> Dataset:
        """Create a sample dataset for demonstration."""
        sample_data = [
            {
                "instruction": "Answer the following insurance-related question accurately and helpfully.",
                "input": "What is comprehensive auto insurance coverage?",
                "output": "Comprehensive auto insurance coverage protects your vehicle against damages not caused by collisions, such as theft, vandalism, natural disasters, falling objects, and animal strikes. It typically covers repairs or replacement of your vehicle minus your deductible."
            },
            {
                "instruction": "Answer the following insurance-related question accurately and helpfully.",
                "input": "How do I choose the right life insurance policy?",
                "output": "When choosing life insurance, consider: 1) Your financial obligations and dependents, 2) Term vs. permanent coverage needs, 3) Coverage amount (typically 10-12x annual income), 4) Your budget for premiums, 5) The insurance company's financial strength rating, and 6) Additional riders you might need."
            },
            {
                "instruction": "Answer the following insurance-related question accurately and helpfully.",
                "input": "What should I do immediately after a car accident?",
                "output": "After a car accident: 1) Ensure safety and call 911 if needed, 2) Move to a safe location if possible, 3) Exchange information with other drivers, 4) Take photos of damages and the scene, 5) Contact your insurance company, 6) Get a police report number, and 7) Seek medical attention if injured."
            }
        ]
        
        return Dataset.from_list(sample_data)
    
    def _format_prompt(self, instruction: str, input_text: str, output_text: str) -> str:
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

    def _preprocess_function(self, example):
        """Preprocess a single example and create labels masked before the response.
        Ensures the response tokens are kept by truncating the prefix from the left if needed.
        """
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = example["output"]
        marker = "### Response:\n"

        # Build prefix up to (and including) the marker
        if input_text.strip():
            prefix = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n{marker}"
        else:
            prefix = f"### Instruction:\n{instruction}\n\n{marker}"

        # Tokenize separately
        prefix_ids = self.tokenizer(
            prefix,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        # Ensure EOS at end of output to help stopping
        eos = self.tokenizer.eos_token or ""
        output_with_eos = output_text + ("" if output_text.endswith(eos) else eos)
        output_ids = self.tokenizer(
            output_with_eos,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        # Guarantee at least 1 token of output
        if len(output_ids) == 0:
            output_ids = [self.tokenizer.eos_token_id]

        # Truncate prefix from the left so the whole output fits into max_length
        max_len = self.data_args.max_length
        max_prefix_len = max_len - len(output_ids)
        if max_prefix_len < 0:
            # Output alone exceeds max length, keep the last max_len tokens of output
            output_ids = output_ids[-max_len:]
            prefix_ids = []
        else:
            if len(prefix_ids) > max_prefix_len:
                # Keep only the last max_prefix_len tokens of the prefix
                prefix_ids = prefix_ids[-max_prefix_len:]

        input_ids = prefix_ids + output_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prefix_ids) + output_ids.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def train(self):
        """Execute the training process."""
        # Initialize wandb if available
        try:
            wandb.init(
                project="insurance-assistant",
                name=f"qlora-{self.model_args.model_name_or_path.split('/')[-1]}",
                config={
                    "model": self.model_args.model_name_or_path,
                    "lora_r": self.lora_args.lora_r,
                    "lora_alpha": self.lora_args.lora_alpha,
                    "max_length": self.data_args.max_length,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
        
        # Data collator (custom to preserve labels)
        data_collator = DataCollatorForCompletionOnlyLM(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.training_args.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.training_args.output_dir)
        
        # Save training metrics
        if trainer.state.log_history:
            with open(os.path.join(self.training_args.output_dir, "training_log.json"), "w") as f:
                json.dump(trainer.state.log_history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune insurance assistant model")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/processed/train.json",
                       help="Training dataset path")
    parser.add_argument("--validation_path", type=str, default="data/processed/validation.json",
                       help="Validation dataset path")
    parser.add_argument("--output_dir", type=str, default="models/insurance-assistant",
                       help="Output directory")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="auto",
                       help="Comma-separated target modules or 'auto' to detect")
    
    args = parser.parse_args()
    
    # Create argument objects
    model_args = ModelArguments(model_name_or_path=args.model_name)
    data_args = DataArguments(
        dataset_path=args.dataset_path,
        validation_path=args.validation_path,
        max_length=args.max_length
    )
    lora_args = LoRAArguments(
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        optim="paged_adamw_8bit", # Or "adamw_8bit"
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        warmup_ratio=0.1,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=["wandb"] if "wandb" in globals() else [],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Initialize trainer
    trainer = InsuranceTrainer(model_args, data_args, lora_args, training_args)
    
    # Setup and train
    trainer.setup_model_and_tokenizer()
    trainer.load_dataset()
    trainer.train()
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()

"""
Improved Chess LLM Training Script
Addresses training instability, overfitting, and gradient issues from previous training
"""

import os
import torch
import warnings
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset
import json
from typing import Dict, List, Optional
import wandb
from pathlib import Path
from dotenv import load_dotenv
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import hashlib
from collections import defaultdict

OPTIMAL_CONFIG_16GB = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 5e-5,
    "num_train_epochs": 1,
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "lora_dropout": 0.2,
    "lora_r": 32,
    "lora_alpha": 64,
    "warmup_steps": 200,
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
    "tokenizer_max_length": 512,  # Reduced for stability
}
OPTIMAL_CONFIG_128GB = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "num_train_epochs": 1,
    "weight_decay": 0.02,
    "max_grad_norm": 0.3,
    "lora_dropout": 0.2, # Increased dropout for regularization
    "lora_r": 32, # Increased rank for better capacity
    "lora_alpha": 64, # Maintains 2:1 alpha/rank ratio
    "warmup_ratio": 0.1,
    "warmup_steps": 500,
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.001,
    "tokenizer_max_length": 1024,  # Reduced for stability
}
OPTIMAL_CONFIG = OPTIMAL_CONFIG_16GB

class ImprovedChessLLMTrainer:
    def __init__(self,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 train_file: str = "chess_train.jsonl",
                 val_file: str = "chess_val.jsonl",
                 output_dir: str = "./improved-chess-llm",
                 checkpoint_path: Optional[str] = None,
                 use_multi_stage: bool = True,
                 learning_rate: float = OPTIMAL_CONFIG["learning_rate"],
                 ):

        self.model_name = model_name
        self.train_file = train_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.checkpoint_path = checkpoint_path
        self.use_multi_stage = use_multi_stage
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

        # Initialize wandb
        self._init_wandb()

    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(
            project="improved-chess-llm",
            name="stable-training-v2",
            config={
                "model": self.model_name,
                "epochs": OPTIMAL_CONFIG["num_train_epochs"],
                "learning_rate": self.learning_rate,
                "batch_size": OPTIMAL_CONFIG["per_device_train_batch_size"],
                "multi_stage": self.use_multi_stage,
                "improvements": [
                    "gradient_clipping_0.5",
                    "weight_decay_0.01",
                    "early_stopping",
                    "cosine_restart_lr",
                    "lora_dropout_0.2",
                    "data_deduplication",
                    "validation_split_15%"
                ]
            }
        )
        print("✓ Wandb initialized successfully")

    def load_model_and_tokenizer(self):
        """Load model and tokenizer with improved stability settings"""
        print(f"Loading model: {self.model_name}")

        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # Better gradient stability
            #bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="right",
            use_fast=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager"  # More stable than flash attention
        )


        self.model.config.use_cache = False
        # Add this after model initialization
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.enable_input_require_grads()  # Force gradient tracking

        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        print("✓ Model and tokenizer loaded successfully")

    def setup_improved_lora(self):
        """Setup LoRA with improved stability configuration"""
        print("Setting up improved LoRA configuration...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=OPTIMAL_CONFIG["lora_r"],  # Increased rank for better capacity
            lora_alpha=OPTIMAL_CONFIG["lora_alpha"],  # Maintains 2:1 alpha/rank ratio
            lora_dropout=OPTIMAL_CONFIG["lora_dropout"],  # Increased dropout for regularization
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # TinyLlama's attention layers
            # target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
            #               "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_rslora=True  # Rank-stabilized LoRA
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # Add this line below to check the first 10 parameters' requires_grad status
        print([p.requires_grad for p in self.model.parameters()][:10])
        print("✓ Improved LoRA setup complete")

    def create_chess_diversity_filter(self, data: List[Dict]) -> List[Dict]:
        """Filter chess data for diversity to reduce overfitting"""
        print("Applying diversity filtering to chess data...")

        # Group by opening moves to ensure diversity
        opening_groups = defaultdict(list)

        for item in data:
            # Extract first few moves as opening signature
            content = item.get("messages", [{}])[0].get("content", "")

            # Simple opening detection (first 3 moves)
            moves = content.split()[:6]  # Usually 3 moves = 6 half-moves
            opening_sig = " ".join(moves) if len(moves) >= 4 else "short_game"
            opening_groups[opening_sig].append(item)

        # Sample from each opening group to maintain diversity
        filtered_data = []
        max_per_opening = max(1, len(data) // (len(opening_groups) * 3))

        for opening, games in opening_groups.items():
            # Sample up to max_per_opening games from each opening
            sampled = games[:max_per_opening]
            filtered_data.extend(sampled)

        print(f"✓ Filtered {len(data)} -> {len(filtered_data)} games for diversity")
        return filtered_data

    def load_and_process_data(self) -> Dict[str, Dataset]:
        """Load and process data with deduplication and diversity filtering"""
        print("Loading and processing training data...")

        def load_jsonl(file_path: str) -> List[Dict]:
            data = []
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                print(f"Warning: {file_path} not found")
                return []
            return data

        def deduplicate_data(data: List[Dict]) -> List[Dict]:
            """Remove duplicate games based on content hash"""
            seen_hashes = set()
            unique_data = []

            for item in data:
                content = str(item.get("messages", []))
                content_hash = hashlib.md5(content.encode()).hexdigest()

                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_data.append(item)

            print(f"✓ Removed {len(data) - len(unique_data)} duplicates")
            return unique_data

        def format_conversation(example: Dict, tokenizer) -> str:
            # Use the tokenizer's chat template for TinyLlama
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False  # For training, don't add extra prompt
            )

        def tokenize_function(examples):
            texts = [format_conversation(ex, self.tokenizer) for ex in examples["conversations"]]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=OPTIMAL_CONFIG["tokenizer_max_length"],
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            #tokenized["labels"] = tokenized["input_ids"].clone()  # Explicit label cloning
            return tokenized

        # Load data
        train_data = load_jsonl(self.train_file)
        val_data = load_jsonl(self.val_file)

        if not train_data and not val_data:
            raise "No data files found, NOT creating sample data..."
            #train_data = self.create_sample_data()
            ## Split sample data for validation
            #split_idx = int(len(train_data) * 0.85)
            #val_data = train_data[split_idx:]
            #train_data = train_data[:split_idx]

        # Deduplicate and filter
        train_data = deduplicate_data(train_data)
        val_data = deduplicate_data(val_data)

        # Apply diversity filtering
        train_data = self.create_chess_diversity_filter(train_data)

        print(f"✓ Final dataset: {len(train_data)} train, {len(val_data)} validation")

        # Create datasets
        train_dataset = Dataset.from_dict({"conversations": train_data})
        val_dataset = Dataset.from_dict({"conversations": val_data})

        # Tokenize
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["conversations"],
            desc="Tokenizing training data"
        )

        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["conversations"],
            desc="Tokenizing validation data"
        )

        return {"train": train_dataset, "validation": val_dataset}

    def create_sample_data(self) -> List[Dict]:
        """Create diverse sample chess training data"""
        sample_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "In this position after 1.e4 e5 2.Nf3 Nc6 3.Bb5, what should Black play and why?"},
                    {"role": "assistant", "content": "This is the Ruy Lopez opening. Black has several strong options: 3...a6 (Morphy Defense) puts immediate pressure on the bishop, 3...Nf6 (Berlin Defense) leads to solid but drawish positions, or 3...f5 (Schliemann Defense) for sharp tactical play. The most popular choice is 3...a6."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What's the key idea behind the Sicilian Defense after 1.e4 c5?"},
                    {"role": "assistant", "content": "The Sicilian Defense aims to create an unbalanced position where Black fights for the initiative. By playing c5, Black controls the d4 square and prepares to develop pieces actively. Common continuations include the Najdorf (5...a6), Dragon (5...g6), or Accelerated Dragon variations."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I improve my endgame technique in King and Pawn endings?"},
                    {"role": "assistant", "content": "Key principles for King and Pawn endgames: 1) Activate your king early, 2) Create passed pawns, 3) Use opposition to restrict the enemy king, 4) Calculate pawn races carefully, 5) Know basic positions like the square rule and triangulation. Practice these fundamental concepts regularly."}
                ]
            }
        ]

        # Create more diverse training samples
        data = []
        for i in range(2000):  # Reduced from 1000 for quality over quantity
            base_conversation = sample_conversations[i % len(sample_conversations)]
            # Add slight variations to prevent exact memorization
            data.append(base_conversation)

        return data

    def create_custom_trainer(self, datasets):
        """Create trainer with improved stability settings"""

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # Training arguments with stability improvements
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=OPTIMAL_CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=OPTIMAL_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=OPTIMAL_CONFIG["gradient_accumulation_steps"],  # Larger effective batch size
            num_train_epochs=OPTIMAL_CONFIG["num_train_epochs"],
            learning_rate=self.learning_rate,
            warmup_steps=OPTIMAL_CONFIG["warmup_steps"],  # Longer warmup
            weight_decay=OPTIMAL_CONFIG["weight_decay"],  # Regularization
            max_grad_norm=OPTIMAL_CONFIG["max_grad_norm"],  # Aggressive gradient clipping
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=50,  # Frequent evaluation
            save_steps=200,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=False,  # Disabled for stability
            bf16=False,
            dataloader_num_workers=2,
            report_to="wandb",
            run_name="improved-chess-llm-stable",
            # gradient_checkpointing=True,
            gradient_checkpointing=False,
            # Disable torch compile for stability
            torch_compile=False,
            label_names=["labels"],  # Explicitly set label names
        )

        # Custom trainer class for learning rate scheduling
        class StableTrainer(Trainer):
            def create_optimizer_and_scheduler(self, num_training_steps):
                # Create optimizer
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.args.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=self.args.weight_decay,
                )

                # Create cosine annealing with warm restarts
                self.lr_scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=500,  # First restart after 500 steps
                    T_mult=2,  # Double period after each restart
                    eta_min=1e-7  # Minimum learning rate
                )

                return self.optimizer, self.lr_scheduler

        # Create trainer with early stopping
        trainer = StableTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            #tokenizer=self.tokenizer, # Deprecated
            processing_class=self.tokenizer,  # RECOMMENDED
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=OPTIMAL_CONFIG["early_stopping_patience"],
                    early_stopping_threshold=OPTIMAL_CONFIG["early_stopping_threshold"]
                )
            ]
        )

        return trainer

    def train_from_checkpoint_if_available(self):
        """Main training function with checkpoint resumption"""
        print("Starting improved chess LLM training...")

        # Load datasets
        datasets = self.load_and_process_data()

        # Create trainer
        trainer = self.create_custom_trainer(datasets)

        # Train from checkpoint if available
        resume_from_checkpoint = None
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            resume_from_checkpoint = self.checkpoint_path
            print(f"✓ Resuming training from {self.checkpoint_path}")

        # Train the model
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"✓ Training complete! Model saved to {self.output_dir}")

    def save_model_for_inference(self):
        """Save model in inference-ready format"""
        print("Preparing model for inference...")

        # Save adapter
        self.model.save_pretrained(self.output_dir)

        # Save configuration
        config = {
            "base_model": self.model_name,
            "adapter_path": self.output_dir,
            "task_type": "chess_llm",
            "training_completed": True,
            "improvements_applied": [
                "gradient_clipping",
                "early_stopping",
                "cosine_restart_lr",
                "data_deduplication",
                "diversity_filtering",
                "weight_decay_regularization"
            ]
        }

        with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print("✓ Model ready for inference!")


def main():
    """Main training function with multi-stage option"""
    print("🚀 Starting Improved Chess LLM Training Pipeline...")

    # Stage 1: Initial training with conservative settings
    trainer = ImprovedChessLLMTrainer(
        # num_train_epochs=1,  # Conservative first stage
        learning_rate=5e-5,
        # batch_size=2
    )

    trainer.load_model_and_tokenizer()
    trainer.setup_improved_lora()
    trainer.train_from_checkpoint_if_available()
    trainer.save_model_for_inference()

    # Stage 2: Continue training with slight adjustments (multi-stage)
    print("🔄 Starting Stage 2: Fine-tuned continuation...")

    stage2_trainer = ImprovedChessLLMTrainer(
        checkpoint_path="../improved-chess-llm",
        # num_train_epochs=1,  # Additional epoch
        learning_rate=2e-5,  # Lower learning rate
        # batch_size=2,
        output_dir="../improved-chess-llm-stage2"
    )

    stage2_trainer.load_model_and_tokenizer()
    stage2_trainer.setup_improved_lora()
    stage2_trainer.train_from_checkpoint_if_available()
    stage2_trainer.save_model_for_inference()

    print("✅ Multi-stage training pipeline complete!")


if __name__ == "__main__":
    main()
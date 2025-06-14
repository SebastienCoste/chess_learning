#!/usr/bin/env python3
"""
Chess LLM Training Script
Fine-tunes Mistral-7B on chess PGN data using LoRA for local deployment.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
import json
from typing import Dict, List
import wandb
from pathlib import Path


class ChessLLMTrainer:
    def __init__(self,
                 model_name: str  = "mistralai/Mistral-7B-Instruct-v0.3",
                 train_file: str  = "chess_train.jsonl",
                 val_file: str    = "chess_val.jsonl",
                 output_dir: str  = "./chess-mistral-7b-lora",
                 num_train_epochs = 3,
                 learning_rate    = 2e-4,
                 batch_size       = 2,
                 ):

        self.model_name = model_name
        self.train_file = train_file
        self.val_file = val_file
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        wandb.init(
            project="chess-llm",
            name="mistral-7b-chess-lora",
            config={
                "model": self.model_name,
                "epochs": self.num_train_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
            }
        )
        print("Wand initialized successfully")

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with quantization"""
        print(f"Loading model: {self.model_name}")

        # remove it if no nVidia GPU
        # Configure quantization for memory efficiency
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4"
        # )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="right",
            use_fast=False
        )

        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # quantization_config=quantization_config,
            # device_map="auto",
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

        print("Model and tokenizer loaded successfully")

    def setup_lora(self):
        """Setup LoRA configuration for parameter-efficient fine-tuning"""
        print("Setting up LoRA...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        print("LoRA setup complete")

    def load_and_process_data(self) -> Dict[str, Dataset]:
        """Load and process the training data"""
        print("Loading training data...")

        def load_jsonl(file_path: str) -> List[Dict]:
            """Load data from JSONL file"""
            data = []
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                print(f"Warning: {file_path} not found. Creating sample data...")
                # Create sample data if file doesn't exist
                data = self.create_sample_data()
            return data

        def format_conversation(example: Dict) -> str:
            """Format conversation for training"""
            conversation = ""
            for message in example["messages"]:
                if message["role"] == "user":
                    conversation += f"<s>[INST] {message['content']} [/INST] "
                elif message["role"] == "assistant":
                    conversation += f"{message['content']}</s>"
            return conversation

        def tokenize_function(examples):
            """Tokenize the conversations"""
            texts = [format_conversation(ex) for ex in examples["conversations"]]

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",  # Add proper padding
                #padding=False,
                max_length=512,
                return_tensors=None
            )

            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Load training and validation data
        train_data = load_jsonl(self.train_file)
        val_data = load_jsonl(self.val_file)

        print(f"Loaded {len(train_data)} training examples")
        print(f"Loaded {len(val_data)} validation examples")

        # Create datasets
        train_dataset = Dataset.from_dict({"conversations": train_data})
        val_dataset = Dataset.from_dict({"conversations": val_data})

        # Tokenize datasets
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
        """Create sample training data if files don't exist"""
        print("Creating sample training data...")

        sample_conversations = [
            {
                "messages": [
                    {"role": "user", "content": "What's the best opening move in chess?"},
                    {"role": "assistant",
                     "content": "The most popular opening moves are 1.e4 and 1.d4. 1.e4 (King's pawn) leads to more tactical games, while 1.d4 (Queen's pawn) often leads to positional games."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "In this position: 1.e4 e5 2.Nf3 Nc6 3.Bb5 - what should Black play?"},
                    {"role": "assistant",
                     "content": "This is the Ruy Lopez opening. Black has several good options: 3...a6 (Morphy Defense), 3...Nf6 (Berlin Defense), or 3...f5 (Schliemann Defense). The most popular is 3...a6."}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "How do I checkmate with Queen and King vs King?"},
                    {"role": "assistant",
                     "content": "To checkmate with Queen and King: 1) Use your King to help drive the enemy King to the edge, 2) Use the Queen to cut off escape squares, 3) Force the King to the corner or edge, 4) Deliver checkmate with the Queen while your King provides support."}
                ]
            }
        ]

        # Replicate to create more training data
        data = []
        for i in range(1000):
            data.append(sample_conversations[i % len(sample_conversations)])

        return data

    def train(self):
        """Train the model"""
        print("Starting training...")

        # Load datasets
        datasets = self.load_and_process_data()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size= self.batch_size,  # Reduced from 4 for CPU training
            per_device_eval_batch_size= self.batch_size,  # Reduced from 4 for CPU training
            gradient_accumulation_steps=8,  # Increased from 4 to compensate for smaller batch size
            num_train_epochs= self.num_train_epochs,
            learning_rate= self.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            fp16=False,                     # Disabled fp16 for CPU training
            report_to="wandb",              # Enable wandb reporting
            run_name="chess-llm-training",  # Name for the wandb run
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train(resume_from_checkpoint=True)

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Training complete! Model saved to {self.output_dir}")

    def save_model_for_inference(self):
        """Save the model in a format ready for inference"""
        print("Saving model for inference...")

        # Save the LoRA adapter
        self.model.save_pretrained(self.output_dir)

        # Save base model name for loading later
        config = {
            "base_model": self.model_name,
            "adapter_path": self.output_dir,
            "task_type": "chess_llm"
        }

        with open(os.path.join(self.output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print("Model ready for inference!")


def main():
    """Main training function"""
    print("Chess LLM Training Starting...")

    # Initialize trainer
    trainer = ChessLLMTrainer()

    # Load model and setup LoRA
    trainer.load_model_and_tokenizer()
    trainer.setup_lora()

    # Train the model
    trainer.train()

    # Save for inference
    trainer.save_model_for_inference()

    print("Training pipeline complete!")


if __name__ == "__main__":
    main()
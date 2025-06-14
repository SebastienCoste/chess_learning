from llama_cpp import Llama
from transformers import AutoTokenizer


class ChessLLM:
    def __init__(self, model_path):
        # Configure for CPU-only inference
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Context window size
            n_threads=8,  # Use 8 CPU threads
            n_gpu_layers=0,  # Disable GPU layers
            offload_kqv=True,  # Optimize memory usage
            main_gpu=0,  # Not used but required param
            vocab_only=False
        )

        # Load original tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            padding_side="right",
            use_fast=False
        )

        # Handle padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_move(self, prompt):
        # Tokenization and generation remains similar
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=32,
            temperature=0.7,
            top_p=0.9
        )
        return output["choices"][0]["text"]

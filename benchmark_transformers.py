import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from titans_transformer import TitansTransformer
import math
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


# Positional Encoding for Standard Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


# Define Standard Transformer
class StandardTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, num_tokens)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer.encoder(src, src_mask)
        return self.fc_out(output)


# Function to load data
def get_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_data = torch.tensor(tokenized_dataset["train"]["input_ids"], dtype=torch.long)
    val_data = torch.tensor(tokenized_dataset["validation"]["input_ids"], dtype=torch.long)
    test_data = torch.tensor(tokenized_dataset["test"]["input_ids"], dtype=torch.long)

    vocab_size = tokenizer.vocab_size
    return train_data, val_data, test_data, vocab_size


# Benchmarking function
def benchmark_models(sequence_lengths=[128, 256, 512, 1024], batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "standard": {"time": [], "memory": [], "perplexity": []},
        "titans": {"time": [], "memory": [], "perplexity": []},
    }

    train_data, val_data, test_data, vocab_size = get_data()
    criterion = nn.CrossEntropyLoss()

    for seq_len in sequence_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Initialize models
        standard_transformer = StandardTransformer(num_tokens=vocab_size).to(device)
        titans_transformer = TitansTransformer(num_tokens=vocab_size, memory_size=512).to(device)

        # Prepare test batch
        test_batch = test_data[:seq_len].unsqueeze(0).to(device)

        # Test Standard Transformer
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            standard_transformer.eval()
            output_standard = standard_transformer(test_batch)
            perplexity_standard = torch.exp(
                criterion(output_standard.view(-1, output_standard.size(-1)), test_batch.view(-1))
            )
        end_time = time.time()

        results["standard"]["time"].append(end_time - start_time)
        results["standard"]["perplexity"].append(perplexity_standard.item())

        # Test Titans Transformer
        torch.cuda.empty_cache()
        start_time = time.time()
        with torch.no_grad():
            titans_transformer.eval()
            output_titans = titans_transformer(test_batch)
            perplexity_titans = torch.exp(
                criterion(output_titans.view(-1, output_titans.size(-1)), test_batch.view(-1))
            )
        end_time = time.time()

        results["titans"]["time"].append(end_time - start_time)
        results["titans"]["perplexity"].append(perplexity_titans.item())

        print(f"Standard Transformer - Time: {results['standard']['time'][-1]:.4f}s, "
              f"Perplexity: {results['standard']['perplexity'][-1]:.2f}")
        print(f"Titans Transformer - Time: {results['titans']['time'][-1]:.4f}s, "
              f"Perplexity: {results['titans']['perplexity'][-1]:.2f}")

    return results, sequence_lengths


# Plot benchmark results
def plot_benchmark_results(results, sequence_lengths):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Inference time
    axes[0].plot(sequence_lengths, results["standard"]["time"], label="Standard Transformer", color="blue")
    axes[0].plot(sequence_lengths, results["titans"]["time"], label="Titans Transformer", color="red")
    axes[0].set_title("Inference Time")
    axes[0].set_xlabel("Sequence Length")
    axes[0].set_ylabel("Time (s)")
    axes[0].legend()

    # Perplexity
    axes[1].plot(sequence_lengths, results["standard"]["perplexity"], label="Standard Transformer", color="blue")
    axes[1].plot(sequence_lengths, results["titans"]["perplexity"], label="Titans Transformer", color="red")
    axes[1].set_title("Perplexity")
    axes[1].set_xlabel("Sequence Length")
    axes[1].set_ylabel("Perplexity")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.close()


# Main function
if __name__ == "__main__":
    print("\nRunning benchmarks...")
    results, sequence_lengths = benchmark_models()
    plot_benchmark_results(results, sequence_lengths)
    print("\nBenchmark results saved to 'benchmark_results.png'.")

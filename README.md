# Titans Transformer Implementation

This repository contains an experimental implementation of the **Titans Transformer** architecture for sequence modeling tasks. The code is a personal exploration and may include errors or inefficiencies as I am currently in the learning stage. It is inspired by the ideas presented in the original **Titans Transformer** paper, and I highly recommend referring to the paper for accurate and detailed information.

## Overview
The **Titans Transformer** introduces a novel architecture designed to enhance long-sequence modeling by incorporating a memory mechanism. This repository includes:

1. A custom implementation of the **Titans Transformer**.
2. Benchmarking code comparing the **Titans Transformer** with a standard Transformer.
3. Training and evaluation scripts on the Wikitext-2 dataset.
4. Visualization of benchmark results.

> **Note**: This repository is for educational and experimental purposes only and is not a production-ready implementation.

---

## Features

- **Titans Transformer Architecture**: Implements memory mechanisms for improved sequence modeling.
- **Standard Transformer**: Baseline implementation of the original Transformer for comparison.
- **Benchmarking**: Evaluates inference time and perplexity across different sequence lengths.
- **Training**: Customizable training loop with data preprocessing, batching, and evaluation.

---

## Prerequisites

### Dependencies

Ensure you have the following installed:

- Python 3.8+
- PyTorch
- Datasets
- Matplotlib

You can install the required libraries using:
```bash
pip install torch datasets matplotlib
```

---

## Usage

### Clone the Repository
```bash
git clone https://github.com/codewithdark-git/titans-transformer.git
cd titans-transformer
```

### Run Training
To train the Titans Transformer model on the Wikitext-2 dataset, execute the training script:
```bash
python train_titans_transformer.py
```

### Benchmark Models
To compare the Titans Transformer and Standard Transformer, run:
```bash
python benchmark_transformers.py
```
This will generate a plot of inference time and perplexity for different sequence lengths.

---

## Results
The repository includes a benchmarking script to compare:

- **Inference Time**: The time taken to process a batch of sequences.
- **Perplexity**: A measure of the model's ability to predict the next token in a sequence.

The results are visualized in a plot saved as `benchmark_results.png`.

---

## Disclaimer

This implementation is an educational attempt to experiment with the Titans Transformer. **It is not guaranteed to be error-free or optimized**. Please refer to the original paper for accurate and detailed information. I am in the learning phase, and this project is part of my journey to better understand advanced Transformer architectures.

Feedback and suggestions for improvement are always welcome!

---

## References
- Original Paper: [Titans Transformer](https://arxiv.org/abs/2501.00663)

---

## Contact
Feel free to reach out if you have any questions or feedback:

**GitHub**: [Codewithdark-git](https://github.com/codewithdark-git)

---

### Thank You for Visiting!

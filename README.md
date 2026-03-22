# nanoGPT

> A minimal, clean, and fast implementation of GPT for training and fine-tuning medium-sized language models — by [Andrej Karpathy](https://github.com/karpathy).

---

## 📌 Overview

**nanoGPT** is the simplest and fastest repository for training and fine-tuning GPT-2 style models. It is designed to be readable, hackable, and educational — stripping away complexity while remaining fully functional for real-world use cases.

Whether you're a researcher, student, or developer, nanoGPT gives you a clean foundation to understand how GPT models work under the hood.

---

## ✨ Features

- Minimal codebase — easy to read and modify
- Trains GPT-2 (small to XL) from scratch
- Fine-tune on any custom text dataset
- Supports multi-GPU training via PyTorch DDP
- Mixed precision training (float16 / bfloat16)
- Reproduces GPT-2 results on OpenWebText
- Compatible with OpenAI's pretrained GPT-2 weights

---

## 📁 Project Structure
```
nanoGPT/
│
├── model.py          # GPT model definition (transformer architecture)
├── train.py          # Training loop
├── sample.py         # Text generation / sampling script
├── config/           # Configuration files for different training runs
├── data/             # Data preparation scripts
│   ├── shakespeare_char/
│   └── openwebtext/
└── bench.py          # Benchmarking script
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch 2.0+
- numpy
- transformers (for GPT-2 tokenizer)
- datasets (for OpenWebText)
- tiktoken (OpenAI's tokenizer)

Install all dependencies:
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

---

## 🚀 Quick Start

### 1. Train on Shakespeare (Character-level)

A fast demo you can run on any laptop in a few minutes:
```bash
# Prepare the dataset
python data/shakespeare_char/prepare.py

# Train the model
python train.py config/train_shakespeare_char.py
```

### 2. Sample / Generate Text
```bash
python sample.py --out_dir=out-shakespeare-char
```

### 3. Fine-tune GPT-2 on Custom Data
```bash
# Prepare your dataset (place your text in data/mydata/input.txt)
python data/shakespeare_char/prepare.py

# Fine-tune using GPT-2 pretrained weights
python train.py config/finetune_shakespeare.py
```

---

## 🧠 Model Architecture

nanoGPT implements a standard **decoder-only Transformer** (same as GPT-2):

| Component | Description |
|---|---|
| Embedding | Token + Positional Embeddings |
| Attention | Causal Multi-Head Self-Attention |
| FFN | 2-layer MLP with GELU activation |
| Normalization | LayerNorm (pre-norm style) |
| Output | Linear projection to vocabulary |

Supported model sizes:

| Model | Layers | Heads | Embedding Dim | Parameters |
|---|---|---|---|---|
| GPT-2 Small | 12 | 12 | 768 | ~124M |
| GPT-2 Medium | 24 | 16 | 1024 | ~350M |
| GPT-2 Large | 36 | 20 | 1280 | ~774M |
| GPT-2 XL | 48 | 25 | 1600 | ~1.5B |

---

## 🖥️ Multi-GPU Training

nanoGPT supports distributed training using **PyTorch DDP**:
```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py
```

This will use 4 GPUs on a single machine.

---

## 📊 Reproducing GPT-2 on OpenWebText
```bash
# Prepare OpenWebText dataset (requires ~54GB disk space)
python data/openwebtext/prepare.py

# Train GPT-2 (124M) — requires ~8x A100 GPUs
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

Expected result: **~2.85 validation loss** on OpenWebText (matches the original GPT-2 paper).

---

## 🔧 Configuration

Training runs are configured via Python config files in the `config/` folder. Key parameters:
```python
# model
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.0

# training
batch_size = 12
max_iters = 600000
learning_rate = 6e-4
```

---

## 💡 Key Concepts for Beginners

| Term | What It Means |
|---|---|
| **Token** | A chunk of text (word or character) the model reads |
| **Embedding** | A number vector that represents a token |
| **Attention** | Mechanism that lets tokens "look at" each other |
| **Loss** | How wrong the model's predictions are (lower = better) |
| **Fine-tuning** | Training a pretrained model on your own data |

---

## 📚 Learning Resources

- [Andrej Karpathy's YouTube — "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Original GPT-2 Paper — OpenAI](https://openai.com/research/language-unsupervised)
- [Attention Is All You Need — Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)

---

## 🙏 Credits

Built and maintained by **[Andrej Karpathy](https://github.com/karpathy)**.

Inspired by the original [GPT-2](https://github.com/openai/gpt-2) and [minGPT](https://github.com/karpathy/minGPT) repositories.

---

## 📄 License

MIT License — free to use, modify, and distribute.

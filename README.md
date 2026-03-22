# 🧠 NanoGPT — Transformer from Scratch

This project builds a **mini GPT model from scratch using PyTorch** to understand the inner workings of modern Large Language Models (LLMs).

Instead of using large pretrained models, this project focuses on **learning the core architecture** behind systems like ChatGPT, Claude, and Gemini.

---

## 📌 About the Project

This is a **character-level language model** that learns from raw text and predicts the next character in a sequence. 

Rather than simple classification, this is a **sequence prediction problem**, where the model learns to generate coherent text one token at a time.

---

## 🏗️ Architecture Overview

The data flow follows this pipeline:
**Raw Text** → **Encoding** → **Embeddings** → **Transformer Blocks** → **Output (Predictions)**

### Key Components Included:
* **Token Embeddings:** Mapping characters to vectors.
* **Positional Embeddings:** Giving the model a sense of "where" a character is in a sentence.
* **Multi-Head Self-Attention:** Allowing tokens to communicate with each other.
* **Feed Forward Layers:** Processing the information gathered by attention.
* **Residual Connections:** Helping gradients flow during deep training.
* **Layer Normalization:** Keeping the internal math stable.

---

## ⚙️ Model Configuration

The following hyperparameters are used to balance training speed and model depth:

```python
batch_size    = 16      # How many independent sequences per batch?
block_size    = 32      # Maximum context length for predictions
n_embed       = 32      # Size of our embedding vectors
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
learning_rate = 1e-3    # Speed of weight updates
max_iters     = 5000    # Total training steps

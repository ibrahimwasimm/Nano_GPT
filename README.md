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

batch_size    = 16      # How many independent sequences per batch?
block_size    = 32      # Maximum context length for predictions
n_embed       = 32      # Size of our embedding vectors
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
learning_rate = 1e-3    # Speed of weight updates
max_iters     = 5000    # Total training steps


🛠️ Tools & Libraries
Python
PyTorch
NumPy


🧠 Core Components
🔹 Self-Attention

Each token looks at previous tokens to understand context using:

Query → what it is looking for
Key → what it contains
Value → what it shares
🔹 Transformer Blocks

Each block performs:

Attention (communication between tokens)
FeedForward (processing information)
Residual connections (retain previous knowledge)
🔄 Training Process

The model learns by repeating:

 --> Sample a batch of text
--> Predict next tokens
--> Calculate loss
--> Backpropagate errors
--> Update weights


📊 Results
Step	Loss
0     ~4.2
500	  ~2.1
5000	~0.8

The decreasing loss shows the model is learning meaningful patterns.

✨ Text Generation

After training, the model generates text like:

Input:  Once upon a time
Output: Once upon a time there was a king...

Text is generated one token at a time.

🙌 Acknowledgment

Inspired by Andrej Karpathy’s NanoGPT.

📌 Final Note

This project is a mini version of GPT, but the same architecture powers today’s most advanced AI systems.

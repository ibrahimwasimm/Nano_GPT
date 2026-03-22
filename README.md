# 🧠 NanoGPT — Transformer from Scratch

This project builds a **mini GPT model from scratch using PyTorch** to understand how modern language models work.

Instead of using large pretrained models, this project focuses on **learning the core architecture** behind systems like ChatGPT, Claude, and Gemini.

---

## 📌 About the Project

This is a **character-level language model** that learns from raw text and predicts the next character.

Rather than classification, this is a **sequence prediction problem**, where the model generates text one token at a time.

---

## 🏗️ Architecture Overview

Raw Text → Encoding → Embeddings → Transformer Blocks → Output


It includes:

- Token Embeddings  
- Positional Embeddings  
- Multi-Head Self-Attention  
- Feed Forward Layers  
- Residual Connections  
- Layer Normalization  

---

## ⚙️ Model Configuration

python
batch_size = 16
block_size = 32
n_embed = 32
n_head = 4
n_layer = 4
learning_rate = 1e-3
max_iters = 5000


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

Sample a batch of text
Predict next tokens
Calculate loss
Backpropagate errors
Update weights
📊 Results
Step	Loss
0	~4.2
500	~2.1
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

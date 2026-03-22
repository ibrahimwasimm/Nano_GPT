🧠 NanoGPT — From Scratch Transformer (Educational Project)

A minimal implementation of a GPT-style Transformer model built completely from scratch using PyTorch.

This project is designed to help understand how modern LLMs like ChatGPT, Claude, and Gemini actually work under the hood.

🚀 What This Project Does
Trains a character-level language model
Learns patterns from raw text
Generates new text token by token
Implements the full Transformer architecture
🧩 Key Concepts Covered

This project is not just code — it’s a deep learning playground:

Tokenization (char → integer)
Embeddings (token + positional)
Self-Attention (Key, Query, Value)
Multi-Head Attention
Feed Forward Networks
Residual Connections
Layer Normalization
Loss + Backpropagation
Autoregressive Text Generation
⚙️ Model Configuration
batch_size = 16
block_size = 32
n_embed = 32
n_head = 4
n_layer = 4
learning_rate = 1e-3
max_iters = 5000

👉 Small model, but same architecture as real LLMs (just scaled down)

🏗️ Architecture Overview
Raw Text
   ↓
Encoding (characters → integers)
   ↓
Token Embeddings + Position Embeddings
   ↓
Transformer Blocks (×4)
   ├── Multi-Head Attention (communication)
   ├── FeedForward (thinking)
   ├── Residual Connections
   └── Layer Normalization
   ↓
Linear Layer (logits)
   ↓
Softmax → Probabilities
   ↓
Next Token Prediction
🔍 How It Works (Simple)
Model reads a sequence of characters
Looks at previous context (block_size = 32)
Predicts the next character
Repeats this process to generate text
🧠 Example
Input:  "The cat"
Output: "The cat sat on the mat..."

Text is generated one token at a time, just like real LLMs.

🔄 Training Process

Each training step:

Sample random sequences from dataset
Predict next tokens
Calculate loss (how wrong the model is)
Backpropagate gradients
Update weights using AdamW
📊 Loss Progress
step 0     → loss ~ 4.2  (random predictions)
step 500   → loss ~ 2.1  (learning patterns)
step 5000  → loss ~ 0.8  (good predictions)
✨ Text Generation

The model uses an autoregressive loop:

next_token = model.predict(previous_tokens)

Repeated hundreds of times to generate full text.

🛠️ Tech Stack
Python
PyTorch
NumPy

🙌 Acknowledgment

Inspired by Andrej Karpathy’s NanoGPT and Transformer research.

📌 Final Note

This project is a mini version of GPT, but the concept scales to the most powerful AI systems in the world.

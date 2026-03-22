🧠 NanoGPT — Transformer from Scratch

This project builds a mini GPT model from scratch using PyTorch to understand the inner workings of modern Large Language Models (LLMs).

Instead of using large pretrained models, this project focuses on learning the core architecture behind systems like ChatGPT, Claude, and Gemini.

📌 About the Project

This is a character-level language model that learns from raw text and predicts the next character in a sequence.

Rather than simple classification, this is a sequence prediction problem, where the model learns to generate coherent text one token at a time.

🏗️ Architecture Overview

The data flow follows this pipeline:
Raw Text → Encoding → Embeddings → Transformer Blocks → Output (Predictions)

Key Components Included:

Token Embeddings: Mapping characters to vectors.

Positional Embeddings: Giving the model a sense of "where" a character is in a sentence.

Multi-Head Self-Attention: Allowing tokens to communicate with each other.

Feed Forward Layers: Processing the information gathered by attention.

Residual Connections: Helping gradients flow during deep training.

Layer Normalization: Keeping the internal math stable.

⚙️ Model Configuration

The following hyperparameters are used to balance training speed and model depth:

batch_size    = 16      # How many independent sequences per batch?
block_size    = 32      # Maximum context length for predictions
n_embed       = 32      # Size of our embedding vectors
n_head        = 4       # Number of attention heads
n_layer       = 4       # Number of transformer blocks
learning_rate = 1e-3    # Speed of weight updates
max_iters     = 5000    # Total training steps


🛠️ Tools & Libraries

Python: Core language for development.

PyTorch: Deep learning framework for tensor operations.

NumPy: Numerical processing and data handling.

🧠 Core Components

🔹 Self-Attention

Each token looks at previous tokens to understand context using:

Query → What it is looking for.

Key → What it contains (its "profile").

Value → What it shares (the actual content).

🔹 Transformer Blocks

Each block performs:

Attention: Communication between tokens to gather context.

FeedForward: Processing information on a per-token basis.

Residual connections: Retaining previous knowledge to avoid vanishing gradients.

🔄 Training Process

The model learns by repeating the following loop:

Sample a batch of text from the training set.

Predict the next tokens in the sequence.

Calculate the loss (cross-entropy) between prediction and reality.

Backpropagate errors through the network.

Update weights using the AdamW optimizer.

📊 Results

Step

Loss

Status

0

~4.2

Initial random guessing

500

~2.1

Learning basic character patterns

5000

~0.8

Generating meaningful, coherent patterns

Note: The decreasing loss shows the model is effectively learning meaningful patterns from the text.

✨ Text Generation

After training, the model can generate text:

Input: Once upon a time

Output: Once upon a time there was a king...

Text is generated one token at a time (autoregressive).

🙌 Acknowledgment

Inspired by Andrej Karpathy’s NanoGPT and his educational series on building Transformers.

📌 Final Note

This project is a mini version of GPT, but the same fundamental architecture powers today’s most advanced AI systems like GPT-4 and Gemini.

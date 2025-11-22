import os
os.environ["HUGGINGFACE_TOKEN"] = "***********"
os.environ["GEMINI_API_KEY"] = "***********"
os.environ["OPENAI_API_KEY"] = "***********"

# OPENAI TO GEMINI
import openai
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from embedding_adapters import EmbeddingAdapter, list_adapters

from openai import OpenAI
client = OpenAI()

# --- 3) Device selection ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Some example texts to embed
texts = [
    "What is the capital of France?",
    "Explain contrastive learning in simple terms.",
    "How do I fine-tune an embedding adapter?",
    "Best practices for semantic search indexing.",
    "What are retrieval-augmented generation systems?",
    "Describe cosine similarity vs dot product.",
    "Tips for improving query embeddings.",
    "Explain vector databases to a beginner.",
    "How does Mahalanobis distance work?",
    "What is the difference between Gemini and OpenAI embeddings?",
]

print(f"Requesting {len(texts)} embeddings from OpenAI...")

# --- 5) Call OpenAI embeddings API ---
start = time.time()
openai_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
elapsed_ms = (time.time() - start) * 1000.0
print(f"Time to get openai embeddings")
print(f"Elapsed time for {len(texts)} embeddings in batch: {elapsed_ms:.2f} ms")
print(f"Average per embedding: {(elapsed_ms / len(texts)):.2f} ms")


emb_list = [emb.embedding for emb in openai_embedding.data]
source_emb_np = np.asarray(emb_list, dtype="float32")
print("openai embeddings array shape:", source_emb_np.shape)  # (N, 768)

# --- 6) Load your new adapter from the registry ---
# NOTE: make sure your registry has this entry:
#   source="intfloat/e5-base-v2", target="text-embedding-3-small", flavor="generic"
print("Available adapters:", list_adapters())
adapter = EmbeddingAdapter.from_pair(
    source="openai/text-embedding-3-small",
    target="gemini/text-embedding-004",
    flavor="generic",
    device='cpu',
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)

print(f"Adapter in_dim: {adapter.in_dim}, out_dim: {adapter.out_dim}")

if adapter.in_dim != source_emb_np.shape[1]:
    raise ValueError(
        f"Dimension mismatch: adapter.in_dim={adapter.in_dim}, "
        f"but Gemini embeddings have dim={source_emb_np.shape[1]}"
    )

# --- 7) Run embeddings through the adapter on GPU ---
start = time.time()
translated = adapter.encode_embeddings(source_emb_np)  # (N, out_dim)
elapsed_ms = (time.time() - start) * 1000.0

print(f"[Device: {device}]")
print(f"Elapsed time for {len(texts)} embeddings in batch: {elapsed_ms:.2f} ms")
print(f"Average per embedding: {(elapsed_ms / len(texts)):.2f} ms")
print("Translated to Gemini embeddings shape:", translated.shape)
print("First 8 dims of first translated emb:", translated[0][:8])

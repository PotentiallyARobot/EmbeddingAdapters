import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()

import torch
from embedding_adapters import EmbeddingAdapter, list_adapters

#IF YOU WOULD LIKE YOU CAN DISABLE THE REGISTRY FROM LOADING THE REMOTE BY RUNNING THIS FIRST
#os.environ['EMBEDDING_ADAPTERS_DISABLE_REMOTE'] = '1'

print("Available adapters:")
for a in list_adapters():
    print(a["source"], " -> ", a["target"])

device = "cuda" if torch.cuda.is_available() else "cpu"

adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="openai/text_embedding_3_small",
    flavor="linear",
    device=device,
    load_source_encoder=True,
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)
import time

text = "Do you have any documents about swiss cheese?"
# Start timing
start = time.time()

num_calls = 100
for _ in range(num_calls):
    adapter.encode(text)

elapsed = time.time() - start

# Report stats
elapsed_ms = elapsed * 1000
avg_ms = elapsed_ms / num_calls

print(f"[Device: {device}]")
print(f"Elapsed time for {num_calls} calls: {elapsed_ms:.2f} ms")
print(f"Average time per call: {avg_ms:.2f} ms <- NOTE: Open AI embedding calls take 200-300ms typically for most remote api calls to go roundtrip")
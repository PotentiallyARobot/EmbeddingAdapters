
# EmbeddingAdapters 🧠 → 🧠

**Map embeddings across models — from local, cheap embeddings to richer, cloud-based embedding spaces.**

---

## 🚀 What is EmbeddingAdapters?

`EmbeddingAdapters` is a lightweight library and model collection that lets you transform embeddings from one model's space into another's. Want to use a local model like `e5-base-v2` and simulate OpenAI's `text-embedding-3-small` embeddings? This tool has you covered — without needing to call expensive APIs.

It’s particularly useful when you want:

- 💸 **Cheaper embeddings**: Use fantastic open-source models locally, then map your vectors into richer cloud-native spaces.
- 🚀 **Faster local inference**: Skip network latency by embedding on-device.
- 🌉 **Cross-model compatibility**: Use embeddings from one ecosystem (e.g. Hugging Face/Google Gemini) in another (e.g. OpenAI APIs).
- 🧩 **Composable adapters**: Stack multiple embedding spaces together for things like hybrid retrieval or intelligent routing.

---

## 🧠 Why Use This?

With EmbeddingAdapters, you can:

- **Simulate OpenAI embeddings** using cheaper or local models like `e5-base-v2`.
- Build search and retrieval systems that work with *any* embedding model.
- Use **quality scoring** to determine whether a query is in-distribution — and decide whether to:
  - 🚀 **Use a local adapter** (fast, cheap, offline), or
  - ☁️ **Call a remote API** for more difficult queries.
- Save significantly on embedding costs (compared to OpenAI/Google APIs) — especially at high volumes.
- 🏠 Run them client-side to avoid network latency for most queries.
- 🔜 **Coming soon**: A hosted API with **intelligent routing** — dynamically route queries between local and cloud embeddings based on confidence and data similarity.

---

## 📦 Install

First, install the library:

```bash
pip install embedding-adapters
```

---

## 🛠️ Usage

Here’s how you can compute embeddings with `e5-base-v2` and map them into the OpenAI `text-embedding-3-small` space:

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

# 1) Compute source embeddings with e5-base-v2
src_model = SentenceTransformer("intfloat/e5-base-v2")
texts = [
    "NASA announces discovery of Earth-like exoplanet.",
    "Local team wins the championship after dramatic overtime.",
]

e5_embs = src_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,  # important: match adapter training setup
)

# 2) Load the adapter
adapter = EmbeddingAdapter.from_registry(
    source="intfloat/e5-base-v2",
    target="text-embedding-3-small",
    flavor="generic",
    device="cpu",
)

# 3) Project into OpenAI embedding space
openai_embs = adapter(e5_embs)
print(openai_embs.shape)  # e.g. (2, 1536)
```

---

## 🎯 Evaluation Example

On a subset of the **AG News** dataset:

| Setting                                           | R@1  | R@5 | R@10 |
|---------------------------------------------------|------|-----|------|
| OpenAI embeddings → OpenAI corpus                 | 1.00 | 1.00| 1.00 |
| e5-base-v2 → *adapter* → OpenAI corpus            | 0.86 | 1.00| 1.00 |

📌 *R@1 is the hit rate of retrieving the top OpenAI match. The adapter reliably retains the top-10 semantic neighborhood!*

---

## 🤖 Quality Scoring

You can assess whether embeddings are in or out-of-distribution using adapter quality metrics like Mahalanobis distance and KNN:

```python
from embedding_adapters.quality import interpret_quality

texts = ["Where can I get a burger?", "asdfasdfasdfasdf"]
src_embs = adapter.encode(texts, normalize=True, return_source=True)

scores = adapter.score_source(src_embs)
print(interpret_quality(texts, scores, space_label="source"))
```

This tells you whether the adapter is likely to perform well on your data.

---

## 🔑 Requirements

You need a Hugging Face token to use some adapters (e.g. to load certain source models):

1. Get yours from: https://huggingface.co/settings/tokens  
2. Replace `huggingface_token="YOUR_TOKEN"` in your code:

```python
adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="text-embedding-3-small",
    huggingface_token="YOUR_TOKEN"
)
```

---

## 🧩 Roadmap

- 🔜 Support for Gemini and Claude embedding spaces
- 🔀 Automatic routing based on cosine similarity
- 🛠️ REST API for hosted adapters with richer model support

---

## ⚠️ License & Note

- Adapter weights are **proprietary encrypted files** (`adapter_fp16.enc`) — see `license.txt`.
- The Python code is under the MIT license.
- You cannot distribute or extract the model weights in decrypted form.

---

## 📬 Feedback / Contribute

Suggestions, issues, or want to contribute a new adapter? Feel free to open an issue or pull request!

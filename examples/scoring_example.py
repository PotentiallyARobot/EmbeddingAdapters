import os
import sys
from dotenv import load_dotenv
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
load_dotenv()
# SCORING EXAMPLE
import sys
from pathlib import Path
import numpy as np
import torch
from embedding_adapters import EmbeddingAdapter
from embedding_adapters.quality import interpret_quality

# -------------------------------------------------------------------------
# 1) Load adapter (with quality stats) and source encoder
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

adapter = EmbeddingAdapter.from_pair(
    source="intfloat/e5-base-v2",
    target="openai/text_embedding_3_small",
    flavor="linear",
    device=device,
    load_source_encoder=True,
    huggingface_token=os.environ['HUGGINGFACE_TOKEN']
)

# -------------------------------------------------------------------------
# 2) Example texts to score
# -------------------------------------------------------------------------
texts = [
    "Where can I get a cheeseburger near my house",
    "disney world fireworks are amazing",
    "how to fix a docker networking issue on windows",
    "asdfasdfasdfasdfasdfasdfasdfasdfasdfasdf",
]

# Get *source-space* embeddings (e5-base-v2) from the adapter
src_embs = adapter.encode(
    texts,
    as_numpy=True,
    normalize=True,
    return_source=True,
)

# -------------------------------------------------------------------------
# 3) Get quality scores (numeric)
# -------------------------------------------------------------------------
scores = adapter.score_source(src_embs)

# scores is a dict of numpy arrays with shape (N,)
maha = scores["mahalanobis"]
knn = scores["knn_distance"]
conf_maha = scores["conf_maha"]
conf_knn = scores["conf_knn"]
conf = scores["confidence"]

print("=== Raw numeric scores (source space) ===")
for i, t in enumerate(texts):
    print(f"Example {i+1}: {t!r}")
    print(f"  mahalanobis   : {maha[i]:.4f}")
    print(f"  kNN distance  : {knn[i]:.4f}")
    print(f"  conf_maha     : {conf_maha[i]:.3f}")
    print(f"  conf_knn      : {conf_knn[i]:.3f}")
    print(f"  confidence    : {conf[i]:.3f}")
    print()

# Batch-level numeric summary
print("Batch confidence stats:")
print(f"  mean: {float(conf.mean()):.3f}")
print(f"  min : {float(conf.min()):.3f}")
print(f"  max : {float(conf.max()):.3f}")
print()

# -------------------------------------------------------------------------
# 4) Human-readable interpretation
# -------------------------------------------------------------------------
print("=== English interpretation ===")
print(interpret_quality(texts, scores, space_label="source"))

# -------------------------------------------------------------------------
# 5) (Optional) Example of scoring target-space embeddings
# -------------------------------------------------------------------------
# If you already have raw OpenAI embeddings (shape: (N, d_tgt)), you can do:
#
#   openai_embs = np.array([...], dtype=np.float32)  # (N, d_tgt)
#   tgt_scores = adapter.score_target(openai_embs)
#   print(interpret_quality(texts, tgt_scores, space_label="target"))
#
# This is useful if you’re scoring production queries from the
# target model rather than from the source encoder.

print("\nDone.")

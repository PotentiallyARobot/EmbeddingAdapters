// ── Documentation (standalone page) ──

const API_DOCS_BASE = "https://embedding-adapters-api.embedding-adapters.workers.dev";

const docSections = [
  { id: "overview",     icon: "◈", label: "Overview" },
  { id: "python_sdk",   icon: "⬡", label: "Python SDK" },
  { id: "paths",        icon: "⬡", label: "Adapter Paths" },
  { id: "registry",     icon: "▦", label: "Adapter Registry" },
  { id: "embed",        icon: "↗", label: "Embed API" },
  { id: "quality",      icon: "◎", label: "Quality Routing" },
  { id: "batch",        icon: "▤", label: "Batch Processing" },
  { id: "adapters_doc", icon: "◇", label: "Custom Adapters" },
  { id: "models_ref",   icon: "◈", label: "Models Reference" },
  { id: "errors",       icon: "⚠", label: "Errors" },
];

let currentSection = "overview";

document.addEventListener("DOMContentLoaded", () => {
  const hash = window.location.hash.replace("#", "");
  if (docSections.find(s => s.id === hash)) currentSection = hash;
  renderDocsSidebar();
  renderDocsContent();
});

function switchDoc(id) {
  currentSection = id;
  window.location.hash = id;
  $$(".sidebar-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  renderDocsContent();
}

function renderDocsSidebar() {
  const sb = $("#sidebar");
  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <div class="nav-logo-icon"><svg viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="7" fill="#2563eb"/><path d="M7 14h5l3-6 3 12 3-6h5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <span class="nav-logo-text">EmbeddingAdapters</span>
      </a>
    </div>
    <div style="padding:12px 16px 8px; font-size:11px; font-weight:700; color:#52525b; text-transform:uppercase; letter-spacing:0.06em;">Documentation</div>
    <div class="sidebar-nav">
      ${docSections.map(s => `
        <button class="sidebar-item ${s.id === currentSection ? 'active' : ''}" data-tab="${s.id}" onclick="switchDoc('${s.id}')">
          <span class="icon">${s.icon}</span> ${s.label}
        </button>
      `).join("")}
    </div>
    <div style="padding:12px 16px 8px; font-size:11px; font-weight:700; color:#52525b; text-transform:uppercase; letter-spacing:0.06em;">Navigation</div>
    <div class="sidebar-nav">
      <a href="dashboard.html" class="sidebar-item" style="text-decoration:none;">
        <span class="icon">◧</span> Dashboard
      </a>
      <a href="benchmarks.html" class="sidebar-item" style="text-decoration:none;">
        <span class="icon">▦</span> Benchmarks
      </a>
      <a href="index.html" class="sidebar-item" style="text-decoration:none;">
        <span class="icon">◪</span> Pricing
      </a>
    </div>
    <div class="sidebar-footer">
      <div style="font-size:11px; color:#3f3f46;">v1 API Reference</div>
    </div>
  `;
}

function renderDocsContent() {
  const main = $("#main");
  main.className = "main-content fade-in";
  const renderers = {
    overview: renderDocOverview,
    python_sdk: renderDocPythonSDK,
    paths: renderDocPaths,
    registry: renderDocRegistry,
    embed: renderDocEmbed,
    quality: renderDocQuality,
    batch: renderDocBatch,
    adapters_doc: renderDocAdapters,
    models_ref: renderDocModelsRef,
    errors: renderDocErrors,
  };
  main.innerHTML = (renderers[currentSection] || renderDocOverview)();
}

function codeBlock(lang, code) {
  const id = "cb_" + Math.random().toString(36).slice(2, 8);
  return `
    <div class="code-wrap">
      <div class="code-actions">
        <span class="mono code-lang">${lang}</span>
        <button class="btn-copy" onclick="copyToClipboard(document.getElementById('${id}').textContent, this)">Copy</button>
      </div>
      <pre class="code-block" id="${id}">${escHtml(code)}</pre>
    </div>
  `;
}

// ── Overview ──
function renderDocOverview() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">What is Embedding Adapters?</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Universal embedding-space translation. Run locally or via API.</p>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>Install</h3></div>
      <div class="card-body">
        ${codeBlock("bash", `pip install torch sentence-transformers transformers fastapi uvicorn huggingface_hub httpx bitsandbytes`)}
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-top:12px; margin-bottom:12px;">
          Then clone the repo and start the server:
        </p>
        ${codeBlock("bash", `git clone https://github.com/PotentiallyARobot/EmbeddingAdapters.git
cd EmbeddingAdapters

# Set your keys (optional — only needed for quality routing and private repos)
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."

# Start the local server on port 8787
python main.py`)}
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-top:12px;">
          Models download automatically on first run (~2GB). Requires NVIDIA GPU for full speed. Server runs at <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">http://127.0.0.1:8787</code>
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>Embed with Qwen3-0.6B → TE3</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Send text, get back 3072-d embeddings compatible with OpenAI text-embedding-3-large. Zero API calls at quality=0.
        </p>
        ${tabbedCodeBlock([
          { lang:"python", label:"Local Server", code: `import requests, base64, numpy as np

resp = requests.post("http://127.0.0.1:8787/v1/embed", json={
    "texts": ["How do embedding adapters work?", "Translate vector spaces locally"],
    "model": "qwen06b-te3-adapted",
    "quality": 0,   # 0 = fully local, no API calls
})

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")           # (2, 3072) — TE3-compatible
print(f"Cost:  \${data['usage']['cost']}")  # $0.000000 at quality=0
print(f"Time:  {data['usage']['seconds']}s")` },
          { lang:"python", label:"Hosted API", code: `import requests, base64, numpy as np

resp = requests.post("${API_DOCS_BASE}/v1/embed",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["How do embedding adapters work?", "Translate vector spaces locally"],
        "model": "qwen06b-te3-adapted",
        "quality": 0,
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")  # (2, 3072)` },
          { lang:"bash", label:"cURL", code: `curl http://127.0.0.1:8787/v1/embed \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["How do embedding adapters work?"],
    "model": "qwen06b-te3-adapted",
    "quality": 0
  }'` },
        ])}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>Reverse Adapter: TE3 → Qwen3-8B</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Already have TE3 embeddings? Translate them into Qwen3-8B's vector space. Send <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">embeddings_b64</code> instead of texts.
        </p>
        ${codeBlock("python", `import requests, base64, numpy as np

# Your existing TE3 embeddings (from OpenAI)
te3_embs = np.random.randn(3, 3072).astype(np.float32)

resp = requests.post("http://127.0.0.1:8787/v1/embed", json={
    "model": "te3-qwen3-8b-adapted",
    "embeddings_b64": base64.b64encode(te3_embs.tobytes()).decode(),
    "texts": ["query 1", "query 2", "query 3"],  # optional, for quality routing
    "quality": 30,
})

data = resp.json()
qwen_embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {qwen_embs.shape}")  # (3, 4096) — Qwen3-8B-compatible`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Available Models</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Model ID</th><th>Direction</th><th>Output</th><th>Speed</th><th>Input</th></tr></thead>
          <tbody>
            <tr><td class="green">minilm-te3-adapted</td><td>MiniLM → TE3</td><td>3072d</td><td>18K tok/s</td><td>texts</td></tr>
            <tr><td class="green">qwen06b-te3-adapted</td><td>Qwen3-0.6B → TE3</td><td>3072d</td><td>1.2K tok/s</td><td>texts</td></tr>
            <tr><td class="green">te3-qwen3-8b-adapted</td><td>TE3 → Qwen3-8B</td><td>4096d</td><td>—</td><td>embeddings_b64</td></tr>
            <tr><td class="green">all-MiniLM-L6-v2</td><td>Raw (no adapter)</td><td>384d</td><td>18K tok/s</td><td>texts</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#52525b; margin-top:8px;">See <a href="#" onclick="switchDoc('models_ref'); return false;" style="color:#3b82f6;">Models Reference</a> for full details.</p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>The Problem</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:12px;">
          OpenAI's <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">text-embedding-3-large</code> is the de-facto standard for vector search, RAG, and classification.
          But at $0.13/1M tokens, costs add up fast — especially when you're embedding millions of documents.
        </p>
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          Switching to a cheaper open-source model means re-embedding your entire corpus and losing compatibility
          with existing TE3 vector stores.
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>The Solution</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:12px;">
          Embedding Adapters runs small, fast open-source encoders (MiniLM, Qwen3) and uses trained adapters
          to project their output into any target embedding space. Forward adapters translate local models → provider space. Reverse adapters translate provider embeddings → local model space. The result: <strong style="color:#3b82f6;">vectors that are directly
          compatible across embedding spaces</strong> at 50-99% lower cost.
        </p>
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          A quality routing system automatically detects texts where the adapter is less confident and can
          selectively re-embed those via OpenAI — giving you a cost/quality dial from 0 (all local) to 100 (all OpenAI).
        </p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;">
          <div style="background:#0a0a0f; border:1px solid #1c1c26; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; font-weight:800; color:#3b82f6; margin-bottom:4px;">50-99%</div>
            <div style="font-size:12px; color:#52525b;">Cost Reduction</div>
          </div>
          <div style="background:#0a0a0f; border:1px solid #1c1c26; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; font-weight:800; color:#6366f1; margin-bottom:4px;">Any→Any</div>
            <div style="font-size:12px; color:#52525b;">Cross-Space Translation</div>
          </div>
          <div style="background:#0a0a0f; border:1px solid #1c1c26; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; font-weight:800; color:#f59e0b; margin-bottom:4px;">93-98%</div>
            <div style="font-size:12px; color:#52525b;">Cosine Similarity to TE3</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>How It Works</h3></div>
      <div class="card-body">
        <div style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          <div style="display:flex; gap:12px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">1</div>
            <div><strong style="color:#d4d4d8;">You send texts</strong> to our API with a model choice (e.g. <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">minilm-te3-adapted</code>).</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">2</div>
            <div><strong style="color:#d4d4d8;">We encode locally</strong> using the open-source model, then apply a trained LoRA adapter to project embeddings into TE3's 3072-d space.</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">3</div>
            <div><strong style="color:#d4d4d8;">Quality routing</strong> (optional): a confidence head scores each text. Below your threshold, we re-embed via OpenAI for full fidelity.</div>
          </div>
          <div style="display:flex; gap:12px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">4</div>
            <div><strong style="color:#d4d4d8;">You get back</strong> base64-encoded float32 vectors that drop directly into the target model's vector store — Pinecone, Weaviate, Qdrant, pgvector, etc.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Use Cases</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
          <div>
            <div style="font-size:14px; font-weight:600; color:#d4d4d8; margin-bottom:6px;">RAG / Retrieval Pipelines</div>
            <p style="font-size:13px; color:#71717a; line-height:1.6;">Embed user queries and document chunks for retrieval-augmented generation. Quality routing ensures critical queries get full TE3 accuracy while bulk indexing stays cheap.</p>
          </div>
          <div>
            <div style="font-size:14px; font-weight:600; color:#d4d4d8; margin-bottom:6px;">Semantic Search</div>
            <p style="font-size:13px; color:#71717a; line-height:1.6;">Drop-in replacement for existing TE3-powered search. Mix adapted and native vectors in the same index — cosine similarity stays high.</p>
          </div>
          <div>
            <div style="font-size:14px; font-weight:600; color:#d4d4d8; margin-bottom:6px;">Classification & Clustering</div>
            <p style="font-size:13px; color:#71717a; line-height:1.6;">Embed text into TE3 space and use existing classifiers or cluster boundaries. No retraining needed when switching from native TE3.</p>
          </div>
          <div>
            <div style="font-size:14px; font-weight:600; color:#d4d4d8; margin-bottom:6px;">Batch Document Processing</div>
            <p style="font-size:13px; color:#71717a; line-height:1.6;">Async batch API handles 1K–100K texts. Ideal for initial corpus indexing, periodic re-embedding, or data migration.</p>
          </div>
        </div>
      </div>
    </div>

  `;
}

// ── Python SDK ──
function renderDocPythonSDK() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Python SDK</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Install the <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">embedding-adapters</code> package and run adapters locally.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Install</h3></div>
      <div class="card-body">
        ${codeBlock("bash", `pip install embedding-adapters`)}
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-top:12px;">
          Requires Python 3.9+ and PyTorch. GPU recommended but not required.
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>CLI — Quick Embed</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Embed text from the command line. The adapter downloads automatically from the registry.
        </p>
        ${codeBlock("bash", `embedding-adapters embed \\
  --source sentence-transformers/all-MiniLM-L6-v2 \\
  --target openai/text-embedding-3-small \\
  --flavor large \\
  --text "Where can I get a hamburger?"`)}
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-top:12px;">
          Use <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">--source</code> and <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">--target</code> to specify the adapter pair. <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">--flavor</code> selects the adapter size (small, medium, large, vlarge).
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Python — Full Example</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Load a source model, load an adapter from the registry, encode, and translate:
        </p>
        ${codeBlock("python", `import os, time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

# 1) Load the source model
src_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Load a pre-trained adapter from the registry
adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-small",
    flavor="large",
    device=device,
    huggingface_token=os.environ.get("HUGGINGFACE_TOKEN"),
)

# 3) Texts to embed
texts = [
    "NASA announces discovery of Earth-like exoplanet.",
    "Can you help me find my keys?",
]

# 4) Encode with the source model
start = time.time()
src_embs = src_model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,  # matches adapter training setup
)

# 5) Translate to target space
translated_embs = adapter.encode_embeddings(src_embs)  # (N, out_dim)
elapsed_ms = (time.time() - start) * 1000.0

print(f"[Device: {device}]")
print(f"Elapsed: {elapsed_ms:.2f} ms for {len(texts)} embeddings")
print(f"Per embedding: {(elapsed_ms / len(texts)):.2f} ms")
print(f"Shape: {translated_embs.shape}")
print(f"First 8 dims: {translated_embs[0][:8]}")`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Key Methods</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Method</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">EmbeddingAdapter.from_registry()</td><td>Load adapter by source/target/flavor. Downloads from HuggingFace.</td></tr>
            <tr><td class="green">adapter.encode_embeddings(embs)</td><td>Translate source embeddings → target space. Returns numpy array (N, out_dim).</td></tr>
            <tr><td class="green">adapter.quality_scores(embs)</td><td>Get per-embedding confidence scores (0–1). Higher = more reliable translation.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Environment Variables</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Variable</th><th>Required</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">HUGGINGFACE_TOKEN</td><td>For pro adapters</td><td>HuggingFace token for downloading encrypted/private adapter weights.</td></tr>
            <tr><td class="green">OPENAI_API_KEY</td><td>Optional</td><td>Only needed if using quality routing with OpenAI fallback on the API server.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ── Adapter Paths ──
function renderDocPaths() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Adapter Paths</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Every translation path available today. Pick a source, pick a target — if there's a line between them, there's an adapter.</p>

    <div class="card" style="margin-bottom:24px;">
      <div class="card-body" style="padding:0; overflow:hidden;">
        <svg viewBox="0 0 800 500" style="width:100%; height:auto; display:block;" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#3b82f6" opacity="0.6"/></marker>
            <marker id="ah2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#f59e0b" opacity="0.6"/></marker>
          </defs>

          <!-- Grid -->
          <rect width="800" height="500" fill="#08080c" rx="14"/>
          <line x1="0" y1="160" x2="800" y2="160" stroke="#1c1c26" stroke-width="1"/>
          <text x="400" y="30" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="10" font-weight="600" letter-spacing="0.08em">LOCAL MODELS</text>
          <text x="400" y="190" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="10" font-weight="600" letter-spacing="0.08em">PROVIDER MODELS</text>

          <!-- Nodes: Local -->
          <rect x="40" y="60" width="200" height="70" rx="10" fill="#0d0d12" stroke="#6366f1" stroke-width="1.5"/>
          <text x="140" y="88" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">all-MiniLM-L6-v2</text>
          <text x="140" y="108" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">384d · 22M params</text>

          <rect x="300" y="60" width="200" height="70" rx="10" fill="#0d0d12" stroke="#10b981" stroke-width="1.5"/>
          <text x="400" y="88" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">intfloat/e5-base-v2</text>
          <text x="400" y="108" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">768d · 110M params</text>

          <rect x="560" y="60" width="200" height="70" rx="10" fill="#0d0d12" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="660" y="88" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">Qwen3-Embed-0.6B</text>
          <text x="660" y="108" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">1024d · 600M params</text>

          <!-- Nodes: Providers -->
          <rect x="80" y="230" width="240" height="70" rx="10" fill="#0d0d12" stroke="#3b82f6" stroke-width="1.5"/>
          <text x="200" y="258" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">OpenAI TE3-small</text>
          <text x="200" y="278" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">1536d</text>

          <rect x="400" y="230" width="240" height="70" rx="10" fill="#0d0d12" stroke="#8b5cf6" stroke-width="1.5"/>
          <text x="520" y="258" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">Gemini TE-004</text>
          <text x="520" y="278" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">768d</text>

          <!-- Nodes: Large local -->
          <rect x="200" y="370" width="240" height="70" rx="10" fill="#0d0d12" stroke="#ef4444" stroke-width="1.5"/>
          <text x="320" y="398" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">OpenAI TE3-large</text>
          <text x="320" y="418" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">3072d</text>

          <rect x="500" y="370" width="240" height="70" rx="10" fill="#0d0d12" stroke="#ec4899" stroke-width="1.5"/>
          <text x="620" y="398" text-anchor="middle" fill="#d4d4d8" font-family="DM Sans,sans-serif" font-size="14" font-weight="700">Qwen3-Embed-8B</text>
          <text x="620" y="418" text-anchor="middle" fill="#52525b" font-family="JetBrains Mono,monospace" font-size="11">4096d</text>

          <!-- Edges: MiniLM → targets -->
          <line x1="140" y1="130" x2="180" y2="230" stroke="#3b82f6" stroke-width="2" opacity="0.5" marker-end="url(#ah)"/>
          <line x1="160" y1="130" x2="500" y2="230" stroke="#8b5cf6" stroke-width="1.5" opacity="0.3" stroke-dasharray="4,4" marker-end="url(#ah)"/>
          <line x1="200" y1="130" x2="400" y2="230" stroke="#10b981" stroke-width="1.5" opacity="0.3" stroke-dasharray="4,4"/>

          <!-- MiniLM → E5 -->
          <line x1="240" y1="95" x2="300" y2="95" stroke="#10b981" stroke-width="2" opacity="0.5" marker-end="url(#ah)"/>

          <!-- E5 → targets -->
          <line x1="400" y1="130" x2="220" y2="230" stroke="#3b82f6" stroke-width="2" opacity="0.5" marker-end="url(#ah)"/>
          <line x1="420" y1="130" x2="520" y2="230" stroke="#8b5cf6" stroke-width="1.5" opacity="0.3" stroke-dasharray="4,4" marker-end="url(#ah)"/>

          <!-- Qwen06 → TE3-large -->
          <line x1="660" y1="130" x2="360" y2="370" stroke="#ef4444" stroke-width="2" opacity="0.5" marker-end="url(#ah)"/>

          <!-- MiniLM → TE3-large -->
          <line x1="120" y1="130" x2="280" y2="370" stroke="#ef4444" stroke-width="2" opacity="0.5" marker-end="url(#ah)"/>

          <!-- TE3-large → Qwen3-8B (reverse) -->
          <line x1="440" y1="405" x2="500" y2="405" stroke="#ec4899" stroke-width="2" opacity="0.5" marker-end="url(#ah2)"/>

          <!-- Cross-provider -->
          <line x1="320" y1="265" x2="400" y2="265" stroke="#f59e0b" stroke-width="1.5" opacity="0.4" stroke-dasharray="4,4" marker-end="url(#ah2)"/>
          <line x1="400" y1="275" x2="320" y2="275" stroke="#f59e0b" stroke-width="1.5" opacity="0.4" stroke-dasharray="4,4" marker-end="url(#ah2)"/>

          <!-- Legend -->
          <line x1="40" y1="475" x2="70" y2="475" stroke="#3b82f6" stroke-width="2" opacity="0.5"/>
          <text x="78" y="479" fill="#71717a" font-family="DM Sans,sans-serif" font-size="11">Public</text>
          <line x1="160" y1="475" x2="190" y2="475" stroke="#8b5cf6" stroke-width="1.5" opacity="0.4" stroke-dasharray="4,4"/>
          <text x="198" y="479" fill="#71717a" font-family="DM Sans,sans-serif" font-size="11">Pro (encrypted)</text>
          <circle cx="330" cy="475" r="4" fill="#f59e0b" opacity="0.6"/>
          <text x="340" y="479" fill="#71717a" font-family="DM Sans,sans-serif" font-size="11">Cross-provider / Reverse</text>
        </svg>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>All Available Paths</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>From</th><th></th><th>To</th><th>Flavors</th><th>Access</th></tr></thead>
          <tbody>
            <tr><td style="color:#6366f1;">MiniLM-L6-v2</td><td>→</td><td style="color:#3b82f6;">OpenAI TE3-small</td><td>medium, large, vlarge, generic</td><td>Public</td></tr>
            <tr><td style="color:#6366f1;">MiniLM-L6-v2</td><td>→</td><td style="color:#ef4444;">OpenAI TE3-large</td><td>adapted (API)</td><td>Public</td></tr>
            <tr><td style="color:#6366f1;">MiniLM-L6-v2</td><td>→</td><td style="color:#10b981;">E5-base-v2</td><td>large</td><td>Public</td></tr>
            <tr><td style="color:#6366f1;">MiniLM-L6-v2</td><td>→</td><td style="color:#8b5cf6;">Gemini TE-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr style="border-top:1px solid #1c1c26;"><td style="color:#10b981;">E5-base-v2</td><td>→</td><td style="color:#3b82f6;">OpenAI TE3-small</td><td>small, linear, large, generic</td><td>Mixed</td></tr>
            <tr><td style="color:#10b981;">E5-base-v2</td><td>→</td><td style="color:#8b5cf6;">Gemini TE-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr style="border-top:1px solid #1c1c26;"><td style="color:#f59e0b;">Qwen3-0.6B</td><td>→</td><td style="color:#ef4444;">OpenAI TE3-large</td><td>adapted (API)</td><td>Public</td></tr>
            <tr style="border-top:1px solid #1c1c26;"><td style="color:#3b82f6;">OpenAI TE3-small</td><td>→</td><td style="color:#8b5cf6;">Gemini TE-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td style="color:#ef4444;">OpenAI TE3-large</td><td>→</td><td style="color:#ec4899;">Qwen3-Embed-8B</td><td>adapted (API)</td><td>Public</td></tr>
            <tr><td style="color:#8b5cf6;">Gemini TE-004</td><td>→</td><td style="color:#10b981;">E5-base-v2</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Example: Chain Adapters</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          No direct adapter from MiniLM to Gemini in the public tier? Chain through E5:
        </p>
        ${codeBlock("python", `from embedding_adapters import EmbeddingAdapter
from sentence_transformers import SentenceTransformer

src = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = ["How do I chain adapters?"]

# Step 1: MiniLM → E5 (public)
adapter_1 = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="intfloat/e5-base-v2",
    flavor="large", device="cuda",
)

# Step 2: E5 → OpenAI TE3-small (public)
adapter_2 = EmbeddingAdapter.from_registry(
    source="intfloat/e5-base-v2",
    target="openai/text-embedding-3-small",
    flavor="small", device="cuda",
)

src_embs = src.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
e5_embs = adapter_1.encode_embeddings(src_embs)
te3_embs = adapter_2.encode_embeddings(e5_embs)

print(te3_embs.shape)  # (1, 1536) — OpenAI TE3-small compatible`)}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Choosing a Path</h3></div>
      <div class="card-body">
        <div style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          <div style="display:flex; gap:12px; margin-bottom:12px; align-items:flex-start;">
            <div style="min-width:24px; height:24px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">1</div>
            <div><strong style="color:#d4d4d8;">Fastest & cheapest?</strong> MiniLM → target. 18K tok/s, smallest model, most adapter flavors.</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:12px; align-items:flex-start;">
            <div style="min-width:24px; height:24px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">2</div>
            <div><strong style="color:#d4d4d8;">Highest accuracy?</strong> E5-base-v2 or Qwen3-0.6B → target. Richer source embeddings = better translation.</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:12px; align-items:flex-start;">
            <div style="min-width:24px; height:24px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">3</div>
            <div><strong style="color:#d4d4d8;">Already have provider embeddings?</strong> Use a reverse/cross-provider adapter. TE3-large → Qwen3-8B, OpenAI → Gemini, etc.</div>
          </div>
          <div style="display:flex; gap:12px; align-items:flex-start;">
            <div style="min-width:24px; height:24px; border-radius:50%; background:#3b82f620; color:#3b82f6; display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:700;">4</div>
            <div><strong style="color:#d4d4d8;">Need a path that doesn't exist?</strong> <a href="#" onclick="switchDoc('adapters_doc'); return false;" style="color:#3b82f6;">Train a custom adapter</a> on your own data.</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ── Adapter Registry ──
function renderDocRegistry() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Adapter Registry</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">All available pre-trained adapters. Use any pair with the Python SDK or API.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>How Adapters Work</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:12px;">
          Each adapter translates embeddings from a <strong style="color:#d4d4d8;">source</strong> model's vector space into a <strong style="color:#d4d4d8;">target</strong> model's vector space.
          This lets you run a cheap local model and get embeddings compatible with an expensive provider — or translate between any two embedding spaces.
        </p>
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          Adapters come in different <strong style="color:#d4d4d8;">flavors</strong> (sizes): <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">linear</code> (fastest, least accurate), <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">small</code>, <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">medium</code>, <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">large</code>, <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">vlarge</code> (slowest, most accurate).
          Pro adapters (marked with <span style="color:#3b82f6; font-weight:600;">PRO</span>) use encrypted weights and require a HuggingFace token.
        </p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>MiniLM → OpenAI TE3-small</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate sentence-transformers/all-MiniLM-L6-v2 (384d) → openai/text-embedding-3-small (1536d)</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Flavor</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">medium</td><td>emb_adapter_minilm_to_openai_text-embedding-3-medium_v1</td><td>Public</td></tr>
            <tr><td class="green">large</td><td>emb_adapter_minilm_to_openai_text-embedding-3-large_v1</td><td>Public</td></tr>
            <tr><td class="green">vlarge</td><td>emb_adapter_minilm_to_openai_text-embedding-3-vlarge_v1</td><td>Public</td></tr>
            <tr><td class="green">generic</td><td>emb_adapter_all-MiniLM-L6-v2_to_openai_text_embedding_3_medium</td><td>Public</td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-small",
    flavor="large",
    device=device,
)`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>E5-base-v2 → OpenAI TE3-small</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate intfloat/e5-base-v2 (768d) → openai/text-embedding-3-small (1536d)</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Flavor</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">small</td><td>emb_adapter_e5-base-v2-to-openai_text_embedding_3_small_v1</td><td>Public</td></tr>
            <tr><td class="green">linear</td><td>emb_adapter_e5-base-v2-to-openai_text_embedding_3_small_v1 (linear)</td><td>Public</td></tr>
            <tr><td class="green">large</td><td>emb_adapter_e5-base-v2-to-openai_text_embedding_3_large_v2</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
            <tr><td class="green">generic</td><td>emb_adapter_e5-base-v2_to_text-embedding-3-small-v_0_1_fp16</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="intfloat/e5-base-v2",
    target="openai/text-embedding-3-small",
    flavor="small",
    device=device,
)`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>MiniLM → E5-base-v2</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate sentence-transformers/all-MiniLM-L6-v2 (384d) → intfloat/e5-base-v2 (768d)</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Flavor</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">large</td><td>emb_adapter_minilm_L6_v2-to-e5-base-v2_large_v1</td><td>Public</td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="intfloat/e5-base-v2",
    flavor="large",
    device=device,
)`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>MiniLM → Gemini text-embedding-004</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate sentence-transformers/all-MiniLM-L6-v2 (384d) → gemini/text-embedding-004 (768d)</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Flavor</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">generic</td><td>emb_adapter_all-MiniLM-L6-v2_to_gemini_text_embedding_004_small_v1</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="gemini/text-embedding-004",
    flavor="generic",
    device=device,
    huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
)`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>E5-base-v2 → Gemini text-embedding-004</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate intfloat/e5-base-v2 (768d) → gemini/text-embedding-004 (768d)</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Flavor</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">generic</td><td>emb_adapter_e5-base-v2_to_gemini_text_embedding_004_small_v1</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Cross-Provider Adapters</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:12px;">Translate directly between provider embedding spaces — no local model needed.</p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>Source → Target</th><th>Slug</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td class="green">OpenAI TE3-small → Gemini</td><td>emb_adapter_openai_text_embedding_3_small_to_gemini_text_embedding_004_linear_v1</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
            <tr><td class="green">Gemini → OpenAI TE3-small</td><td>emb_adapter_gemini_text_embedding_004_to_e5-base-v2-to_linear_v1</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
            <tr><td class="green">Gemini → E5-base-v2</td><td>emb_adapter_gemini_text_embedding_004_to_e5-base-v2-to_linear_v1</td><td><span style="color:#3b82f6; font-weight:600;">PRO</span></td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `# Translate OpenAI embeddings into Gemini's space
adapter = EmbeddingAdapter.from_registry(
    source="openai/text-embedding-3-small",
    target="gemini/text-embedding-004",
    flavor="generic",
    device=device,
    huggingface_token=os.environ["HUGGINGFACE_TOKEN"],
)

# Pass in your pre-computed OpenAI embeddings
translated = adapter.encode_embeddings(openai_embs)  # → Gemini-compatible`)}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Full Adapter List</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:11px;">
          <thead><tr><th>Source</th><th>Target</th><th>Flavor</th><th>Type</th></tr></thead>
          <tbody>
            <tr><td>all-MiniLM-L6-v2</td><td>openai/te3-small</td><td>medium</td><td>Public</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td>openai/te3-small</td><td>large</td><td>Public</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td>openai/te3-small</td><td>vlarge</td><td>Public</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td>openai/te3-small</td><td>generic</td><td>Public</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td>intfloat/e5-base-v2</td><td>large</td><td>Public</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td>gemini/te-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td>intfloat/e5-base-v2</td><td>openai/te3-small</td><td>small</td><td>Public</td></tr>
            <tr><td>intfloat/e5-base-v2</td><td>openai/te3-small</td><td>linear</td><td>Public</td></tr>
            <tr><td>intfloat/e5-base-v2</td><td>openai/te3-small</td><td>large</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td>intfloat/e5-base-v2</td><td>openai/te3-small</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td>intfloat/e5-base-v2</td><td>gemini/te-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td>openai/te3-small</td><td>gemini/te-004</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
            <tr><td>gemini/te-004</td><td>intfloat/e5-base-v2</td><td>generic</td><td style="color:#3b82f6;">PRO</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ── Embed Texts ──
function renderDocEmbed() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Embed Texts</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Core endpoint for generating embeddings.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/embed</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:20px;">
          Send texts or pre-computed embeddings, receive adapted embedding vectors. Forward adapters (e.g. <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">minilm-te3-adapted</code>) accept texts and output 3072-d TE3-compatible vectors. Reverse adapters (e.g. <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">te3-qwen3-8b-adapted</code>) accept pre-computed TE3 embeddings and output vectors in the target model's space.
        </p>

        ${tabbedCodeBlock([
          { lang:"bash", label:"cURL", code: `curl -X POST ${API_DOCS_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["hello world"], "model": "qwen06b-te3-adapted", "quality": 0}'` },
          { lang:"python", label:"Python", code: `resp = requests.post("${API_DOCS_BASE}/v1/embed",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={"texts": ["hello world"], "model": "qwen06b-te3-adapted", "quality": 0})` },
        ])}

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Request Headers</div>
        <table class="table mono" style="font-size:12px; margin-bottom:20px;">
          <thead><tr><th>Header</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td class="green">Authorization</td><td>Bearer YOUR_API_KEY</td></tr>
            <tr><td class="green">Content-Type</td><td>application/json</td></tr>
          </tbody>
        </table>

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Request Body</div>
        <table class="table mono" style="font-size:12px; margin-bottom:20px;">
          <thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">texts</td><td>string[]</td><td>✓*</td><td>—</td><td>Array of texts to embed. Max 8192. Required for forward adapters. Optional for reverse adapters (used for quality routing).</td></tr>
            <tr><td class="green">embeddings_b64</td><td>string</td><td>✓*</td><td>—</td><td>Base64-encoded float32 source embeddings. Required for reverse adapters (e.g. te3-qwen3-8b-adapted). Not used for forward adapters.</td></tr>
            <tr><td class="green">model</td><td>string</td><td></td><td>minilm-te3-adapted</td><td>Model ID. See Models Reference.</td></tr>
            <tr><td class="green">quality</td><td>integer</td><td></td><td>0</td><td>0–100. Controls quality routing threshold. For reverse adapters, routes to local fallback model.</td></tr>
            <tr><td class="green">include_quality</td><td>boolean</td><td></td><td>false</td><td>If true, returns per-text quality confidence scores.</td></tr>
            <tr><td class="green">adapter_id</td><td>string</td><td></td><td>—</td><td>Custom LoRA adapter ID.</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#52525b; margin-bottom:20px;">* Forward adapters require <code>texts</code>. Reverse adapters require <code>embeddings_b64</code> and optionally accept <code>texts</code> for quality routing.</p>

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Response</div>
        ${codeBlock("json", `{
  "id": "emb_a1b2c3d4",
  "model": "minilm-te3-adapted",
  "embeddings_b64": "base64...",
  "n": 5,
  "dim": 3072,
  "quality_scores": [0.82, 0.67, 0.91, 0.45, 0.73],
  "usage": {
    "tokens": 340,
    "adapted": 5,
    "reembedded": 0,
    "cost": 0.000022
  }
}`)}
        <table class="table mono" style="font-size:12px; margin-top:16px; margin-bottom:20px;">
          <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">id</td><td>string</td><td>Unique request ID</td></tr>
            <tr><td class="green">model</td><td>string</td><td>Model used</td></tr>
            <tr><td class="green">embeddings_b64</td><td>string</td><td>Base64-encoded float32 array (n × dim)</td></tr>
            <tr><td class="green">n</td><td>integer</td><td>Number of embeddings returned</td></tr>
            <tr><td class="green">dim</td><td>integer</td><td>Embedding dimensions (3072 for TE3-adapted, 4096 for Qwen3-8B-adapted, 384 for raw MiniLM)</td></tr>
            <tr><td class="green">quality_scores</td><td>float[]</td><td>Per-text confidence (only if include_quality=true)</td></tr>
            <tr><td class="green">usage.tokens</td><td>integer</td><td>Tokens consumed</td></tr>
            <tr><td class="green">usage.adapted</td><td>integer</td><td>Texts handled by local adapter</td></tr>
            <tr><td class="green">usage.reembedded</td><td>integer</td><td>Texts routed to OpenAI</td></tr>
            <tr><td class="green">usage.cost</td><td>float</td><td>USD cost for this request</td></tr>
          </tbody>
        </table>

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Decode Embeddings</div>
        ${codeBlock("python", `import base64, numpy as np

embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])  # (5, 3072)`)}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Full Examples</h3></div>
      <div class="card-body">
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

resp = requests.post("${API_DOCS_BASE}/v1/embed",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["Quantum computing uses qubits"],
        "model": "minilm-te3-adapted",
        "quality": 0,
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")     # (1, 3072)
print(f"Cost: \${data['usage']['cost']}")` },
          { lang:"javascript", label:"JavaScript", code: `const resp = await fetch("${API_DOCS_BASE}/v1/embed", {
  method: "POST",
  headers: {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    texts: ["Quantum computing uses qubits"],
    model: "minilm-te3-adapted",
    quality: 0,
  }),
});

const data = await resp.json();
console.log(data.n, data.dim);  // 1, 3072` },
          { lang:"bash", label:"cURL", code: `curl ${API_DOCS_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Quantum computing uses qubits"],
    "model": "minilm-te3-adapted",
    "quality": 0
  }'` },
        ])}
      </div>
    </div>
  `;
}

// ── Quality Routing ──
function renderDocQuality() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Quality Routing</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Adaptive routing between local adapters and OpenAI.</p>

    <div class="card">
      <div class="card-header"><h3>How It Works</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          A lightweight quality head predicts how well the local adapter will handle each text.
          The <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality</code> parameter
          sets the confidence threshold — texts below it get re-embedded via OpenAI for full TE3 fidelity.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:16px;">
          <thead><tr><th>quality</th><th>Behavior</th><th>Cost</th><th>Accuracy</th></tr></thead>
          <tbody>
            <tr><td class="green">0</td><td>All adapted locally</td><td>Lowest</td><td>~93-97% of TE3</td></tr>
            <tr><td class="green">25-75</td><td>Hybrid routing</td><td>Medium</td><td>~97-99% of TE3</td></tr>
            <tr><td class="green">100</td><td>All via OpenAI</td><td>Highest</td><td>100% TE3 native</td></tr>
          </tbody>
        </table>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests

url = "${API_DOCS_BASE}/v1/embed"
h = {"Authorization": "Bearer YOUR_API_KEY", "Content-Type": "application/json"}

# Cheapest — all local
resp = requests.post(url, headers=h, json={
    "texts": texts, "model": "minilm-te3-adapted", "quality": 0
})

# Balanced — hybrid routing
resp = requests.post(url, headers=h, json={
    "texts": texts, "model": "minilm-te3-adapted", "quality": 50
})

# Maximum accuracy — all OpenAI
resp = requests.post(url, headers=h, json={
    "texts": texts, "model": "minilm-te3-adapted", "quality": 100
})` },
          { lang:"bash", label:"cURL", code: `# Cheapest — quality=0
curl ${API_DOCS_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["Hello"],"model":"minilm-te3-adapted","quality":0}'

# Balanced — quality=50
curl ${API_DOCS_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["Hello"],"model":"minilm-te3-adapted","quality":50}'` },
        ])}
      </div>
    </div>
  `;
}

// ── Batch Processing ──
function renderDocBatch() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Batch Processing</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Async API for large-scale embedding jobs.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/batch/submit</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Submit 1K–100K texts for asynchronous processing. Returns a job ID to poll.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:20px;">
          <thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">texts</td><td>string[]</td><td>✓</td><td>Array of texts (1K–100K)</td></tr>
            <tr><td class="green">model</td><td>string</td><td></td><td>Model ID</td></tr>
            <tr><td class="green">quality</td><td>integer</td><td></td><td>Quality routing threshold</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>GET /v1/batch/:id</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          Poll job status. Returns <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">pending</code>,
          <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">processing</code>,
          <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">done</code>, or
          <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">failed</code>.
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>GET /v1/batch/:id/results</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Retrieve completed embeddings. Supports <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">format=binary</code> or <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">format=json</code>.
        </p>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, time, numpy as np

url = "${API_DOCS_BASE}"
h = {"Authorization": "Bearer YOUR_API_KEY", "Content-Type": "application/json"}

# 1. Submit
resp = requests.post(f"{url}/v1/batch/submit", headers=h, json={
    "texts": large_list, "model": "minilm-te3-adapted", "quality": 0
})
job_id = resp.json()["id"]

# 2. Poll
while True:
    s = requests.get(f"{url}/v1/batch/{job_id}?api_key=KEY").json()
    if s["status"] in ("done", "failed"): break
    time.sleep(5)

# 3. Retrieve as binary
r = requests.get(f"{url}/v1/batch/{job_id}/results?api_key=KEY&format=binary")
embs = np.frombuffer(r.content, dtype=np.float32).reshape(-1, 3072)` },
          { lang:"bash", label:"cURL", code: `# 1. Submit
curl -X POST ${API_DOCS_BASE}/v1/batch/submit \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["text1","text2","..."],"model":"minilm-te3-adapted"}'

# 2. Poll
curl "${API_DOCS_BASE}/v1/batch/JOB_ID?api_key=YOUR_API_KEY"

# 3. Retrieve
curl "${API_DOCS_BASE}/v1/batch/JOB_ID/results?api_key=YOUR_API_KEY&format=binary" -o embeddings.bin` },
        ])}
      </div>
    </div>
  `;
}

// ── Custom Adapters ──
function renderDocAdapters() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Custom Adapters</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Train LoRA adapters on your own data.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/adapters</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Create a custom LoRA adapter. The system auto-trains weights when you embed with quality routing enabled.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:20px;">
          <thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">source_model</td><td>string</td><td>✓</td><td>Base encoder (e.g. all-MiniLM-L6-v2)</td></tr>
            <tr><td class="green">target_model</td><td>string</td><td>✓</td><td>Target space (e.g. text-embedding-3-large)</td></tr>
            <tr><td class="green">name</td><td>string</td><td></td><td>Human-readable name</td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `resp = requests.post(f"{url}/v1/adapters", headers=h, json={
    "source_model": "all-MiniLM-L6-v2",
    "target_model": "text-embedding-3-large",
    "name": "my-domain-adapter",
})
adapter_id = resp.json()["id"]`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>GET /v1/adapters</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7;">
          List all your adapters. Returns an array with each adapter's ID, name, source/target models, pair count, LoRA generation, and calibration status.
        </p>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Using an Adapter</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Pass <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">adapter_id</code> when embedding. With quality &gt; 0, the system collects training pairs automatically.
        </p>
        ${codeBlock("python", `resp = requests.post(f"{url}/v1/embed", headers=h, json={
    "texts": texts,
    "model": "minilm-te3-adapted",
    "adapter_id": adapter_id,
    "quality": 50,
})`)}
      </div>
    </div>
  `;
}

// ── Models Reference ──
function renderDocModelsRef() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Models Reference</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Available models and their specifications.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Forward Adapters — Source → TE3</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-bottom:16px;">Run a local model, get TE3-compatible embeddings. Send <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">texts</code> and receive 3072-d vectors that drop into any TE3 index.</p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>minilm-te3-adapted</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">MiniLM → TE3</div>
          <div class="mono" style="color:#3b82f6; font-weight:700;">$0.065 /1M tokens</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>all-MiniLM-L6-v2 (22M params, 384d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>text-embedding-3-large (3072d)</td></tr>
            <tr><td style="color:#52525b;">Adapter</td><td>27M param LoRA projection</td></tr>
            <tr><td style="color:#52525b;">Speed</td><td>18,000 tok/s on a single GPU</td></tr>
            <tr><td style="color:#52525b;">Quality</td><td>93-97% of native TE3</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>3072</td></tr>
            <tr><td style="color:#52525b;">Input</td><td><code>texts</code></td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Fastest adapted model. Best for high-throughput indexing, RAG retrieval, classification, and clustering.</p>
        ${codeBlock("bash", \`curl \${API_DOCS_BASE}/v1/embed \\\\
  -H "Authorization: Bearer YOUR_KEY" \\\\
  -d '{"texts": ["your query"], "model": "minilm-te3-adapted", "quality": 0}'\`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>qwen06b-te3-adapted</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">Qwen3-0.6B → TE3</div>
          <div class="mono" style="color:#3b82f6; font-weight:700;">$0.040 /1M tokens</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>Qwen3-Embedding-0.6B (600M params, 1024d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>text-embedding-3-large (3072d)</td></tr>
            <tr><td style="color:#52525b;">Speed</td><td>1,200 tok/s on a single GPU</td></tr>
            <tr><td style="color:#52525b;">Quality</td><td>95-98% of native TE3</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>3072</td></tr>
            <tr><td style="color:#52525b;">Input</td><td><code>texts</code></td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Higher-fidelity adapted model. Richer semantics, longer contexts, multilingual. Ideal for complex document retrieval, legal/medical search, and cross-lingual embedding.</p>
        ${codeBlock("bash", \`curl \${API_DOCS_BASE}/v1/embed \\\\
  -H "Authorization: Bearer YOUR_KEY" \\\\
  -d '{"texts": ["your query"], "model": "qwen06b-te3-adapted", "quality": 30}'\`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>Reverse Adapters — TE3 → Target</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-bottom:8px;">You already have TE3 embeddings. Send them to the adapter to translate into a different model's vector space. This lets you search a corpus indexed with a different model without re-embedding.</p>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Reverse adapters accept <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">embeddings_b64</code> (base64-encoded float32 TE3 3072d vectors) instead of <code>texts</code>. Optionally include <code>texts</code> for quality routing to a local fallback model.</p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>te3-qwen3-8b-adapted</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">TE3 → Qwen3-Embedding-8B</div>
          <div class="mono" style="color:#3b82f6; font-weight:700;">$0.080 /1M tokens</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>text-embedding-3-large (3072d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>Qwen3-Embedding-8B (4096d)</td></tr>
            <tr><td style="color:#52525b;">Adapter</td><td>TylerF/text-embedding-3-large-to-qwen3-embedding-8b</td></tr>
            <tr><td style="color:#52525b;">Quality routing</td><td>Falls back to local quantized Qwen3-8B (4-bit)</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>4096</td></tr>
            <tr><td style="color:#52525b;">Input</td><td><code>embeddings_b64</code> (required) + <code>texts</code> (optional, for routing)</td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-bottom:12px;">Reverse adapter. Translates pre-computed TE3 embeddings into Qwen3-8B's vector space. Use this when your corpus is indexed with Qwen3-8B and you want to query it using TE3 embeddings.</p>
        <p style="font-size:13px; color:#71717a; line-height:1.6; margin-bottom:12px;">If you include <code>texts</code> and set <code>quality > 0</code>, low-confidence texts are re-embedded with a local quantized Qwen3-8B instead of routing to an external API.</p>

        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: \`import requests, base64, numpy as np

# You already have TE3 embeddings (e.g. from OpenAI)
te3_embeddings = np.random.randn(5, 3072).astype(np.float32)

resp = requests.post("\${API_DOCS_BASE}/v1/embed",
    headers={"Authorization": "Bearer YOUR_KEY"},
    json={
        "model": "te3-qwen3-8b-adapted",
        "embeddings_b64": base64.b64encode(te3_embeddings.tobytes()).decode(),
        # Optional: include texts for quality routing to local Qwen3-8B
        "texts": ["query 1", "query 2", "query 3", "query 4", "query 5"],
        "quality": 30,
    })

data = resp.json()
qwen_embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {qwen_embs.shape}")  # (5, 4096)\` },
          { lang:"bash", label:"cURL", code: \`# First base64-encode your TE3 embeddings
# Then send them to the adapter

curl \${API_DOCS_BASE}/v1/embed \\\\
  -H "Authorization: Bearer YOUR_KEY" \\\\
  -H "Content-Type: application/json" \\\\
  -d '{
    "model": "te3-qwen3-8b-adapted",
    "embeddings_b64": "<base64-encoded float32 TE3 3072d vectors>",
    "texts": ["optional texts for routing"],
    "quality": 30
  }'\` },
        ])}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>all-MiniLM-L6-v2</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">MiniLM Raw</div>
          <div class="mono" style="color:#3b82f6; font-weight:700;">$0.010 /1M tokens</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>all-MiniLM-L6-v2 (22M params, 384d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>— (no adapter)</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>384</td></tr>
            <tr><td style="color:#52525b;">Input</td><td><code>texts</code></td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Raw 384-d embeddings, no adapter. Not TE3-compatible. Cheapest option for prototyping.</p>
      </div>
    </div>
  `;
}

// ── Errors ──
function renderDocErrors() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Errors</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Error response format and codes.</p>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Error Format</h3></div>
      <div class="card-body">
        ${codeBlock("json", `{
  "error": {
    "type": "insufficient_balance",
    "message": "Balance $0.0012 insufficient."
  }
}`)}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Error Codes</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Status</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="yellow">400</td><td class="red">texts_required</td><td>Missing or empty texts array</td></tr>
            <tr><td class="yellow">400</td><td class="red">batch_too_large</td><td>Exceeds 8192 texts per request</td></tr>
            <tr><td class="yellow">400</td><td class="red">unknown_model</td><td>Invalid model name</td></tr>
            <tr><td class="yellow">401</td><td class="red">authentication</td><td>Missing or invalid API key</td></tr>
            <tr><td class="yellow">402</td><td class="red">insufficient_balance</td><td>Not enough credits</td></tr>
            <tr><td class="yellow">429</td><td class="red">rate_limited</td><td>Too many requests — back off and retry</td></tr>
            <tr><td class="yellow">500</td><td class="red">internal_error</td><td>Server error — retry or contact support</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

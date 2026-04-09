// ── Documentation (standalone page) ──

const API_DOCS_BASE = "https://embedding-adapters-api.embedding-adapters.workers.dev";

const docSections = [
  { id: "overview",     icon: "◈", label: "Overview" },
  { id: "embed",        icon: "↗", label: "Embed Texts" },
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
        <div class="nav-logo-icon mono">E</div>
        <span class="nav-logo-text">Embedding Adapters</span>
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">A drop-in replacement for OpenAI embeddings at a fraction of the cost.</p>

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
          Embedding Adapters runs small, fast open-source encoders (MiniLM, Qwen3) and uses trained LoRA adapters
          to project their output into OpenAI's TE3 embedding space. The result: <strong style="color:#10b981;">vectors that are directly
          compatible with your existing TE3 data</strong> at 50-92% lower cost.
        </p>
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          A quality routing system automatically detects texts where the adapter is less confident and can
          selectively re-embed those via OpenAI — giving you a cost/quality dial from 0 (all local) to 100 (all OpenAI).
        </p>
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;">
          <div style="background:#0a0a0f; border:1px solid #1c1c26; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; font-weight:800; color:#10b981; margin-bottom:4px;">50-92%</div>
            <div style="font-size:12px; color:#52525b;">Cost Reduction</div>
          </div>
          <div style="background:#0a0a0f; border:1px solid #1c1c26; border-radius:10px; padding:16px; text-align:center;">
            <div style="font-size:28px; font-weight:800; color:#6366f1; margin-bottom:4px;">3072-d</div>
            <div style="font-size:12px; color:#52525b;">TE3-Compatible Vectors</div>
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
            <div style="min-width:28px; height:28px; border-radius:50%; background:#10b98120; color:#10b981; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">1</div>
            <div><strong style="color:#d4d4d8;">You send texts</strong> to our API with a model choice (e.g. <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">minilm-te3-adapted</code>).</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#10b98120; color:#10b981; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">2</div>
            <div><strong style="color:#d4d4d8;">We encode locally</strong> using the open-source model, then apply a trained LoRA adapter to project embeddings into TE3's 3072-d space.</div>
          </div>
          <div style="display:flex; gap:12px; margin-bottom:16px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#10b98120; color:#10b981; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">3</div>
            <div><strong style="color:#d4d4d8;">Quality routing</strong> (optional): a confidence head scores each text. Below your threshold, we re-embed via OpenAI for full fidelity.</div>
          </div>
          <div style="display:flex; gap:12px; align-items:flex-start;">
            <div style="min-width:28px; height:28px; border-radius:50%; background:#10b98120; color:#10b981; display:flex; align-items:center; justify-content:center; font-size:13px; font-weight:700;">4</div>
            <div><strong style="color:#d4d4d8;">You get back</strong> base64-encoded float32 vectors that drop directly into any TE3 vector store — Pinecone, Weaviate, Qdrant, pgvector, etc.</div>
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

    <div class="card">
      <div class="card-header"><h3>Quick Start</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Create an API key from the <a href="dashboard.html" style="color:#10b981;">dashboard</a>, then embed your first texts:
        </p>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

API_KEY = "YOUR_API_KEY"
url = "${API_DOCS_BASE}/v1/embed"

resp = requests.post(url,
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "texts": ["Embedding adapters save money", "Drop-in TE3 replacement"],
        "model": "minilm-te3-adapted",
        "quality": 0,   # 0 = all local, cheapest
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")         # (2, 3072)
print(f"Cost:  \${data['usage']['cost']}")` },
          { lang:"javascript", label:"JavaScript", code: `const resp = await fetch("${API_DOCS_BASE}/v1/embed", {
  method: "POST",
  headers: {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    texts: ["Embedding adapters save money", "Drop-in TE3 replacement"],
    model: "minilm-te3-adapted",
    quality: 0,
  }),
});

const data = await resp.json();
console.log(data.n, data.dim);  // 2, 3072` },
          { lang:"bash", label:"cURL", code: `curl ${API_DOCS_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Embedding adapters save money", "Drop-in TE3 replacement"],
    "model": "minilm-te3-adapted",
    "quality": 0
  }'` },
        ])}
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
          Send an array of texts, receive base64-encoded float32 embedding vectors. All adapted models return 3072-d vectors compatible with OpenAI <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">text-embedding-3-large</code>.
        </p>

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
            <tr><td class="green">texts</td><td>string[]</td><td>✓</td><td>—</td><td>Array of texts to embed. Max 8192 per request.</td></tr>
            <tr><td class="green">model</td><td>string</td><td></td><td>minilm-te3-adapted</td><td>Model ID. See Models Reference.</td></tr>
            <tr><td class="green">quality</td><td>integer</td><td></td><td>0</td><td>0–100. Controls quality routing threshold.</td></tr>
            <tr><td class="green">include_quality</td><td>boolean</td><td></td><td>false</td><td>If true, returns per-text quality confidence scores.</td></tr>
            <tr><td class="green">adapter_id</td><td>string</td><td></td><td>—</td><td>Custom LoRA adapter ID.</td></tr>
          </tbody>
        </table>

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
            <tr><td class="green">dim</td><td>integer</td><td>Embedding dimensions (3072 for adapted, 384 for raw)</td></tr>
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
      <div class="card-header"><h3>minilm-te3-adapted</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">MiniLM → TE3</div>
          <div class="mono" style="color:#10b981; font-weight:700;">$0.065 /1M — 50% cheaper</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>all-MiniLM-L6-v2 (22M params, 384d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>text-embedding-3-large (3072d)</td></tr>
            <tr><td style="color:#52525b;">Adapter</td><td>27M param LoRA projection</td></tr>
            <tr><td style="color:#52525b;">Quality</td><td>93-97% of native TE3</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>3072</td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Fastest adapted model. Best for high-throughput semantic search, RAG retrieval, classification, and clustering.</p>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>qwen06b-te3-adapted</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">Qwen3-0.6B → TE3</div>
          <div class="mono" style="color:#10b981; font-weight:700;">$0.040 /1M — 69% cheaper</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>Qwen3-Embedding-0.6B (600M params, 1024d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>text-embedding-3-large (3072d)</td></tr>
            <tr><td style="color:#52525b;">Quality</td><td>95-98% of native TE3</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>3072</td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Higher-fidelity adapted model. The 600M-param Qwen3 encoder captures richer semantics, longer contexts, and multilingual text. Ideal for complex document retrieval, legal/medical search, cross-lingual embedding, and fine-grained similarity.</p>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>all-MiniLM-L6-v2</h3></div>
      <div class="card-body">
        <div class="flex justify-between items-center" style="margin-bottom:12px;">
          <div style="font-size:16px; font-weight:700;">MiniLM Raw</div>
          <div class="mono" style="color:#10b981; font-weight:700;">$0.010 /1M — 92% cheaper</div>
        </div>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <tbody>
            <tr><td style="color:#52525b; width:140px;">Source</td><td>all-MiniLM-L6-v2 (22M params, 384d)</td></tr>
            <tr><td style="color:#52525b;">Target</td><td>— (no projection)</td></tr>
            <tr><td style="color:#52525b;">Quality</td><td>MiniLM native</td></tr>
            <tr><td style="color:#52525b;">Output dim</td><td>384</td></tr>
          </tbody>
        </table>
        <p style="font-size:13px; color:#71717a; line-height:1.6;">Raw 384-d embeddings, no adapter. Not TE3-compatible. Cheapest option for prototyping, internal tools, and lightweight recommendations.</p>
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

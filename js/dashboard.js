let session = null;
let userData = null;
let currentTab = "overview";

document.addEventListener("DOMContentLoaded", async () => {
  const params = new URLSearchParams(window.location.search);

  // Handle Google OAuth redirect (session token in URL)
  if (params.get("session")) {
    const s = { session_token: params.get("session"), email: params.get("email") || "" };
    setSessionStorage(s);
    window.history.replaceState({}, "", window.location.pathname);
  }

  session = getSession();
  if (!session) { window.location.href = "subscribe.html"; return; }

  await loadUserData();

  // Check for Stripe redirect — billing top-up
  const sid = params.get("session_id");
  if (sid && userData && userData.api_key) {
    await apiFetch("POST", "/v1/billing/confirm", { session_id: sid }, userData.api_key).catch(() => {});
    window.history.replaceState({}, "", window.location.pathname);
    await loadUserData();
  }

  // Check for model purchase redirect
  const purchaseSuccess = params.get("purchase_success");
  if (purchaseSuccess && session) {
    await apiFetch("POST", "/v1/models/confirm", { model_id: purchaseSuccess }, session.session_token).catch(() => {});
    window.history.replaceState({}, "", window.location.pathname);
    await loadUserData();
    currentTab = "models";
  }

  renderSidebar();
  renderTab();
});

async function loadUserData() {
  const me = await apiFetch("GET", "/auth/me", null, session.session_token);
  if (me.error) { clearSession(); window.location.href = "subscribe.html"; return; }
  userData = me;

  // Load purchased models
  try {
    const purchases = await apiFetch("GET", "/v1/models/purchased", null, session.session_token);
    userData.purchased_models = (purchases.models || []).map(m => m.model_id);
  } catch (e) {
    userData.purchased_models = [];
  }
}

function logout() {
  clearSession();
  window.location.href = "subscribe.html";
}

function switchTab(tab) {
  currentTab = tab;
  $$(".sidebar-item").forEach(el => el.classList.toggle("active", el.dataset.tab === tab));
  renderTab();
}

function renderSidebar() {
  const sb = $("#sidebar");
  const mn = $("#mobile-nav");
  const tabs = [
    { id: "overview", icon: "◈", label: "Overview", short: "Home" },
    { id: "apidocs", icon: "↗", label: "API Docs", short: "API" },
    { id: "keys", icon: "⚿", label: "API Keys", short: "Keys" },
    { id: "billing", icon: "◎", label: "Billing", short: "Billing" },
    { id: "models", icon: "◇", label: "Models", short: "Models" },
    { id: "adapters", icon: "◉", label: "Adapters", short: "Adapters" },
  ];

  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <div class="nav-logo-icon"><svg viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="7" fill="#2563eb"/><path d="M7 14h5l3-6 3 12 3-6h5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <span class="nav-logo-text">EmbeddingAdapters</span>
      </a>
    </div>
    <div class="sidebar-nav">
      ${tabs.map(t => `
        <button class="sidebar-item ${t.id === currentTab ? 'active' : ''}" data-tab="${t.id}" onclick="switchTab('${t.id}')">
          <span class="icon">${t.icon}</span> ${t.label}
        </button>
      `).join("")}
      <a href="benchmarks.html" class="sidebar-item" style="text-decoration:none;">
        <span class="icon">▦</span> Benchmarks
      </a>
    </div>
    <div class="sidebar-footer">
      <div class="sidebar-email mono">${escHtml(session.email)}</div>
      <button class="btn-ghost" style="width:100%;" onclick="logout()">Log out</button>
    </div>
  `;

  // Mobile bottom nav
  if (mn) {
    mn.innerHTML = tabs.map(t => `
      <button class="mobile-nav-btn ${t.id === currentTab ? 'active' : ''}" onclick="switchTab('${t.id}')">
        <span class="icon">${t.icon}</span>
        ${t.short}
      </button>
    `).join("");
  }
}

function renderTab() {
  const main = $("#main");
  main.className = "main-content fade-in";
  switch (currentTab) {
    case "overview": renderOverview(main); break;
    case "apidocs": renderApiDocs(main); break;
    case "keys": renderKeys(main); break;
    case "billing": renderBilling(main); break;
    case "models": renderModels(main); break;
    case "adapters": renderAdapters(main); break;
  }
}

// ── Overview ──
function renderOverview(el) {
  const d = userData;
  if (!d.has_api_key) {
    el.innerHTML = `
      <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Welcome!</h2>
      <div class="card fade-up">
        <div class="card-body text-center" style="padding:48px 24px;">
          <div style="font-size:48px; margin-bottom:16px;">🔑</div>
          <h3 style="font-size:20px; font-weight:700; margin-bottom:8px;">Create your API key</h3>
          <p style="color:#71717a; line-height:1.6; margin-bottom:8px; max-width:400px; margin-left:auto; margin-right:auto;">
            Get started with <strong style="color:#3b82f6;">10,000 free tokens</strong> — enough to embed ~100 texts and test every model.
          </p>
          <p style="color:#52525b; font-size:13px; margin-bottom:24px;">No credit card required.</p>
          <button class="btn btn-primary" onclick="createApiKey()" id="create-key-btn" style="font-size:16px; padding:14px 32px;">
            Create API Key — Free Trial
          </button>
        </div>
      </div>
      <div class="card fade-up delay-1" style="margin-top:16px;">
        <div class="card-header"><h3>What you get</h3></div>
        <div class="card-body" style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
          <div>
            <div style="font-size:24px; margin-bottom:8px;">⚡</div>
            <div style="font-size:14px; font-weight:600; margin-bottom:4px;">3 Models</div>
            <div style="font-size:13px; color:#71717a;">MiniLM→TE3, Qwen06→TE3, MiniLM raw</div>
          </div>
          <div>
            <div style="font-size:24px; margin-bottom:8px;">💰</div>
            <div style="font-size:14px; font-weight:600; margin-bottom:4px;">50-92% Cheaper</div>
            <div style="font-size:13px; color:#71717a;">Than OpenAI text-embedding-3-large</div>
          </div>
          <div>
            <div style="font-size:24px; margin-bottom:8px;">🎯</div>
            <div style="font-size:14px; font-weight:600; margin-bottom:4px;">Quality Routing</div>
            <div style="font-size:13px; color:#71717a;">Adaptive quality with per-text confidence</div>
          </div>
        </div>
      </div>
    `;
    return;
  }

  el.innerHTML = `
    <!-- ── API Key bar ── -->
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; flex-wrap:wrap; gap:10px;">
      <div>
        <h2 style="font-size:22px; font-weight:800; margin-bottom:2px;">Dashboard</h2>
        <div style="font-size:12px; color:#52525b;">Balance: <span class="mono" style="color:#3b82f6;">$${d.balance.toFixed(4)}</span> · ${d.total_passages.toLocaleString()} embeddings generated</div>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <div class="mono" style="font-size:11px; color:#3f3f46; background:#08080c; border:1px solid #1c1c26; border-radius:8px; padding:8px 12px; cursor:pointer; max-width:220px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" onclick="copyToClipboard('${userData.api_key}', this)" title="Click to copy">${userData.api_key.slice(0, 12)}...${userData.api_key.slice(-4)}</div>
        <button class="btn-copy" onclick="copyToClipboard('${userData.api_key}', this)">Copy Key</button>
      </div>
    </div>

    <!-- ── POST /v1/embed ── -->
    <div class="card fade-up" style="margin-bottom:16px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;">
        <h3 style="color:#3b82f6;">POST /v1/embed</h3>
        <div style="font-size:12px; color:#71717a;">Generate openai/text-embedding-3-large vectors without OpenAI</div>
      </div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:14px;">
          Send texts, get 3072-d embeddings compatible with openai/text-embedding-3-large. Drop directly into Pinecone, Weaviate, Qdrant, or pgvector. Your key is pre-filled — copy and run.
        </p>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

resp = requests.post("${API_BASE}/v1/embed",
    headers={"Authorization": "Bearer ${userData.api_key}"},
    json={
        "texts": ["How do neural networks learn?"],
        "model": "qwen06b-te3-adapted",
        "quality": 0,
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(embs.shape)  # (1, 3072)` },
          { lang:"bash", label:"cURL", code: `curl -X POST ${API_BASE}/v1/embed \\
  -H "Authorization: Bearer ${userData.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["How do neural networks learn?"],"model":"qwen06b-te3-adapted","quality":0}'` },
        ])}
        <div style="display:flex; gap:8px; margin-top:12px; flex-wrap:wrap;">
          <button class="btn-sm btn-secondary" onclick="switchTab('apidocs')">Full API Docs →</button>
          <button class="btn-sm btn-ghost" onclick="switchTab('billing')">Add Credits</button>
        </div>
      </div>
    </div>

    <!-- ── API Models ── -->
    <div class="card fade-up delay-1" style="margin-bottom:16px;">
      <div class="card-header"><h3>API Models</h3></div>
      <div class="card-body" style="padding:14px 20px;">
        <div style="display:grid; grid-template-columns:1fr auto auto; gap:6px 16px; font-size:12px; align-items:center;">
          <div class="mono green" style="font-weight:700;">qwen06b-te3-adapted</div>
          <div style="color:#71717a;">Qwen3-0.6B → openai/text-embedding-3-large · 3072d</div>
          <div class="mono" style="color:#10b981;">$0.040/1M</div>

          <div class="mono green" style="font-weight:700;">minilm-te3-adapted</div>
          <div style="color:#71717a;">MiniLM-L6-v2 → openai/text-embedding-3-large · 3072d</div>
          <div class="mono" style="color:#10b981;">$0.065/1M</div>

          <div class="mono green" style="font-weight:700;">te3-qwen3-8b-adapted</div>
          <div style="color:#71717a;">openai/text-embedding-3-large → Qwen3-8B · 4096d</div>
          <div class="mono" style="color:#10b981;">$0.080/1M</div>
        </div>
        <div style="font-size:11px; color:#52525b; margin-top:10px;">OpenAI text-embedding-3-large costs $0.130/1M tokens. These adapters produce the same vectors for 50-69% less.</div>
      </div>
    </div>

    <!-- ── Local SDK ── -->
    <div class="card fade-up delay-2" style="margin-bottom:16px; border-color:#10b98120;">
      <div class="card-header" style="border-color:#10b98120;">
        <h3 style="color:#10b981;">Run Locally — $10/month</h3>
        <button class="btn-sm btn-primary" onclick="switchTab('models')">Subscribe</button>
      </div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:12px;">
          Unlimited inference on your own GPU. No per-token cost. No API calls.
        </p>
        ${codeBlock("python", `pip install embedding-adapters

from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

src = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-large",
    flavor="v2", device="cuda",
)

embs = src.encode(["your text"], normalize_embeddings=True)
out = adapter.encode_embeddings(embs)
print(out.shape)  # (1, 3072)`)}
        <div style="font-size:12px; color:#71717a; margin-top:10px;">
          Available v2 adapters:<br>
          <span class="mono" style="font-size:11px;">sentence-transformers/all-MiniLM-L6-v2 → openai/text-embedding-3-large (3072d)</span><br>
          <span class="mono" style="font-size:11px;">Qwen/Qwen3-Embedding-0.6B → openai/text-embedding-3-large (3072d)</span>
        </div>
      </div>
    </div>

    <!-- ── Migrate off OpenAI ── -->
    <div class="card fade-up delay-3" style="margin-bottom:16px; border-color:#f59e0b20;">
      <div class="card-header" style="border-color:#f59e0b20;">
        <h3 style="color:#f59e0b;">Migrate off OpenAI</h3>
      </div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:12px;">
          Already have an index built with openai/text-embedding-3-large? Translate those vectors into open-source Qwen3-Embedding-8B space — no re-embedding your corpus.
        </p>
        ${codeBlock("python", `resp = requests.post("${API_BASE}/v1/embed",
    headers={"Authorization": "Bearer ${userData.api_key}"},
    json={
        "model": "te3-qwen3-8b-adapted",
        "embeddings_b64": base64.b64encode(your_te3_vectors.tobytes()).decode(),
        "texts": ["query"],
        "quality": 0,
    })
# Returns 4096-d Qwen3-8B vectors from your existing TE3 embeddings`)}
      </div>
    </div>
  `;
}

// ── Create API Key ──
async function createApiKey() {
  const btn = $("#create-key-btn");
  if (btn) { btn.textContent = "Creating..."; btn.disabled = true; }
  try {
    const data = await apiFetch("POST", "/v1/keys/create", null, session.session_token);
    if (data.error) { alert(data.error.message); if (btn) { btn.textContent = "Create API Key — Free Trial"; btn.disabled = false; } return; }
    // Update userData and session
    userData.api_key = data.api_key;
    userData.has_api_key = true;
    userData.balance = data.balance;
    setSessionStorage(session);
    await loadUserData();
    // Show the key
    currentTab = "keys";
    renderSidebar();
    renderTab();
  } catch (e) {
    alert("Failed to create key. Try again.");
    if (btn) { btn.textContent = "Create API Key — Free Trial"; btn.disabled = false; }
  }
}

// ── Keys ──
let keyVisible = false;
function toggleKey() { keyVisible = !keyVisible; renderTab(); }

function renderKeys(el) {
  if (!userData.has_api_key) {
    el.innerHTML = `
      <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">API Keys</h2>
      <div class="card">
        <div class="card-body text-center" style="padding:40px 20px;">
          <div style="font-size:32px; margin-bottom:12px;">🔑</div>
          <div style="font-size:15px; font-weight:600; margin-bottom:8px;">No API key yet</div>
          <p style="font-size:13px; color:#52525b; margin-bottom:20px;">Create your key to start embedding texts.</p>
          <button class="btn btn-primary" onclick="createApiKey()" id="create-key-btn">Create API Key — 10K Free Tokens</button>
        </div>
      </div>
    `;
    return;
  }

  const apiKey = userData.api_key;
  const masked = apiKey.slice(0, 10) + "•".repeat(24) + apiKey.slice(-4);
  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">API Keys</h2>
    <div class="card">
      <div class="card-header">
        <h3>Your API Key</h3>
        <div class="flex gap-8">
          <button class="btn-copy" onclick="toggleKey()">${keyVisible ? "Hide" : "Reveal"}</button>
          <button class="btn-copy" onclick="copyToClipboard('${userData.api_key}', this)">Copy</button>
        </div>
      </div>
      <div class="card-body">
        <div class="mono" style="font-size:14px; color:${keyVisible ? "#3b82f6" : "#3f3f46"};
          word-break:break-all; user-select:${keyVisible ? "all" : "none"};
          padding:12px 16px; background:#08080c; border-radius:8px; border:1px solid #1c1c26;">
          ${keyVisible ? escHtml(userData.api_key) : masked}
        </div>
        <p style="font-size:13px; color:#52525b; margin-top:12px; line-height:1.6;">
          Use this key in the <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">Authorization: Bearer</code> header.
        </p>
      </div>
    </div>
    <div class="card" style="margin-top:16px;">
      <div class="card-header"><h3>Usage Examples</h3></div>
      <div class="card-body">
        <div style="margin-bottom:20px;">
          <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:6px;">Python</div>
          ${codeBlock("python", `import requests, base64, numpy as np

resp = requests.post("${API_BASE}/v1/embed",
    headers={"Authorization": "Bearer ${keyVisible ? userData.api_key : "sk-ea-..."}"},
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
print(f"Cost: \${data['usage']['cost']}")`)}
        </div>
        <div style="margin-bottom:20px;">
          <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:6px;">JavaScript</div>
          ${codeBlock("javascript", `const resp = await fetch("${API_BASE}/v1/embed", {
  method: "POST",
  headers: {
    "Authorization": "Bearer ${keyVisible ? userData.api_key : "sk-ea-..."}",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    texts: ["Quantum computing uses qubits"],
    model: "minilm-te3-adapted",
    quality: 0,
  }),
});

const data = await resp.json();
console.log(data.n, data.dim);  // 1, 3072`)}
        </div>
        <div>
          <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:6px;">cURL</div>
          ${codeBlock("bash", `curl ${API_BASE}/v1/embed \\
  -H "Authorization: Bearer ${keyVisible ? userData.api_key : "sk-ea-..."}" \\
  -H "Content-Type: application/json" \\
  -d '{
    "texts": ["Quantum computing uses qubits"],
    "model": "minilm-te3-adapted",
    "quality": 0
  }'`)}
        </div>
      </div>
    </div>
  `;
}

// ── Billing ──
function renderBilling(el) {
  if (!userData.has_api_key) {
    el.innerHTML = `<h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Billing</h2>
      <div class="card"><div class="card-body text-center" style="padding:32px;">
        <p style="color:#71717a;">Create an API key first to manage billing.</p>
        <button class="btn btn-primary" style="margin-top:16px;" onclick="switchTab('overview')">Create API Key</button>
      </div></div>`;
    return;
  }
  const d = userData;
  const tokensLeft = Math.floor(d.balance / 0.065 * 1_000_000);
  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Billing</h2>
    <div class="grid-2" style="margin-bottom:24px;">
      <div class="stat fade-up">
        <div class="stat-label">Current Balance</div>
        <div class="stat-value mono" style="font-size:32px; color:#3b82f6;">$${d.balance.toFixed(4)}</div>
        <div class="stat-sub mono">≈ ${tokensLeft.toLocaleString()} tokens remaining</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">Lifetime Spend</div>
        <div class="stat-value mono" style="font-size:32px; color:#f59e0b;">$${d.total_spent.toFixed(4)}</div>
        <div class="stat-sub mono">${d.total_passages.toLocaleString()} passages embedded</div>
      </div>
    </div>
    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Add Credits</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#71717a; margin-bottom:16px;">Payments processed securely via Stripe.</p>
        <div class="flex gap-8" style="flex-wrap:wrap; margin-bottom:16px;">
          ${[5, 10, 25, 50, 100].map(v => `
            <button class="deposit-btn" data-amt="${v}" onclick="selectDeposit(${v})">$${v}</button>
          `).join("")}
        </div>
        <div class="flex gap-12">
          <div style="position:relative; flex:1;">
            <span class="mono" style="position:absolute; left:14px; top:50%; transform:translateY(-50%); color:#52525b; font-size:15px;">$</span>
            <input type="number" id="deposit-input" min="1" placeholder="Amount" class="input mono"
              style="padding-left:28px;" oninput="onDepositInput()">
          </div>
          <button id="deposit-submit" class="btn btn-primary" onclick="submitDeposit()" disabled>Pay with Stripe →</button>
        </div>
      </div>
    </div>
    <div class="card fade-up delay-3" style="margin-top:16px;">
      <div class="card-header"><h3>Pricing</h3></div>
      <div class="card-body">
        <table class="table mono">
          <thead><tr><th>Model</th><th>Rate /1M tokens</th><th>vs OpenAI TE3</th></tr></thead>
          <tbody>
            <tr><td>minilm-te3-adapted</td><td class="green">$0.065</td><td class="green" style="font-weight:600;">50% cheaper</td></tr>
            <tr><td>qwen06b-te3-adapted</td><td class="green">$0.040</td><td class="green" style="font-weight:600;">69% cheaper</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td class="green">$0.010</td><td class="green" style="font-weight:600;">92% cheaper</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function selectDeposit(amt) {
  const inp = $("#deposit-input");
  inp.value = amt;
  $$(".deposit-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.amt) === amt));
  $("#deposit-submit").disabled = false;
}

function onDepositInput() {
  const val = parseFloat($("#deposit-input").value);
  $$(".deposit-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.amt) === val));
  $("#deposit-submit").disabled = !val || val < 1;
}

async function submitDeposit() {
  const amt = parseFloat($("#deposit-input").value);
  if (!amt || amt < 1) return;
  const btn = $("#deposit-submit");
  btn.textContent = "Processing...";
  btn.disabled = true;
  const d = await apiFetch("POST", "/v1/billing/deposit", { amount: amt }, userData.api_key);
  if (d.checkout_url) window.location.href = d.checkout_url;
  else { btn.textContent = "Pay with Stripe →"; btn.disabled = false; }
}

// ── Models ──
function renderModels(el) {
  const models = [
    { id:"minilm-te3-adapted", name:"sentence-transformers/all-MiniLM-L6-v2 → openai/text-embedding-3-large", shortName:"MiniLM → TE3-large", src:"384d → 3072d", speed:"18,000 tok/s", quality:"93-97%", price:10, period:"month", stripe_price:"price_minilm_te3_monthly", desc:"Fastest adapter. 18K tok/s throughput. Best for high-volume indexing and RAG.", paid:true },
    { id:"qwen06b-te3-adapted", name:"Qwen/Qwen3-Embedding-0.6B → openai/text-embedding-3-large", shortName:"Qwen3-0.6B → TE3-large", src:"1024d → 3072d", speed:"1,200 tok/s", quality:"95-98%", price:10, period:"month", stripe_price:"price_qwen06_te3_monthly", desc:"Higher accuracy. Richer semantics, multilingual, longer contexts.", paid:true },
    { id:"all-MiniLM-L6-v2", name:"sentence-transformers/all-MiniLM-L6-v2 (raw)", shortName:"MiniLM Raw", src:"384d", speed:"18,000 tok/s", quality:"Native", price:0, period:null, stripe_price:null, desc:"Raw MiniLM embeddings. No adapter. Free forever.", paid:false },
  ];

  // Check which models user has active subscriptions for
  const purchased = userData.purchased_models || [];
  models.forEach(m => {
    if (!m.paid || purchased.includes(m.id)) m.active = true;
    else m.active = false;
  });

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:8px;">Models</h2>
    <p style="font-size:14px; color:#a1a1aa; margin-bottom:24px; line-height:1.6;">Subscribe to adapter models for local inference via the Python SDK. Cancel anytime.</p>

    ${models.map((m, i) => `
      <div class="card fade-up delay-${i + 1}" style="margin-bottom:16px; ${m.active ? 'border-color:#3b82f630;' : ''}">
        <div class="card-body">
          <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap;">
            <div style="flex:1; min-width:200px;">
              <div style="font-size:15px; font-weight:700; margin-bottom:2px;">${m.shortName}</div>
              <div class="mono" style="font-size:11px; color:#71717a; margin-bottom:8px; word-break:break-all;">${m.name}</div>
              <p style="font-size:13px; color:#a1a1aa; line-height:1.5; margin-bottom:10px;">${m.desc}</p>
              <div style="display:flex; gap:16px; font-size:12px; color:#71717a; flex-wrap:wrap;">
                <span>Dims: <span style="color:#a1a1aa;">${m.src}</span></span>
                <span>Speed: <span style="color:#a1a1aa;">${m.speed}</span></span>
                <span>Quality: <span style="color:#a1a1aa;">${m.quality}</span></span>
              </div>
            </div>
            <div style="text-align:right; min-width:130px;">
              ${m.active ? `
                <div style="color:#10b981; font-weight:700; font-size:14px; margin-bottom:4px;">✓ ${m.paid ? 'Subscribed' : 'Free'}</div>
                ${m.paid ? `<div style="font-size:11px; color:#71717a; margin-bottom:8px;">$${m.price}/month · auto-renews</div>` : ''}
                <div style="font-size:12px; color:#71717a; margin-top:4px;">Ready to use in SDK</div>
              ` : `
                <div class="mono" style="font-size:22px; font-weight:800; color:#e4e4e7; margin-bottom:2px;">$${m.price}</div>
                <div style="font-size:11px; color:#71717a; margin-bottom:10px;">per month · cancel anytime</div>
                <button class="btn btn-primary" style="font-size:13px; padding:8px 20px;" onclick="subscribeModel('${m.id}', '${m.stripe_price}')">Subscribe</button>
              `}
            </div>
          </div>
        </div>
      </div>
    `).join("")}

    <div class="card fade-up delay-4" style="margin-top:8px;">
      <div class="card-header"><h3>How it works</h3></div>
      <div class="card-body" style="font-size:13px; color:#a1a1aa; line-height:1.7;">
        <div style="margin-bottom:8px;"><strong style="color:#d4d4d8;">1.</strong> Subscribe to an adapter above</div>
        <div style="margin-bottom:8px;"><strong style="color:#d4d4d8;">2.</strong> Log in with your API key: <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">embedding-adapters login</code></div>
        <div style="margin-bottom:8px;"><strong style="color:#d4d4d8;">3.</strong> Load the adapter — the SDK validates your license automatically (cached 24hrs)</div>
        <div style="margin-bottom:8px;"><strong style="color:#d4d4d8;">4.</strong> Run completely offline after validation. No per-token metering.</div>
        <div style="color:#71717a; margin-top:12px;">Cancel anytime from Stripe. Adapter stops loading when the subscription expires.</div>
      </div>
    </div>

    <div class="card fade-up delay-5" style="margin-top:16px;">
      <div class="card-header"><h3>Usage</h3></div>
      <div class="card-body">
        ${codeBlock("bash", `pip install embedding-adapters
embedding-adapters login  # paste your API key`)}
        ${codeBlock("python", `from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

src = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-large",
    flavor="large",
    device="cuda",
)
# License validated automatically — runs offline for 24hrs

embs = src.encode(["your text"], normalize_embeddings=True)
translated = adapter.encode_embeddings(embs)
print(translated.shape)  # (1, 3072)`)}
      </div>
    </div>
  `;
}

async function subscribeModel(modelId, stripePriceId) {
  try {
    const resp = await apiFetch("POST", "/v1/models/purchase", {
      model_id: modelId,
      price_id: stripePriceId,
    }, session.session_token);

    if (resp.error) { alert(resp.error.message); return; }
    if (resp.checkout_url) {
      window.location.href = resp.checkout_url;
    }
  } catch (e) {
    alert("Failed to start checkout. Please try again.");
  }
}

// ── API Docs ──
function renderApiDocs(el) {
  const key = userData?.api_key || "YOUR_API_KEY";
  const base = API_BASE;

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:8px;">API Documentation</h2>
    <p style="font-size:14px; color:#a1a1aa; margin-bottom:24px; line-height:1.6;">Your API key is pre-filled in all examples below. Copy and run.</p>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>POST /v1/embed</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:16px;">
          Send texts (or pre-computed embeddings for reverse adapters), receive base64-encoded float32 vectors.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:16px;">
          <thead><tr><th>Field</th><th>Type</th><th>Default</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">texts</td><td>string[]</td><td>—</td><td>Texts to embed. Required for forward adapters.</td></tr>
            <tr><td class="green">model</td><td>string</td><td>qwen06b-te3-adapted</td><td>Model ID (see below).</td></tr>
            <tr><td class="green">quality</td><td>int</td><td>0</td><td>0–100. Quality routing threshold.</td></tr>
            <tr><td class="green">embeddings_b64</td><td>string</td><td>—</td><td>Base64 float32 embeddings. Required for reverse adapters.</td></tr>
            <tr><td class="green">include_quality</td><td>bool</td><td>false</td><td>Return per-text quality scores.</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>qwen06b-te3-adapted</h3></div>
      <div class="card-body">
        <div style="display:flex; justify-content:space-between; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div>
            <div style="font-size:15px; font-weight:700;">Qwen/Qwen3-Embedding-0.6B → openai/text-embedding-3-large</div>
            <div style="font-size:12px; color:#71717a;">1024d → 3072d · 1,200 tok/s · $0.040/1M tokens · Higher accuracy</div>
          </div>
        </div>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

resp = requests.post("${base}/v1/embed",
    headers={"Authorization": "Bearer ${key}"},
    json={
        "texts": ["Quantum computing uses qubits for parallel computation"],
        "model": "qwen06b-te3-adapted",
        "quality": 30,   # route lowest-confidence texts to provider
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")  # (1, 3072)
print(f"Adapted: {data['usage']['adapted']}, Routed: {data['usage']['reembedded']}")` },
          { lang:"bash", label:"cURL", code: `curl -X POST ${base}/v1/embed \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Quantum computing uses qubits"], "model": "qwen06b-te3-adapted", "quality": 30}'` },
        ])}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>minilm-te3-adapted</h3></div>
      <div class="card-body">
        <div style="display:flex; justify-content:space-between; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div>
            <div style="font-size:15px; font-weight:700;">sentence-transformers/all-MiniLM-L6-v2 → openai/text-embedding-3-large</div>
            <div style="font-size:12px; color:#71717a;">384d → 3072d · 18,000 tok/s · $0.065/1M tokens · Fastest</div>
          </div>
        </div>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

resp = requests.post("${base}/v1/embed",
    headers={"Authorization": "Bearer ${key}"},
    json={
        "texts": ["NASA discovers new exoplanet", "Help me find my keys"],
        "model": "minilm-te3-adapted",
        "quality": 0,
    })

data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")         # (2, 3072)
print(f"Cost:  \${data['usage']['cost']}")` },
          { lang:"bash", label:"cURL", code: `curl -X POST ${base}/v1/embed \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["NASA discovers new exoplanet"], "model": "minilm-te3-adapted", "quality": 0}'` },
        ])}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>te3-qwen3-8b-adapted</h3></div>
      <div class="card-body">
        <div style="display:flex; justify-content:space-between; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div>
            <div style="font-size:15px; font-weight:700;">openai/text-embedding-3-large → Qwen/Qwen3-Embedding-8B</div>
            <div style="font-size:12px; color:#71717a;">3072d → 4096d · Reverse adapter · Send embeddings_b64 instead of texts</div>
          </div>
        </div>
        ${tabbedCodeBlock([
          { lang:"python", label:"Python", code: `import requests, base64, numpy as np

# You already have TE3 embeddings (e.g. from OpenAI)
te3_embs = np.random.randn(3, 3072).astype(np.float32)

resp = requests.post("${base}/v1/embed",
    headers={"Authorization": "Bearer ${key}"},
    json={
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

print(f"Shape: {qwen_embs.shape}")  # (3, 4096)` },
          { lang:"bash", label:"cURL", code: `curl -X POST ${base}/v1/embed \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "te3-qwen3-8b-adapted", "embeddings_b64": "<base64>", "quality": 30}'` },
        ])}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>all-MiniLM-L6-v2 (raw)</h3></div>
      <div class="card-body">
        <div style="display:flex; justify-content:space-between; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
          <div>
            <div style="font-size:15px; font-weight:700;">sentence-transformers/all-MiniLM-L6-v2 — No adapter</div>
            <div style="font-size:12px; color:#71717a;">384d · $0.010/1M tokens · Not TE3-compatible · For prototyping</div>
          </div>
        </div>
        ${codeBlock("bash", `curl -X POST ${base}/v1/embed \\
  -H "Authorization: Bearer ${key}" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["Hello world"], "model": "all-MiniLM-L6-v2"}'`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px; border-color:#3b82f630;">
      <div class="card-header" style="border-color:#3b82f620;"><h3>Python SDK — Local Inference</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:16px;">
          Run adapters on your own GPU. No API calls after license validation (cached 24hrs).
        </p>
        ${codeBlock("bash", `pip install embedding-adapters
embedding-adapters login   # paste your API key: ${key.slice(0, 12)}...`)}

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin:16px 0 8px;">v2: MiniLM → openai/text-embedding-3-large ($10/month)</div>
        ${codeBlock("python", `from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

src = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-large",
    flavor="v2",
    device="cuda",
)

embs = src.encode(["Hello world"], normalize_embeddings=True)
translated = adapter.encode_embeddings(embs)
print(translated.shape)  # (1, 3072)

# Built-in quality head
_, quality = adapter.score_v2(embs)
print(quality)  # [0.74]`)}

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin:16px 0 8px;">v1: MiniLM → openai/text-embedding-3-small (free)</div>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="openai/text-embedding-3-small",
    flavor="large",
    device="cuda",
)

translated = adapter.encode_embeddings(embs)
print(translated.shape)  # (1, 1536)`)}

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin:16px 0 8px;">v1: E5-base-v2 → openai/text-embedding-3-small (free)</div>
        ${codeBlock("python", `from sentence_transformers import SentenceTransformer
from embedding_adapters import EmbeddingAdapter

src = SentenceTransformer("intfloat/e5-base-v2")
adapter = EmbeddingAdapter.from_registry(
    source="intfloat/e5-base-v2",
    target="openai/text-embedding-3-small",
    flavor="small",
    device="cuda",
)

embs = src.encode(["Hello world"], normalize_embeddings=True)
translated = adapter.encode_embeddings(embs)
print(translated.shape)  # (1, 1536)`)}

        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin:16px 0 8px;">v1: MiniLM → intfloat/e5-base-v2 (free)</div>
        ${codeBlock("python", `adapter = EmbeddingAdapter.from_registry(
    source="sentence-transformers/all-MiniLM-L6-v2",
    target="intfloat/e5-base-v2",
    flavor="large",
    device="cuda",
)

translated = adapter.encode_embeddings(embs)
print(translated.shape)  # (1, 768)`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Routing</h3></div>
      <div class="card-body">
        <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:12px;">
          The <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality</code> parameter controls how aggressively low-confidence texts get re-embedded by the provider.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:12px;">
          <thead><tr><th>quality</th><th>Behavior</th></tr></thead>
          <tbody>
            <tr><td class="green">0</td><td>Everything local. Zero provider calls. Cheapest.</td></tr>
            <tr><td class="green">30</td><td>~8-21% routed to provider. Good balance.</td></tr>
            <tr><td class="green">50</td><td>~17-56% routed. Higher quality.</td></tr>
            <tr><td class="green">70</td><td>~44-82% routed. Near-provider quality.</td></tr>
            <tr><td class="green">100</td><td>Everything routed to provider.</td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `resp = requests.post("${base}/v1/embed",
    headers={"Authorization": "Bearer ${key}"},
    json={
        "texts": ["complex medical terminology query"],
        "model": "qwen06b-te3-adapted",
        "quality": 50,
        "include_quality": True,
    })

data = resp.json()
print(data["quality_scores"])  # [0.42] — this one was routed to OpenAI`)}
      </div>
    </div>

    <div class="card" style="margin-bottom:20px;">
      <div class="card-header"><h3>Decode Response</h3></div>
      <div class="card-body">
        ${codeBlock("python", `import base64, numpy as np

# Response contains base64-encoded float32 vectors
data = resp.json()
embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])

print(f"Shape: {embs.shape}")              # (N, 3072) or (N, 4096)
print(f"Tokens: {data['usage']['tokens']}")
print(f"Cost: \${data['usage']['cost']}")
print(f"Adapted: {data['usage']['adapted']}")
print(f"Routed: {data['usage']['reembedded']}")`)}
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Errors</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Status</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td style="color:#f59e0b;">401</td><td style="color:#ef4444;">authentication</td><td>Missing or invalid API key</td></tr>
            <tr><td style="color:#f59e0b;">402</td><td style="color:#ef4444;">insufficient_balance</td><td>Add credits in Billing tab</td></tr>
            <tr><td style="color:#f59e0b;">400</td><td style="color:#ef4444;">validation</td><td>Invalid request body (empty texts, bad model, etc.)</td></tr>
            <tr><td style="color:#f59e0b;">429</td><td style="color:#ef4444;">rate_limit</td><td>Too many requests. Slow down.</td></tr>
            <tr><td style="color:#f59e0b;">500</td><td style="color:#ef4444;">internal</td><td>Server error. Try again.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ── Adapters ──
async function renderAdapters(el) {
  el.innerHTML = `<h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Your Adapters</h2>
    <div style="color:#52525b;">Loading...</div>`;

  const adapters = await apiFetch("GET", "/v1/adapters", null, userData.api_key);

  if (!Array.isArray(adapters) || adapters.length === 0) {
    el.innerHTML = `
      <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Your Adapters</h2>
      <div class="card">
        <div class="card-body text-center" style="padding:40px 20px;">
          <div style="font-size:32px; margin-bottom:12px;">◇</div>
          <div style="font-size:15px; font-weight:600; margin-bottom:8px;">No adapters yet</div>
          <p style="font-size:13px; color:#52525b; line-height:1.6;">
            Create a custom LoRA adapter to improve quality for your specific data.
            <br>See the <a href="docs.html#adapters_doc" style="color:#3b82f6;">documentation</a>.
          </p>
        </div>
      </div>
    `;
    return;
  }

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Your Adapters</h2>
    ${adapters.map(a => `
      <div class="card" style="margin-bottom:12px;">
        <div class="card-body">
          <div class="flex justify-between items-center">
            <div>
              <div style="font-size:15px; font-weight:600;">${escHtml(a.name || a.id)}</div>
              <div class="mono" style="font-size:12px; color:#52525b; margin-top:4px;">${escHtml(a.source_model)} → ${escHtml(a.target_model)}</div>
            </div>
            <div class="flex gap-16" style="text-align:right;">
              <div><div class="mono" style="font-size:14px; color:#3b82f6; font-weight:600;">${a.n_pairs ?? 0}</div><div style="font-size:11px; color:#52525b;">pairs</div></div>
              <div><div class="mono" style="font-size:14px; color:#6366f1; font-weight:600;">${a.lora_generation ?? 0}</div><div style="font-size:11px; color:#52525b;">gen</div></div>
              <div><div style="font-size:14px; color:${a.calibrated ? "#3b82f6" : "#52525b"};">${a.calibrated ? "✓" : "—"}</div><div style="font-size:11px; color:#52525b;">calibrated</div></div>
            </div>
          </div>
        </div>
      </div>
    `).join("")}
  `;
}

// ── Code Block Helper ──
function codeBlock(lang, code) {
  const id = "cb_" + Math.random().toString(36).slice(2, 8);
  return `
    <div class="code-wrap">
      <div class="code-actions">
        <span class="mono code-lang">${lang}</span>
        <button class="btn-copy" onclick="copyToClipboard(document.getElementById('${id}').textContent, this)">Copy</button>
      </div>
      <pre class="code-block" id="${id}">${highlightCode(code, lang)}</pre>
    </div>
  `;
}

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

  // Check for Stripe redirect (after loadUserData so we have api_key)
  const sid = params.get("session_id");
  if (sid && userData && userData.api_key) {
    await apiFetch("POST", "/v1/billing/confirm", { session_id: sid }, userData.api_key).catch(() => {});
    window.history.replaceState({}, "", window.location.pathname);
    await loadUserData(); // reload balance
  }

  renderSidebar();
  renderTab();
});

async function loadUserData() {
  const me = await apiFetch("GET", "/auth/me", null, session.session_token);
  if (me.error) { clearSession(); window.location.href = "subscribe.html"; return; }
  userData = me;
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
  const tabs = [
    { id: "overview", icon: "◈", label: "Overview" },
    { id: "keys", icon: "⚿", label: "API Keys" },
    { id: "billing", icon: "◎", label: "Billing" },
    { id: "docs", icon: "◆", label: "Documentation" },
    { id: "models", icon: "◇", label: "Models" },
    { id: "adapters", icon: "◉", label: "Adapters" },
  ];

  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <svg style="width:28px;height:28px;" viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="8" fill="#2563eb"/><path d="M8 10.5C8 10.5 10.5 7 14 7s6 3.5 6 3.5M8 14c0 0 2.5-3.5 6-3.5s6 3.5 6 3.5M8 17.5c0 0 2.5-3.5 6-3.5s6 3.5 6 3.5" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/></svg>
        <span class="nav-logo-text">EmbeddingAdapters</span>
      </a>
    </div>
    <div class="sidebar-nav">
      ${tabs.map(t => `
        <button class="sidebar-item ${t.id === currentTab ? 'active' : ''}" data-tab="${t.id}" onclick="switchTab('${t.id}')">
          <span class="icon">${t.icon}</span> ${t.label}
        </button>
      `).join("")}
    </div>
    <div class="sidebar-footer">
      <div class="sidebar-email mono">${escHtml(session.email)}</div>
      <button class="btn-ghost" style="width:100%;" onclick="logout()">Log out</button>
    </div>
  `;
}

function renderTab() {
  const main = $("#main");
  main.className = "main-content fade-in";
  switch (currentTab) {
    case "overview": renderOverview(main); break;
    case "keys": renderKeys(main); break;
    case "billing": renderBilling(main); break;
    case "docs": renderDocs(main); break;
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
            Get started with <strong style="color:#2563eb;">10,000 free tokens</strong> — enough to embed ~100 texts and test every model.
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

  const lowBalance = d.balance < 0.01;
  const tokensLeft = Math.floor(d.balance / 0.065 * 1_000_000);

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Overview</h2>

    ${lowBalance ? `
    <div class="card fade-up" style="border-color:#d97706; background:linear-gradient(135deg, #fffbeb, #fef3c7); margin-bottom:20px;">
      <div class="card-body" style="display:flex; align-items:center; justify-content:space-between;">
        <div>
          <div style="font-size:15px; font-weight:700; color:#92400e; margin-bottom:4px;">Low balance — ${tokensLeft.toLocaleString()} tokens remaining</div>
          <div style="font-size:13px; color:#a16207;">Add credits to keep embedding. Plans start at $5.</div>
        </div>
        <button class="btn btn-primary" onclick="switchTab('billing')" style="white-space:nowrap;">Add Credits →</button>
      </div>
    </div>
    ` : ''}

    <div class="grid-3" style="margin-bottom:28px;">
      <div class="stat fade-up">
        <div class="stat-label">Balance</div>
        <div class="stat-value mono" style="color:#16a34a;">$${d.balance.toFixed(4)}</div>
        <div class="stat-sub mono">${tokensLeft.toLocaleString()} tokens remaining</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">Total Spent</div>
        <div class="stat-value mono" style="color:#f59e0b;">$${d.total_spent.toFixed(4)}</div>
        <div class="stat-sub mono">Lifetime usage</div>
      </div>
      <div class="stat fade-up delay-2">
        <div class="stat-label">Passages Embedded</div>
        <div class="stat-value mono" style="color:#6366f1;">${d.total_passages.toLocaleString()}</div>
        <div class="stat-sub mono">Total API calls</div>
      </div>
    </div>
    <div class="card fade-up delay-3">
      <div class="card-header"><h3>Quick Start</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:var(--text-secondary); line-height:1.6; margin-bottom:16px;">
          Send texts, get TE3-compatible embeddings. 50-92% cheaper than OpenAI.
        </p>
        ${codeBlock("bash", `curl ${API_BASE}/v1/embed \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["Hello world"],"model":"minilm-te3-adapted"}'`)}
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
        <div class="mono" style="font-size:14px; color:${keyVisible ? "#2563eb" : "#3f3f46"};
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
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Billing & Plans</h2>
    <div class="grid-2" style="margin-bottom:28px;">
      <div class="stat fade-up">
        <div class="stat-label">Current Balance</div>
        <div class="stat-value mono" style="font-size:32px; color:#16a34a;">$${d.balance.toFixed(4)}</div>
        <div class="stat-sub mono">≈ ${tokensLeft.toLocaleString()} tokens remaining</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">Lifetime Spend</div>
        <div class="stat-value mono" style="font-size:32px; color:#d97706;">$${d.total_spent.toFixed(4)}</div>
        <div class="stat-sub mono">${d.total_passages.toLocaleString()} passages embedded</div>
      </div>
    </div>

    <!-- Plans -->
    <div class="grid-3 fade-up delay-2" style="margin-bottom:24px;">
      ${[
        { name:"Starter", price:5, tokens:"77K", texts:"~770", desc:"Test & prototype" },
        { name:"Developer", price:25, tokens:"385K", texts:"~3,850", desc:"Build & ship", popular:true },
        { name:"Growth", price:100, tokens:"1.54M", texts:"~15,400", desc:"Scale production" },
      ].map(p => `
        <div class="card" style="margin-bottom:0; ${p.popular ? 'border:2px solid var(--accent);' : ''}">
          ${p.popular ? '<div style="background:var(--accent); color:#fff; text-align:center; padding:4px; font-size:11px; font-weight:700; letter-spacing:0.05em;">MOST POPULAR</div>' : ''}
          <div class="card-body text-center" style="padding:28px 20px;">
            <div style="font-size:13px; font-weight:600; color:var(--text-muted); margin-bottom:8px;">${p.name}</div>
            <div style="font-size:36px; font-weight:800; color:var(--text); margin-bottom:4px;">$${p.price}</div>
            <div class="mono" style="font-size:13px; color:var(--accent); font-weight:600; margin-bottom:4px;">${p.tokens} tokens</div>
            <div style="font-size:12px; color:var(--text-muted); margin-bottom:16px;">${p.texts} texts · ${p.desc}</div>
            <button class="btn ${p.popular ? 'btn-primary' : 'btn-secondary'}" onclick="selectDeposit(${p.price}); submitDeposit();" style="width:100%; justify-content:center;">
              Buy $${p.price}
            </button>
          </div>
        </div>
      `).join("")}
    </div>

    <!-- Custom amount -->
    <div class="card fade-up delay-3">
      <div class="card-header"><h3>Custom Amount</h3></div>
      <div class="card-body">
        <div class="flex gap-8" style="flex-wrap:wrap; margin-bottom:16px;">
          ${[5, 10, 25, 50, 100, 250, 500].map(v => `
            <button class="deposit-btn" data-amt="${v}" onclick="selectDeposit(${v})">$${v}</button>
          `).join("")}
        </div>
        <div class="flex gap-12">
          <div style="position:relative; flex:1;">
            <span class="mono" style="position:absolute; left:14px; top:50%; transform:translateY(-50%); color:var(--text-muted); font-size:15px;">$</span>
            <input type="number" id="deposit-input" min="1" placeholder="Enter amount" class="input mono"
              style="padding-left:28px;" oninput="onDepositInput()">
          </div>
          <button id="deposit-submit" class="btn btn-primary" onclick="submitDeposit()" disabled>Pay with Stripe →</button>
        </div>
        <div id="deposit-preview" style="margin-top:12px; font-size:13px; color:var(--text-muted);"></div>
      </div>
    </div>

    <!-- Pricing table -->
    <div class="card fade-up delay-4" style="margin-top:16px;">
      <div class="card-header"><h3>Per-Token Rates</h3></div>
      <div class="card-body">
        <table class="table mono">
          <thead><tr><th>Model</th><th>Rate /1M tokens</th><th>OpenAI TE3</th><th>You Save</th></tr></thead>
          <tbody>
            <tr><td>minilm-te3-adapted</td><td style="color:var(--text);">$0.065</td><td style="color:var(--text-muted);">$0.130</td><td class="green" style="font-weight:600;">50%</td></tr>
            <tr><td>qwen06b-te3-adapted</td><td style="color:var(--text);">$0.040</td><td style="color:var(--text-muted);">$0.130</td><td class="green" style="font-weight:600;">69%</td></tr>
            <tr><td>all-MiniLM-L6-v2</td><td style="color:var(--text);">$0.010</td><td style="color:var(--text-muted);">$0.130</td><td class="green" style="font-weight:600;">92%</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

function selectDeposit(amt) {
  const inp = $("#deposit-input");
  if (inp) inp.value = amt;
  $$(".deposit-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.amt) === amt));
  const sub = $("#deposit-submit");
  if (sub) sub.disabled = false;
  updateDepositPreview(amt);
}

function onDepositInput() {
  const val = parseFloat($("#deposit-input").value);
  $$(".deposit-btn").forEach(b => b.classList.toggle("active", parseInt(b.dataset.amt) === val));
  $("#deposit-submit").disabled = !val || val < 1;
  updateDepositPreview(val);
}

function updateDepositPreview(amt) {
  const el = $("#deposit-preview");
  if (!el || !amt || amt < 1) { if(el) el.innerHTML = ""; return; }
  const tokens = Math.floor(amt / 0.065 * 1_000_000);
  const texts = Math.floor(tokens / 13);
  const oaiCost = (tokens * 0.130 / 1_000_000).toFixed(2);
  el.innerHTML = `<span class="mono">≈ ${tokens.toLocaleString()} tokens · ~${texts.toLocaleString()} texts · <span style="color:var(--green);">saves $${(oaiCost - amt).toFixed(2)} vs OpenAI</span></span>`;
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

// ── Docs ──
let docsSection = "embed";
function switchDoc(sec) { docsSection = sec; renderTab(); }

function renderDocs(el) {
  const sections = [
    { id: "embed", label: "Embed Texts" },
    { id: "quality", label: "Quality Routing" },
    { id: "batch", label: "Batch Processing" },
    { id: "adapters_doc", label: "Custom Adapters" },
    { id: "errors", label: "Errors" },
  ];

  let content = "";
  if (docsSection === "embed") content = renderDocEmbed();
  else if (docsSection === "quality") content = renderDocQuality();
  else if (docsSection === "batch") content = renderDocBatch();
  else if (docsSection === "adapters_doc") content = renderDocAdapters();
  else if (docsSection === "errors") content = renderDocErrors();

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Documentation</h2>
    <div class="flex gap-8" style="margin-bottom:24px; flex-wrap:wrap;">
      ${sections.map(s => `<button class="pill ${docsSection === s.id ? 'active' : ''}" onclick="switchDoc('${s.id}')">${s.label}</button>`).join("")}
    </div>
    <div class="fade-in">${content}</div>
  `;
}

function renderDocEmbed() {
  return `
    <div class="card">
      <div class="card-header"><h3>POST /v1/embed</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Send texts, receive embeddings. Returns base64-encoded float32 arrays.
        </p>
        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Request Body</div>
        <table class="table mono" style="font-size:12px; margin-bottom:20px;">
          <thead><tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="green">texts</td><td>string[]</td><td>✓</td><td>List of texts (max 8192)</td></tr>
            <tr><td class="green">model</td><td>string</td><td></td><td>Default: minilm-te3-adapted</td></tr>
            <tr><td class="green">quality</td><td>integer</td><td></td><td>0-100. 0=adapted, 100=OpenAI</td></tr>
            <tr><td class="green">include_quality</td><td>boolean</td><td></td><td>Return quality scores</td></tr>
            <tr><td class="green">adapter_id</td><td>string</td><td></td><td>Custom LoRA adapter</td></tr>
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
        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin:20px 0 8px;">Decode Embeddings</div>
        ${codeBlock("python", `import base64, numpy as np

embs = np.frombuffer(
    base64.b64decode(data["embeddings_b64"]),
    dtype=np.float32
).reshape(data["n"], data["dim"])  # (5, 3072)`)}
      </div>
    </div>
  `;
}

function renderDocQuality() {
  return `
    <div class="card">
      <div class="card-header"><h3>Quality-Based Routing</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          The quality head predicts adaptation fidelity. Low-quality texts get re-embedded via OpenAI.
        </p>
        <table class="table mono" style="font-size:12px; margin-bottom:16px;">
          <thead><tr><th>quality</th><th>Behavior</th><th>Cost</th><th>Accuracy</th></tr></thead>
          <tbody>
            <tr><td class="green">0</td><td>All adapted locally</td><td>Lowest</td><td>~93-97% of TE3</td></tr>
            <tr><td class="green">25-75</td><td>Hybrid routing</td><td>Medium</td><td>~97-99% of TE3</td></tr>
            <tr><td class="green">100</td><td>All via OpenAI</td><td>Highest</td><td>100% TE3 native</td></tr>
          </tbody>
        </table>
        ${codeBlock("python", `# Cheapest
resp = post(url, json={"texts": texts, "model": "minilm-te3-adapted", "quality": 0})

# Balanced
resp = post(url, json={"texts": texts, "model": "minilm-te3-adapted", "quality": 50})

# Maximum accuracy
resp = post(url, json={"texts": texts, "model": "minilm-te3-adapted", "quality": 100})`)}
      </div>
    </div>
  `;
}

function renderDocBatch() {
  return `
    <div class="card">
      <div class="card-header"><h3>Batch Processing</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          For large jobs (1K-100K texts), use the async batch API.
        </p>
        ${codeBlock("python", `# Submit
resp = post(f"{url}/v1/batch/submit", headers=h, json={
    "texts": large_list, "model": "minilm-te3-adapted", "quality": 0
})
job_id = resp.json()["id"]

# Poll
while True:
    s = get(f"{url}/v1/batch/{job_id}?api_key=KEY").json()
    if s["status"] in ("done","failed"): break
    time.sleep(5)

# Retrieve
r = get(f"{url}/v1/batch/{job_id}/results?api_key=KEY&format=binary")
embs = np.frombuffer(r.content, dtype=np.float32).reshape(-1, 3072)`)}
      </div>
    </div>
  `;
}

function renderDocAdapters() {
  return `
    <div class="card">
      <div class="card-header"><h3>Custom LoRA Adapters</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Create adapters that learn from your data. The system auto-trains LoRA weights when you use quality routing.
        </p>
        ${codeBlock("python", `# Create
resp = post(f"{url}/v1/adapters", headers=h, json={
    "source_model": "all-MiniLM-L6-v2",
    "target_model": "text-embedding-3-large",
    "name": "my-adapter",
})
adapter_id = resp.json()["id"]

# Use (quality > 0 triggers learning)
resp = post(f"{url}/v1/embed", headers=h, json={
    "texts": texts, "model": "minilm-te3-adapted",
    "adapter_id": adapter_id, "quality": 50,
})`)}
      </div>
    </div>
  `;
}

function renderDocErrors() {
  return `
    <div class="card">
      <div class="card-header"><h3>Error Codes</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Status</th><th>Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td class="yellow">400</td><td class="red">texts_required</td><td>Missing texts array</td></tr>
            <tr><td class="yellow">400</td><td class="red">batch_too_large</td><td>Exceeds 8192 texts</td></tr>
            <tr><td class="yellow">400</td><td class="red">unknown_model</td><td>Invalid model name</td></tr>
            <tr><td class="yellow">401</td><td class="red">authentication</td><td>Invalid API key</td></tr>
            <tr><td class="yellow">402</td><td class="red">insufficient_balance</td><td>Not enough credits</td></tr>
            <tr><td class="yellow">429</td><td class="red">rate_limited</td><td>Too many requests</td></tr>
          </tbody>
        </table>
        ${codeBlock("json", `{
  "error": {
    "type": "insufficient_balance",
    "message": "Balance $0.0012 insufficient."
  }
}`)}
      </div>
    </div>
  `;
}

// ── Models ──
function renderModels(el) {
  const models = [
    { id:"minilm-te3-adapted", name:"MiniLM → TE3", src:"all-MiniLM-L6-v2 (22M, 384d)", tgt:"text-embedding-3-large (3072d)", rate:"$0.065", save:"50%", desc:"Fastest. Tiny MiniLM encoder + 27M adapter. Best for throughput.", quality:"93-97% of TE3" },
    { id:"qwen06b-te3-adapted", name:"Qwen3-0.6B → TE3", src:"Qwen3-Embedding-0.6B (600M, 1024d)", tgt:"text-embedding-3-large (3072d)", rate:"$0.040", save:"69%", desc:"Higher source quality from 600M-param encoder. Better nuance.", quality:"95-98% of TE3" },
    { id:"all-MiniLM-L6-v2", name:"MiniLM Raw", src:"all-MiniLM-L6-v2 (22M, 384d)", tgt:"—", rate:"$0.010", save:"92%", desc:"Raw 384d embeddings. No TE3 projection. Cheapest option.", quality:"MiniLM native" },
  ];

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Models</h2>
    ${models.map((m, i) => `
      <div class="card fade-up delay-${i + 1}">
        <div class="card-body">
          <div class="flex justify-between" style="align-items:flex-start; margin-bottom:12px;">
            <div>
              <div style="font-size:16px; font-weight:700; margin-bottom:4px;">${m.name}</div>
              <div class="mono" style="font-size:12px; color:#52525b;">${m.id}</div>
            </div>
            <div style="text-align:right;">
              <div class="mono" style="font-size:14px; color:#2563eb; font-weight:700;">${m.rate} /1M</div>
              <div style="font-size:12px; color:#2563eb;">${m.save} cheaper</div>
            </div>
          </div>
          <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:12px;">${m.desc}</p>
          <div class="flex gap-16" style="font-size:12px; color:#52525b;">
            <span>Source: <span style="color:#71717a;">${m.src}</span></span>
            <span>Quality: <span style="color:#71717a;">${m.quality}</span></span>
          </div>
        </div>
      </div>
    `).join("")}
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
            <br>See the <span style="color:#2563eb; cursor:pointer;" onclick="switchDoc('adapters_doc'); switchTab('docs');">documentation</span>.
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
              <div><div class="mono" style="font-size:14px; color:#2563eb; font-weight:600;">${a.n_pairs ?? 0}</div><div style="font-size:11px; color:#52525b;">pairs</div></div>
              <div><div class="mono" style="font-size:14px; color:#6366f1; font-weight:600;">${a.lora_generation ?? 0}</div><div style="font-size:11px; color:#52525b;">gen</div></div>
              <div><div style="font-size:14px; color:${a.calibrated ? "#2563eb" : "#52525b"};">${a.calibrated ? "✓" : "—"}</div><div style="font-size:11px; color:#52525b;">calibrated</div></div>
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
      <pre class="code-block" id="${id}">${escHtml(code)}</pre>
    </div>
  `;
}

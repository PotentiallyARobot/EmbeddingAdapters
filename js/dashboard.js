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
    { id: "models", icon: "◇", label: "Models" },
    { id: "adapters", icon: "◉", label: "Adapters" },
  ];

  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <div class="nav-logo-icon mono">E</div>
        <span class="nav-logo-text">Embedding Adapters</span>
      </a>
    </div>
    <div class="sidebar-nav">
      ${tabs.map(t => `
        <button class="sidebar-item ${t.id === currentTab ? 'active' : ''}" data-tab="${t.id}" onclick="switchTab('${t.id}')">
          <span class="icon">${t.icon}</span> ${t.label}
        </button>
      `).join("")}
      <a href="docs.html" class="sidebar-item" style="text-decoration:none;">
        <span class="icon">◆</span> Documentation
      </a>
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
            Get started with <strong style="color:#10b981;">10,000 free tokens</strong> — enough to embed ~100 texts and test every model.
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
    <h2 style="font-size:22px; font-weight:800; margin-bottom:24px;">Overview</h2>
    <div class="grid-3" style="margin-bottom:28px;">
      <div class="stat fade-up">
        <div class="stat-label">Balance</div>
        <div class="stat-value mono" style="color:#10b981;">$${d.balance.toFixed(4)}</div>
        <div class="stat-sub mono">Available credits</div>
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
        <p style="font-size:14px; color:#71717a; line-height:1.6; margin-bottom:16px;">
          Send texts, get TE3-compatible embeddings. 50-92% cheaper than OpenAI.
        </p>
        <div style="font-size:12px; color:#52525b; margin-bottom:8px;">Your API key is pre-filled below — copy and run:</div>
        ${codeBlock("bash", `curl ${API_BASE}/v1/embed \\
  -H "Authorization: Bearer ${userData.api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{"texts":["Hello world"],"model":"minilm-te3-adapted"}'`)}
        <div style="margin-top:16px;">
          <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:6px;">Your API Key</div>
          <div class="flex gap-8 items-center">
            <div class="mono" style="font-size:13px; color:#10b981; background:#08080c; border:1px solid #1c1c26; border-radius:8px; padding:10px 14px; flex:1; word-break:break-all; user-select:all;">${escHtml(userData.api_key)}</div>
            <button class="btn-copy" onclick="copyToClipboard('${userData.api_key}', this)">Copy</button>
          </div>
        </div>
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
        <div class="mono" style="font-size:14px; color:${keyVisible ? "#10b981" : "#3f3f46"};
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
        <div class="stat-value mono" style="font-size:32px; color:#10b981;">$${d.balance.toFixed(4)}</div>
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
    { id:"minilm-te3-adapted", name:"MiniLM → TE3", src:"all-MiniLM-L6-v2 (22M, 384d)", tgt:"text-embedding-3-large (3072d)", rate:"$0.065", save:"50%", desc:"Fastest adapted model. Tiny 22M-param MiniLM encoder with a 27M-param LoRA adapter projects into TE3's 3072-d space. Best when you need high throughput at low cost and TE3-level compatibility.", quality:"93-97% of TE3", useCases:["Semantic search over large corpora","RAG retrieval pipelines","Near-real-time classification","Clustering & deduplication"] },
    { id:"qwen06b-te3-adapted", name:"Qwen3-0.6B → TE3", src:"Qwen3-Embedding-0.6B (600M, 1024d)", tgt:"text-embedding-3-large (3072d)", rate:"$0.040", save:"69%", desc:"Higher-fidelity adapted model. The 600M-param Qwen3 encoder captures richer semantics — longer contexts, nuanced meaning, multilingual text — then projects to TE3-compatible 3072-d vectors. Ideal when quality matters more than latency.", quality:"95-98% of TE3", useCases:["Complex document retrieval & legal/medical search","Multilingual & cross-lingual embedding","Fine-grained similarity (paraphrase detection, plagiarism)","High-stakes classification where accuracy is critical"] },
    { id:"all-MiniLM-L6-v2", name:"MiniLM Raw", src:"all-MiniLM-L6-v2 (22M, 384d)", tgt:"—", rate:"$0.010", save:"92%", desc:"Raw 384-d MiniLM embeddings with no adapter projection. Cheapest option by far. Use when you don't need TE3 compatibility and just want fast, good-enough embeddings.", quality:"MiniLM native (384d)", useCases:["Prototyping & experimentation","Internal tools with cost constraints","Simple keyword-level similarity","Lightweight recommendations"] },
  ];

  el.innerHTML = `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:8px;">Models</h2>
    <p style="font-size:14px; color:#71717a; margin-bottom:24px; line-height:1.6;">All adapted models output TE3-compatible 3072-d vectors — drop-in replacements for OpenAI <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">text-embedding-3-large</code>.</p>
    ${models.map((m, i) => `
      <div class="card fade-up delay-${i + 1}" style="margin-bottom:16px;">
        <div class="card-body">
          <div class="flex justify-between" style="align-items:flex-start; margin-bottom:12px;">
            <div>
              <div style="font-size:16px; font-weight:700; margin-bottom:4px;">${m.name}</div>
              <div class="mono" style="font-size:12px; color:#52525b;">${m.id}</div>
            </div>
            <div style="text-align:right;">
              <div class="mono" style="font-size:14px; color:#10b981; font-weight:700;">${m.rate} /1M</div>
              <div style="font-size:12px; color:#10b981;">${m.save} cheaper</div>
            </div>
          </div>
          <p style="font-size:13px; color:#a1a1aa; line-height:1.6; margin-bottom:14px;">${m.desc}</p>
          <div style="margin-bottom:14px;">
            <div style="font-size:12px; font-weight:700; color:#52525b; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.05em;">Best for</div>
            <div style="display:flex; flex-wrap:wrap; gap:6px;">
              ${m.useCases.map(u => `<span style="font-size:12px; padding:4px 10px; border-radius:6px; background:#10b98110; color:#10b981; border:1px solid #10b98125;">${u}</span>`).join("")}
            </div>
          </div>
          <div class="flex gap-16" style="font-size:12px; color:#52525b;">
            <span>Source: <span style="color:#71717a;">${m.src}</span></span>
            <span>Target: <span style="color:#71717a;">${m.tgt}</span></span>
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
            <br>See the <a href="docs.html#adapters_doc" style="color:#10b981;">documentation</a>.
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
              <div><div class="mono" style="font-size:14px; color:#10b981; font-weight:600;">${a.n_pairs ?? 0}</div><div style="font-size:11px; color:#52525b;">pairs</div></div>
              <div><div class="mono" style="font-size:14px; color:#6366f1; font-weight:600;">${a.lora_generation ?? 0}</div><div style="font-size:11px; color:#52525b;">gen</div></div>
              <div><div style="font-size:14px; color:${a.calibrated ? "#10b981" : "#52525b"};">${a.calibrated ? "✓" : "—"}</div><div style="font-size:11px; color:#52525b;">calibrated</div></div>
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

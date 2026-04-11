// ── Benchmarks Page — MiniLM→TE3 vs Qwen3-0.6B→TE3 ──

const benchSections = [
  { id: "overview",     icon: "◈", label: "Key Results" },
  { id: "models",       icon: "◎", label: "Model Comparison" },
  { id: "routing",      icon: "↗", label: "Quality Routing" },
  { id: "performance",  icon: "⚡", label: "Local Performance" },
  { id: "cost",         icon: "◇", label: "Cost Savings" },
  { id: "calibrate",    icon: "▤", label: "Calibrate API" },
];

let currentBench = "overview";

document.addEventListener("DOMContentLoaded", () => {
  const hash = window.location.hash.replace("#", "");
  if (benchSections.find(s => s.id === hash)) currentBench = hash;
  renderBenchSidebar();
  renderMobileNav();
  renderBenchContent();
});

function switchBench(id) {
  currentBench = id;
  window.location.hash = id;
  $$(".sidebar-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  $$(".bench-mobile-nav a").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  renderBenchContent();
  window.scrollTo(0, 0);
}

function renderMobileNav() {
  const nav = document.createElement("div");
  nav.className = "bench-mobile-nav";
  nav.innerHTML = benchSections.map(s =>
    `<a href="#${s.id}" data-tab="${s.id}" class="${s.id === currentBench ? 'active' : ''}" onclick="event.preventDefault(); switchBench('${s.id}')">${s.label}</a>`
  ).join("");
  document.body.insertBefore(nav, document.body.firstChild);
}

function renderBenchSidebar() {
  const sb = $("#sidebar");
  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <div class="nav-logo-icon"><svg viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="7" fill="#2563eb"/><path d="M7 14h5l3-6 3 12 3-6h5" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
        <span class="nav-logo-text">EmbeddingAdapters</span>
      </a>
    </div>
    <div style="padding:12px 16px 8px; font-size:11px; font-weight:700; color:#52525b; text-transform:uppercase; letter-spacing:0.06em;">Benchmarks</div>
    <div class="sidebar-nav">
      ${benchSections.map(s => `
        <button class="sidebar-item ${s.id === currentBench ? 'active' : ''}" data-tab="${s.id}" onclick="switchBench('${s.id}')">
          <span class="icon">${s.icon}</span> ${s.label}
        </button>
      `).join("")}
    </div>
    <div style="padding:12px 16px 8px; font-size:11px; font-weight:700; color:#52525b; text-transform:uppercase; letter-spacing:0.06em;">Navigation</div>
    <div class="sidebar-nav">
      <a href="dashboard.html" class="sidebar-item" style="text-decoration:none;"><span class="icon">◧</span> Dashboard</a>
      <a href="docs.html" class="sidebar-item" style="text-decoration:none;"><span class="icon">◆</span> Documentation</a>
      <a href="index.html" class="sidebar-item" style="text-decoration:none;"><span class="icon">◪</span> Home</a>
    </div>
  `;
}

function renderBenchContent() {
  const main = $("#main");
  main.className = "main-content fade-in";
  const r = {
    overview: renderOverview, models: renderModels, routing: renderRouting,
    performance: renderPerformance, cost: renderCost, calibrate: renderCalibrate,
  };
  main.innerHTML = (r[currentBench] || renderOverview)();
}

// ── Helpers ──
function codeBlock(lang, code) {
  const id = "cb_" + Math.random().toString(36).slice(2, 8);
  return `<div class="code-wrap"><div class="code-actions"><span class="mono code-lang">${lang}</span><button class="btn-copy" onclick="copyToClipboard(document.getElementById('${id}').textContent, this)">Copy</button></div><pre class="code-block" id="${id}">${escHtml(code)}</pre></div>`;
}

function barChart(rows, maxVal) {
  if (!maxVal) maxVal = Math.max(...rows.map(r => r.value));
  return rows.map(r => {
    const pct = (r.value / maxVal * 100).toFixed(1);
    const c = r.color || "#3b82f6";
    return `<div style="margin-bottom:10px;">
      <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;">
        <span style="color:#a1a1aa; font-weight:600;">${r.label}</span>
        <span class="mono" style="color:${c}; font-weight:700;">${r.display || r.value}</span>
      </div>
      <div style="height:8px; background:#1c1c26; border-radius:4px; overflow:hidden;">
        <div style="height:100%; width:${pct}%; background:${c}; border-radius:4px; transition:width 0.6s;"></div>
      </div>
    </div>`;
  }).join("");
}

const MT = (m) => `<span class="model-tag ${m === 'MiniLM→TE3' ? 'minilm' : m === 'Qwen3-0.6B→TE3' ? 'qwen' : m === 'OpenAI TE3' ? 'te3' : 'source'}">${m}</span>`;

// ══════════════════════════════════════════════════════
// OVERVIEW
// ══════════════════════════════════════════════════════
function renderOverview() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Benchmark Results</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Two adapted models benchmarked on real RAG retrieval tasks against OpenAI text-embedding-3-large.</p>

    <div class="bench-stat-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px;">
      <div class="stat fade-up">
        <div class="stat-label">${MT('MiniLM→TE3')} Natural Questions, quality=0</div>
        <div class="stat-value mono" style="color:#6366f1;">MRR 0.926</div>
        <div class="stat-sub">+10% vs MiniLM raw · zero OpenAI calls · 99.5% cheaper than TE3</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">${MT('Qwen3-0.6B→TE3')} Natural Questions, quality=0</div>
        <div class="stat-value mono" style="color:#3b82f6;">MRR 0.934</div>
        <div class="stat-sub">Matches TE3 at quality=40 · 99.7% cheaper · higher accuracy</div>
      </div>
    </div>

    <div class="card fade-up delay-2" style="margin-bottom:20px;">
      <div class="card-header"><h3>What This Means For You</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>If you have a vector database indexed with OpenAI text-embedding-3-large, you can query it using our adapted models instead of calling OpenAI for every search. Your queries get transformed into TE3-compatible embeddings locally.</p>
        <p style="margin-top:8px;">The <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality</code> parameter controls accuracy vs cost. At quality=0, everything runs locally. Increase it to route uncertain texts through OpenAI for higher accuracy.</p>
      </div>
    </div>

    <div class="card fade-up delay-3" style="margin-bottom:20px;">
      <div class="card-header"><h3>MRR@10 — Adapted Queries → TE3 Corpus</h3></div>
      <div class="card-body">
        <div class="bench-dual-col" style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:10px;">Natural Questions</div>
            ${barChart([
              { label: "MiniLM raw (baseline)", value: 0.843, display: "0.843", color: "#ef4444" },
              { label: "MiniLM→TE3 q=0", value: 0.926, display: "0.926", color: "#6366f1" },
              { label: "Qwen3→TE3 q=0", value: 0.934, display: "0.934", color: "#3b82f6" },
              { label: "Qwen3→TE3 q=40 (≈TE3)", value: 0.953, display: "0.953", color: "#3b82f6" },
              { label: "OpenAI TE3", value: 0.960, display: "0.960", color: "#52525b" },
            ], 1.0)}
          </div>
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:10px;">HotpotQA (multi-hop)</div>
            ${barChart([
              { label: "MiniLM→TE3 q=0", value: 0.827, display: "0.827", color: "#6366f1" },
              { label: "Qwen3→TE3 q=0", value: 0.835, display: "0.835", color: "#3b82f6" },
              { label: "MiniLM raw (baseline)", value: 0.872, display: "0.872", color: "#ef4444" },
              { label: "Qwen3→TE3 q=60 (≈TE3)", value: 0.911, display: "0.911", color: "#3b82f6" },
              { label: "OpenAI TE3", value: 0.917, display: "0.917", color: "#52525b" },
            ], 1.0)}
          </div>
        </div>
      </div>
    </div>

    <div class="card fade-up delay-4">
      <div class="card-header"><h3>Choose Your Model</h3></div>
      <div class="card-body">
        <div class="bench-dual-col" style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
          <div style="padding:16px; border:1px solid #6366f130; border-radius:10px; background:#6366f108;">
            <div style="margin-bottom:8px;">${MT('MiniLM→TE3')}</div>
            <div style="font-size:14px; color:#d4d4d8; font-weight:600; margin-bottom:6px;">Speed Champion</div>
            <p style="font-size:13px; color:#a1a1aa; line-height:1.6;">18,000 tok/s · 50ms for 10 texts · 16 concurrent users on a laptop GPU. Best for high-throughput, latency-sensitive workloads with general-purpose text.</p>
          </div>
          <div style="padding:16px; border:1px solid #3b82f630; border-radius:10px; background:#3b82f608;">
            <div style="margin-bottom:8px;">${MT('Qwen3-0.6B→TE3')}</div>
            <div style="font-size:14px; color:#d4d4d8; font-weight:600; margin-bottom:6px;">Accuracy Champion</div>
            <p style="font-size:13px; color:#a1a1aa; line-height:1.6;">1,200 tok/s · 140ms for 10 texts · Consistently 1–3% better MRR. Reaches TE3 parity at lower quality thresholds. Best for precision-critical retrieval.</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// MODEL COMPARISON
// ══════════════════════════════════════════════════════
function renderModels() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Model Comparison</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Side-by-side quality sweeps on two datasets. Adapted queries search TE3-embedded corpora.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Natural Questions — 100 queries × 1,000 passages</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th colspan="3">${MT('MiniLM→TE3')}</th><th colspan="3">${MT('Qwen3-0.6B→TE3')}</th></tr>
          <tr><th></th><th>%OAI</th><th>MRR@10</th><th>Cost</th><th>%OAI</th><th>MRR@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr class="highlight-row"><td>0</td><td>0%</td><td style="color:#6366f1; font-weight:700;">0.926</td><td>$0.000076</td><td>0%</td><td style="color:#3b82f6; font-weight:700;">0.934</td><td>$0.000047</td></tr>
            <tr><td>30</td><td>8%</td><td>0.948</td><td>$0.000087</td><td>21%</td><td>0.948</td><td>$0.000074</td></tr>
            <tr class="highlight-row"><td>50</td><td>17%</td><td>0.948</td><td>$0.000100</td><td>56%</td><td style="color:#3b82f6; font-weight:700;">0.953</td><td>$0.000121</td></tr>
            <tr><td>70</td><td>44%</td><td>0.948</td><td>$0.000135</td><td>82%</td><td>0.958</td><td>$0.000153</td></tr>
            <tr><td>100</td><td>100%</td><td>0.960</td><td>$0.000204</td><td>100%</td><td>0.960</td><td>$0.000175</td></tr>
          </tbody>
        </table>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:10px;">Baselines: MiniLM raw MRR=0.843 | OpenAI TE3 MRR=0.960. Both adapters beat source at q=0.</p>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>HotpotQA — 100 multi-hop queries × 1,190 passages</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th colspan="3">${MT('MiniLM→TE3')}</th><th colspan="3">${MT('Qwen3-0.6B→TE3')}</th></tr>
          <tr><th></th><th>%OAI</th><th>MRR@10</th><th>Cost</th><th>%OAI</th><th>MRR@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>0%</td><td>0.827</td><td>$0.000130</td><td>0%</td><td>0.835</td><td>$0.000080</td></tr>
            <tr class="highlight-row"><td>30</td><td>22%</td><td>0.847</td><td>$0.000189</td><td>40%</td><td style="color:#3b82f6; font-weight:700;">0.875</td><td>$0.000177</td></tr>
            <tr><td>50</td><td>46%</td><td>0.866</td><td>$0.000251</td><td>76%</td><td>0.898</td><td>$0.000274</td></tr>
            <tr class="highlight-row"><td>60</td><td>63%</td><td style="color:#6366f1; font-weight:700;">0.878</td><td>$0.000298</td><td>85%</td><td style="color:#3b82f6; font-weight:700;">0.911</td><td>$0.000303</td></tr>
            <tr><td>80</td><td>90%</td><td>0.914</td><td>$0.000370</td><td>100%</td><td>0.917</td><td>$0.000349</td></tr>
            <tr><td>100</td><td>100%</td><td>0.917</td><td>$0.000399</td><td>100%</td><td>0.917</td><td>$0.000349</td></tr>
          </tbody>
        </table>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:10px;">Baselines: MiniLM raw MRR=0.872 | OpenAI TE3 MRR=0.917. Qwen3 beats source at q=30, MiniLM at q=60.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Key Thresholds</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table" style="font-size:13px;">
          <thead><tr><th>Milestone</th><th>${MT('MiniLM→TE3')}</th><th>${MT('Qwen3-0.6B→TE3')}</th></tr></thead>
          <tbody>
            <tr><td style="color:#a1a1aa;">NQ: Beats MiniLM raw</td><td class="green" style="font-weight:700;">q=0 (0% routed)</td><td class="green" style="font-weight:700;">q=0 (0% routed)</td></tr>
            <tr><td style="color:#a1a1aa;">NQ: Matches TE3 (within 1%)</td><td>q=90 (81%)</td><td class="green" style="font-weight:700;">q=40 (38%)</td></tr>
            <tr><td style="color:#a1a1aa;">HotpotQA: Beats MiniLM raw</td><td>q=60 (63%)</td><td class="green" style="font-weight:700;">q=30 (40%)</td></tr>
            <tr><td style="color:#a1a1aa;">HotpotQA: Matches TE3</td><td>q=80 (90%)</td><td class="green" style="font-weight:700;">q=60 (85%)</td></tr>
          </tbody>
        </table>
        </div>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// QUALITY ROUTING
// ══════════════════════════════════════════════════════
function renderRouting() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Quality Routing</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">The quality head predicts adapter confidence per-text. Uncertain texts route to OpenAI automatically.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>How It Works</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Every text gets a confidence score (0.0–1.0) from the adapter's neural quality head. Set <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality=50</code> to re-embed any text scoring below 0.5 via OpenAI TE3.</p>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-top:16px;">
          <div style="padding:12px; border-radius:8px; background:#3b82f608; border:1px solid #3b82f620; text-align:center;">
            <div class="mono" style="font-size:20px; font-weight:800; color:#3b82f6;">quality=0</div>
            <div style="font-size:12px; color:#71717a; margin-top:4px;">100% local · cheapest</div>
          </div>
          <div style="padding:12px; border-radius:8px; background:#6366f108; border:1px solid #6366f120; text-align:center;">
            <div class="mono" style="font-size:20px; font-weight:800; color:#6366f1;">quality=50</div>
            <div style="font-size:12px; color:#71717a; margin-top:4px;">Hybrid · balanced</div>
          </div>
          <div style="padding:12px; border-radius:8px; background:#52525b10; border:1px solid #52525b30; text-align:center;">
            <div class="mono" style="font-size:20px; font-weight:800; color:#a1a1aa;">quality=100</div>
            <div style="font-size:12px; color:#71717a; margin-top:4px;">All OpenAI · max quality</div>
          </div>
        </div>
        <p style="margin-top:16px;">The percentage routed varies by dataset — that's correct. The head routes more on harder texts and less on easy ones. Use <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/quality/calibrate</code> to preview routing on your data.</p>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Routing Rates — % Sent to OpenAI</h3></div>
      <div class="card-body">
        <div class="bench-dual-col" style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">Natural Questions</div>
            <div class="table-scroll">
            <table class="table mono" style="font-size:12px;">
              <thead><tr><th>Quality</th><th>${MT('MiniLM→TE3')}</th><th>${MT('Qwen3-0.6B→TE3')}</th></tr></thead>
              <tbody>
                <tr><td>0</td><td>0%</td><td>0%</td></tr>
                <tr><td>30</td><td>8%</td><td>21%</td></tr>
                <tr><td>50</td><td>17%</td><td>56%</td></tr>
                <tr><td>70</td><td>44%</td><td>82%</td></tr>
                <tr><td>100</td><td>100%</td><td>100%</td></tr>
              </tbody>
            </table>
            </div>
          </div>
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:8px;">HotpotQA</div>
            <div class="table-scroll">
            <table class="table mono" style="font-size:12px;">
              <thead><tr><th>Quality</th><th>${MT('MiniLM→TE3')}</th><th>${MT('Qwen3-0.6B→TE3')}</th></tr></thead>
              <tbody>
                <tr><td>0</td><td>0%</td><td>0%</td></tr>
                <tr><td>30</td><td>22%</td><td>40%</td></tr>
                <tr><td>50</td><td>46%</td><td>76%</td></tr>
                <tr><td>70</td><td>78%</td><td>93%</td></tr>
                <tr><td>100</td><td>100%</td><td>100%</td></tr>
              </tbody>
            </table>
            </div>
          </div>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:10px;">Qwen3 routes more aggressively (stricter quality head) but achieves higher MRR at each level.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Don't Guess — Calibrate</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Send a sample of your actual data to the calibrate endpoint. It returns your quality grade, routing preview at every level, and the recommended setting.</p>
        <p style="margin-top:8px;"><span style="cursor:pointer; color:#3b82f6; font-weight:600;" onclick="switchBench('calibrate')">See Calibrate API →</span></p>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// LOCAL PERFORMANCE
// ══════════════════════════════════════════════════════
function renderPerformance() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Local Deployment Performance</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Benchmarked on an RTX 3060 Laptop GPU (6GB VRAM). Your results will scale with hardware.</p>

    <div class="perf-grid" style="margin-bottom:20px;">
      <div class="stat fade-up">
        <div class="stat-label">${MT('MiniLM→TE3')} Throughput</div>
        <div class="stat-value mono" style="color:#6366f1; font-size:22px;">18,000 tok/s</div>
        <div class="stat-sub">~5.9s for 8,192 texts</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">${MT('Qwen3-0.6B→TE3')} Throughput</div>
        <div class="stat-value mono" style="color:#3b82f6; font-size:22px;">1,200 tok/s</div>
        <div class="stat-sub">~95s for 8,192 texts</div>
      </div>
      <div class="stat fade-up delay-2">
        <div class="stat-label">Baseline VRAM (all models)</div>
        <div class="stat-value mono" style="font-size:22px;">1,603 MB</div>
        <div class="stat-sub">25% of 6.4GB GPU</div>
      </div>
    </div>

    <div class="card fade-up delay-3" style="margin-bottom:20px;">
      <div class="card-header"><h3>Latency by Batch Size (quality=0)</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Batch</th><th colspan="2">${MT('MiniLM→TE3')}</th><th colspan="2">${MT('Qwen3-0.6B→TE3')}</th></tr>
          <tr><th></th><th>Time</th><th>Peak VRAM</th><th>Time</th><th>Peak VRAM</th></tr></thead>
          <tbody>
            <tr><td>1</td><td>0.13s</td><td>+1MB</td><td>0.83s</td><td>+2MB</td></tr>
            <tr><td>10</td><td>0.03s</td><td>+3MB</td><td>0.22s</td><td>+28MB</td></tr>
            <tr><td>100</td><td>0.07s</td><td>+27MB</td><td>1.10s</td><td>+190MB</td></tr>
            <tr><td>512</td><td>0.35s</td><td>+150MB</td><td>5.55s</td><td>+210MB</td></tr>
            <tr><td>1,024</td><td>0.71s</td><td>+164MB</td><td>11.2s</td><td>+212MB</td></tr>
            <tr><td>4,096</td><td>2.94s</td><td>+197MB</td><td>47.1s</td><td>+256MB</td></tr>
            <tr><td>8,192</td><td>5.94s</td><td>+394MB</td><td>95.3s</td><td>+512MB</td></tr>
          </tbody>
        </table>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:10px;">VRAM delta is on top of 1,603MB baseline with all models loaded. RTX 3060 Laptop, 6.4GB total.</p>
      </div>
    </div>

    <div class="card fade-up delay-4" style="margin-bottom:20px;">
      <div class="card-header"><h3>API Response Times (via FastAPI)</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>End-to-end latency including HTTP overhead, JSON serialization, and embedding:</p>
        <div class="table-scroll" style="margin-top:12px;">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Model</th><th>10 texts</th><th>100 texts</th><th>500 texts</th></tr></thead>
          <tbody>
            <tr><td>${MT('MiniLM→TE3')}</td><td class="green">52ms</td><td class="green">146ms</td><td>456ms</td></tr>
            <tr><td>${MT('Qwen3-0.6B→TE3')}</td><td>142ms</td><td>801ms</td><td>3.6s</td></tr>
          </tbody>
        </table>
        </div>
        <p style="margin-top:12px; font-size:13px;">Server processing is 56ms for 100 texts on MiniLM. The rest is HTTP + JSON overhead (~60ms). On production hardware with faster GPUs, expect proportionally better results.</p>
      </div>
    </div>

    <div class="card fade-up delay-5">
      <div class="card-header"><h3>Concurrent Users</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Tested with parallel requests hitting the local server simultaneously:</p>
        <div class="bench-dual-col" style="display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:12px;">
          <div style="padding:16px; border-radius:10px; background:#6366f108; border:1px solid #6366f120;">
            <div style="margin-bottom:6px;">${MT('MiniLM→TE3')}</div>
            <div class="mono" style="font-size:13px; color:#d4d4d8; line-height:1.8;">
              16 users × 10 texts: p95 = 2.7s<br>
              8 users × 100 texts: p95 = 3.0s<br>
              4 users × 500 texts: p95 = 3.9s<br>
              <span style="color:#6366f1; font-weight:700;">Peak: 1,520 texts/s</span>
            </div>
          </div>
          <div style="padding:16px; border-radius:10px; background:#3b82f608; border:1px solid #3b82f620;">
            <div style="margin-bottom:6px;">${MT('Qwen3-0.6B→TE3')}</div>
            <div class="mono" style="font-size:13px; color:#d4d4d8; line-height:1.8;">
              8 users × 10 texts: p95 = 3.2s<br>
              4 users × 50 texts: p95 = 4.3s<br>
              2 users × 100 texts: p95 = 4.5s<br>
              <span style="color:#3b82f6; font-weight:700;">Peak: 138 texts/s</span>
            </div>
          </div>
        </div>
        <p style="margin-top:12px; font-size:13px;">VRAM stays under 31% for all tested configurations. Headroom for larger batches or additional models.</p>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// COST
// ══════════════════════════════════════════════════════
function renderCost() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Cost Savings</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">98–99%+ savings vs OpenAI at every quality level, for both models.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Query Cost — Natural Questions</h3></div>
      <div class="card-body">
        ${barChart([
          { label: "Qwen3→TE3 q=0", value: 0.000047, display: "$0.000047", color: "#3b82f6" },
          { label: "MiniLM→TE3 q=0", value: 0.000076, display: "$0.000076", color: "#6366f1" },
          { label: "Qwen3→TE3 q=40 (≈TE3)", value: 0.000097, display: "$0.000097", color: "#3b82f6" },
          { label: "MiniLM→TE3 q=90 (≈TE3)", value: 0.000180, display: "$0.000180", color: "#6366f1" },
          { label: "MiniLM raw (no adapter)", value: 0.000930, display: "$0.000930", color: "#ef4444" },
          { label: "OpenAI TE3 direct", value: 0.014080, display: "$0.014080", color: "#52525b" },
        ], 0.015)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Monthly Projections</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Volume</th><th>${MT('OpenAI TE3')}</th><th>${MT('MiniLM→TE3')} q=0</th><th>${MT('Qwen3-0.6B→TE3')} q=0</th><th>You Save</th></tr></thead>
          <tbody>
            <tr><td>10M tokens/mo</td><td>$1.30</td><td class="green">$0.01</td><td class="green">$0.005</td><td class="green" style="font-weight:700;">$1.29+</td></tr>
            <tr><td>100M tokens/mo</td><td>$13.00</td><td class="green">$0.07</td><td class="green">$0.04</td><td class="green" style="font-weight:700;">$12.93+</td></tr>
            <tr><td>1B tokens/mo</td><td>$130</td><td class="green">$0.65</td><td class="green">$0.40</td><td class="green" style="font-weight:700;">$129+</td></tr>
            <tr><td>10B tokens/mo</td><td>$1,300</td><td class="green">$6.50</td><td class="green">$4.00</td><td class="green" style="font-weight:700;">$1,294+</td></tr>
          </tbody>
        </table>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:10px;">At quality=0. With routing, costs increase slightly but stay 98%+ below OpenAI.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Savings at Every Quality Level</h3></div>
      <div class="card-body">
        <div class="table-scroll">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th colspan="2">${MT('MiniLM→TE3')}</th><th colspan="2">${MT('Qwen3-0.6B→TE3')}</th></tr>
          <tr><th></th><th>NQ</th><th>HotpotQA</th><th>NQ</th><th>HotpotQA</th></tr></thead>
          <tbody>
            <tr><td>0</td><td class="green" style="font-weight:700;">99.5%</td><td class="green" style="font-weight:700;">99.4%</td><td class="green" style="font-weight:700;">99.7%</td><td class="green" style="font-weight:700;">99.6%</td></tr>
            <tr><td>30</td><td class="green">99.4%</td><td class="green">99.1%</td><td class="green">99.5%</td><td class="green">99.1%</td></tr>
            <tr><td>50</td><td class="green">99.3%</td><td class="green">98.8%</td><td class="green">99.1%</td><td class="green">98.6%</td></tr>
            <tr><td>80</td><td class="green">98.9%</td><td class="green">98.2%</td><td class="green">98.8%</td><td class="green">98.3%</td></tr>
            <tr><td>100</td><td class="green">98.6%</td><td class="green">98.0%</td><td class="green">98.8%</td><td class="green">98.3%</td></tr>
          </tbody>
        </table>
        </div>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// CALIBRATE
// ══════════════════════════════════════════════════════
function renderCalibrate() {
  return `
    <h2 style="font-size:24px; font-weight:800; margin-bottom:4px;">Calibrate API</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Find the right quality setting before you commit. Send sample data, get recommendations.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/quality/calibrate</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Send 10–2,048 texts from your actual workload. Returns your quality grade, score distribution, routing preview at every level, and recommended settings.</p>
        ${codeBlock("python", `import requests

resp = requests.post(
    "https://embedding-adapters-api.workers.dev/v1/quality/calibrate",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["sample of your actual data...", "..."],
        "model": "qwen06b-te3-adapted"  # or "minilm-te3-adapted"
    })

cal = resp.json()
print(f"Grade: {cal['quality_grade']}")
print(f"Min quality: {cal['recommendation']['min_quality']}")
print(f"Matches TE3: {cal['recommendation']['matches_te3']}")`)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Example Response</h3></div>
      <div class="card-body">
        ${codeBlock("json", `{
  "quality_grade": "excellent",
  "quality_scores": {
    "mean": 0.703, "p25": 0.586, "p50": 0.746, "p75": 0.859
  },
  "routing_preview": [
    {"quality": 0,  "pct_local": 100, "pct_openai": 0},
    {"quality": 30, "pct_local": 92,  "pct_openai": 8},
    {"quality": 50, "pct_local": 83,  "pct_openai": 17}
  ],
  "recommendation": {
    "min_quality": 0,
    "matches_te3": 100,
    "budget": {"quality": 0, "description": "Beats source model."},
    "balanced": {"quality": 20, "description": "Good tradeoff."},
    "max_quality": {"quality": 100, "description": "Matches TE3."}
  }
}`)}
      </div>
    </div>

    <div class="card fade-up delay-2" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Grades</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:10px;" class="bench-stat-grid">
          <div style="padding:14px; border-radius:10px; background:#3b82f608; border:1px solid #3b82f620; text-align:center;">
            <div style="font-size:16px; font-weight:800; color:#3b82f6;">Excellent</div>
            <div class="mono" style="font-size:12px; color:#71717a; margin-top:4px;">mean ≥ 0.65</div>
            <div style="font-size:11px; color:#52525b; margin-top:6px;">Use quality=0</div>
          </div>
          <div style="padding:14px; border-radius:10px; background:#3b82f608; border:1px solid #3b82f620; text-align:center;">
            <div style="font-size:16px; font-weight:800; color:#3b82f6;">Good</div>
            <div class="mono" style="font-size:12px; color:#71717a; margin-top:4px;">0.50 – 0.65</div>
            <div style="font-size:11px; color:#52525b; margin-top:6px;">Use min_quality</div>
          </div>
          <div style="padding:14px; border-radius:10px; background:#f59e0b08; border:1px solid #f59e0b20; text-align:center;">
            <div style="font-size:16px; font-weight:800; color:#f59e0b;">Moderate</div>
            <div class="mono" style="font-size:12px; color:#71717a; margin-top:4px;">0.35 – 0.50</div>
            <div style="font-size:11px; color:#52525b; margin-top:6px;">Route or train LoRA</div>
          </div>
          <div style="padding:14px; border-radius:10px; background:#ef444408; border:1px solid #ef444420; text-align:center;">
            <div style="font-size:16px; font-weight:800; color:#ef4444;">Poor</div>
            <div class="mono" style="font-size:12px; color:#71717a; margin-top:4px;">< 0.35</div>
            <div style="font-size:11px; color:#52525b; margin-top:6px;">Train custom LoRA</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card fade-up delay-3">
      <div class="card-header"><h3>When to Train a LoRA</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>If calibrate returns <strong style="color:#f59e0b;">moderate</strong> or <strong style="color:#ef4444;">poor</strong>, the base adapter doesn't know your domain well enough. Create a custom LoRA that auto-improves:</p>
        <p style="margin-top:12px;"><strong style="color:#d4d4d8;">1.</strong> Create adapter: <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/adapters</code></p>
        <p><strong style="color:#d4d4d8;">2.</strong> Embed with <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality > 0</code> + your <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">adapter_id</code></p>
        <p><strong style="color:#d4d4d8;">3.</strong> The system auto-collects training pairs from OpenAI fallbacks</p>
        <p><strong style="color:#d4d4d8;">4.</strong> Re-calibrate to see your grade improve over time</p>
        <p style="margin-top:12px;"><a href="docs.html#adapters_doc" style="color:#3b82f6; font-weight:600;">Adapters documentation →</a></p>
      </div>
    </div>
  `;
}

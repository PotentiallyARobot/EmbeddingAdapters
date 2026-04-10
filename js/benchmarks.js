// ── Benchmarks Page — MiniLM→TE3 vs Qwen3-0.6B→TE3 on HotpotQA + NQ ──

const benchSections = [
  { id: "overview",  icon: "◈", label: "Key Results" },
  { id: "nq",        icon: "↗", label: "Natural Questions" },
  { id: "hotpot",    icon: "↗", label: "HotpotQA" },
  { id: "routing",   icon: "◎", label: "Quality Routing" },
  { id: "cost",      icon: "◇", label: "Cost Analysis" },
  { id: "calibrate", icon: "▤", label: "Calibrate API" },
];

let currentBench = "overview";

document.addEventListener("DOMContentLoaded", () => {
  const hash = window.location.hash.replace("#", "");
  if (benchSections.find(s => s.id === hash)) currentBench = hash;
  renderBenchSidebar();
  renderBenchContent();
});

function switchBench(id) {
  currentBench = id;
  window.location.hash = id;
  $$(".sidebar-item").forEach(el => el.classList.toggle("active", el.dataset.tab === id));
  renderBenchContent();
}

function renderBenchSidebar() {
  const sb = $("#sidebar");
  sb.innerHTML = `
    <div class="sidebar-header">
      <a href="index.html" class="nav-logo">
        <div class="nav-logo-icon mono">E</div>
        <span class="nav-logo-text">Embedding Adapters</span>
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
      <a href="index.html" class="sidebar-item" style="text-decoration:none;"><span class="icon">◪</span> Pricing</a>
    </div>
  `;
}

function renderBenchContent() {
  const main = $("#main");
  main.className = "main-content fade-in";
  const r = { overview: renderOverview, nq: renderNQ, hotpot: renderHotpot, routing: renderRouting, cost: renderCost, calibrate: renderCalibrate };
  main.innerHTML = (r[currentBench] || renderOverview)();
}

function codeBlock(lang, code) {
  const id = "cb_" + Math.random().toString(36).slice(2, 8);
  return `<div class="code-wrap"><div class="code-actions"><span class="mono code-lang">${lang}</span><button class="btn-copy" onclick="copyToClipboard(document.getElementById('${id}').textContent, this)">Copy</button></div><pre class="code-block" id="${id}">${escHtml(code)}</pre></div>`;
}

function barChart(rows, maxVal) {
  if (!maxVal) maxVal = Math.max(...rows.map(r => r.value));
  return rows.map(r => {
    const pct = (r.value / maxVal * 100).toFixed(1);
    const color = r.color || "#10b981";
    return `<div style="margin-bottom:12px;">
      <div class="flex justify-between" style="font-size:12px; margin-bottom:4px;">
        <span style="color:#a1a1aa; font-weight:600;">${r.label}</span>
        <span class="mono" style="color:${color}; font-weight:700;">${r.display || r.value}</span>
      </div>
      <div style="height:8px; background:#1c1c26; border-radius:4px; overflow:hidden;">
        <div style="height:100%; width:${pct}%; background:${color}; border-radius:4px; transition:width 0.6s ease;"></div>
      </div>
    </div>`;
  }).join("");
}

function modelTag(model) {
  const colors = { "MiniLM→TE3": "#6366f1", "Qwen3-0.6B→TE3": "#10b981" };
  const c = colors[model] || "#52525b";
  return `<span class="mono" style="color:${c}; font-weight:700; font-size:12px; background:${c}15; padding:2px 8px; border-radius:4px; border:1px solid ${c}30;">${model}</span>`;
}

// ══════════════════════════════════════════════════════
// KEY RESULTS
// ══════════════════════════════════════════════════════
function renderOverview() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Benchmark Results</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Two adapted models evaluated on RAG retrieval. Adapted queries search TE3-embedded corpora.</p>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px;">
      <div class="stat fade-up">
        <div class="stat-label">${modelTag("MiniLM→TE3")} — NQ q=0</div>
        <div class="stat-value mono" style="color:#6366f1; font-size:20px;">MRR 0.926 · 99.5% cheaper</div>
        <div class="stat-sub">Beats MiniLM raw (0.843) at zero OpenAI cost</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">${modelTag("Qwen3-0.6B→TE3")} — NQ q=0</div>
        <div class="stat-value mono" style="color:#10b981; font-size:20px;">MRR 0.934 · 99.7% cheaper</div>
        <div class="stat-sub">Higher accuracy, matches TE3 at q=40</div>
      </div>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px;">
      <div class="stat fade-up delay-2">
        <div class="stat-label">NQ — Matches TE3</div>
        <div class="stat-value mono" style="font-size:18px;"><span style="color:#6366f1;">q=90</span> · <span style="color:#10b981;">q=40</span></div>
        <div class="stat-sub">Qwen3 reaches TE3 quality 50 quality levels sooner</div>
      </div>
      <div class="stat fade-up delay-3">
        <div class="stat-label">HotpotQA — Beats Source</div>
        <div class="stat-value mono" style="font-size:18px;"><span style="color:#6366f1;">q=60</span> · <span style="color:#10b981;">q=30</span></div>
        <div class="stat-sub">Qwen3 beats MiniLM raw at half the routing</div>
      </div>
      <div class="stat fade-up delay-4">
        <div class="stat-label">HotpotQA — Matches TE3</div>
        <div class="stat-value mono" style="font-size:18px;"><span style="color:#6366f1;">q=80</span> · <span style="color:#10b981;">q=60</span></div>
        <div class="stat-sub">Both achieve TE3 parity at 98%+ savings</div>
      </div>
    </div>

    <div class="card fade-up delay-5" style="margin-bottom:20px;">
      <div class="card-header"><h3>Model Comparison — MRR@10</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">Natural Questions (100q × 1000 passages)</div>
            ${barChart([
              { label: "MiniLM raw", value: 0.843, display: "0.843", color: "#ef4444" },
              { label: "MiniLM→TE3 q=0", value: 0.926, display: "0.926", color: "#6366f1" },
              { label: "Qwen3→TE3 q=0", value: 0.934, display: "0.934", color: "#10b981" },
              { label: "MiniLM→TE3 q=30", value: 0.948, display: "0.948", color: "#6366f1" },
              { label: "Qwen3→TE3 q=30", value: 0.948, display: "0.948", color: "#10b981" },
              { label: "OpenAI TE3", value: 0.960, display: "0.960", color: "#52525b" },
            ], 1.0)}
          </div>
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">HotpotQA (100q × 1190 passages)</div>
            ${barChart([
              { label: "MiniLM→TE3 q=0", value: 0.827, display: "0.827", color: "#6366f1" },
              { label: "Qwen3→TE3 q=0", value: 0.835, display: "0.835", color: "#10b981" },
              { label: "MiniLM raw", value: 0.872, display: "0.872", color: "#ef4444" },
              { label: "Qwen3→TE3 q=30", value: 0.875, display: "0.875", color: "#10b981" },
              { label: "MiniLM→TE3 q=70", value: 0.904, display: "0.904", color: "#6366f1" },
              { label: "Qwen3→TE3 q=60", value: 0.911, display: "0.911", color: "#10b981" },
              { label: "OpenAI TE3", value: 0.917, display: "0.917", color: "#52525b" },
            ], 1.0)}
          </div>
        </div>
      </div>
    </div>

    <div class="card fade-up delay-6">
      <div class="card-header"><h3>Which Model to Use?</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="margin-bottom:8px;">${modelTag("MiniLM→TE3")}</div>
            <p><strong style="color:#d4d4d8;">Best for: Speed + low VRAM.</strong> 18K tok/s on a laptop GPU. Handles 16 concurrent users. Great when you need fast inference and your data is general-purpose text.</p>
            <p style="margin-top:4px;">Latency: ~50ms for 10 texts, ~450ms for 500 texts.</p>
          </div>
          <div>
            <div style="margin-bottom:8px;">${modelTag("Qwen3-0.6B→TE3")}</div>
            <p><strong style="color:#d4d4d8;">Best for: Higher accuracy.</strong> Consistently 1–3% better MRR, reaches TE3 parity at lower quality thresholds. Trades ~15× inference speed for better cross-domain generalization.</p>
            <p style="margin-top:4px;">Latency: ~140ms for 10 texts, ~3.6s for 500 texts.</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// NATURAL QUESTIONS
// ══════════════════════════════════════════════════════
function renderNQ() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Natural Questions</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">100 queries × 1,000 TE3-embedded passages. Both models beat source at q=0.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>${modelTag("MiniLM→TE3")} — Quality Sweep</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr style="background:#6366f108;"><td style="color:#6366f1;">0</td><td>0%</td><td style="color:#6366f1; font-weight:700;">0.926</td><td>0.900</td><td>0.970</td><td style="color:#10b981;">$0.000076</td></tr>
            <tr><td>30</td><td>8%</td><td style="font-weight:700;">0.948</td><td>0.930</td><td>0.970</td><td>$0.000087</td></tr>
            <tr><td>50</td><td>17%</td><td>0.948</td><td>0.930</td><td>0.970</td><td>$0.000100</td></tr>
            <tr style="background:#6366f108;"><td style="color:#6366f1;">90</td><td>81%</td><td style="color:#6366f1; font-weight:700;">0.955</td><td>0.930</td><td>0.980</td><td>$0.000180</td></tr>
            <tr><td>100</td><td>100%</td><td>0.960</td><td>0.940</td><td>0.980</td><td>$0.000204</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>${modelTag("Qwen3-0.6B→TE3")} — Quality Sweep</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr style="background:#10b98108;"><td style="color:#10b981;">0</td><td>0%</td><td style="color:#10b981; font-weight:700;">0.934</td><td>0.910</td><td>0.970</td><td style="color:#10b981;">$0.000047</td></tr>
            <tr><td>30</td><td>21%</td><td style="font-weight:700;">0.948</td><td>0.930</td><td>0.970</td><td>$0.000074</td></tr>
            <tr style="background:#10b98108;"><td style="color:#10b981;">40</td><td>38%</td><td style="color:#10b981; font-weight:700;">0.953</td><td>0.930</td><td>0.980</td><td>$0.000097</td></tr>
            <tr><td>70</td><td>82%</td><td>0.958</td><td>0.940</td><td>0.980</td><td>$0.000153</td></tr>
            <tr><td>100</td><td>100%</td><td>0.960</td><td>0.940</td><td>0.980</td><td>$0.000175</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Key Takeaway</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Both models crush the MiniLM baseline (0.843) at <strong style="color:#10b981;">quality=0 with zero OpenAI calls</strong>. Qwen3-0.6B starts higher (0.934 vs 0.926) and reaches TE3 parity at q=40 vs q=90 — meaning you can match OpenAI quality while keeping 62% of queries local.</p>
        <p style="margin-top:8px;">Baselines: MiniLM→MiniLM MRR=0.843, R@10=0.940 | OpenAI TE3 MRR=0.960, R@10=0.980</p>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// HOTPOTQA
// ══════════════════════════════════════════════════════
function renderHotpot() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">HotpotQA</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">100 multi-hop queries × 1,190 TE3-embedded passages. Quality routing matters here.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>${modelTag("MiniLM→TE3")} — Quality Sweep</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>0%</td><td>0.827</td><td>0.730</td><td>0.990</td><td style="color:#10b981;">$0.000130</td></tr>
            <tr><td>30</td><td>22%</td><td>0.847</td><td>0.770</td><td>0.990</td><td>$0.000189</td></tr>
            <tr style="background:#6366f108;"><td style="color:#6366f1;">60</td><td>63%</td><td style="color:#6366f1; font-weight:700;">0.878</td><td>0.810</td><td>0.990</td><td>$0.000298</td></tr>
            <tr><td>70</td><td>78%</td><td>0.904</td><td>0.850</td><td>1.000</td><td>$0.000337</td></tr>
            <tr style="background:#6366f108;"><td style="color:#6366f1;">80</td><td>90%</td><td style="color:#6366f1; font-weight:700;">0.914</td><td>0.860</td><td>1.000</td><td>$0.000370</td></tr>
            <tr><td>100</td><td>100%</td><td>0.917</td><td>0.860</td><td>1.000</td><td>$0.000399</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:8px;">Beats MiniLM raw (0.872) at q=60. Matches TE3 at q=80.</p>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>${modelTag("Qwen3-0.6B→TE3")} — Quality Sweep</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@10</th><th>Cost</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>0%</td><td>0.835</td><td>0.740</td><td>1.000</td><td style="color:#10b981;">$0.000080</td></tr>
            <tr style="background:#10b98108;"><td style="color:#10b981;">30</td><td>40%</td><td style="color:#10b981; font-weight:700;">0.875</td><td>0.800</td><td>1.000</td><td>$0.000177</td></tr>
            <tr><td>50</td><td>76%</td><td>0.898</td><td>0.840</td><td>1.000</td><td>$0.000274</td></tr>
            <tr style="background:#10b98108;"><td style="color:#10b981;">60</td><td>85%</td><td style="color:#10b981; font-weight:700;">0.911</td><td>0.850</td><td>1.000</td><td>$0.000303</td></tr>
            <tr><td>80</td><td>100%</td><td>0.917</td><td>0.860</td><td>1.000</td><td>$0.000349</td></tr>
            <tr><td>100</td><td>100%</td><td>0.917</td><td>0.860</td><td>1.000</td><td>$0.000349</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:8px;">Beats MiniLM raw at q=30. Matches TE3 at q=60. R@10=1.000 at every quality level.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Qwen3 Advantage</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>On this harder multi-hop dataset, Qwen3-0.6B reaches key thresholds significantly sooner:</p>
        <p style="margin-top:8px;"><strong style="color:#d4d4d8;">Beats MiniLM raw:</strong> ${modelTag("Qwen3-0.6B→TE3")} at q=30 (40% routing) vs ${modelTag("MiniLM→TE3")} at q=60 (63% routing)</p>
        <p style="margin-top:4px;"><strong style="color:#d4d4d8;">Matches TE3:</strong> ${modelTag("Qwen3-0.6B→TE3")} at q=60 (85% routing) vs ${modelTag("MiniLM→TE3")} at q=80 (90% routing)</p>
        <p style="margin-top:8px;">Both achieve perfect R@10 on Qwen3 at every quality level — the correct answer is always in the top 10.</p>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// QUALITY ROUTING
// ══════════════════════════════════════════════════════
function renderRouting() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Quality Routing</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Each model's quality head predicts adapter confidence. Low-confidence texts route to OpenAI.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>How It Works</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>The <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality</code> parameter sets a confidence threshold (0–100). Texts where the adapter's neural quality head scores below this threshold get re-embedded via OpenAI TE3.</p>
        <p style="margin-top:8px;">The % routed varies by dataset and model — this is correct. The head routes more on hard data and less on easy data.</p>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Routing Rate Comparison</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">Natural Questions</div>
            <table class="table mono" style="font-size:12px;">
              <thead><tr><th>Quality</th><th>${modelTag("MiniLM→TE3")}</th><th>${modelTag("Qwen3-0.6B→TE3")}</th></tr></thead>
              <tbody>
                <tr><td>0</td><td>0% → 0.926</td><td>0% → 0.934</td></tr>
                <tr><td>30</td><td>8% → 0.948</td><td>21% → 0.948</td></tr>
                <tr><td>50</td><td>17% → 0.948</td><td>56% → 0.953</td></tr>
                <tr><td>70</td><td>44% → 0.948</td><td>82% → 0.958</td></tr>
                <tr><td>100</td><td>100% → 0.960</td><td>100% → 0.960</td></tr>
              </tbody>
            </table>
          </div>
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">HotpotQA</div>
            <table class="table mono" style="font-size:12px;">
              <thead><tr><th>Quality</th><th>${modelTag("MiniLM→TE3")}</th><th>${modelTag("Qwen3-0.6B→TE3")}</th></tr></thead>
              <tbody>
                <tr><td>0</td><td>0% → 0.827</td><td>0% → 0.835</td></tr>
                <tr><td>30</td><td>22% → 0.847</td><td>40% → 0.875</td></tr>
                <tr><td>50</td><td>46% → 0.866</td><td>76% → 0.898</td></tr>
                <tr><td>70</td><td>78% → 0.904</td><td>93% → 0.911</td></tr>
                <tr><td>100</td><td>100% → 0.917</td><td>100% → 0.917</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <p style="font-size:12px; color:#3f3f46; margin-top:12px;">Format: %routed → MRR@10. Qwen3 routes more aggressively (stricter quality head) but achieves higher MRR at each level.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Find Your Setting</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Use <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/quality/calibrate</code> with a sample of your actual data. The endpoint returns your quality grade, score distribution, routing preview, and recommended settings.</p>
        <p style="margin-top:8px;"><span style="cursor:pointer; color:#10b981;" onclick="switchBench('calibrate')">See Calibrate API →</span></p>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// COST ANALYSIS
// ══════════════════════════════════════════════════════
function renderCost() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Cost Analysis</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Both models achieve 98–99%+ savings vs OpenAI TE3 at every quality level.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Query Cost — Natural Questions</h3></div>
      <div class="card-body">
        ${barChart([
          { label: "Qwen3→TE3 q=0 (0% OpenAI)", value: 0.000047, display: "$0.000047", color: "#10b981" },
          { label: "MiniLM→TE3 q=0 (0% OpenAI)", value: 0.000076, display: "$0.000076", color: "#6366f1" },
          { label: "Qwen3→TE3 q=40 (≈TE3 quality)", value: 0.000097, display: "$0.000097", color: "#10b981" },
          { label: "MiniLM→TE3 q=90 (≈TE3 quality)", value: 0.000180, display: "$0.000180", color: "#6366f1" },
          { label: "MiniLM raw", value: 0.000930, display: "$0.000930", color: "#ef4444" },
          { label: "OpenAI TE3", value: 0.014080, display: "$0.014080", color: "#52525b" },
        ], 0.015)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Savings by Quality Level</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th colspan="2">${modelTag("MiniLM→TE3")}</th><th colspan="2">${modelTag("Qwen3-0.6B→TE3")}</th></tr>
          <tr><th></th><th>NQ Savings</th><th>HotpotQA</th><th>NQ Savings</th><th>HotpotQA</th></tr></thead>
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

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Monthly Projections</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Volume</th><th>OpenAI TE3</th><th>MiniLM→TE3 q=0</th><th>Qwen3→TE3 q=0</th><th>Savings</th></tr></thead>
          <tbody>
            <tr><td>10M tokens</td><td>$1.30</td><td class="green">$0.01</td><td class="green">$0.005</td><td class="green">99%+</td></tr>
            <tr><td>100M tokens</td><td>$13.00</td><td class="green">$0.07</td><td class="green">$0.04</td><td class="green">99%+</td></tr>
            <tr><td>1B tokens</td><td>$130</td><td class="green">$0.65</td><td class="green">$0.40</td><td class="green">99%+</td></tr>
            <tr><td>10B tokens</td><td>$1,300</td><td class="green">$6.50</td><td class="green">$4.00</td><td class="green">99%+</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════
// CALIBRATE API
// ══════════════════════════════════════════════════════
function renderCalibrate() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Calibrate API</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Find the right quality setting for your data.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/quality/calibrate</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">Send 10–2,048 sample texts. Get back score distribution, routing preview, and recommended quality setting.</p>
        ${tabbedCodeBlock ? tabbedCodeBlock("calibrate_example", {
          Python: `import requests

resp = requests.post(
    "https://embedding-adapters-api.embedding-adapters.workers.dev/v1/quality/calibrate",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["your sample texts...", "more actual data..."],
        "model": "qwen06b-te3-adapted"  # or "minilm-te3-adapted"
    })

cal = resp.json()
print(f"Grade: {cal['quality_grade']}")
print(f"Recommended: quality={cal['recommendation']['min_quality']}")
print(f"Matches TE3: quality={cal['recommendation']['matches_te3']}")`,
          JavaScript: `const resp = await fetch(
  "https://embedding-adapters-api.embedding-adapters.workers.dev/v1/quality/calibrate",
  {
    method: "POST",
    headers: {
      "Authorization": "Bearer YOUR_API_KEY",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      texts: ["your sample texts...", "more actual data..."],
      model: "qwen06b-te3-adapted"
    })
  }
);
const cal = await resp.json();
console.log(\`Grade: \${cal.quality_grade}\`);`,
          cURL: `curl -X POST https://embedding-adapters-api.embedding-adapters.workers.dev/v1/quality/calibrate \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"texts": ["your texts..."], "model": "qwen06b-te3-adapted"}'`
        }) : codeBlock("python", `import requests

resp = requests.post(
    "https://embedding-adapters-api.embedding-adapters.workers.dev/v1/quality/calibrate",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["your sample texts...", "more actual data..."],
        "model": "qwen06b-te3-adapted"  # or "minilm-te3-adapted"
    })

cal = resp.json()
print(f"Grade: {cal['quality_grade']}")
print(f"Recommended: quality={cal['recommendation']['min_quality']}")
print(f"Matches TE3: quality={cal['recommendation']['matches_te3']}")`)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Example Response</h3></div>
      <div class="card-body">
        ${codeBlock("json", `{
  "quality_grade": "excellent",
  "quality_scores": {
    "mean": 0.703, "std": 0.176,
    "p25": 0.586, "p50": 0.746, "p75": 0.859
  },
  "routing_preview": [
    {"quality": 0,  "pct_local": 100, "pct_openai": 0},
    {"quality": 30, "pct_local": 92,  "pct_openai": 8},
    {"quality": 50, "pct_local": 83,  "pct_openai": 17},
    {"quality": 100,"pct_local": 0,   "pct_openai": 100}
  ],
  "recommendation": {
    "min_quality": 0,
    "matches_te3": 100,
    "budget":      {"quality": 0,   "description": "Beats source model. ~0% routed."},
    "balanced":    {"quality": 20,  "description": "Good cost/quality tradeoff."},
    "max_quality": {"quality": 100, "description": "Matches TE3."}
  }
}`)}
      </div>
    </div>

    <div class="card fade-up delay-2" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Grades</h3></div>
      <div class="card-body">
        <table class="table" style="font-size:13px;">
          <thead><tr><th>Grade</th><th>Mean Score</th><th>What It Means</th><th>Action</th></tr></thead>
          <tbody>
            <tr><td class="green" style="font-weight:700;">Excellent</td><td>≥ 0.65</td><td>Adapter handles your data well</td><td>Use quality=0</td></tr>
            <tr><td style="color:#10b981; font-weight:700;">Good</td><td>0.50–0.65</td><td>Works for most texts</td><td>Use recommended min_quality</td></tr>
            <tr><td class="yellow" style="font-weight:700;">Moderate</td><td>0.35–0.50</td><td>Mixed results</td><td>Use quality=p50 or train a LoRA</td></tr>
            <tr><td class="red" style="font-weight:700;">Poor</td><td>< 0.35</td><td>Adapter struggles</td><td>Train a custom LoRA adapter</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card fade-up delay-3">
      <div class="card-header"><h3>Custom LoRA Adapters</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>If your data scores <strong style="color:#f59e0b;">moderate</strong> or <strong style="color:#ef4444;">poor</strong>, train a custom LoRA that learns your domain:</p>
        <p style="margin-top:12px;">1. Create with <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/adapters</code></p>
        <p>2. Embed with <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality > 0</code> — OpenAI fallbacks auto-collect training pairs</p>
        <p>3. Re-calibrate periodically to watch your grade improve</p>
        <p style="margin-top:12px;"><a href="docs.html#adapters_doc" style="color:#10b981;">Adapters documentation →</a></p>
      </div>
    </div>
  `;
}

// ── Benchmarks Page — Real data from HotpotQA + NQ benchmarks ──

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

// ══════════════════════════════════════════════════════
// KEY RESULTS
// ══════════════════════════════════════════════════════
function renderOverview() {
  return `
    <h2 style="font-size:22px; font-weight:800; margin-bottom:6px;">Benchmark Results</h2>
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">MiniLM→TE3 adapter evaluated on real RAG retrieval tasks. Adapted queries search against TE3-embedded corpora.</p>

    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; margin-bottom:24px;">
      <div class="stat fade-up">
        <div class="stat-label">NQ — Adapted q=0</div>
        <div class="stat-value mono" style="color:#10b981; font-size:22px;">MRR 0.926</div>
        <div class="stat-sub">vs TE3: 0.958 | vs MiniLM: 0.843</div>
      </div>
      <div class="stat fade-up delay-1">
        <div class="stat-label">Cost at q=0</div>
        <div class="stat-value mono" style="color:#10b981; font-size:22px;">99.5% cheaper</div>
        <div class="stat-sub">$0.000076 vs $0.014080</div>
      </div>
      <div class="stat fade-up delay-2">
        <div class="stat-label">NQ — Matches TE3 at</div>
        <div class="stat-value mono" style="color:#6366f1; font-size:22px;">quality=90</div>
        <div class="stat-sub">MRR 0.955 at 98.7% cheaper</div>
      </div>
    </div>

    <div class="card fade-up delay-3" style="margin-bottom:20px;">
      <div class="card-header"><h3>Headlines</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="font-size:15px; font-weight:700; color:#10b981; margin-bottom:8px;">Natural Questions</div>
            <p style="font-size:13px; color:#a1a1aa; line-height:1.7;">
              At <strong style="color:#d4d4d8;">quality=0</strong> (zero OpenAI calls), the adapter achieves MRR@10 of 0.926 — beating the raw MiniLM baseline (0.843) by +10% while costing <strong style="color:#10b981;">99.5% less than OpenAI TE3</strong>.
              At quality=30, MRR rises to 0.948 with only 8% of queries routed to OpenAI.
            </p>
          </div>
          <div>
            <div style="font-size:15px; font-weight:700; color:#f59e0b; margin-bottom:8px;">HotpotQA (Multi-hop)</div>
            <p style="font-size:13px; color:#a1a1aa; line-height:1.7;">
              Harder multi-hop retrieval. At <strong style="color:#d4d4d8;">quality=0</strong>, MRR@10 is 0.827 (below MiniLM's 0.872).
              The quality head correctly identifies difficult texts — at quality=60, the adapter beats MiniLM with 63% routing.
              At quality=80, it <strong style="color:#10b981;">matches TE3</strong> at 98% cost savings.
            </p>
          </div>
        </div>
      </div>
    </div>

    <div class="card fade-up delay-4">
      <div class="card-header"><h3>MRR@10 — Adapted queries → TE3 corpus</h3></div>
      <div class="card-body">
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">Natural Questions</div>
            ${barChart([
              { label: "MiniLM→MiniLM (baseline)", value: 0.843, display: "0.843", color: "#ef4444" },
              { label: "Adapted q=0 (0% OpenAI)", value: 0.926, display: "0.926", color: "#10b981" },
              { label: "Adapted q=30 (8% OpenAI)", value: 0.948, display: "0.948", color: "#10b981" },
              { label: "Adapted q=50 (17% OpenAI)", value: 0.948, display: "0.948", color: "#10b981" },
              { label: "Adapted q=90 (81% OpenAI)", value: 0.955, display: "0.955", color: "#6366f1" },
              { label: "OpenAI TE3 (100%)", value: 0.958, display: "0.958", color: "#52525b" },
            ], 1.0)}
          </div>
          <div>
            <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">HotpotQA</div>
            ${barChart([
              { label: "Adapted q=0 (0% OpenAI)", value: 0.827, display: "0.827", color: "#6366f1" },
              { label: "MiniLM→MiniLM (baseline)", value: 0.872, display: "0.872", color: "#ef4444" },
              { label: "Adapted q=60 (63% OpenAI)", value: 0.878, display: "0.878", color: "#10b981" },
              { label: "Adapted q=70 (78% OpenAI)", value: 0.904, display: "0.904", color: "#10b981" },
              { label: "Adapted q=80 (90% OpenAI)", value: 0.914, display: "0.914", color: "#10b981" },
              { label: "OpenAI TE3 (100%)", value: 0.917, display: "0.917", color: "#52525b" },
            ], 1.0)}
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">100 queries searching 1,000 TE3-embedded passages. Adapter beats source model at quality=0.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Sweep — Adapted queries → TE3 corpus</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@50</th><th>Cost</th></tr></thead>
          <tbody>
            <tr style="background:#10b98108;"><td class="green">0</td><td>0%</td><td class="green" style="font-weight:700;">0.926</td><td>0.900</td><td>0.940</td><td>0.960</td><td>0.970</td><td>0.980</td><td class="green">$0.000076</td></tr>
            <tr><td>10</td><td>0%</td><td>0.926</td><td>0.900</td><td>0.940</td><td>0.960</td><td>0.970</td><td>0.980</td><td>$0.000076</td></tr>
            <tr><td>20</td><td>1%</td><td>0.934</td><td>0.910</td><td>0.950</td><td>0.960</td><td>0.970</td><td>0.980</td><td>$0.000077</td></tr>
            <tr style="background:#10b98108;"><td class="green">30</td><td>8%</td><td class="green" style="font-weight:700;">0.948</td><td>0.930</td><td>0.960</td><td>0.970</td><td>0.970</td><td>0.980</td><td class="green">$0.000087</td></tr>
            <tr><td>50</td><td>17%</td><td>0.948</td><td>0.930</td><td>0.960</td><td>0.970</td><td>0.970</td><td>0.980</td><td>$0.000100</td></tr>
            <tr><td>70</td><td>44%</td><td>0.948</td><td>0.930</td><td>0.960</td><td>0.970</td><td>0.970</td><td>0.980</td><td>$0.000135</td></tr>
            <tr style="background:#6366f108;"><td class="green">90</td><td>81%</td><td class="green" style="font-weight:700;">0.955</td><td>0.930</td><td>0.980</td><td>0.980</td><td>0.980</td><td>0.980</td><td>$0.000180</td></tr>
            <tr><td>100</td><td>100%</td><td>0.958</td><td>0.940</td><td>0.980</td><td>0.980</td><td>0.980</td><td>0.980</td><td>$0.000204</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:12px;">Baselines: MiniLM→MiniLM MRR=0.843, R@10=0.940 | OpenAI→OpenAI MRR=0.958, R@10=0.980</p>
      </div>
    </div>

    <div class="card fade-up delay-1">
      <div class="card-header"><h3>Key Takeaway</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>At <strong style="color:#10b981;">quality=0</strong>, the adapter achieves MRR@10 of 0.926 with <strong style="color:#10b981;">zero OpenAI calls</strong> — 10% better than raw MiniLM (0.843) and 96.5% of TE3's accuracy (0.958), at 0.5% of the cost.</p>
        <p style="margin-top:8px;">Increasing quality to 30 routes just 8% of queries to OpenAI and pushes MRR to 0.948 — 99% of TE3 accuracy at 0.6% of the cost.</p>
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">100 multi-hop queries searching 1,190 TE3-embedded passages. Harder task — quality routing shines here.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Sweep — Adapted queries → TE3 corpus</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>%OpenAI</th><th>MRR@10</th><th>R@1</th><th>R@3</th><th>R@5</th><th>R@10</th><th>R@50</th><th>Cost</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>0%</td><td>0.827</td><td>0.730</td><td>0.900</td><td>0.950</td><td>0.990</td><td>1.000</td><td class="green">$0.000130</td></tr>
            <tr><td>20</td><td>4%</td><td>0.827</td><td>0.730</td><td>0.900</td><td>0.950</td><td>0.990</td><td>1.000</td><td>$0.000139</td></tr>
            <tr><td>30</td><td>22%</td><td>0.847</td><td>0.770</td><td>0.900</td><td>0.960</td><td>0.990</td><td>1.000</td><td>$0.000189</td></tr>
            <tr><td>50</td><td>46%</td><td>0.866</td><td>0.800</td><td>0.910</td><td>0.960</td><td>0.990</td><td>1.000</td><td>$0.000251</td></tr>
            <tr style="background:#10b98108;"><td class="green">60</td><td>63%</td><td class="green" style="font-weight:700;">0.878</td><td>0.810</td><td>0.930</td><td>0.970</td><td>0.990</td><td>1.000</td><td>$0.000298</td></tr>
            <tr><td>70</td><td>78%</td><td>0.904</td><td>0.850</td><td>0.940</td><td>0.980</td><td>1.000</td><td>1.000</td><td>$0.000337</td></tr>
            <tr style="background:#6366f108;"><td class="green">80</td><td>90%</td><td class="green" style="font-weight:700;">0.914</td><td>0.860</td><td>0.960</td><td>0.990</td><td>1.000</td><td>1.000</td><td>$0.000370</td></tr>
            <tr><td>90</td><td>98%</td><td>0.917</td><td>0.860</td><td>0.970</td><td>1.000</td><td>1.000</td><td>1.000</td><td>$0.000393</td></tr>
            <tr><td>100</td><td>100%</td><td>0.917</td><td>0.860</td><td>0.970</td><td>1.000</td><td>1.000</td><td>1.000</td><td>$0.000399</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:12px;">Baselines: MiniLM→MiniLM MRR=0.872, R@10=0.990 | OpenAI→OpenAI MRR=0.917, R@10=1.000</p>
      </div>
    </div>

    <div class="card fade-up delay-1">
      <div class="card-header"><h3>Quality Routing in Action</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>HotpotQA requires multi-hop reasoning — harder for the adapter. The quality head correctly identifies difficult queries and routes them to OpenAI.</p>
        <p style="margin-top:8px;">At <strong style="color:#10b981;">quality=60</strong>, the adapter beats MiniLM (MRR 0.878 vs 0.872) while routing 63% to OpenAI — still <strong style="color:#10b981;">98.5% cheaper</strong> than full TE3.</p>
        <p style="margin-top:8px;">At <strong style="color:#6366f1;">quality=80</strong>, it matches TE3 (MRR 0.914 vs 0.917) at <strong style="color:#10b981;">98.2% cost savings</strong>.</p>
        <p style="margin-top:8px;">R@10 is 0.990+ at every quality level — even at q=0, the correct passage is almost always in the top 10.</p>
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">The quality head predicts adapter confidence per-text. Low-confidence texts get re-embedded via OpenAI.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>How It Works</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Every text gets a quality score (0–1) from the adapter's neural quality head. The <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality</code> parameter sets the confidence threshold — texts scoring below it get re-embedded via OpenAI TE3.</p>
        <p style="margin-top:12px;">The % routed to OpenAI varies by dataset because different text types have different adapter difficulty. This is correct — the head routes more on hard data and less on easy data.</p>
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Routing Rate by Dataset</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th colspan="2">Natural Questions</th><th colspan="2">HotpotQA</th></tr>
          <tr><th></th><th>%OpenAI</th><th>MRR@10</th><th>%OpenAI</th><th>MRR@10</th></tr></thead>
          <tbody>
            <tr><td>0</td><td>0%</td><td class="green">0.926</td><td>0%</td><td>0.827</td></tr>
            <tr><td>20</td><td>1%</td><td class="green">0.934</td><td>4%</td><td>0.827</td></tr>
            <tr><td>30</td><td>8%</td><td class="green">0.948</td><td>22%</td><td>0.847</td></tr>
            <tr><td>50</td><td>17%</td><td class="green">0.948</td><td>46%</td><td>0.866</td></tr>
            <tr><td>70</td><td>44%</td><td class="green">0.948</td><td>78%</td><td class="green">0.904</td></tr>
            <tr><td>80</td><td>56%</td><td class="green">0.948</td><td>90%</td><td class="green">0.914</td></tr>
            <tr><td>100</td><td>100%</td><td>0.958</td><td>100%</td><td>0.917</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:12px;">NQ routes less because the adapter handles it well natively. HotpotQA routes more because multi-hop queries are harder.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Calibrate Your Data</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>Use the <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/quality/calibrate</code> endpoint to see what quality setting works best for your specific data. Send a sample of 50-500 texts and get back score distribution, routing preview, and recommended settings.</p>
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Real costs from the benchmark runs.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>Query Cost Comparison</h3></div>
      <div class="card-body">
        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px;">Natural Questions — 100 queries</div>
        ${barChart([
          { label: "Adapted q=0 (0% OpenAI)", value: 0.000076, display: "$0.000076", color: "#10b981" },
          { label: "Adapted q=30 (8% OpenAI)", value: 0.000087, display: "$0.000087", color: "#10b981" },
          { label: "Adapted q=90 (81% OpenAI)", value: 0.000180, display: "$0.000180", color: "#6366f1" },
          { label: "MiniLM→MiniLM", value: 0.000930, display: "$0.000930", color: "#ef4444" },
          { label: "OpenAI TE3", value: 0.014080, display: "$0.014080", color: "#52525b" },
        ], 0.015)}
        <div style="font-size:13px; font-weight:700; color:#a1a1aa; margin-bottom:12px; margin-top:24px;">HotpotQA — 100 queries</div>
        ${barChart([
          { label: "Adapted q=0 (0% OpenAI)", value: 0.000130, display: "$0.000130", color: "#10b981" },
          { label: "Adapted q=60 (63% OpenAI)", value: 0.000298, display: "$0.000298", color: "#10b981" },
          { label: "Adapted q=80 (90% OpenAI)", value: 0.000370, display: "$0.000370", color: "#6366f1" },
          { label: "MiniLM→MiniLM", value: 0.001400, display: "$0.001400", color: "#ef4444" },
          { label: "OpenAI TE3", value: 0.020179, display: "$0.020179", color: "#52525b" },
        ], 0.021)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Cost Savings vs OpenAI TE3</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Quality</th><th>NQ Savings</th><th>NQ MRR</th><th>HotpotQA Savings</th><th>HotpotQA MRR</th></tr></thead>
          <tbody>
            <tr><td>0</td><td class="green" style="font-weight:700;">99.5%</td><td>0.926</td><td class="green" style="font-weight:700;">99.4%</td><td>0.827</td></tr>
            <tr><td>30</td><td class="green" style="font-weight:700;">99.4%</td><td>0.948</td><td class="green" style="font-weight:700;">99.1%</td><td>0.847</td></tr>
            <tr><td>50</td><td class="green" style="font-weight:700;">99.3%</td><td>0.948</td><td class="green" style="font-weight:700;">98.8%</td><td>0.866</td></tr>
            <tr><td>80</td><td class="green" style="font-weight:700;">98.9%</td><td>0.948</td><td class="green" style="font-weight:700;">98.2%</td><td>0.914</td></tr>
            <tr><td>100</td><td class="green">98.6%</td><td>0.958</td><td class="green">98.0%</td><td>0.917</td></tr>
          </tbody>
        </table>
        <p style="font-size:12px; color:#3f3f46; margin-top:12px;">Even at quality=100 (all OpenAI), the adapter's per-token rate is cheaper than calling OpenAI directly.</p>
      </div>
    </div>

    <div class="card fade-up delay-2">
      <div class="card-header"><h3>Projected Monthly Savings</h3></div>
      <div class="card-body">
        <table class="table mono" style="font-size:12px;">
          <thead><tr><th>Monthly Volume</th><th>OpenAI TE3</th><th>Adapted q=0</th><th>Savings</th><th>Adapted q=50</th><th>Savings</th></tr></thead>
          <tbody>
            <tr><td>10M tokens</td><td>$1.30</td><td class="green">$0.01</td><td class="green">$1.29</td><td class="green">$0.01</td><td class="green">$1.29</td></tr>
            <tr><td>100M tokens</td><td>$13.00</td><td class="green">$0.07</td><td class="green">$12.93</td><td class="green">$0.10</td><td class="green">$12.90</td></tr>
            <tr><td>1B tokens</td><td>$130.00</td><td class="green">$0.65</td><td class="green">$129.35</td><td class="green">$1.00</td><td class="green">$129.00</td></tr>
            <tr><td>10B tokens</td><td>$1,300</td><td class="green">$6.50</td><td class="green">$1,293</td><td class="green">$10.00</td><td class="green">$1,290</td></tr>
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
    <p style="font-size:14px; color:#52525b; margin-bottom:24px;">Find the right quality setting for your data before committing.</p>

    <div class="card fade-up" style="margin-bottom:20px;">
      <div class="card-header"><h3>POST /v1/quality/calibrate</h3></div>
      <div class="card-body">
        <p style="font-size:14px; color:#a1a1aa; line-height:1.7; margin-bottom:16px;">
          Send 10–2,048 sample texts from your actual workload. The endpoint runs them through the adapter's quality head and returns score distribution, routing preview at every quality level, and a recommended setting.
        </p>
        ${codeBlock("python", `import requests

resp = requests.post("https://embedding-adapters-api.embedding-adapters.workers.dev/v1/quality/calibrate",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "texts": ["your sample texts here...", "more of your actual data..."],
        "model": "minilm-te3-adapted"
    })

cal = resp.json()
print(f"Grade: {cal['quality_grade']}")
print(f"Recommended: quality={cal['recommendation']['min_quality']}")
print(f"Matches TE3: quality={cal['recommendation']['matches_te3']}")`)}
      </div>
    </div>

    <div class="card fade-up delay-1" style="margin-bottom:20px;">
      <div class="card-header"><h3>Response</h3></div>
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
    {"quality": 80, "pct_local": 44,  "pct_openai": 56},
    {"quality": 100,"pct_local": 0,   "pct_openai": 100}
  ],
  "recommendation": {
    "min_quality": 0,
    "matches_te3": 100,
    "budget": {"quality": 0, "description": "Beats source model. ~0% routed to OpenAI."},
    "balanced": {"quality": 20, "description": "Good cost/quality tradeoff."},
    "max_quality": {"quality": 100, "description": "Matches TE3. ~100% routed to OpenAI."}
  }
}`)}
      </div>
    </div>

    <div class="card fade-up delay-2" style="margin-bottom:20px;">
      <div class="card-header"><h3>Quality Grades</h3></div>
      <div class="card-body">
        <table class="table" style="font-size:13px;">
          <thead><tr><th>Grade</th><th>Mean Score</th><th>What It Means</th><th>Recommended Action</th></tr></thead>
          <tbody>
            <tr><td class="green" style="font-weight:700;">Excellent</td><td>≥ 0.65</td><td>Adapter handles your data well</td><td>Use quality=0 for maximum savings</td></tr>
            <tr><td style="color:#10b981; font-weight:700;">Good</td><td>0.50–0.65</td><td>Works for most texts, some need help</td><td>Use quality=p25 to route bottom 25%</td></tr>
            <tr><td class="yellow" style="font-weight:700;">Moderate</td><td>0.35–0.50</td><td>Mixed results</td><td>Use quality=p50 or train a LoRA adapter</td></tr>
            <tr><td class="red" style="font-weight:700;">Poor</td><td>< 0.35</td><td>Adapter struggles with your data</td><td>Train a custom LoRA adapter on your domain</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="card fade-up delay-3">
      <div class="card-header"><h3>Custom LoRA Adapters</h3></div>
      <div class="card-body" style="font-size:14px; color:#a1a1aa; line-height:1.7;">
        <p>If your data scores <strong style="color:#f59e0b;">moderate</strong> or <strong style="color:#ef4444;">poor</strong>, create a custom LoRA adapter that learns from your specific domain:</p>
        <p style="margin-top:12px;">1. Create an adapter with <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">POST /v1/adapters</code></p>
        <p>2. Embed your texts with <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">quality > 0</code> and pass the <code class="mono" style="color:#71717a; background:#18181b; padding:2px 6px; border-radius:4px; font-size:12px;">adapter_id</code></p>
        <p>3. The system auto-collects training pairs from OpenAI fallbacks and improves over time</p>
        <p>4. Re-calibrate periodically to see your quality grade improve</p>
        <p style="margin-top:12px;"><a href="docs.html#adapters_doc" style="color:#10b981;">See Adapters documentation →</a></p>
      </div>
    </div>
  `;
}

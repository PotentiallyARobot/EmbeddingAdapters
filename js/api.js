const API_BASE = "https://embedding-adapters-api.embedding-adapters.workers.dev";

async function apiFetch(method, path, body = null, token = null) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const opts = { method, headers };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(`${API_BASE}${path}`, opts);
  return r.json();
}

function getSession() {
  try {
    const s = localStorage.getItem("ea_session");
    return s ? JSON.parse(s) : null;
  } catch { return null; }
}

function setSessionStorage(session) {
  localStorage.setItem("ea_session", JSON.stringify(session));
}

function clearSession() {
  const s = getSession();
  if (s) apiFetch("POST", "/auth/logout", null, s.session_token).catch(() => {});
  localStorage.removeItem("ea_session");
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text);
  if (btn) {
    const orig = btn.textContent;
    btn.textContent = "✓ Copied";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = orig; btn.classList.remove("copied"); }, 1500);
  }
}

function escHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

// Tabbed code block: tabbedCodeBlock([{lang:"python",label:"Python",code:"..."}, ...])
function tabbedCodeBlock(tabs) {
  const gid = "tcb_" + Math.random().toString(36).slice(2, 8);
  const tabBtns = tabs.map((t, i) =>
    `<button class="code-tab ${i === 0 ? 'active' : ''}" onclick="switchCodeTab('${gid}',${i},this)">${t.label || t.lang}</button>`
  ).join("");
  const panels = tabs.map((t, i) => {
    const pid = gid + "_" + i;
    return `<div class="code-tab-panel ${i === 0 ? '' : 'hidden'}" data-tcb="${gid}" data-idx="${i}">
      <div class="code-wrap" style="margin-top:0;">
        <div class="code-actions">
          <span class="mono code-lang">${t.lang}</span>
          <button class="btn-copy" onclick="copyToClipboard(document.getElementById('${pid}').textContent, this)">Copy</button>
        </div>
        <pre class="code-block" id="${pid}">${highlightCode(t.code, t.lang)}</pre>
      </div>
    </div>`;
  }).join("");
  return `<div class="code-tabs-wrap">
    <div class="code-tabs-bar">${tabBtns}</div>
    ${panels}
  </div>`;
}

function switchCodeTab(gid, idx, btn) {
  btn.parentElement.querySelectorAll('.code-tab').forEach((b, i) => b.classList.toggle('active', i === idx));
  document.querySelectorAll(`[data-tcb="${gid}"]`).forEach((p, i) => p.classList.toggle('hidden', i !== idx));
}

function highlightCode(code, lang) {
  // Escape only <, >, & for HTML safety in <pre> tags. Leave quotes unescaped.
  var h = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Single-pass: one regex with alternation groups. The engine picks the first
  // matching alternative at each position, so no group ever sees another's output.

  if (lang === "python" || lang === "py") {
    h = h.replace(
      /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(import|from|as|def|class|return|if|else|elif|for|in|while|with|try|except|raise|not|and|or|is|None|True|False|print|async|await)\b|\b(np|requests|base64|torch|os|time|json)\b|\b(\d+\.?\d*)\b/g,
      function(m, cm, st, kw, lb, nm) {
        if (cm != null && cm !== "") return '<span class="hl-cm">' + cm + '</span>';
        if (st != null && st !== "") return '<span class="hl-st">' + st + '</span>';
        if (kw != null && kw !== "") return '<span class="hl-kw">' + kw + '</span>';
        if (lb != null && lb !== "") return '<span class="hl-lb">' + lb + '</span>';
        if (nm != null && nm !== "") return '<span class="hl-nm">' + nm + '</span>';
        return m;
      }
    );
  } else if (lang === "bash" || lang === "sh") {
    h = h.replace(
      /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'[^']*')|\b(curl|pip|git|cd|export|python|set)\b|(--?\w[\w-]*)/g,
      function(m, cm, st, kw, fl) {
        if (cm != null && cm !== "") return '<span class="hl-cm">' + cm + '</span>';
        if (st != null && st !== "") return '<span class="hl-st">' + st + '</span>';
        if (kw != null && kw !== "") return '<span class="hl-kw">' + kw + '</span>';
        if (fl != null && fl !== "") return '<span class="hl-lb">' + fl + '</span>';
        return m;
      }
    );
  } else if (lang === "javascript" || lang === "js") {
    h = h.replace(
      /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(const|let|var|function|return|if|else|for|while|await|async|new|true|false|null|undefined)\b|\b(\d+\.?\d*)\b/g,
      function(m, cm, st, kw, nm) {
        if (cm != null && cm !== "") return '<span class="hl-cm">' + cm + '</span>';
        if (st != null && st !== "") return '<span class="hl-st">' + st + '</span>';
        if (kw != null && kw !== "") return '<span class="hl-kw">' + kw + '</span>';
        if (nm != null && nm !== "") return '<span class="hl-nm">' + nm + '</span>';
        return m;
      }
    );
  }

  return h;
}

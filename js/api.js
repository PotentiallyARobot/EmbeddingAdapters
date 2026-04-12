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
  let h = escHtml(code);
  if (lang === "python" || lang === "py") {
    h = h.replace(/(#.*)/g, '<span class=hl-cm>$1</span>');
    h = h.replace(/(["'])(?:(?!\1).)*?\1/g, '<span class=hl-st>$&</span>');
    h = h.replace(/\b(import|from|as|def|class|return|if|else|elif|for|in|while|with|try|except|raise|not|and|or|is|None|True|False|print|async|await)\b/g, '<span class=hl-kw>$1</span>');
    h = h.replace(/\b(\d+\.?\d*)\b/g, '<span class=hl-nm>$1</span>');
    h = h.replace(/\b(np|requests|base64|torch|os|time|json)\b/g, '<span class=hl-lb>$1</span>');
  } else if (lang === "bash" || lang === "sh") {
    h = h.replace(/(#.*)/g, '<span class=hl-cm>$1</span>');
    h = h.replace(/(["'])(?:(?!\1).)*?\1/g, '<span class=hl-st>$&</span>');
    h = h.replace(/\b(curl|pip|git|cd|export|python|set|embedding-adapters)\b/g, '<span class=hl-kw>$1</span>');
    h = h.replace(/(--?\w[\w-]*)/g, '<span class=hl-lb>$1</span>');
  } else if (lang === "javascript" || lang === "js") {
    h = h.replace(/(\/\/.*)/g, '<span class=hl-cm>$1</span>');
    h = h.replace(/(["'`])(?:(?!\1).)*?\1/g, '<span class=hl-st>$&</span>');
    h = h.replace(/\b(const|let|var|function|return|if|else|for|while|await|async|new|true|false|null|undefined)\b/g, '<span class=hl-kw>$1</span>');
    h = h.replace(/\b(\d+\.?\d*)\b/g, '<span class=hl-nm>$1</span>');
  } else if (lang === "json") {
    h = h.replace(/(["'])(\w+)\1\s*:/g, '<span class=hl-lb>$&</span>');
    h = h.replace(/:\s*(["'])(?:(?!\1).)*?\1/g, '<span class=hl-st>$&</span>');
    h = h.replace(/:\s*(\d+\.?\d*)/g, ': <span class=hl-nm>$1</span>');
    h = h.replace(/\b(true|false|null)\b/g, '<span class=hl-kw>$1</span>');
  }
  return h;
}

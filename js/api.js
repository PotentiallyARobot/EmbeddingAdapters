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

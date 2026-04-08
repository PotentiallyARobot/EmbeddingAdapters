document.addEventListener("DOMContentLoaded", () => {
  const session = getSession();
  if (session) { window.location.href = "dashboard.html"; return; }

  // Handle Google OAuth redirect
  const params = new URLSearchParams(window.location.search);
  if (params.get("session")) {
    setSessionStorage({ session_token: params.get("session"), email: params.get("email") });
    window.history.replaceState({}, "", window.location.pathname);
    window.location.href = "dashboard.html";
    return;
  }

  if (params.get("error")) {
    showForm("login", "Google sign-in failed. Try email instead.");
    window.history.replaceState({}, "", window.location.pathname);
    return;
  }

  showForm("login");
});

let currentMode = "login";

function showForm(mode, errorMsg) {
  currentMode = mode;
  const isLogin = mode === "login";
  const app = $("#app");

  app.innerHTML = `
    <div class="auth-wrap">
      <div class="auth-box fade-up">
        <h2 class="text-center" style="font-size:28px; font-weight:700; margin-bottom:8px;">
          ${isLogin ? "Welcome back" : "Create an account"}
        </h2>
        <p class="text-center" style="color:#71717a; margin-bottom:32px;">
          ${isLogin ? "Sign in to your dashboard" : "Get started in seconds"}
        </p>

        <button onclick="googleSignIn()" style="
          width:100%; padding:13px; border-radius:10px; border:1px solid #27272a;
          background:#18181b; color:#d4d4d8; font-size:15px; font-weight:600;
          display:flex; align-items:center; justify-content:center; gap:10px; margin-bottom:24px;
        ">
          <svg width="18" height="18" viewBox="0 0 18 18"><path fill="#4285F4" d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z"/><path fill="#34A853" d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"/><path fill="#FBBC05" d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.997 8.997 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z"/><path fill="#EA4335" d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 2.58 9 2.58z"/></svg>
          Continue with Google
        </button>

        <div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
          <div style="flex:1; height:1px; background:#27272a;"></div>
          <span style="font-size:12px; color:#52525b; font-weight:500;">OR</span>
          <div style="flex:1; height:1px; background:#27272a;"></div>
        </div>

        <form id="auth-form" style="text-align:left;">
          <label class="input-label">Email</label>
          <input type="email" id="auth-email" class="input" placeholder="you@company.com" required style="margin-bottom:12px;">
          <label class="input-label">Password</label>
          <input type="password" id="auth-password" class="input" placeholder="${isLogin ? 'Enter password' : 'Min 8 characters'}"
            required minlength="${isLogin ? 1 : 8}" style="margin-bottom:${isLogin ? 16 : 12}px;">
          ${!isLogin ? `
            <label class="input-label">Confirm Password</label>
            <input type="password" id="auth-confirm" class="input" placeholder="Confirm password" required style="margin-bottom:16px;">
          ` : ""}
          <div id="auth-error" class="${errorMsg ? '' : 'hidden'}" style="color:#ef4444; font-size:13px; margin-bottom:12px;">${errorMsg || ''}</div>
          <button type="submit" id="auth-submit" class="btn btn-primary" style="width:100%; justify-content:center;">
            ${isLogin ? "Sign In" : "Create Account"}
          </button>
        </form>

        <p class="text-center" style="margin-top:20px; font-size:13px; color:#52525b;">
          ${isLogin
            ? `Don't have an account? <span style="color:#2563eb; cursor:pointer;" onclick="showForm('signup')">Sign up</span>`
            : `Already have an account? <span style="color:#2563eb; cursor:pointer;" onclick="showForm('login')">Sign in</span>`}
        </p>
      </div>
    </div>
  `;

  $("#auth-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const email = $("#auth-email").value.trim();
    const password = $("#auth-password").value;
    const errEl = $("#auth-error");
    const btn = $("#auth-submit");

    if (!isLogin) {
      const confirm = $("#auth-confirm").value;
      if (password !== confirm) { errEl.textContent = "Passwords don't match"; errEl.classList.remove("hidden"); return; }
    }

    btn.textContent = isLogin ? "Signing in..." : "Creating account...";
    btn.disabled = true;
    errEl.classList.add("hidden");

    try {
      const data = await apiFetch("POST", isLogin ? "/auth/login" : "/auth/signup", { email, password });
      if (data.error) {
        errEl.textContent = data.error.message;
        errEl.classList.remove("hidden");
        btn.textContent = isLogin ? "Sign In" : "Create Account";
        btn.disabled = false;
        return;
      }
      setSessionStorage({ session_token: data.session_token, email: data.email });
      window.location.href = "dashboard.html";
    } catch {
      errEl.textContent = "Connection failed. Try again.";
      errEl.classList.remove("hidden");
      btn.textContent = isLogin ? "Sign In" : "Create Account";
      btn.disabled = false;
    }
  });
}

function googleSignIn() {
  window.location.href = `${API_BASE}/auth/google`;
}

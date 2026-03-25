/* app.js — Spectral Face Recognition frontend */

const API = "http://localhost:8000";

/* ── Utility ─────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

function showResult(el, type, html) {
  el.className = `result-area ${type}`;
  el.innerHTML = html;
  el.classList.remove("hidden");
}

function setLoading(btn, loading, defaultText) {
  if (loading) {
    btn.disabled = true;
    btn.classList.add("loading");
    btn.innerHTML = `<span class="spinner"></span> Processing…`;
  } else {
    btn.disabled = false;
    btn.classList.remove("loading");
    btn.textContent = defaultText;
  }
}

function initials(name) {
  return name.trim().split(/\s+/).map(w => w[0].toUpperCase()).slice(0, 2).join("");
}

/* ── API status ──────────────────────────────────────────────────── */
async function checkHealth() {
  const dot = $("status-dot");
  try {
    const r = await fetch(`${API}/health`);
    if (r.ok) {
      dot.className = "status-dot online";
      dot.title = "API online";
    } else {
      dot.className = "status-dot offline";
    }
  } catch {
    dot.className = "status-dot offline";
    dot.title = "API offline — is the server running?";
  }
}
checkHealth();
setInterval(checkHealth, 10_000);

/* ── Tabs ────────────────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => {
      p.classList.add("hidden");
      p.classList.remove("active");
    });
    tab.classList.add("active");
    const panel = $(`tab-${tab.dataset.tab}`);
    panel.classList.remove("hidden");
    panel.classList.add("active");
    if (tab.dataset.tab === "register") loadUsers();
  });
});

/* ── Upload areas ────────────────────────────────────────────────── */
function setupUpload({ dropId, inputId, placeholderId, previewId, btnId }) {
  const drop    = $(dropId);
  const input   = $(inputId);
  const preview = $(previewId);
  const ph      = $(placeholderId);
  const btn     = $(btnId);

  let selectedFile = null;

  function setFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    selectedFile = file;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.classList.remove("hidden");
    ph.style.display = "none";
    btn.disabled = false;
  }

  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => setFile(input.files[0]));
  drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("drag-over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
  drop.addEventListener("drop", e => {
    e.preventDefault();
    drop.classList.remove("drag-over");
    setFile(e.dataTransfer.files[0]);
  });

  return () => selectedFile;
}

const getRegFile = setupUpload({
  dropId: "reg-drop", inputId: "reg-file",
  placeholderId: "reg-placeholder", previewId: "reg-preview",
  btnId: "reg-btn",
});

const getVerFile = setupUpload({
  dropId: "ver-drop", inputId: "ver-file",
  placeholderId: "ver-placeholder", previewId: "ver-preview",
  btnId: "ver-btn",
});

/* ── Register ────────────────────────────────────────────────────── */
$("reg-btn").addEventListener("click", async () => {
  const file = getRegFile();
  const name = $("reg-name").value.trim();
  const btn  = $("reg-btn");
  const res  = $("reg-result");

  if (!file)        return showResult(res, "error", "Please select an image.");
  if (!name)        return showResult(res, "error", "Please enter a name.");

  setLoading(btn, true, "Register face");

  try {
    const form = new FormData();
    form.append("name", name);
    form.append("image", file);

    const r    = await fetch(`${API}/register`, { method: "POST", body: form });
    const data = await r.json();

    if (r.ok && data.success) {
      showResult(res, "success",
        `<div class="result-name">✓ Registered successfully</div>
         <div class="result-meta">${data.message}</div>`
      );
      $("reg-name").value = "";
      loadUsers();
    } else {
      showResult(res, "error", `✗ ${data.detail || data.message || "Registration failed."}`);
    }
  } catch (err) {
    showResult(res, "error", "✗ Could not reach the server. Is it running?");
  } finally {
    setLoading(btn, false, "Register face");
  }
});

/* Enable register button when name is typed even before image */
$("reg-name").addEventListener("input", () => {
  const file = getRegFile();
  const name = $("reg-name").value.trim();
  $("reg-btn").disabled = !(file && name);
});

/* ── Verify ──────────────────────────────────────────────────────── */
$("ver-btn").addEventListener("click", async () => {
  const file = getVerFile();
  const btn  = $("ver-btn");
  const res  = $("ver-result");

  if (!file) return showResult(res, "error", "Please select an image.");

  setLoading(btn, true, "Check identity");

  try {
    const form = new FormData();
    form.append("image", file);

    const r    = await fetch(`${API}/verify`, { method: "POST", body: form });
    const data = await r.json();

    if (!r.ok) {
      showResult(res, "error", `✗ ${data.detail || "Verification failed."}`);
      return;
    }

    if (data.match) {
      const pct = Math.round(data.confidence * 100);
      showResult(res, "success",
        `<div class="result-name">✓ Identity matched</div>
         <div class="result-meta">Recognised as <strong>${data.name}</strong> &nbsp;·&nbsp; distance: ${data.distance}</div>
         <div style="font-size:13px;margin-bottom:4px;color:inherit;">Confidence: ${pct}%</div>
         <div class="conf-bar-wrap">
           <div class="conf-bar-fill" style="width:${pct}%"></div>
         </div>`
      );
    } else {
      showResult(res, "error",
        `<div class="result-name">✗ Unknown face</div>
         <div class="result-meta">No registered user matched this face. (distance: ${data.distance})</div>`
      );
    }
  } catch {
    showResult(res, "error", "✗ Could not reach the server. Is it running?");
  } finally {
    setLoading(btn, false, "Check identity");
  }
});

/* ── Users list ──────────────────────────────────────────────────── */
async function loadUsers() {
  const list = $("users-list");
  try {
    const r    = await fetch(`${API}/users`);
    const data = await r.json();

    if (!data.users || data.users.length === 0) {
      list.innerHTML = `<p class="empty-hint">No users registered yet.</p>`;
      return;
    }

    list.innerHTML = data.users.map(name => `
      <div class="user-row" data-name="${name}">
        <div class="user-name">
          <div class="avatar">${initials(name)}</div>
          <span>${name}</span>
        </div>
        <button class="btn-delete" title="Remove ${name}" onclick="deleteUser('${name}')">✕</button>
      </div>
    `).join("");

  } catch {
    list.innerHTML = `<p class="empty-hint">Could not load users.</p>`;
  }
}

async function deleteUser(name) {
  if (!confirm(`Remove "${name}" from the system?`)) return;

  try {
    const r    = await fetch(`${API}/users/${encodeURIComponent(name)}`, { method: "DELETE" });
    const data = await r.json();

    if (r.ok) {
      loadUsers();
    } else {
      alert(data.detail || "Delete failed.");
    }
  } catch {
    alert("Could not reach the server.");
  }
}

$("refresh-users").addEventListener("click", loadUsers);

// Load users immediately on page open
loadUsers();

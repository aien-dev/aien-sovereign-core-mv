// ========================================================================= //
// ALLEN HARNESS FRONT END (SPARK COCKPIT)                                    //
// Every figure on the page is read from the server; nothing is typed in.    //
// ========================================================================= //

let currentThreadId = "main";
let activeView = "chat";
let isStreaming = false;
let termWs = null;
let activeArtifact = null;

// Initialize on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initNavigation();
  initChat();
  initApprovals();
  initTelemetry();
  initLive();
  initDrawer();
  initGoals();
  initMemory();
});

// ========================================================================= //
// THEME & NAVIGATION                                                        //
// ========================================================================= //
function initTheme() {
  // A link can choose the theme (?theme=light); otherwise the last choice on this browser, else dark.
  const fromLink = new URLSearchParams(window.location.search).get("theme");
  const savedTheme = fromLink || localStorage.getItem("allen_theme") || localStorage.getItem("muse_theme") || "dark";
  document.body.className = savedTheme === "light" ? "light-theme" : "dark-theme";

  const btn = document.getElementById("btn-theme-toggle");
  if (btn) {
    btn.addEventListener("click", () => {
      const isLight = document.body.classList.contains("light-theme");
      const nextTheme = isLight ? "dark" : "light";
      document.body.className = nextTheme + "-theme";
      localStorage.setItem("allen_theme", nextTheme);
    });
  }

  // Wide screens: the menu button narrows the rail to icons. Phones (under 820 px): the rail
  // sits off canvas and the same button slides it in over a backdrop.
  const sidebarToggle = document.getElementById("btn-sidebar-toggle");
  const sidebar = document.getElementById("muse-sidebar");
  const backdrop = document.getElementById("sidebar-backdrop");
  const phone = window.matchMedia("(max-width: 820px)");
  const closeRail = () => {
    sidebar.classList.remove("open");
    if (backdrop) backdrop.classList.remove("show");
  };
  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", () => {
      if (phone.matches) {
        const open = sidebar.classList.toggle("open");
        if (backdrop) backdrop.classList.toggle("show", open);
      } else {
        sidebar.classList.toggle("collapsed");
      }
    });
    if (backdrop) backdrop.addEventListener("click", closeRail);
    sidebar.querySelectorAll(".nav-item, .thread-item").forEach(el => {
      el.addEventListener("click", () => { if (phone.matches) closeRail(); });
    });
    phone.addEventListener("change", closeRail);
  }
}

function initNavigation() {
  const navItems = document.querySelectorAll(".nav-item");
  navItems.forEach(item => {
    item.addEventListener("click", () => {
      const viewName = item.getAttribute("data-view");
      switchView(viewName);
    });
  });

  const btnNewChat = document.getElementById("btn-new-chat");
  if (btnNewChat) {
    btnNewChat.addEventListener("click", () => {
      switchView("chat");
      clearChat();
    });
  }
}

function switchView(viewName) {
  activeView = viewName;
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.getAttribute("data-view") === viewName);
  });

  document.querySelectorAll(".muse-view").forEach(el => {
    el.classList.toggle("active", el.id === `view-${viewName}`);
  });

  if (viewName === "feed") loadFeed();
  if (viewName === "goals") loadGoals();
  if (viewName === "artifacts") loadArtifacts();
  if (viewName === "memory") loadMemory();
  if (viewName === "terminal") initTerminal();
}

// ========================================================================= //
// REAL-TIME TELEMETRY                                                       //
// ========================================================================= //
async function initTelemetry() {
  async function fetchTelemetry() {
    try {
      const res = await fetch("/api/telemetry/live");
      if (!res.ok) return;
      const data = await res.json();
      
      // /api/telemetry/live answers {gpu: {temperature_c}, memory: {used_kb, total_kb}}.
      // The pills stay hidden until a real reading arrives.
      const temp = data.gpu && data.gpu.temperature_c;
      const usedKb = data.memory && data.memory.used_kb;
      const totalKb = data.memory && data.memory.total_kb;
      let shown = false;

      const gpuTempEl = document.getElementById("gpu-temp");
      const gpuPill = document.getElementById("pill-gpu");
      if (gpuTempEl && typeof temp === "number" && temp > 0) {
        gpuTempEl.textContent = `GB10 ${Math.round(temp)}°C`;
        if (gpuPill) gpuPill.hidden = false;
        shown = true;
      } else if (gpuPill) {
        gpuPill.hidden = true;
      }

      const ramUsedEl = document.getElementById("ram-used");
      const ramPill = document.getElementById("pill-ram");
      if (ramUsedEl && typeof usedKb === "number" && typeof totalKb === "number" && totalKb > 0) {
        const usedGb = usedKb / 1048576;
        const totalGb = Math.round(totalKb / 1048576);
        ramUsedEl.textContent = `${usedGb.toFixed(1)} of ${totalGb} GB`;
        if (ramPill) ramPill.hidden = false;
        shown = true;
      } else if (ramPill) {
        ramPill.hidden = true;
      }

      const pills = document.getElementById("system-pills");
      if (pills) pills.hidden = !shown;
    } catch (_) {}
  }

  fetchTelemetry();
  setInterval(fetchTelemetry, 3000);
}

// ========================================================================= //
// LIVE STATE ON THE WELCOME (seat, approvals, goals, Spark judge)            //
// ========================================================================= //
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function greetingForHour(hour) {
  if (hour < 5) return "Good night";
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

async function readJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function refreshSeat() {
  const badge = document.getElementById("seat-model-badge");
  try {
    const data = await readJson("/api/models");
    const first = (data.active_models || [])[0] || {};
    const online = first.status === "ONLINE" && first.id;
    const text = online ? first.id : "seat offline";
    if (badge) {
      badge.textContent = text;
      badge.classList.toggle("is-offline", !online);
    }
    setText("live-seat", text);
    const card = document.getElementById("live-card-seat");
    if (card) card.classList.toggle("is-bad", !online);
  } catch (_) {
    if (badge) badge.textContent = "seat unknown";
    setText("live-seat", "not readable");
  }
}

async function refreshCounts() {
  try {
    const data = await readJson("/api/approvals");
    const pending = (data.approvals || []).filter(a => a.status === "pending").length;
    setText("live-approvals", String(pending));
    const card = document.getElementById("live-card-approvals");
    if (card) card.classList.toggle("is-attention", pending > 0);
  } catch (_) {
    setText("live-approvals", "not readable");
  }
  try {
    const data = await readJson("/api/goals");
    setText("live-goals", String((data.goals || []).length));
  } catch (_) {
    setText("live-goals", "not readable");
  }
}

function judgeWord(v) {
  if (v === "PASS") return "PASS";
  if (v === "FAIL") return "FAIL";
  if (!v || v === "NO VERDICT") return "NO VERDICT";
  return "BLOCKED";
}

async function refreshJudge() {
  const value = document.getElementById("live-judge");
  const note = document.getElementById("live-judge-note");
  const card = document.getElementById("live-card-judge");
  try {
    const j = await readJson("/api/judge");
    const word = judgeWord(j.verdict);
    if (value) {
      value.textContent = word;
      value.className = `live-value verdict verdict-${word.toLowerCase().replace(" ", "-")}`;
    }
    if (card) card.classList.toggle("is-bad", word === "FAIL");
    if (note) {
      if (j.commit) {
        const when = (j.finished_utc || "").replace("T", " ").replace("Z", " UTC");
        note.textContent = `commit ${j.commit.slice(0, 7)}: ${j.checks_pass} of ${j.checks_total} checks PASS, ${j.checks_fail} FAIL, ${j.checks_blocked} blocked. Judged ${when}.`;
      } else {
        note.textContent = j.note || "No Spark judge bundle could be read on this machine.";
      }
    }
  } catch (_) {
    if (value) value.textContent = "not readable";
    if (note) note.textContent = "The judge's bundle store could not be read.";
  }
}

function initLive() {
  setText("hero-greeting", `${greetingForHour(new Date().getHours())}. ALLEN is ready.`);
  refreshSeat();
  refreshCounts();
  refreshJudge();
  setInterval(refreshSeat, 15000);
  setInterval(refreshCounts, 15000);
  setInterval(refreshJudge, 60000);
}

// ========================================================================= //
// CHAT & STREAMING (SSE)                                                    //
// ========================================================================= //
function initChat() {
  const input = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const btnClear = document.getElementById("btn-clear-chat");

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    input.addEventListener("input", () => {
      input.style.height = "auto";
      input.style.height = Math.min(input.scrollHeight, 160) + "px";
    });
  }

  if (btnSend) {
    btnSend.addEventListener("click", () => sendMessage());
  }

  if (btnClear) {
    btnClear.addEventListener("click", () => clearChat());
  }

  // Suggestion chips in hero
  document.querySelectorAll(".suggestion-chip").forEach(chip => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (input && prompt) {
        input.value = prompt;
        sendMessage();
      }
    });
  });
}

function clearChat() {
  const list = document.getElementById("messages-list");
  if (list) list.innerHTML = "";
  const hero = document.getElementById("chat-hero");
  if (hero) hero.style.display = "flex";
  setAgentStatus("Ready");
}

function setAgentStatus(statusText) {
  const phrase = document.getElementById("status-phrase");
  if (phrase) phrase.textContent = statusText;
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  if (!input || isStreaming) return;
  const text = input.value.trim();
  if (!text) return;

  // Hide hero
  const hero = document.getElementById("chat-hero");
  if (hero) hero.style.display = "none";

  // Append user bubble
  appendUserMessage(text);
  input.value = "";
  input.style.height = "auto";

  // Append agent placeholder
  const { row, bubble, thoughtContent, textContent } = createAgentMessagePlaceholder();
  isStreaming = true;
  setAgentStatus("Reasoning...");

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "atlas-lightning-omni",
        messages: [{ role: "user", content: text }]
      })
    });

    if (!res.ok) {
      textContent.textContent = `Error connecting to model seat: HTTP ${res.status}`;
      isStreaming = false;
      setAgentStatus("Ready");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let accumulatedContent = "";
    let accumulatedThought = "";
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const dataStr = trimmed.slice(5).trim();
        if (dataStr === "[DONE]") break;

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.reasoning) {
            accumulatedThought += parsed.reasoning;
            thoughtContent.textContent = accumulatedThought;
            const thoughtBox = row.querySelector(".thought-box");
            if (thoughtBox) thoughtBox.style.display = "block";
          }
          if (parsed.content) {
            accumulatedContent += parsed.content;
            renderMarkdown(textContent, accumulatedContent);
          }
        } catch (_) {}
      }
      scrollToBottom();
    }
  } catch (err) {
    textContent.textContent = `Network exception during stream: ${err.message}`;
  } finally {
    isStreaming = false;
    setAgentStatus("Ready");
    scrollToBottom();
  }
}

function appendUserMessage(text) {
  const list = document.getElementById("messages-list");
  if (!list) return;

  const row = document.createElement("div");
  row.className = "message-row user-row";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;

  row.appendChild(bubble);
  list.appendChild(row);
  scrollToBottom();
}

function createAgentMessagePlaceholder() {
  const list = document.getElementById("messages-list");
  const row = document.createElement("div");
  row.className = "message-row agent-row";

  const avatar = document.createElement("img");
  avatar.className = "msg-avatar";
  avatar.src = "/avatar.jpg";
  avatar.onerror = () => { avatar.src = "/icon-192.png"; };

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  const thoughtBox = document.createElement("div");
  thoughtBox.className = "thought-box";
  thoughtBox.style.display = "none";
  thoughtBox.innerHTML = `
    <div class="thought-header">
      <span>⚡ Reasoning Process</span>
      <span>▼</span>
    </div>
    <div class="thought-content"></div>
  `;
  const thoughtHeader = thoughtBox.querySelector(".thought-header");
  const thoughtContent = thoughtBox.querySelector(".thought-content");
  thoughtHeader.addEventListener("click", () => {
    const isVisible = thoughtContent.style.display !== "none";
    thoughtContent.style.display = isVisible ? "none" : "block";
    thoughtHeader.querySelector("span:last-child").textContent = isVisible ? "▶" : "▼";
  });

  const textContent = document.createElement("div");
  textContent.className = "agent-markdown-body";
  textContent.textContent = "Thinking...";

  bubble.appendChild(thoughtBox);
  bubble.appendChild(textContent);
  row.appendChild(avatar);
  row.appendChild(bubble);
  list.appendChild(row);

  scrollToBottom();
  return { row, bubble, thoughtContent, textContent };
}

function renderMarkdown(container, rawText) {
  // Simple, high-speed compiled markdown formatter without external runtime bloat
  let formatted = rawText
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks: ```lang ... ```
  formatted = formatted.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `<pre class="code-canvas"><code class="language-${lang}">${code}</code></pre>`;
  });

  // Inline code: `code`
  formatted = formatted.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold: **text**
  formatted = formatted.replace(/\*\*([^\*]+)\*\*/g, "<strong>$1</strong>");

  // Headers: ### Head
  formatted = formatted.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  formatted = formatted.replace(/^## (.*$)/gim, "<h2>$1</h2>");
  formatted = formatted.replace(/^# (.*$)/gim, "<h1>$1</h1>");

  // Lists: - item
  formatted = formatted.replace(/^\s*-\s*(.*$)/gim, "<li>$1</li>");
  formatted = formatted.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");

  // Paragraphs
  formatted = formatted.replace(/\n\n/g, "<br><br>");

  container.innerHTML = formatted;
}

function scrollToBottom() {
  const area = document.getElementById("chat-scroll-area");
  if (area) {
    area.scrollTop = area.scrollHeight;
  }
}

// ========================================================================= //
// PROACTIVE FEED VIEW                                                       //
// ========================================================================= //
async function loadFeed() {
  const grid = document.getElementById("feed-grid");
  if (!grid) return;
  grid.innerHTML = "<div style='color:var(--text-muted);'>Loading proactive feed updates...</div>";

  try {
    const res = await fetch("/api/feed");
    if (!res.ok) throw new Error("Feed request failed");
    const data = await res.json();
    grid.innerHTML = "";

    (data.feed || []).forEach(item => {
      const card = document.createElement("div");
      card.className = "feed-card";
      card.innerHTML = `
        <div>
          <div class="feed-card-header">
            <span class="feed-card-badge ${item.badge.toLowerCase()}">${item.badge}</span>
            <span style="font-size:11px; color:var(--text-muted);">${item.category.toUpperCase()}</span>
          </div>
          <h3 class="feed-card-title" style="margin-top:8px;">${item.title}</h3>
          <p class="feed-card-summary" style="margin-top:6px;">${item.summary}</p>
        </div>
        <div style="display:flex; justify-content:flex-end;">
          <button class="stage-btn" data-target="${item.action_target}">${item.action_label}</button>
        </div>
      `;
      const btn = card.querySelector("button");
      btn.addEventListener("click", () => {
        switchView(item.action_target);
      });
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<div style="color:var(--status-crimson);">Failed to load feed: ${err.message}</div>`;
  }
}

// ========================================================================= //
// GOALS DECK                                                                //
// ========================================================================= //
function initGoals() {
  const btnNewGoal = document.getElementById("btn-open-new-goal-modal");
  const modal = document.getElementById("modal-new-goal");
  const btnSubmit = document.getElementById("btn-submit-goal");

  if (btnNewGoal && modal) {
    btnNewGoal.addEventListener("click", () => {
      modal.style.display = "flex";
    });
  }

  if (btnSubmit) {
    btnSubmit.addEventListener("click", async () => {
      const titleInput = document.getElementById("goal-input-title");
      const msInput = document.getElementById("goal-input-milestones");
      const title = titleInput.value.trim();
      if (!title) return;

      const milestones = msInput.value
        .split(",")
        .map(s => s.trim())
        .filter(s => s.length > 0);

      try {
        await fetch("/api/goals", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title, milestones })
        });
        modal.style.display = "none";
        titleInput.value = "";
        msInput.value = "";
        loadGoals();
      } catch (err) {
        alert("Failed to create goal: " + err.message);
      }
    });
  }
}

async function loadGoals() {
  const container = document.getElementById("goals-container");
  if (!container) return;
  container.innerHTML = "<div style='color:var(--text-muted);'>Retrieving objectives...</div>";

  try {
    const res = await fetch("/api/goals");
    if (!res.ok) throw new Error("Failed to load goals");
    const data = await res.json();
    container.innerHTML = "";

    const goals = data.goals || [];
    const badge = document.getElementById("goals-badge");
    if (badge) badge.textContent = goals.length;

    if (goals.length === 0) {
      container.innerHTML = `
        <div class="goal-card" style="grid-column: 1 / -1; text-align:center; padding: 40px;">
          <h3 style="font-size:16px; margin-bottom:8px;">No Active Objectives Tracked</h3>
          <p style="color:var(--text-secondary); margin-bottom:16px;">Deploy your first autonomous engineering or systems objective.</p>
          <div><button class="stage-btn primary" onclick="document.getElementById('modal-new-goal').style.display='flex'">+ New Goal</button></div>
        </div>
      `;
      return;
    }

    goals.forEach(goal => {
      const card = document.createElement("div");
      card.className = "goal-card";
      
      const milestones = goal.milestones || [];
      const completedCount = milestones.filter(m => m.completed).length;
      const pct = milestones.length > 0 ? Math.round((completedCount / milestones.length) * 100) : 0;

      card.innerHTML = `
        <div class="goal-card-top">
          <span class="goal-title">${goal.title}</span>
          <span class="feed-card-badge ${pct === 100 ? "online" : "pending"}">${pct}%</span>
        </div>
        <div class="goal-progress-wrap">
          <div class="goal-progress-bar-track">
            <div class="goal-progress-bar-fill" style="width: ${pct}%;"></div>
          </div>
          <div style="font-size:11px; color:var(--text-muted); display:flex; justify-content:space-between;">
            <span>${completedCount} of ${milestones.length} milestones complete</span>
          </div>
        </div>
        <div class="goal-milestones-list">
          ${milestones.map(m => `
            <label class="milestone-row">
              <input type="checkbox" class="milestone-check" ${m.completed ? "checked" : ""}>
              <span style="${m.completed ? "text-decoration:line-through; opacity:0.6;" : ""}">${m.name || m.title || m}</span>
            </label>
          `).join("")}
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div style="color:var(--status-crimson);">Goals error: ${err.message}</div>`;
  }
}

// ========================================================================= //
// ARTIFACTS LIBRARY                                                         //
// ========================================================================= //
async function loadArtifacts() {
  const grid = document.getElementById("artifacts-grid");
  if (!grid) return;
  grid.innerHTML = "<div style='color:var(--text-muted);'>Inspecting artifacts catalog...</div>";

  try {
    const res = await fetch("/api/artifacts");
    if (!res.ok) throw new Error("Artifacts fetch failed");
    const data = await res.json();
    grid.innerHTML = "";

    const items = data.artifacts || [];
    const badge = document.getElementById("artifacts-badge");
    if (badge) badge.textContent = items.length;

    if (items.length === 0) {
      grid.innerHTML = "<div style='color:var(--text-muted); padding:20px;'>No artifacts generated yet. Output files will appear here automatically.</div>";
      return;
    }

    items.forEach(item => {
      const card = document.createElement("div");
      card.className = "artifact-card";
      card.innerHTML = `
        <div class="artifact-card-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
        </div>
        <div class="artifact-card-title">${item.filename}</div>
        <div style="font-size:11px; color:var(--text-muted); display:flex; justify-content:space-between;">
          <span>${item.ext.toUpperCase()}</span>
          <span>${(item.size / 1024).toFixed(1)} KB</span>
        </div>
      `;
      card.addEventListener("click", () => {
        openArtifactInDrawer(item.filename);
      });
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<div style="color:var(--status-crimson);">Failed to load artifacts: ${err.message}</div>`;
  }
}

async function openArtifactInDrawer(filename) {
  const drawer = document.getElementById("muse-drawer");
  const emptyState = document.getElementById("artifact-empty-state");
  const viewer = document.getElementById("artifact-content-viewer");
  const titleEl = document.getElementById("artifact-active-filename");
  const codeEl = document.getElementById("artifact-active-code");

  if (drawer) drawer.classList.add("open");
  switchDrawerTab("artifact");

  try {
    const res = await fetch(`/api/artifacts/${filename}`);
    if (!res.ok) throw new Error("Could not retrieve artifact file");
    const text = await res.text();

    activeArtifact = { filename, content: text };
    if (titleEl) titleEl.textContent = filename;
    if (codeEl) codeEl.textContent = text;
    if (emptyState) emptyState.style.display = "none";
    if (viewer) viewer.style.display = "block";
  } catch (err) {
    alert("Error opening artifact: " + err.message);
  }
}

// ========================================================================= //
// APPROVALS QUEUE                                                           //
// ========================================================================= //
function initApprovals() {
  const btnStageApprovals = document.getElementById("btn-approvals-queue");
  if (btnStageApprovals) {
    btnStageApprovals.addEventListener("click", () => {
      const drawer = document.getElementById("muse-drawer");
      if (drawer) drawer.classList.add("open");
      switchDrawerTab("approvals");
      loadApprovals();
    });
  }
  loadApprovals();
}

async function loadApprovals() {
  try {
    const res = await fetch("/api/approvals");
    if (!res.ok) return;
    const data = await res.json();
    const approvals = data.approvals || [];

    const badgeTop = document.getElementById("approvals-count-badge");
    const badgeDrawer = document.getElementById("drawer-appr-badge");
    const pendingCount = approvals.filter(a => a.status === "pending").length;

    if (badgeTop) badgeTop.textContent = pendingCount;
    if (badgeDrawer) badgeDrawer.textContent = pendingCount;

    const list = document.getElementById("drawer-approvals-list");
    if (!list) return;

    if (approvals.length === 0) {
      list.innerHTML = "<div style='color:var(--text-muted); text-align:center; padding:30px;'>The cockpit holds no approvals. The approval desk issues them (aien compose authorize).</div>";
      return;
    }

    list.innerHTML = "";
    approvals.forEach(appr => {
      const card = document.createElement("div");
      card.className = "approval-card";
      card.innerHTML = `
        <div class="approval-header">
          <span class="approval-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px;color:var(--status-amber);"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            ${appr.title}
          </span>
          <span class="approval-badge">${appr.status.toUpperCase()}</span>
        </div>
        <p class="approval-desc">${appr.description}</p>
        ${appr.command_or_diff ? `<pre class="approval-cmd">${appr.command_or_diff}</pre>` : ""}
        ${appr.status === "pending" ? `
          <div class="approval-actions">
            <button class="btn-approve" data-id="${appr.id}">Approve</button>
            <button class="btn-reject" data-id="${appr.id}">Reject</button>
          </div>
        ` : ""}
      `;

      const btnApprove = card.querySelector(".btn-approve");
      const btnReject = card.querySelector(".btn-reject");

      if (btnApprove) {
        btnApprove.addEventListener("click", () => resolveApproval(appr.id, "approve"));
      }
      if (btnReject) {
        btnReject.addEventListener("click", () => resolveApproval(appr.id, "reject"));
      }

      list.appendChild(card);
    });
  } catch (_) {}
}

async function resolveApproval(id, action) {
  try {
    const res = await fetch(`/api/approvals/${id}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action })
    });
    if (res.ok) {
      loadApprovals();
    }
  } catch (err) {
    alert("Approval error: " + err.message);
  }
}

// ========================================================================= //
// MEMORY INSPECTOR (SPARK CORTEX)                                           //
// ========================================================================= //
function initMemory() {
  const btnSearch = document.getElementById("btn-search-memory");
  const input = document.getElementById("memory-search-input");

  if (btnSearch && input) {
    btnSearch.addEventListener("click", () => {
      loadMemory(input.value.trim());
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") loadMemory(input.value.trim());
    });
  }
}

async function loadMemory(query = "") {
  const grid = document.getElementById("memory-grid");
  if (!grid) return;
  grid.innerHTML = "<div style='color:var(--text-muted);'>Querying Spark Cortex space atlas-memory...</div>";

  try {
    const url = query ? `/api/cortex?q=${encodeURIComponent(query)}` : "/api/cortex";
    const res = await fetch(url);
    if (!res.ok) throw new Error("Cortex query failed");
    const data = await res.json();
    grid.innerHTML = "";

    const results = data.results || [];
    if (results.length === 0) {
      grid.innerHTML = "<div style='color:var(--text-muted); padding:20px;'>No memory records found matching query in space atlas-memory.</div>";
      return;
    }

    results.forEach(item => {
      const card = document.createElement("div");
      card.className = "memory-card";
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="memory-name">${item.canonical_name || item.name || "Entity"}</span>
          <span class="feed-card-badge online">${item.confidence ? Math.round(item.confidence * 100) + "%" : "STORED"}</span>
        </div>
        <p class="memory-content">${item.content || item.summary || ""}</p>
      `;
      grid.appendChild(card);
    });
  } catch (err) {
    grid.innerHTML = `<div style="color:var(--status-crimson);">Cortex memory error: ${err.message}</div>`;
  }
}

// ========================================================================= //
// WORKSPACE DRAWER TABS                                                     //
// ========================================================================= //
function initDrawer() {
  const drawer = document.getElementById("muse-drawer");
  const btnClose = document.getElementById("btn-close-drawer");
  const btnToggle = document.getElementById("btn-toggle-artifacts-drawer");

  if (btnClose && drawer) {
    btnClose.addEventListener("click", () => drawer.classList.remove("open"));
  }

  if (btnToggle && drawer) {
    btnToggle.addEventListener("click", () => drawer.classList.toggle("open"));
  }

  document.querySelectorAll(".drawer-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const tabName = tab.getAttribute("data-drawer-tab");
      switchDrawerTab(tabName);
    });
  });

  const btnCopy = document.getElementById("btn-copy-artifact");
  if (btnCopy) {
    btnCopy.addEventListener("click", () => {
      if (activeArtifact && activeArtifact.content) {
        navigator.clipboard.writeText(activeArtifact.content);
        btnCopy.textContent = "Copied!";
        setTimeout(() => { btnCopy.textContent = "Copy"; }, 2000);
      }
    });
  }
}

function switchDrawerTab(tabName) {
  document.querySelectorAll(".drawer-tab").forEach(t => {
    t.classList.toggle("active", t.getAttribute("data-drawer-tab") === tabName);
  });
  document.querySelectorAll(".drawer-pane").forEach(p => {
    p.classList.toggle("active", p.id === `drawer-pane-${tabName}`);
  });
}

// ========================================================================= //
// INTERACTIVE PTY TERMINAL (BASH ON SPARK)                                  //
// ========================================================================= //
function initTerminal() {
  const output = document.getElementById("term-output");
  const input = document.getElementById("term-input");
  const status = document.getElementById("term-status-indicator");
  const btnClear = document.getElementById("btn-term-clear");

  if (btnClear && output) {
    btnClear.addEventListener("click", () => { output.textContent = ""; });
  }

  if (termWs && termWs.readyState === WebSocket.OPEN) return;

  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${location.host}/ws/terminal`;
  termWs = new WebSocket(wsUrl);

  termWs.onopen = () => {
    if (status) {
      status.textContent = "Connected";
      status.style.color = "var(--status-emerald)";
    }
  };

  termWs.onmessage = (event) => {
    if (output) {
      output.textContent += event.data;
      output.scrollTop = output.scrollHeight;
    }
  };

  termWs.onclose = () => {
    if (status) {
      status.textContent = "Disconnected";
      status.style.color = "var(--status-crimson)";
    }
  };

  if (input) {
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const cmd = input.value + "\n";
        if (termWs && termWs.readyState === WebSocket.OPEN) {
          termWs.send(cmd);
        }
        input.value = "";
      }
    });
  }
}

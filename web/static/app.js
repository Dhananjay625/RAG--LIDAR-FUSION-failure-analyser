/* Chat UI for the LiDAR failure analyzer. Conversations live in
   localStorage; the backend is stateless and answers one question at a time. */

const STORAGE_KEY = "lidar-chats-v1";
const REQUEST_TIMEOUT_MS = 180000;
const TITLE_MAX_CHARS = 48;

const EXAMPLES = [
  "My BEV fusion model drops accuracy at night in heavy fog, though each sensor works alone.",
  "LiDAR reports false positives in dense traffic while radar and camera agree the road is clear.",
  "After rain, droplets on the camera lens cut fusion confidence 40% and it never recovers.",
  "Why does sensor fusion degrade at oblique viewing angles?",
];

const el = (id) => document.getElementById(id);
const messagesEl = el("messages");
const chatListEl = el("chat-list");
const inputEl = el("input");
const sendEl = el("send");

let chats = loadChats();
let activeId = chats[0]?.id ?? null;
let isBusy = false;

/* ---------- persistence (immutable updates) ---------- */

function loadChats() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveChats(next) {
  chats = next;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    /* quota or private mode - UI still works for this session */
  }
}

const activeChat = () => chats.find((c) => c.id === activeId) ?? null;

function newChat() {
  const chat = { id: crypto.randomUUID(), title: "New chat", messages: [] };
  saveChats([chat, ...chats]);
  activeId = chat.id;
  render();
}

function deleteChat(id) {
  saveChats(chats.filter((c) => c.id !== id));
  if (activeId === id) activeId = chats[0]?.id ?? null;
  render();
}

function appendMessage(chatId, message) {
  saveChats(
    chats.map((c) =>
      c.id === chatId
        ? {
            ...c,
            title: c.messages.length === 0 && message.role === "user" ? titleFrom(message.text) : c.title,
            messages: [...c.messages, message],
          }
        : c
    )
  );
}

const titleFrom = (text) =>
  text.length > TITLE_MAX_CHARS ? `${text.slice(0, TITLE_MAX_CHARS)}…` : text;

/* ---------- rendering ---------- */

function render() {
  renderChatList();
  renderThread();
  el("chat-title").textContent = activeChat()?.title ?? "New chat";
}

function renderChatList() {
  chatListEl.replaceChildren(
    ...chats.map((chat) => {
      const item = document.createElement("div");
      item.className = `chat-item${chat.id === activeId ? " active" : ""}`;

      const label = document.createElement("span");
      label.textContent = chat.title;
      label.onclick = () => {
        activeId = chat.id;
        render();
      };

      const del = document.createElement("button");
      del.textContent = "×";
      del.title = "Delete chat";
      del.onclick = (e) => {
        e.stopPropagation();
        deleteChat(chat.id);
      };

      item.append(label, del);
      return item;
    })
  );
}

function renderThread() {
  const chat = activeChat();
  if (!chat || chat.messages.length === 0) {
    messagesEl.replaceChildren(emptyState());
    return;
  }
  const thread = document.createElement("div");
  thread.className = "thread";
  chat.messages.forEach((m) => thread.append(messageNode(m)));
  messagesEl.replaceChildren(thread);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function emptyState() {
  const wrap = document.createElement("div");
  wrap.className = "empty";
  wrap.innerHTML =
    "<h2>Diagnose a sensor-fusion failure</h2>" +
    "<p>Ask in plain language. Answers are grounded in your indexed papers.</p>";

  const examples = document.createElement("div");
  examples.className = "examples";
  EXAMPLES.forEach((text) => {
    const btn = document.createElement("div");
    btn.className = "example";
    btn.textContent = text;
    btn.onclick = () => {
      inputEl.value = text;
      autoGrow();
      inputEl.focus();
    };
    examples.append(btn);
  });
  wrap.append(examples);
  return wrap;
}

function messageNode(m) {
  const node = document.createElement("div");
  node.className = `msg ${m.role}${m.isError ? " error" : ""}`;

  const role = document.createElement("div");
  role.className = "role";
  role.textContent = m.role === "user" ? "You" : "Analyzer";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = m.text;

  node.append(role, bubble);
  if (m.role === "assistant" && !m.isError) node.append(metaNode(m), sourcesNode(m));
  if (m.isError && m.retryOf) node.append(retryNode(m.retryOf));
  return node;
}

function retryNode(question) {
  const wrap = document.createElement("div");
  wrap.className = "meta";
  const btn = document.createElement("button");
  btn.className = "retry";
  btn.textContent = "Retry";
  btn.onclick = () => submit(question);
  wrap.append(btn);
  return wrap;
}

function metaNode(m) {
  const meta = document.createElement("div");
  meta.className = "meta";

  if (typeof m.confidence === "number") {
    meta.append(badge(`Confidence ${(m.confidence * 100).toFixed(0)}%`, m.confidence >= 0.75 ? "good" : ""));
  }
  meta.append(badge(m.isValidated ? "Citations validated" : "Not validated", m.isValidated ? "good" : "warn"));
  if (typeof m.elapsed === "number") meta.append(badge(`${m.elapsed.toFixed(1)}s`));
  (m.citations ?? []).forEach((c) => meta.append(badge(c)));
  return meta;
}

function badge(text, variant = "") {
  const b = document.createElement("span");
  b.className = `badge ${variant}`.trim();
  b.textContent = text;
  return b;
}

function sourcesNode(m) {
  const sources = m.sources ?? [];
  const details = document.createElement("details");
  details.className = "sources";
  if (sources.length === 0) {
    details.style.display = "none";
    return details;
  }

  const summary = document.createElement("summary");
  summary.textContent = `${sources.length} retrieved passage${sources.length === 1 ? "" : "s"}`;
  details.append(summary);

  sources.forEach((s) => {
    const row = document.createElement("div");
    row.className = "source";

    const file = document.createElement("div");
    file.className = "file";
    const score = typeof s.score === "number" ? ` · score ${s.score.toFixed(3)}` : "";
    file.textContent = `${s.filename} (chunk ${s.chunk_id ?? "?"})${score}`;

    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.textContent = s.text;

    row.append(file, snippet);
    details.append(row);
  });
  return details;
}

function showThinking() {
  const thread = messagesEl.querySelector(".thread") ?? messagesEl;
  const node = document.createElement("div");
  node.className = "msg assistant";
  node.id = "thinking";
  node.innerHTML =
    '<div class="role">Analyzer</div>' +
    '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span> retrieving papers and reasoning…</div>';
  thread.append(node);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

/* ---------- networking ---------- */

async function ask(question) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = body.detail;
      const message =
        typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `HTTP ${res.status}`;
      throw new Error(message);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function submit(question) {
  if (isBusy || !question) return;
  if (!activeChat()) newChat();

  const chatId = activeId;
  appendMessage(chatId, { role: "user", text: question });
  render();

  isBusy = true;
  sendEl.disabled = true;
  showThinking();

  try {
    const data = await ask(question);
    appendMessage(chatId, {
      role: "assistant",
      text: data.answer,
      confidence: data.confidence,
      citations: data.citations,
      sources: data.sources,
      isValidated: data.is_validated,
      elapsed: data.elapsed_seconds,
    });
  } catch (err) {
    const text =
      err.name === "AbortError"
        ? "The request timed out. The agent can retry retrieval several times on low-confidence answers — check the server logs."
        : `Request failed: ${err.message}`;
    appendMessage(chatId, { role: "assistant", text, isError: true, retryOf: question });
  } finally {
    isBusy = false;
    sendEl.disabled = false;
    render();
    inputEl.focus();
  }
}

async function checkHealth() {
  const dot = el("health-dot");
  const text = el("health-text");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    dot.className = "dot ok";
    text.textContent = data.agent_loaded ? "agent ready" : "server up · agent loads on first ask";
  } catch {
    dot.className = "dot bad";
    text.textContent = "server unreachable";
  }
}

/* ---------- events ---------- */

function autoGrow() {
  inputEl.style.height = "auto";
  inputEl.style.height = `${Math.min(inputEl.scrollHeight, 200)}px`;
}

el("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const question = inputEl.value.trim();
  if (!question) return;
  inputEl.value = "";
  autoGrow();
  submit(question);
});

inputEl.addEventListener("input", autoGrow);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    el("composer").requestSubmit();
  }
});

el("new-chat").onclick = newChat;
el("toggle-sidebar").onclick = () => el("sidebar").classList.toggle("hidden");

render();
checkHealth();

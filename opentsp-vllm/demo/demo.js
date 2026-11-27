"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("request");
const prompt = byId("prompt");
const limit = byId("limit");
const generate = byId("generate");
const stop = byId("stop");
const copy = byId("copy");
const completion = byId("completion");
const status = byId("run-status");
const connection = byId("connection");
const presets = [...document.querySelectorAll("[data-prompt]")];
let active = null;
let modelReady = false;

// The qualified route uses a single four-chip node.
const groups = [
  {list: "layers-first", chips: [0]},
  {list: "layers-return", chips: [1]},
  {list: "layers-last", chips: [2]},
  {list: "heads", chips: [3]},
];
groups.forEach(({list, chips}) => chips.forEach((chip) => {
  const tile = document.createElement("li");
  const stage = document.createElement("strong");
  const physical = document.createElement("span");
  const isHead = list === "heads";
  const layer = 4 * chip + 1;
  const range = `${String(layer).padStart(2, "0")}–${String(layer + 3).padStart(2, "0")}`;
  const label = isHead ? "Head" : `L${range}`;
  stage.textContent = label;
  physical.textContent = `C${chip}`;
  tile.className = "chip";
  const role = isHead ? "Full vocabulary projection" : `Transformer layers ${range}`;
  tile.setAttribute("aria-label", `${role}, chip ${chip}`);
  tile.append(stage, physical);
  byId(list).append(tile);
}));
function showStatus(message, error = false) {
  status.textContent = message;
  status.dataset.error = error;
}

function updateControls() {
  generate.disabled = !modelReady || active !== null;
  stop.disabled = active === null;
  prompt.readOnly = active !== null;
  limit.disabled = active !== null;
  for (const button of presets) button.disabled = active !== null;
  copy.disabled = !completion.textContent;
}

async function checkConnection() {
  connection.disabled = true;
  connection.dataset.state = "checking";
  byId("connection-label").textContent = "Checking vLLM";
  try {
    const response = await fetch("/api/models", {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error("Unavailable");
    const models = await response.json();
    modelReady = models.data.some((model) => model.id === "gpt2");
    if (!modelReady) throw new Error("GPT-2 unavailable");
    connection.dataset.state = "ready";
    byId("connection-label").textContent = "GPT-2 connected";
    if (!active && !completion.textContent) showStatus("Ready to generate.");
  } catch {
    modelReady = false;
    connection.dataset.state = "offline";
    byId("connection-label").textContent = "Reconnect to vLLM";
    if (!active) showStatus("Start the local GPT-2 vLLM server, then reconnect.", true);
  } finally {
    connection.disabled = false;
    updateControls();
  }
}

async function consumeStream(stream, onData) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) throw new Error("The stream ended before vLLM signaled completion.");
      pending += decoder.decode(value, {stream: true});
      let separator;
      while ((separator = /\r?\n\r?\n/.exec(pending))) {
        const event = pending.slice(0, separator.index);
        pending = pending.slice(separator.index + separator[0].length);
        const data = event.split(/\r?\n/).filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart()).join("\n");
        if (!data) continue;
        if (data === "[DONE]") return;
        const chunk = JSON.parse(data);
        if (chunk.error) throw new Error(chunk.error.message || "vLLM reported an error.");
        onData(chunk, performance.now());
      }
    }
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

async function requestCompletion(text, maxTokens, signal, onData, id) {
  const response = await fetch("/api/completions", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({prompt: text, max_tokens: maxTokens, request_id: id}),
    signal,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error?.message || `vLLM returned HTTP ${response.status}.`);
  }
  if (!response.headers.get("Content-Type")?.startsWith("text/event-stream") || !response.body) {
    throw new Error("The server did not return a completion stream.");
  }
  return consumeStream(response.body, onData);
}

const formatSeconds = (ms) => Number.isFinite(ms) && ms >= 0 ? `${(ms / 1000).toFixed(2)} s` : "—";
const formatRate = (tokens, ms) => Number.isInteger(tokens) && tokens >= 0 && ms > 0
  ? `${(1000 * tokens / ms).toFixed(2)} tok/s` : "—";

function observeCompletion(timing, text, usage, receivedAt) {
  if (usage) timing.usage = usage;
  if (!text) return;
  if (timing.firstAt === undefined) {
    timing.firstAt = receivedAt;
    timing.firstCount = usage?.completion_tokens;
  }
  timing.lastAt = receivedAt;
  timing.lastCount = usage?.completion_tokens;
}

function decodeRate(timing) {
  return formatRate(timing.lastCount - timing.firstCount, timing.lastAt - timing.firstAt);
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!modelReady || active) return;
  if (!prompt.value.trim()) {
    showStatus("Enter a prompt to generate text.", true);
    prompt.focus();
    return;
  }

  const controller = new AbortController();
  active = controller;
  completion.textContent = "";
  singleProjection = null;
  renderSingleProjection();
  for (const value of document.querySelectorAll("#generation-view .metrics dd")) value.textContent = "—";
  showStatus("Waiting for the first text from vLLM…");
  const requestId = crypto.randomUUID().replaceAll("-", "");
  const timing = {start: performance.now()};
  updateControls();
  try {
    await requestCompletion(prompt.value, Number(limit.value), controller.signal, (chunk, receivedAt) => {
      const text = chunk.choices?.[0]?.text;
      observeCompletion(timing, text, chunk.usage, receivedAt);
      if (text) {
        byId("metric-first").textContent = formatSeconds(timing.firstAt - timing.start);
        completion.textContent += text;
        completion.scrollTop = completion.scrollHeight;
        showStatus("Streaming from vLLM…");
        updateControls();
      }
      if (Number.isInteger(chunk.usage?.completion_tokens)) {
        byId("metric-tokens").textContent = String(chunk.usage.completion_tokens);
      }
    }, requestId);
    const duration = performance.now() - timing.start;
    byId("metric-total").textContent = formatSeconds(duration);
    byId("metric-input-rate").textContent = formatRate(timing.usage?.prompt_tokens, duration);
    byId("metric-output-rate").textContent = formatRate(timing.usage?.completion_tokens, duration);
    byId("metric-decode-rate").textContent = decodeRate(timing);
    showSingleProjection(requestId, timing.usage);
    showStatus(completion.textContent ? "Generation complete." : "vLLM returned no visible text.");
  } catch (error) {
    showStatus(controller.signal.aborted ? "Generation stopped." : error.message, !controller.signal.aborted);
  } finally {
    if (byId("metric-total").textContent === "—") {
      byId("metric-total").textContent = formatSeconds(performance.now() - timing.start);
    }
    active = null;
    updateControls();
  }
});

for (const button of presets) {
  button.addEventListener("click", () => {
    prompt.value = button.dataset.prompt;
    prompt.focus();
  });
}
prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});
stop.addEventListener("click", () => active?.abort());
copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(completion.textContent);
    copy.textContent = "Copied";
    setTimeout(() => { copy.textContent = "Copy"; }, 1300);
  } catch {
    showStatus("Clipboard unavailable. Select the continuation to copy it.", true);
  }
});
connection.addEventListener("click", checkConnection);
checkConnection();
setInterval(() => {
  if (!modelReady && !connection.disabled && !active) checkConnection();
}, 4000);

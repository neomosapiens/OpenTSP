"use strict";

const lab = byId("serving-lab");
const labForm = byId("lab-form");
const labStatus = byId("lab-status");
const actionButtons = [...lab.querySelectorAll("[data-action]")];
const requestCards = byId("request-cards");
const requestDetails = byId("request-details");
const stopWave = byId("stop-wave");
const execution = byId("live-execution");
const compact = byId("compact-execution");
const latestBatch = byId("latest-batch");
const chipStops = [...byId("live-chip-stops").querySelectorAll(".chip-users")];
const pageGrid = byId("kv-pages");
const wavePerformance = byId("wave-performance");
let waveProjection = null;
let liveThroughput = null;
const sharedPrefix = "The old train crossed the bridge at dusk while the town below turned on its lights";
const requests = new Map();
const maxInFlight = 8;
const wavePrompts = ["Once", "Today", "Suddenly", "Tomorrow", "Meanwhile", "Perhaps", "Yesterday", "After"];
const waveTokens = 6;
const prefixHits = new Map();
let requestNumber = 0;
let history = [];
let eventId = null;
let visiblePage = null;
let capacity = 16;
let waveController = null;
let liveWave = null;
let focusedWave = null;
let liveAdmission = null;
let traceGap = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setView(showLab) {
  lab.hidden = !showLab;
  byId("generation-view").hidden = showLab;
  byId("lab-view").setAttribute("aria-pressed", showLab);
  byId("generate-view").setAttribute("aria-pressed", !showLab);
  byId(showLab ? "lab-clock" : "single-clock").append(byId("clock-settings"));
  if (showLab) refreshLab();
}

byId("generate-view").addEventListener("click", () => setView(false));
byId("lab-view").addEventListener("click", () => setView(true));
compact.addEventListener("click", () => {
  const collapsed = execution.toggleAttribute("data-compact");
  compact.setAttribute("aria-expanded", !collapsed);
  compact.textContent = collapsed ? "Expand" : "Compact";
});
stopWave.addEventListener("click", () => waveController?.abort());

function updateLab() {
  const inFlight = [...requests.values()].filter((request) => request.running).length;
  const completed = [...requests.values()].filter((request) => request.card.dataset.state === "complete").length;
  byId("lab-active").textContent = inFlight ? `${inFlight} active` : completed ? `${completed} completed` : "Ready to start";
  stopWave.hidden = !waveController;
  byId("lab-send").disabled = !modelReady || inFlight >= maxInFlight;
  for (const button of actionButtons) {
    button.disabled = !modelReady || (["pair", "eight"].includes(button.dataset.action) ? !!waveController || inFlight !== 0 : inFlight >= maxInFlight);
  }
}

function streamCard(text, index, budget, waiting) {
  const label = `U${index}`;
  const hue = (index - 1) % maxInFlight;
  const card = element("article", "request-card");
  card.dataset.hue = hue;
  card.dataset.state = waiting ? "ready" : "queued";
  const head = element("div", "request-card-head");
  const state = element("span", "request-state", waiting ? "Ready" : "Queued");
  const counter = element("span", "request-count", `0 / ${budget} tokens`);
  const metadata = element("div", "request-meta");
  metadata.append(state, counter);
  head.append(element("h3", "", `User ${index}`), metadata);
  const textLine = element("p", "request-text");
  const input = element("strong", "request-prompt", text);
  const output = element("span", "request-output");
  output.setAttribute("aria-label", `Generated text for User ${index}`);
  textLine.append(input, output);
  const footer = element("div", "request-card-foot");
  const provenance = element("p", "request-provenance");
  card.append(head, textLine, provenance, footer);
  return {card, label, hue, state, output, counter, provenance, footer};
}

function addLabRequest(text, title, controller, id, budget, wave = false) {
  requestCards.querySelectorAll("[data-example]").forEach((node) => node.remove());
  const stream = streamCard(text, ++requestNumber, budget, false);
  const {card, label, hue, footer} = stream;
  if (wave) card.dataset.waveRequest = "true";
  let stopRequest;
  if (!wave) {
    stopRequest = element("button", "request-stop", "Stop");
    stopRequest.type = "button";
    stopRequest.setAttribute("aria-label", `Stop User ${requestNumber}`);
    stopRequest.addEventListener("click", () => controller.abort());
    footer.append(stopRequest);
  }
  const detail = element("article", "request-detail");
  detail.dataset.hue = hue;
  const facts = element("p", "request-facts", "Awaiting scheduler");
  const progress = element("p", "request-progress");
  const performanceLine = element("p", "request-timing");
  const siliconLine = element("p", "request-timing request-projection");
  detail.append(element("strong", "", `${label} · ${title}`), facts, progress, performanceLine, siliconLine);
  requestCards.append(card);
  requestDetails.append(detail);
  const request = {...stream, id, detail, facts, progress, stopRequest, performanceLine, siliconLine, budget,
    timing: {}, points: [], peak: 0, running: true};
  requests.set(id, request);
  if (requests.size > maxInFlight) {
    const oldest = [...requests.values()].find((entry) => !entry.running);
    oldest.card.remove();
    oldest.detail.remove();
    requests.delete(oldest.id);
    prefixHits.delete(oldest.id);
  }
  return request;
}

function finishLab(request, message, finishedAt = performance.now()) {
  const timing = request.timing;
  const count = timing.usage?.completion_tokens ?? null;
  request.state.textContent = message === undefined ? "Done" : message === "Stopped" ? "Stopped" : "Error";
  request.state.title = message ?? "Generation complete";
  request.card.dataset.state = message === undefined ? "complete" : message === "Stopped" ? "stopped" : "error";
  if (message && message !== "Stopped") request.footer.append(element("p", "request-error", message));
  request.running = false;
  if (request.stopRequest) request.stopRequest.disabled = true;
  if (message === undefined) {
    const duration = finishedAt - timing.start;
    request.performanceLine.replaceChildren(
      element("strong", "", "Software ISS"),
      element("span", "", `${formatSeconds(timing.firstAt - timing.start)} first token`),
      element("span", "", `${formatRate(timing.usage?.prompt_tokens, duration)} in`),
      element("span", "", `${formatRate(count, duration)} out`),
      element("span", "", `${decodeRate(timing)} decode`),
    );
  }
  updateLab();
}

function addLabText(request, text, usage, receivedAt, output) {
  if (output) {
    request.provenance.dataset.sequence = output.seq;
    if (output.verified) {
      const first = output.ordinal;
      const last = first + output.ids.length - 1;
      request.firstOutput ??= {batch: output.batches[0], ordinal: first};
      request.lastOutput = {batch: output.batches.at(-1), ordinal: last};
      const batches = [...new Set(output.batches)].map(batch => String(batch).padStart(2, "0")).join("/");
      request.provenance.textContent = `B${batches} · E${String(output.seq).padStart(2, "0")} · Token ${first === last ? first : `${first}–${last}`}`;
    } else {
      request.provenance.textContent = "Output trace unavailable";
    }
  }
  if (text && request.timing.firstAt === undefined) {
    request.performanceLine.textContent = `Software ISS · ${formatSeconds(receivedAt - request.timing.start)} first token`;
  }
  observeCompletion(request.timing, text, usage, receivedAt);
  if (Number.isInteger(request.timing.usage?.completion_tokens)) {
    request.counter.textContent = `${request.timing.usage.completion_tokens} / ${request.budget} tokens`;
  }
  if (text) {
    request.output.append(element("span", "arrived-text", text));
    request.state.textContent = "Generating";
    request.card.dataset.state = "streaming";
  }
  if (output) renderPath();
}

function renderRequestProjection(request) {
  if (!request.projection) return;
  const {choice, start} = request.projection;
  request.siliconLine.replaceChildren(element("strong", "", "Projected TSP silicon"));
  if (!choice) {
    request.siliconLine.append(element("span", "", "Projection unavailable"));
    return;
  }
  const fields = projectedChoice(choice, request.timing.usage, start);
  request.siliconLine.append(
    element("span", "", `${fields.first} first token`),
    element("span", "", `${fields.input} in`),
    element("span", "", `${fields.output} out`),
    element("span", "", `${fields.decode} decode`),
  );
}

function renderLiveThroughput(rate = liveThroughput) {
  const rateNumber = (value) => Number.isFinite(value)
    ? value.toLocaleString("en-US", {minimumFractionDigits: 2, maximumFractionDigits: 2}) : "—";
  const iss = rate?.wallMs > 0 ? 1000 * rate.tokens / rate.wallMs : NaN;
  const siliconMs = rate?.endCycle === undefined ? NaN : milliseconds(rate.endCycle - rate.startCycle);
  const silicon = siliconMs > 0 && rate.projectedTokens === rate.tokens ? 1000 * rate.tokens / siliconMs : NaN;
  const users = rate?.users;
  byId("iss-tps").textContent = rateNumber(iss);
  byId("iss-tps-user").textContent = rateNumber(iss / users);
  byId("silicon-tps").textContent = rateNumber(silicon);
  byId("silicon-tps-user").textContent = rateNumber(silicon / users);
  byId("silicon-clock").textContent = projectionClock.validity.valid ? `· ${projectionClock.valueAsNumber} MHz` : "";
}

function updateLiveThroughput(rate, outputs, now) {
  for (const output of outputs ?? []) {
    const tokens = output.ids.length;
    rate.tokens += tokens;
    const ends = output.batches.map(step => rate.cycleEnds.get(step));
    if (output.verified && ends.length === tokens && ends.every(Number.isSafeInteger)) {
      rate.projectedTokens += tokens;
      rate.endCycle = Math.max(rate.endCycle ?? 0, ...ends);
    }
  }
  if (rate.tokens) {
    rate.wallMs = now - rate.startMs;
    renderLiveThroughput(rate);
  }
}

function renderWaveProjection() {
  const choices = Object.values(waveProjection?.choices ?? {});
  const span = choices.length ? Math.max(...choices.map(row => row.outputs.at(-1))) - waveProjection.start : NaN;
  const fields = projectedRates(waveProjection?.usage, milliseconds(span));
  for (const key of ["input", "output", "total"]) byId(`wave-projection-${key}`).textContent = fields[key];
}

projectionClock.addEventListener("input", () => {
  renderLiveThroughput();
  renderWaveProjection();
  for (const request of requests.values()) renderRequestProjection(request);
});

async function showLabProjection(tag, wave, aggregateUsage) {
  const state = {usage: aggregateUsage};
  if (aggregateUsage) {
    waveProjection = state;
    renderWaveProjection();
  }
  const choices = await requestProjection(tag, wave.map(request => request.timing.usage?.completion_tokens));
  const start = choices ? Math.min(...Object.values(choices).map(row => row.start)) : undefined;
  wave.forEach((request, index) => {
    request.projection = {choice: choices?.[index], start};
    renderRequestProjection(request);
  });
  if (waveProjection === state) {
    state.choices = choices;
    state.start = start;
    renderWaveProjection();
  }
}

async function submitLab(text, maxTokens, title = "Custom", afterFirstText) {
  if (!modelReady || [...requests.values()].filter((request) => request.running).length >= maxInFlight) return;
  if (!waveController) {
    liveThroughput = null;
    renderLiveThroughput();
  }
  const controller = new AbortController();
  const id = crypto.randomUUID().replaceAll("-", "");
  const request = addLabRequest(text, title, controller, id, maxTokens);
  updateLab();
  try {
    request.timing.start = performance.now();
    await requestCompletion(text, maxTokens, controller.signal, (chunk, receivedAt) => {
      if (chunk.opentsp_events) applyLiveEvents(chunk.opentsp_events);
      const next = chunk.choices?.[0]?.text;
      if (!chunk.opentsp_events) addLabText(request, next, chunk.usage, receivedAt, chunk.opentsp_output?.[0]);
      if (next && afterFirstText) {
        afterFirstText();
        afterFirstText = null;
      }
    }, id);
    finishLab(request);
    void showLabProjection(id, [request]);
  } catch (error) {
    finishLab(request, controller.signal.aborted ? "Stopped" : error.message);
  } finally {
    updateLab();
    if (afterFirstText && !controller.signal.aborted) labStatus.textContent = "Pair did not start. The first request produced no visible text.";
  }
}

async function submitWave() {
  if (!modelReady || waveController || [...requests.values()].some((request) => request.running)) return;
  const controller = new AbortController();
  waveController = controller;
  const tag = crypto.randomUUID().replaceAll("-", "");
  requestCards.replaceChildren();
  requestDetails.replaceChildren();
  requests.clear();
  prefixHits.clear();
  requestNumber = 0;
  const wave = wavePrompts.map((prompt, index) => addLabRequest(prompt, "Concurrent run", controller, `${tag}/${index}`, waveTokens, true));
  liveWave = focusedWave = liveAdmission = traceGap = null;
  showLiveChips();
  renderLiveBatch();
  wavePerformance.hidden = true;
  waveProjection = null;
  liveThroughput = null;
  renderLiveThroughput();
  renderWaveProjection();
  updateLab();
  try {
    // One HTTP stream avoids the browser's HTTP/1.1 connection limit. vLLM
    // queues one engine request per prompt and indexes the streamed choices.
    const start = performance.now();
    const rate = {tag, startMs: start, tokens: 0, projectedTokens: 0, users: wave.length, cycleEnds: new Map()};
    liveThroughput = rate;
    for (const request of wave) request.timing.start = start;
    let usage;
    await requestCompletion(wavePrompts, waveTokens, controller.signal, (chunk, receivedAt) => {
      if (chunk.opentsp_events) {
        applyLiveEvents(chunk.opentsp_events);
        return;
      }
      updateLiveThroughput(rate, chunk.opentsp_output, receivedAt);
      if (!chunk.choices?.length && chunk.usage) usage = chunk.usage;
      for (const choice of chunk.choices ?? []) {
        const request = wave[choice.index];
        if (!request) continue;
        addLabText(request, choice.text, chunk.usage, receivedAt, chunk.opentsp_output?.find(item => item.index === choice.index));
        if (choice.finish_reason) finishLab(request, undefined, receivedAt);
      }
    }, tag);
    if (usage && wave.every((request) => !request.running)) {
      const duration = performance.now() - start;
      rate.tokens = usage.completion_tokens;
      rate.wallMs = duration;
      renderLiveThroughput(rate);
      byId("wave-input-rate").textContent = formatRate(usage.prompt_tokens, duration);
      byId("wave-output-rate").textContent = formatRate(usage.completion_tokens, duration);
      byId("wave-total").textContent = formatSeconds(duration);
      wavePerformance.hidden = false;
      void showLabProjection(tag, wave, usage);
    }
    for (const request of wave) {
      if (request.running) finishLab(request, "Stream ended before this request completed.");
    }
  } catch (error) {
    for (const request of wave) {
      if (request.running) finishLab(request, controller.signal.aborted ? "Stopped" : error.message);
    }
  } finally {
    waveController = null;
    updateLab();
  }
}

labForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = byId("lab-prompt").value.trim();
  if (text) submitLab(text, Number(byId("lab-limit").value));
  else byId("lab-prompt").focus();
});
for (const button of actionButtons) {
  button.addEventListener("click", () => {
    switch (button.dataset.action) {
      case "warm": submitLab(`${sharedPrefix} and`, 1, "Warm prefix"); break;
      case "pair":
        submitLab(`${sharedPrefix} before`, 5, "Pair A", () => submitLab(`${sharedPrefix} again`, 3, "Pair B"));
        break;
      case "eight": submitWave(); break;
      case "grow":
        submitLab("A small blue boat drifted across the lake as the sun sank behind distant hills", 4, "Grow KV");
        break;
      case "cold": submitLab("A quiet garden", 2, "Cold control"); break;
    }
  });
}

const eventKey = (event) => `${event.step}/${event.phase}`;
const requestName = (id) => requests.get(id)?.label ?? `External ${id.slice(0, 5)}`;
const pageState = (refs, hashed) => refs > 1 ? "shared" : refs ? "active" : hashed ? "retained" : "free";
const labDiagnostics = byId("lab-diagnostics");
let batchKey = null;
let chipKey = null;

function userBadge(id, kind) {
  const request = requests.get(id);
  const badge = element("span", "user-badge");
  badge.dataset.hue = request?.hue ?? "external";
  badge.dataset.kind = kind ?? "?";
  const icon = element("b", "kind-icon", badge.dataset.kind);
  icon.dataset.kind = badge.dataset.kind;
  icon.setAttribute("aria-hidden", "true");
  const label = request?.label ?? "Other";
  badge.append(icon, document.createTextNode(label));
  badge.title = `${label} · ${kind === "P" ? "Prefill" : kind === "D" ? "Decode" : "Phase pending"}`;
  return badge;
}

function batchCounts(frame) {
  const rows = frame?.requests ?? [];
  return [rows.reduce((sum, row) => sum + row.prefill, 0), rows.reduce((sum, row) => sum + row.decode, 0)];
}

function renderLiveBatch() {
  const frame = liveAdmission;
  const key = frame?.step ?? -1;
  if (batchKey === key) return;
  batchKey = key;
  const tokens = [];
  if (frame) latestBatch.dataset.step = frame.step;
  else delete latestBatch.dataset.step;
  for (const row of frame?.requests ?? []) {
    for (const [count, kind] of [[row.prefill, "P"], [row.decode, "D"]]) {
      if (!count) continue;
      const badge = userBadge(row.id, kind);
      if (count > 1) badge.append(document.createTextNode(` ×${count}`));
      badge.title += ` · ${count} row${count === 1 ? "" : "s"}`;
      tokens.push(badge);
    }
  }
  latestBatch.replaceChildren(...tokens);
  const [p, d] = batchCounts(frame);
  byId("latest-batch-label").textContent = frame
    ? `B${String(frame.step).padStart(2, "0")} · ${p}P + ${d}D` : "Waiting for a batch";
}

function classifyWave(batch, indexes, positions) {
  if (!batch || indexes.length !== positions?.length) return null;
  const pending = new Map(batch.requests.filter(r => r.prefill + r.decode > 0)
    .map(row => [row.id, row]));
  if (indexes.length !== [...pending.values()].reduce((total, row) => total + row.prefill + row.decode, 0)) return null;
  const seen = new Set();
  const kinds = indexes.map((id, i) => {
    const scheduled = pending.get(id);
    const position = positions[i];
    const key = `${id}/${position}`;
    if (!scheduled || seen.has(key) || !Number.isInteger(position) ||
        position < scheduled.filled || position >= scheduled.filled + scheduled.prefill + scheduled.decode) return null;
    seen.add(key);
    return position < scheduled.prompt ? "P" : "D";
  });
  return kinds.some(kind => kind === null) ? null : kinds;
}

function renderPath() {
  const title = byId("live-path-title");
  const path = byId("live-path-steps");
  const request = [...requests.values()].find(r => r.label === "U1");
  if (request && liveWave?.row_ids.includes(request.id)) focusedWave = liveWave;
  const wave = focusedWave;
  const current = wave?.row_ids.findIndex((id, row) => id === request?.id && wave.row_logit?.[row]);
  const row = current >= 0 ? current : wave?.row_ids.indexOf(request?.id) ?? -1;
  const stage = row >= 0 ? wave.stages[row] : -1;
  const phase = row >= 0 ? wave.kinds?.[row] : null;
  const output = request?.lastOutput;
  const firstOutput = request?.firstOutput;
  const admitted = wave?.batch.requests.find(r => r.id === request?.id);
  const token = current >= 0 && admitted ? wave.row_positions[current] - admitted.prompt + 2 : null;
  const batch = value => `B${String(value).padStart(2, "0")}`;
  const from = firstOutput && phase === "D" ? `P ${batch(firstOutput.batch)} → sampled token 1 → ` : "";
  const pass = phase ? `${phase} ${batch(wave.step)}` : "Phase unavailable";
  title.textContent = stage < 0 ? "Follow U1 through the pipeline"
    : `Follow U1 · ${from}${pass}${token === null ? "" : ` · token ${token}`}`;
  const doneText = output && output.batch === wave?.step && output.ordinal >= token;
  const labels = ["Admitted", "C0", "C1", "C2", "C3", doneText ? "Text streamed" : stage === 5 ? "Logits ready" : "Text pending"];
  path.replaceChildren(...labels.map((name, index) => {
    const item = element("li", "", name);
    item.toggleAttribute("data-reached", index <= stage);
    item.toggleAttribute("data-current", index === stage);
    return item;
  }));
}

function showLiveChips(wave, changes = []) {
  const stops = chipStops.map(() => []);
  const handoffs = new Map(changes.filter(move => move[1] === wave?.stages[move[0]]).map(move => [move[0], move[1]]));
  wave?.row_ids.forEach((id, row) => {
    const stage = wave.stages[row];
    stops[stage].push({id, kind: wave.kinds?.[row], key: `${wave.step}/${row}`, arrival: handoffs.has(row)});
  });
  const key = JSON.stringify([stops, traceGap]);
  if (chipKey === key) return;
  chipKey = key;
  for (const stop of chipStops) {
    stop.parentElement.removeAttribute("data-handoff");
    delete stop.parentElement.dataset.handoffHue;
  }
  for (const [index, rows] of stops.entries()) {
    const stop = chipStops[index];
    stop.replaceChildren(...rows.map(row => {
      const badge = userBadge(row.id, row.kind);
      badge.dataset.row = row.key;
      if (row.arrival && index > 0) {
        badge.dataset.arrived = "";
        badge.title += ` · arrived from ${index === 1 ? "host" : `C${index - 2}`}`;
        const source = chipStops[index - 1].parentElement;
        source.dataset.handoff = "";
        source.dataset.handoffHue = badge.dataset.hue;
      }
      return badge;
    }));
    stop.parentElement.toggleAttribute("data-occupied", rows.length > 0);
    stop.toggleAttribute("data-packed", rows.length > 4);
    stop.parentElement.toggleAttribute("data-followed", rows.some(row => requests.get(row.id)?.label === "U1"));
  }
  byId("live-chip-status").textContent = traceGap ? "Trace incomplete · last confirmed"
    : wave ? `Batch ${wave.step} · ${wave.done ? "logits ready" : "observed simulator progress"}` : "Waiting for input";
  byId("live-chip-status").title = traceGap ?? "";
  renderPath();
}

function applyLiveEvents(events) {
  for (const event of events) {
    if (liveThroughput && event.phase === "pipeline" &&
        event.row_ids.some(id => id.startsWith(liveThroughput.tag + "/")) &&
        Number.isSafeInteger(event.wave_start) && Number.isSafeInteger(event.cycles) && event.cycles > 0) {
      liveThroughput.startCycle ??= event.wave_start;
      liveThroughput.cycleEnds.set(event.step, event.wave_start + event.cycles);
    }
    if (event.phase === "trace-gap") {
      traceGap = event.reason;
      showLiveChips(liveWave);
    } else if (event.phase === "scheduled" && event.step >= (liveAdmission?.step ?? 0)) {
      liveAdmission = event;
      renderLiveBatch();
    } else if (event.phase === "wave-plan" && event.step === liveAdmission?.step &&
               event.step > (liveWave?.step ?? 0)) {
      const kinds = classifyWave(liveAdmission, event.row_ids, event.row_positions);
      traceGap = kinds ? null : "The worker rows did not match the admitted batch";
      liveWave = {...event, batch: liveAdmission, kinds, stages: event.row_ids.map(() => 0), positionsByCycle: new Map()};
      showLiveChips(liveWave);
    } else if (event.phase === "wave-positions" && event.step === liveWave?.step) {
      const stages = event.stages;
      const prior = liveWave.positionsByCycle.get(event.cycle);
      // Concurrent request streams carry the same observed checkpoint. A late
      // copy can arrive after another stream has already shown the logits.
      if (prior && Array.isArray(stages) && stages.length === prior.length &&
          stages.every((stage, row) => stage === prior[row])) continue;
      if (prior || !Number.isSafeInteger(event.cycle) || !Array.isArray(stages) ||
          stages.length !== liveWave.stages.length ||
          stages.some((stage, row) => !Number.isInteger(stage) || stage < liveWave.stages[row] || stage > 4)) {
        traceGap = "A receive checkpoint is inconsistent with the preceding checkpoint";
        showLiveChips(liveWave);
        continue;
      }
      liveWave.positionsByCycle.set(event.cycle, stages);
      liveWave.stages = stages;
      showLiveChips(liveWave, event.moves);
    } else if (event.phase === "pipeline" && event.step === liveWave?.step) {
      liveWave.done = true;
      if (!traceGap && liveWave.stages.every(stage => stage === 4)) {
        liveWave.stages = liveWave.stages.map(() => 5);
      }
      showLiveChips(liveWave);
    }
  }
}

const pageButtons = Array.from({length: 64}, (_, index) => {
  const button = element("button", "page-button", String(index));
  button.type = "button";
  button.dataset.page = index;
  button.addEventListener("click", () => {
    visiblePage = index;
    renderEvents();
  });
  pageGrid.append(button);
  return button;
});

function renderBatches(index) {
  const rows = [];
  const batches = history.slice(0, index + 1).filter((item) => item.phase === "scheduled").slice(-5);
  for (const frame of batches) {
    const batch = element("div", "batch-row");
    const tokens = element("div", "batch-tokens");
    const count = frame.requests.filter((row) => row.prefill + row.decode > 0).length;
    batch.dataset.requests = count;
    batch.append(element("span", "batch-number", `#${frame.step}`), tokens,
      element("span", "batch-count", `${count} req`));
    for (const row of frame.requests) {
      for (const [howMany, label] of [[row.prefill, "P"], [row.decode, "D"]]) {
        for (let j = 0; j < howMany; j++) {
          const request = requests.get(row.id);
          const token = element("span", "batch-token", request ? `${request.label} ${label}` : label);
          token.dataset.hue = requests.get(row.id)?.hue ?? "external";
          token.title = `${requestName(row.id)} · ${label === "P" ? "Prefill" : "Decode"}`;
          token.setAttribute("aria-label", token.title);
          tokens.append(token);
        }
      }
    }
    rows.push(batch);
  }
  byId("batch-rows").replaceChildren(...rows);
}

function renderPages(frame) {
  const pageCount = frame?.capacity ?? capacity;
  if (visiblePage >= pageCount) visiblePage = null;
  const pages = new Map(frame?.pages?.map(([id, refs, hashed]) => [id, {refs, hashed}]) ?? []);
  for (const [index, button] of pageButtons.entries()) {
    button.hidden = index >= pageCount;
    const {refs, hashed} = pages.get(index + 1) ?? {refs: 0, hashed: false};
    const state = frame ? pageState(refs, hashed) : "unknown";
    button.dataset.pageState = state;
    button.setAttribute("aria-pressed", index === visiblePage);
    button.setAttribute("aria-label", `Page ${index}, ${state === "active" ? "held" : state}`);
  }
  const blocks = [...pages.values()];
  byId("kv-usage").textContent = frame
    ? `${blocks.filter((page) => page.refs).length} held · ${blocks.filter((page) => !page.refs).length} retained / ${pageCount}`
    : `${pageCount} pages · 16 slots each`;

  if (visiblePage === null) return;
  const page = visiblePage;
  const block = page + 1;
  const slot0 = page * 16;
  const keyWord = 0x1400 + Math.floor(page / 8) * 16 + (page % 8) * 2;
  const hex = (word) => `0x${word.toString(16)}`;
  byId("page-heading").textContent = `Page ${page} · block ${block}`;
  byId("page-info").textContent = `Slots ${slot0}–${slot0 + 15} in each layer's resident cache.`;
  const range = (first, words) => `${hex(first)}–${hex(first + words - 1)}`;
  const keys = Array.from({length: 4}, (_, layer) => range(keyWord + 0x100 * layer, 2));
  const values = Array.from({length: 4}, (_, layer) => range(0x100 + slot0 + 0x100 * layer, 16));
  byId("page-address").textContent = `K tile 0 · slices 44–59 · ${keys.join(" / ")}. `
    + `V · alternating slice pairs within 76–87 · ${values.join(" / ")}. Each range is one layer.`;
  const owners = frame?.requests.filter((row) => row.blocks.includes(block));
  const held = pages.get(block);
  byId("page-owners").textContent = held?.refs
    ? `${held.refs} reference${held.refs === 1 ? "" : "s"}${owners?.length ? ` · ${owners.map((row) => requestName(row.id)).join(", ")}` : ""}`
    : held?.hashed ? "Retained for prefix reuse · zero live references" : frame ? "Free to allocate" : "Residency data pending";
}

function renderEvents() {
  renderLiveBatch();
  if (!labDiagnostics.open) return;
  let index = eventId === null ? history.length - 1 : history.findIndex((event) => eventKey(event) === eventId);
  if (index < 0 && history.length) {
    eventId = null;
    index = history.length - 1;
  }
  const frame = history[index];
  byId("step-prev").disabled = index < 1;
  byId("step-next").disabled = eventId === null || index >= history.length - 1;
  byId("step-live").setAttribute("aria-pressed", eventId === null);
  byId("step-label").textContent = frame
    ? `Step ${frame.step} · ${frame.phase === "scheduled" ? "admitted" : "settled"} · ${eventId === null ? "live" : "paused"}`
    : "Waiting for a scheduler step.";
  renderBatches(index);
  renderPages(frame);
  for (const request of requests.values()) {
    const row = frame?.requests.find((item) => item.id === request.id);
    const prefix = prefixHits.get(request.id);
    const parts = [];
    if (prefix !== undefined && (prefix > 0 || !request.card.dataset.waveRequest)) parts.push(`${prefix} prefix tokens reused`);
    if (row) parts.push(`${row.filled} KV positions · ${row.blocks.filter(Boolean).length} held pages`);
    if (parts.length) request.facts.textContent = parts.join(" · ");
    else request.facts.textContent = request.running ? "Awaiting scheduler" : "No scheduler observation";
  }
}

function scrub(delta) {
  const current = eventId === null ? history.length - 1 : history.findIndex((event) => eventKey(event) === eventId);
  const next = history[current + delta];
  if (next) eventId = eventKey(next);
  renderEvents();
}
byId("step-prev").addEventListener("click", () => scrub(-1));
byId("step-next").addEventListener("click", () => scrub(1));
byId("step-live").addEventListener("click", () => { eventId = null; renderEvents(); });
labDiagnostics.addEventListener("toggle", renderEvents);

let refreshing = false;
async function refreshLab() {
  if (refreshing) return;
  refreshing = true;
  updateLab();
  try {
    const response = await fetch("/api/events", {signal: AbortSignal.timeout(3000)});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || "Unable to read live scheduler events.");
    capacity = data.capacity ?? 16;
    history = data.events.filter((event) => event.phase === "scheduled" || event.phase === "settled");
    for (const event of history) {
      for (const row of event.requests) {
        const request = requests.get(row.id);
        if (!request) continue;
        if (row.prefix !== undefined) prefixHits.set(row.id, row.prefix);
        if (request.lastObservation === undefined || event.step * 2 + Number(event.phase === "settled") > request.lastObservation) {
          request.lastObservation = event.step * 2 + Number(event.phase === "settled");
          const pages = row.blocks.filter(Boolean).length;
          if (request.points.at(-1) !== row.filled) request.points.push(row.filled);
          if (request.points.length > 10) request.points.shift();
          request.peak = Math.max(request.peak, pages);
        }
      }
    }
    for (const request of requests.values()) {
      request.progress.textContent = request.points.length > 1
        ? `KV positions ${request.points.join(" → ")} · peak ${request.peak} held` : "";
    }
    if (visiblePage === null && history.at(-1)?.pages.length) visiblePage = history.at(-1).pages[0][0] - 1;
    labStatus.textContent = !data.available ? "Start the browser with just gpt2-demo to see live execution."
      : data.limited ? "Execution trace is full. Restart the demo to capture new activity." : "";
    labStatus.hidden = !labStatus.textContent;
    renderEvents();
  } catch (error) {
    labStatus.textContent = error.message;
    labStatus.hidden = false;
  } finally {
    refreshing = false;
  }
}
wavePrompts.forEach((text, index) => {
  const {card} = streamCard(text, index + 1, waveTokens, true);
  card.setAttribute("data-example", "");
  requestCards.append(card);
});
setView(true);
setInterval(() => { if (!lab.hidden) refreshLab(); }, 180);
renderEvents();

"use strict";

const projectionClock = byId("projection-clock");
let singleProjection = null;

async function requestProjection(tag, tokenCounts) {
  try {
    const response = await fetch(`/api/projection?request_id=${tag}`, {signal: AbortSignal.timeout(5000)});
    if (!response.ok) return null;
    const {choices} = await response.json();
    return tokenCounts.every((count, index) =>
      count > 0 && choices?.[index]?.outputs.length === count) ? choices : null;
  } catch {
    return null;
  }
}

const milliseconds = (cycles) => projectionClock.validity.valid
  ? cycles / (projectionClock.valueAsNumber * 1000) : NaN;
const formatMilliseconds = (ms) => Number.isFinite(ms) ? `${ms.toFixed(3)} ms` : "—";

function projectedRates(usage, duration, first = NaN, last = NaN) {
  return {
    first: formatMilliseconds(first),
    total: formatMilliseconds(duration),
    input: formatRate(usage?.prompt_tokens, duration),
    output: formatRate(usage?.completion_tokens, duration),
    decode: formatRate(usage?.completion_tokens - 1, last - first),
  };
}

function projectedChoice(choice, usage, start = choice?.start) {
  if (!choice) return projectedRates();
  const first = milliseconds(choice.outputs[0] - start);
  const last = milliseconds(choice.outputs.at(-1) - start);
  return projectedRates(usage, last, first, last);
}

function renderSingleProjection() {
  const choice = singleProjection?.choices?.[0];
  const fields = projectedChoice(choice, singleProjection?.usage);
  fields.cycles = choice ? (choice.outputs.at(-1) - choice.start).toLocaleString("en-US") : "—";
  for (const [key, value] of Object.entries(fields)) byId(`projection-${key}`).textContent = value;
  byId("projection-status").textContent = !singleProjection ? "Generate a response to project its device timing."
    : singleProjection.choices === undefined ? "Reading simulation cycles…"
    : !choice ? "Projection unavailable for this response."
    : !Number.isFinite(milliseconds(1)) ? "Enter a clock from 1 to 10,000 MHz."
    : "Projection preserves wave completion barriers. It estimates device time at the selected clock.";
}

function showSingleProjection(tag, usage) {
  const state = {usage};
  singleProjection = state;
  renderSingleProjection();
  requestProjection(tag, [usage?.completion_tokens]).then(choices => {
    state.choices = choices;
    if (singleProjection === state) renderSingleProjection();
  });
}

projectionClock.addEventListener("input", renderSingleProjection);

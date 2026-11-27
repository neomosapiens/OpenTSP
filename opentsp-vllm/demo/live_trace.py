"""Correlate actual engine output with the same run's completion stream."""
from collections import defaultdict, deque
import json
import time


class LiveTrace:
    def __init__(self, path, tag, wave):
        self.path = path
        self.tag = tag
        self.wave = wave
        self.offset = path.stat().st_size
        self.scheduled = {}
        self.egress = {}
        self.sampled = {}
        self.tokens = defaultdict(deque)
        self.ordinals = defaultdict(int)
        self.sequence = 0

    def includes(self, request):
        return request == self.tag or request.startswith(self.tag + "/")

    def events(self):
        with self.path.open("rb") as file:
            file.seek(self.offset)
            while line := file.readline():
                if not line.endswith(b"\n"):
                    break
                self.offset += len(line)
                event = json.loads(line)
                step = event.get("step")
                phase = event["phase"]
                if phase == "scheduled":
                    if not any(self.includes(row["id"]) and row["prefill"] + row["decode"] for row in event["requests"]):
                        continue
                    self.scheduled[step] = {row["id"] for row in event["requests"] if row["prefill"] + row["decode"]}
                elif step not in self.scheduled:
                    continue
                if phase == "pipeline":
                    self.egress[step] = {
                        row for row, flag in zip(event["row_ids"], event["logit_rows"], strict=True) if flag
                    }
                elif phase == "sampled":
                    for span in event["tokens"]:
                        request = span["id"]
                        for offset, token in enumerate(span["ids"]):
                            self.sampled[request, span["ordinal"] + offset] = (token, step)
                    continue
                elif phase == "tokens":
                    for span in event["tokens"]:
                        request = span["id"]
                        if not self.includes(request):
                            continue
                        valid = request in self.scheduled[step] and request in self.egress.get(step, ())
                        for offset, token in enumerate(span["ids"]):
                            ordinal = span["ordinal"] + offset
                            sampled = self.sampled.pop((request, ordinal), None)
                            self.tokens[request].append((ordinal, token, step, valid and sampled == (token, step)))
                    continue
                if phase in {"scheduled", "wave-plan", "wave-positions", "pipeline", "trace-gap", "settled"}:
                    yield event

    def annotate(self, chunk):
        outputs = []
        for choice in chunk.get("choices", []):
            ids = choice.get("token_ids") or []
            if not ids and not choice.get("text"):
                continue
            index = choice["index"]
            request = f"{self.tag}/{index}" if self.wave else self.tag
            queue = self.tokens[request]
            ordinal = self.ordinals[request] + 1
            rows = [queue.popleft() for _ in range(min(len(ids), len(queue)))]
            valid = bool(ids) and len(rows) == len(ids) and all(
                available and token == ids[i] and number == ordinal + i
                for i, (number, token, _, available) in enumerate(rows)
            )
            self.ordinals[request] += len(ids)
            self.sequence += 1
            outputs.append(dict(index=index, seq=self.sequence, ordinal=ordinal, ids=ids,
                                batches=[row[2] for row in rows] if valid else [],
                                verified=valid, relayed_ns=time.monotonic_ns()))
        if outputs:
            chunk["opentsp_output"] = outputs
        return outputs

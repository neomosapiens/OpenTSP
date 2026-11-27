"""Serve the GPT-2 pipeline demo and relay its requests to local vLLM."""
import argparse
import http.client
import json
import os
import re
from queue import Empty, Queue
from threading import Event, Thread
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/demo.css": ("demo.css", "text/css; charset=utf-8"),
    "/demo.js": ("demo.js", "text/javascript; charset=utf-8"),
    "/projection.js": ("projection.js", "text/javascript; charset=utf-8"),
    "/lab.js": ("lab.js", "text/javascript; charset=utf-8"),
    "/lab.css": ("lab.css", "text/css; charset=utf-8"),
}
DIRECTORY = Path(__file__).resolve().parent
TRACE_LIMIT = 16 * 1024 * 1024


def project_request_cycles(events, tag):
    """Project when each selected logit becomes available at its wave barrier."""
    choices = {}
    start = 0
    for event in events:
        if event["phase"] != "pipeline":
            continue
        cycles = event["cycles"]
        if type(cycles) is not int or cycles <= 0 or event.get("wave_start") != start:
            return None
        rows = event["row_ids"]
        if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
            return None
        for index, request_id in enumerate(rows):
            if request_id == tag:
                choice = "0"
            elif request_id.startswith(tag + "/"):
                choice = request_id[len(tag) + 1:]
            else:
                continue
            egress = event.get("completion_cycles")
            flags = event.get("logit_rows")
            if not isinstance(flags, list) or not isinstance(egress, list) or len(flags) != len(rows) or len(egress) != len(rows):
                return None
            completion = egress[index]
            if type(flags[index]) is not bool or type(completion) is not int or not 0 < completion <= cycles:
                return None
            row = choices.setdefault(choice, {"start": start, "outputs": []})
            if flags[index]:
                row["outputs"].append(start + cycles)
        start += cycles
    return choices


class DemoHandler(BaseHTTPRequestHandler):
    def log_request(self, code="-", size="-"):
        if self.path == "/api/events" and code == 200:
            return
        super().log_request(code, size)

    def local_request(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def headers_for(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; form-action 'self'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def send_json(self, status, message):
        data = json.dumps({"error": {"message": message}}).encode()
        self.headers_for(status, "application/json", len(data))
        self.wfile.write(data)

    def events(self):
        trace = self.server.demo_trace
        events = []
        limited = False
        if trace is not None:
            try:
                with trace.open("rb") as stream:
                    total = stream.seek(0, os.SEEK_END)
                    limited = total >= TRACE_LIMIT
                    stream.seek(max(0, total - 256 * 1024))
                    if total > 256 * 1024:
                        stream.readline()  # drop a partial first record
                    for line in stream.read(256 * 1024).splitlines(keepends=True):
                        if line.endswith(b"\n"):
                            events.append(json.loads(line))
            except (OSError, ValueError):
                self.send_json(503, "The local trace file is unreadable.")
                return
        body = json.dumps(dict(available=trace is not None, limited=limited, capacity=self.server.demo_capacity, events=events[-120:])).encode()
        self.headers_for(200, "application/json", len(body))
        self.wfile.write(body)

    def projection(self, tag):
        choices = None
        trace = self.server.demo_trace
        if trace is not None:
            try:
                with trace.open("rb") as stream:
                    if os.fstat(stream.fileno()).st_size <= TRACE_LIMIT:
                        choices = project_request_cycles(
                            (json.loads(line) for line in stream if line.endswith(b"\n")), tag)
            except (OSError, ValueError, TypeError, KeyError):
                self.send_json(503, "The local projection trace is unreadable.")
                return
        body = json.dumps({"choices": choices}).encode()
        self.headers_for(200, "application/json", len(body))
        self.wfile.write(body)

    def do_GET(self):
        if not self.local_request():
            self.send_error(403)
            return
        projection = re.fullmatch(r"/api/projection\?request_id=([a-f0-9]{32})", self.path)
        if projection:
            self.projection(projection[1])
            return
        if self.path == "/api/events":
            self.events()
            return
        if self.path == "/api/models":
            self.relay("GET", "/v1/models")
            return
        if self.path not in ASSETS:
            self.send_error(404)
            return
        file, content_type = ASSETS[self.path]
        data = (DIRECTORY / file).read_bytes()
        self.headers_for(200, content_type, len(data))
        self.wfile.write(data)

    def do_POST(self):
        if not self.local_request() or self.headers.get("Origin") != f"http://{self.headers.get('Host')}":
            self.send_json(403, "Make requests from the local demo page.")
            return
        if self.path != "/api/completions":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
            if not 0 < length <= 16384:
                raise ValueError
            data = json.loads(self.rfile.read(length))
            prompt = data["prompt"]
            count = data["max_tokens"]
            wave = isinstance(prompt, list)
            prompts = prompt if wave else [prompt]
            if not 1 <= len(prompts) <= 8 or any(not isinstance(text, str) or not text.strip() or len(text) > 800 for text in prompts):
                raise ValueError
            if type(count) is not int or not 1 <= count <= 32:
                raise ValueError
            request_id = data.get("request_id")
            options = {}
            if request_id is not None:
                if not isinstance(request_id, str) or not re.fullmatch("[a-f0-9]{32}", request_id):
                    raise ValueError
                kind = "wave" if wave else "demo"
                options["request_id"] = f"opentsp-{kind}-{request_id}"
            payload = json.dumps({
                "model": "gpt2", "prompt": prompt, "max_tokens": count, "temperature": 0,
                "stream": True, "return_token_ids": self.server.demo_trace is not None, "stream_options": {"include_usage": True, "continuous_usage_stats": True}, **options,
            }).encode()
        except (KeyError, TypeError, ValueError):
            self.send_json(400, "Enter a prompt of 1–800 characters and request 1–32 tokens.")
            return
        if request_id and self.server.demo_trace is not None:
            from live_trace import LiveTrace

            trace = LiveTrace(self.server.demo_trace, request_id, wave)
            self.relay("POST", "/v1/completions", payload, trace)
        else:
            self.relay("POST", "/v1/completions", payload)

    def relay(self, method, path, payload=None, trace=None):
        upstream = http.client.HTTPConnection("127.0.0.1", self.server.vllm_port, timeout=120)
        try:
            try:
                upstream.request(method, path, body=payload, headers={"Content-Type": "application/json"})
                response = upstream.getresponse()
            except (OSError, http.client.HTTPException):
                self.send_json(502, "Cannot reach local vLLM. Start the GPT-2 server and reconnect.")
                return
            self.headers_for(response.status, response.getheader("Content-Type", "application/json"))
            if trace is not None and response.status == 200 and response.getheader("Content-Type", "").startswith("text/event-stream"):
                self.relay_live(response, trace)
            else:
                # HTTP/1.0 framing closes the response at EOF.
                while chunk := response.read1(16384):
                    self.wfile.write(chunk)
        except (OSError, http.client.HTTPException):
            # A failed downstream write closes the upstream request.
            pass
        finally:
            upstream.close()


    def relay_live(self, response, trace):
        messages = Queue(maxsize=32)
        stopped = Event()

        def publish(message):
            from queue import Full

            while not stopped.is_set():
                try:
                    messages.put(message, timeout=0.1)
                    return
                except Full:
                    continue

        def read_upstream():
            frame = bytearray()
            try:
                while line := response.readline(16384):
                    if line in (b"\n", b"\r\n"):
                        publish(bytes(frame))
                        frame.clear()
                    else:
                        frame.extend(line)
                        if len(frame) > 256 * 1024:
                            break
            except (OSError, http.client.HTTPException):
                pass
            finally:
                publish(None)

        def send(data):
            self.wfile.write(b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n")
            self.wfile.flush()

        reader = Thread(target=read_upstream, daemon=True)
        reader.start()
        trace_available = True
        run = trace.tag[:8]
        try:
            while True:
                try:
                    frame = messages.get(timeout=0.012)
                except Empty:
                    frame = b""
                if trace_available:
                    try:
                        events = list(trace.events())
                    except (OSError, ValueError, KeyError) as error:
                        trace_available = False
                        events = [dict(phase="trace-gap", reason=f"Trace unavailable: {error}")]
                    if events:
                        send({"opentsp_events": events})
                        for event in events:
                            step, phase = event.get("step"), event["phase"]
                            if phase == "scheduled":
                                prefill = sum(r["prefill"] for r in event["requests"])
                                decode = sum(r["decode"] for r in event["requests"])
                                print(f"{run}  Batch {step:02} admit   {prefill} P + {decode} D", flush=True)
                            elif phase == "wave-positions":
                                # One line per observed frontier with completed receives per chip.
                                counts = " ".join(f"C{i}:{event['stages'].count(i + 1)}" for i in range(4))
                                print(f"{run}  Batch {step:02} chips   {counts}", flush=True)
                            elif phase == "pipeline":
                                print(f"{run}  Batch {step:02} host    logits ready", flush=True)
                            elif phase == "trace-gap":
                                print(f"{run}  trace unavailable: {event['reason']}", flush=True)
                if frame is None:
                    return
                data = b"\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith(b"data:"))
                if not data:
                    continue
                if data == b"[DONE]":
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                chunk = json.loads(data)
                outputs = trace.annotate(chunk)
                for output in outputs:
                    index, seq, ordinal = output["index"], output["seq"], output["ordinal"]
                    batches = "/".join(f"{step:02}" for step in dict.fromkeys(output["batches"])) or "?"
                    print(f"{run}  E{seq:02} B{batches} U{index + 1} token {ordinal}  IDs {output['ids']}", flush=True)
                send(chunk)
        finally:
            stopped.set()


def run_stacks(server, args, trace):
    server.demo_trace = trace
    server.demo_capacity = args.kv_chunks * 8
    server.vllm_port = args.vllm_port
    url = f"http://127.0.0.1:{args.port}"
    if not args.launch:
        print(f"OpenTSP GPT-2 demo at {url}", flush=True)
        server.serve_forever()
        return 0

    # Catch common port conflicts before handing ownership of the backend to its launcher.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.vllm_port))
    cmd = [sys.executable, "-m", "opentsp_vllm.serve", "--tsp-model", "gpt2", "--tsp-chips", "4",
           "--host", "127.0.0.1", "--port", str(args.vllm_port), "--enable-prefix-caching", "--max-num-seqs", "8",
           "--tsp-kv-chunks", str(args.kv_chunks)]
    if args.programs:
        cmd.extend(["--tsp-programs", args.programs])
    print(f"OpenTSP GPT-2 demo at {url} (waiting for the backend to load)", flush=True)
    env = dict(os.environ, OPENTSP_DEMO_TRACE=str(trace))
    env.setdefault("HF_HUB_OFFLINE", "1")
    # The demo owns its simulator instance on this port.
    env.setdefault("OPENTSP_RAS_DOMAIN", f"opentsp-gpt2-demo:127.0.0.1:{args.vllm_port}")
    backend = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        server.timeout = 0.3
        while backend.poll() is None:
            server.handle_request()
        print(f"GPT-2 backend exited with status {backend.returncode}. Check the serving log above.", file=sys.stderr)
        return backend.returncode or 1
    finally:
        if backend.poll() is None:
            backend.terminate()  # launcher forwards it and waits for its owned process group
            try:
                backend.wait(timeout=8)
            except subprocess.TimeoutExpired:
                backend.kill()
                backend.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18765, help="local browser port (default 18765)")
    parser.add_argument("--vllm-port", type=int, default=8000, help="local vLLM port (default 8000)")
    parser.add_argument("--launch", action="store_true", help="start and stop a dedicated vLLM server")
    parser.add_argument("--kv-chunks", type=int, choices=(2,), default=2, help="two 128-slot chunks for the four-chip route")
    parser.add_argument("--programs", help="published stage package or cache root for --launch")
    parser.add_argument("--trace", type=Path, help="local event log (must be a new file when launching)")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.vllm_port <= 65535 or args.port == args.vllm_port:
        parser.error("choose distinct ports in 1–65535")
    if args.programs and not args.launch:
        parser.error("--programs requires --launch")
    if args.trace and not args.launch and not args.trace.is_file():
        parser.error(f"trace file does not exist: {args.trace}")
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler) as server:
            if not args.launch:
                return run_stacks(server, args, args.trace)
            with TemporaryDirectory(prefix="opentsp-demo-") as temp:
                trace = args.trace or Path(temp, "events.jsonl")
                trace = trace.resolve()
                trace.touch(mode=0o600, exist_ok=False)
                return run_stacks(server, args, trace)
    except KeyboardInterrupt:
        return 0
    except OSError as error:
        parser.exit(1, f"Cannot start the demo: {error}. Check the ports and trace path.\n")


if __name__ == "__main__":
    sys.exit(main())

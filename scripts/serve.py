#!/usr/bin/env python3
"""Local demo server: serves the web page and relays chat requests to the model on Modal.

    uv run scripts/train.py --action deploy     # once, publishes the model service
    uv run scripts/serve.py                                     # then open http://127.0.0.1:8000

Built for a single presenter on one machine. It listens on localhost only, keeps no
history, and handles one model request at a time. It is not meant for public traffic.

Cold start: the GPU container on Modal shuts down after [serve] scaledown_seconds without
requests. The first request after that waits for a new container and a model load. The
page asks this server to wake the model early (/api/warm) and shows the state while it waits.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from another_me.config import ROOT, load_config, load_profile, resolve  # noqa: E402
from another_me.schema import REGISTERS, read_jsonl, system_prompt  # noqa: E402

CFG = load_config()  # also loads .env, which carries the Modal token
SERVE = CFG["serve"]
PROFILE = load_profile(CFG)
MAX_CHARS = 2000
MAX_TURNS = 12


class Model:
    """Tracks whether the Modal container is asleep, waking or ready, and calls it."""

    def __init__(self):
        self.lock = threading.Lock()   # state changes
        self.busy = threading.Lock()   # one model request at a time
        self.waking_since: float | None = None
        self.last_ok: float | None = None
        self.last_cold_start: float | None = None
        self.error: str | None = None
        self.last_reply: dict | None = None
        self._twin = None

    def twin(self):
        if self._twin is None:
            import modal

            self._twin = modal.Cls.from_name(CFG["modal"]["app_name"], "Twin")()
        return self._twin

    def state(self) -> str:
        if self.waking_since is not None:
            return "waking"
        if self.last_ok and time.time() - self.last_ok < SERVE["scaledown_seconds"] - 15:
            return "ready"
        return "asleep"

    def status(self) -> dict:
        with self.lock:
            return {
                "state": self.state(),
                "waking_seconds": round(time.time() - self.waking_since, 1) if self.waking_since else None,
                "last_cold_start_seconds": self.last_cold_start,
                "idle_shutdown_seconds": SERVE["scaledown_seconds"],
                "seconds_until_sleep": max(0, round(SERVE["scaledown_seconds"] - (time.time() - self.last_ok)))
                                       if self.last_ok else None,
                "last_reply": self.last_reply,
                "error": self.error,
            }

    def warm(self) -> None:
        """Start waking the container in the background unless it is already up or waking."""
        with self.lock:
            if self.state() != "asleep":
                return
            self.waking_since = time.time()
            self.error = None

        def run():
            started = time.time()
            try:
                self.twin().ping.remote()
                with self.lock:
                    self.last_ok = time.time()
                    self.last_cold_start = round(time.time() - started, 1)
                print(f"[model] awake after {self.last_cold_start}s", flush=True)
            except Exception as e:  # shown on the page
                with self.lock:
                    self.error = f"{type(e).__name__}: {e}"[:300]
            finally:
                with self.lock:
                    self.waking_since = None

        threading.Thread(target=run, daemon=True).start()

    def reply(self, messages: list[dict], arms: list[str], seed: int) -> dict:
        started = time.time()
        was = self.state()
        with self.busy:
            result = self.twin().reply.remote(messages, arms, SERVE["max_tokens"], seed)
        with self.lock:
            self.last_ok = time.time()
            self.error = None
        result["total_seconds"] = round(time.time() - started, 2)
        result["was_cold"] = was == "asleep"
        with self.lock:
            self.last_reply = {"arms": arms, "generate_seconds": result["seconds"],
                               "total_seconds": result["total_seconds"], "was_cold": result["was_cold"]}
        print(f"[model] {'+'.join(arms)}: generated in {result['seconds']}s, "
              f"{result['total_seconds']}s end to end{' (model was asleep)' if result['was_cold'] else ''}",
              flush=True)
        return result


MODEL = Model()


def page_config() -> dict:
    """Everything the page shows that depends on this installation."""
    name = CFG["persona"]["name"]
    info = {
        "name": name,
        "base_model": CFG["model"]["hf_id"],
        "adapter": SERVE["adapter"],
        "warm_on_load": SERVE["warm_on_load"],
        "idle_shutdown_seconds": SERVE["scaledown_seconds"],
        "dataset": None,
        "training": None,
    }
    raw = resolve(CFG["data"]["raw"])
    if raw.exists():
        counts: dict[str, int] = {}
        for pair in read_jsonl(raw):
            key = pair["register"] if pair["kind"] == "answer" else pair["kind"]
            counts[key] = counts.get(key, 0) + 1
        info["dataset"] = {"total": sum(counts.values()), "counts": counts}
    run = ROOT / "runs" / f"{SERVE['adapter']}.modal.json"
    if run.exists():
        r = json.loads(run.read_text())
        info["training"] = {k: r.get(k) for k in (
            "gpu_name", "train_pairs", "epochs_run", "best_epoch", "train_seconds",
            "tokens_per_second", "trainable_percent", "trainable_params", "valid_loss")}
    return info


def clean_messages(payload: dict, notes: bool = True) -> tuple[str, list[dict]]:
    """Validate the conversation and put the system prompt in front.

    notes=False leaves the profile notes out: that is what the base model gets in the
    comparison, so it answers as downloaded, with nothing it was told about the person.
    """
    register = payload.get("register")
    if register not in REGISTERS:
        register = "professional"
    turns = []
    for m in (payload.get("messages") or [])[-MAX_TURNS:]:
        role, content = m.get("role"), str(m.get("content", "")).strip()[:MAX_CHARS]
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content})
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("The last message must be from the user.")
    system = system_prompt(CFG["persona"]["name"], register, profile=PROFILE if notes else "")
    return register, [{"role": "system", "content": system}] + turns


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def log_message(self, fmt, *args):  # keep the terminal quiet except for API calls
        if self.path.startswith("/api/") and "/api/status" not in self.path:
            super().log_message(fmt, *args)

    def send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/config":
            return self.send_json(page_config())
        if self.path == "/api/status":
            return self.send_json(MODEL.status())
        return super().do_GET()

    def do_POST(self):
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 200_000)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/warm":
                MODEL.warm()
                return self.send_json(MODEL.status())
            if self.path == "/api/chat":
                _, messages = clean_messages(payload)
                return self.send_json(MODEL.reply(messages, ["tuned"], int(time.time()) % 100000))
            if self.path == "/api/compare":
                question = str(payload.get("question", "")).strip()
                ask = {"register": payload.get("register"),
                       "messages": [{"role": "user", "content": question}]}
                # The tuned model answers as it does in chat. The base model gets the same
                # question without the profile notes. Same random seed for both.
                seed = int(time.time()) % 100000
                tuned = MODEL.reply(clean_messages(ask)[1], ["tuned"], seed)
                base = MODEL.reply(clean_messages(ask, notes=False)[1], ["base"], seed)
                return self.send_json({
                    "replies": {"tuned": tuned["replies"]["tuned"], "base": base["replies"]["base"]},
                    "seconds": round(tuned["seconds"] + base["seconds"], 2),
                    "total_seconds": round(tuned["total_seconds"] + base["total_seconds"], 2),
                    "was_cold": tuned["was_cold"],
                })
            return self.send_json({"error": "not found"}, 404)
        except ValueError as e:
            return self.send_json({"error": str(e)}, 400)
        except Exception as e:
            hint = ""
            if "NotFound" in type(e).__name__:
                hint = " Deploy the model service first: scripts/train.py --action deploy"
            return self.send_json({"error": f"{type(e).__name__}: {e}"[:300] + hint}, 502)


def main() -> None:
    server = ThreadingHTTPServer((SERVE["host"], SERVE["port"]), Handler)
    print(f"another-me demo: http://{SERVE['host']}:{SERVE['port']}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

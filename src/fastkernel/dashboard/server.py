"""Zero-dependency live dashboard: ThreadingHTTPServer + Server-Sent Events over the SQLite event log."""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..campaign import Campaign
from .data import campaign_state, campaigns_index, experiment_detail

STATIC = Path(__file__).parent / "static"


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self._campaigns: dict[str, Campaign] = {}
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            for campaign in Campaign.discover_all(self.root):
                self._campaigns.setdefault(campaign.name, campaign)
            direct = Campaign(self.root)
            if direct.exists:
                self._campaigns.setdefault(direct.name, direct)

    def get(self, name: str) -> Campaign | None:
        with self._lock:
            campaign = self._campaigns.get(name)
        if campaign is None:
            self.refresh()
            with self._lock:
                campaign = self._campaigns.get(name)
        return campaign

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._campaigns)


def make_handler(registry: Registry):
    class Handler(BaseHTTPRequestHandler):
        server_version = "fast-kernel/0.1"

        def log_message(self, format, *args):  # noqa: A002
            pass

        def _json(self, payload, status=200):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, name: str):
            path = (STATIC / name).resolve()
            if not str(path).startswith(str(STATIC.resolve())) or not path.exists():
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            if not parts or parts == ["index.html"]:
                return self._static("index.html")
            if parts[0] == "static" and len(parts) == 2:
                return self._static(parts[1])
            if parts == ["api", "campaigns"]:
                registry.refresh()
                return self._json({"campaigns": campaigns_index(registry.root), "names": registry.names()})
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "c":
                campaign = registry.get(parts[2])
                if campaign is None:
                    return self._json({"error": f"unknown campaign {parts[2]}"}, 404)
                rest = parts[3:]
                if rest == ["state"]:
                    return self._json(campaign_state(campaign))
                if rest == ["events"]:
                    after = int(query.get("after", ["0"])[0])
                    return self._json({"events": campaign.store.events_after(after, limit=int(query.get("limit", ["300"])[0]))})
                if len(rest) == 2 and rest[0] == "experiments":
                    detail = experiment_detail(campaign, int(rest[1]))
                    return self._json(detail or {"error": "not found"}, 200 if detail else 404)
                if rest == ["plan"]:
                    return self._json({"plan": campaign.plan_path.read_text(encoding="utf-8") if campaign.plan_path.exists() else "",
                                       "knowledge": campaign.knowledge_path.read_text(encoding="utf-8") if campaign.knowledge_path.exists() else "",
                                       "goal": campaign.goal_path.read_text(encoding="utf-8")})
                if rest == ["stream"]:
                    return self._stream(campaign, int(query.get("after", ["0"])[0]))
            self.send_error(404)

        def do_POST(self):  # noqa: N802
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if len(parts) == 4 and parts[:2] == ["api", "c"] and parts[3] == "control":
                campaign = registry.get(parts[2])
                if campaign is None:
                    return self._json({"error": "unknown campaign"}, 404)
                action = body.get("action")
                if action == "pause":
                    campaign.set_flag("paused")
                elif action == "resume":
                    campaign.clear_flag("paused")
                elif action == "stop":
                    campaign.set_flag("stop")
                    campaign.clear_flag("loop.active")
                elif action == "start-loop":
                    campaign.set_flag("loop.active", "dashboard")
                    campaign.clear_flag("stop")
                elif action == "note":
                    from .. import knowledge
                    knowledge.add_note(campaign.knowledge_path, str(body.get("text", "")), body.get("tags") or [])
                else:
                    return self._json({"error": "unknown action"}, 400)
                campaign.store.event("control", action=action, source="dashboard")
                return self._json({"ok": True, "summary": campaign.summary()})
            self.send_error(404)

        def _stream(self, campaign: Campaign, after: int):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last = after
            last_beat = time.time()
            try:
                self.wfile.write(b"event: hello\ndata: {}\n\n")
                self.wfile.flush()
                while True:
                    events = campaign.store.events_after(last, limit=200)
                    if events:
                        last = events[-1]["id"]
                        payload = json.dumps({"events": events}, default=str)
                        self.wfile.write(f"id: {last}\nevent: events\ndata: {payload}\n\n".encode())
                        self.wfile.flush()
                    elif time.time() - last_beat > 15:
                        self.wfile.write(b"event: heartbeat\ndata: {}\n\n")
                        self.wfile.flush()
                        last_beat = time.time()
                    time.sleep(0.5)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

    return Handler


def serve(root: Path, port: int = 8765, host: str = "127.0.0.1", open_browser: bool = False) -> None:
    registry = Registry(Path(root).resolve())
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), make_handler(registry))
            port = candidate
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError("no free port found")
    server.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"fast-kernel dashboard: {url}  (campaigns: {', '.join(registry.names()) or 'none found under ' + str(root)})", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

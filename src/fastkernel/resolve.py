"""Turn a plain sentence into a deterministic plan: which model, which folder, which steps remain.

    fast-kernel resolve "Optimize the Mimi codec model."
    -> {"action": "optimize", "model": "mimi", "campaign": "/repo/campaigns/mimi", "exists": true, "steps": [...]}

This is the bridge between the user's text and the file system so the agent never guesses a folder:
the repository root is found by walking up from the current directory (pyproject.toml of fast-kernel or
.claude/skills/fk-optimize), the campaign is always <root>/campaigns/<model> (or an existing campaign
that already optimizes that model), and the remaining steps are computed from what is on disk.
"""
from __future__ import annotations

import importlib.util
import os
import re
import socket
from pathlib import Path
from typing import Any

from .campaign import Campaign
from .util import read_json

MODEL_RULES: list[tuple[str, str, str]] = [
    # (model name, regex over the lower-cased sentence, human display)
    ("lfm-audio", r"lfm[\s\-_.]*2(\.5)?[\s\-_]*audio|lfm[\s\-_]*audio|liquid[\s\-_]*audio|speech[\s\-]*to[\s\-]*speech", "LFM2-Audio (LiquidAI/LFM2.5-Audio-1.5B)"),
    ("lfm25", r"lfm[\s\-_.]*2(\.5)?\b|lfm25|liquid", "LFM2.5 (LiquidAI/LFM2.5-1.2B-Instruct)"),
    ("mimi", r"\bmimi\b|\bcodec\b|kyutai", "Mimi codec (kyutai/mimi)"),
    ("yolo", r"\byolo|\bdetect(ion|or)?\b|ultralytics", "YOLO (Ultralytics YOLO26n)"),
]
EXTRAS = {"lfm-audio": [("liquid_audio", "audio")], "yolo": [("ultralytics", "yolo")]}
# "end" is intentionally excluded: it collides with ordinary phrasing ("end-to-end", "in the end").
STOP_RE = re.compile(r"\b(stop|halt|finish|pause|cancel)\b")
STATUS_RE = re.compile(r"\b(status|progress|how (is|are|far)|where are we|report|summar)")
PATH_RE = re.compile(r"(?:^|\s)((?:\.{1,2}/|/|~/)?[\w\-./]+\.py|(?:\.{1,2}/|/|~/)[\w\-./]+)")


def find_root(start: Path | None = None) -> Path:
    here = Path(start or os.getcwd()).resolve()
    env = os.environ.get("FAST_KERNEL_HOME")
    if env and (Path(env) / ".claude").exists():
        return Path(env).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".claude" / "skills" / "fk-optimize").exists():
            return candidate
        pyproject = candidate / "pyproject.toml"
        if pyproject.exists() and 'name = "fast-kernel"' in pyproject.read_text(encoding="utf-8", errors="ignore"):
            return candidate
    return here


def _dashboard_url(ports: range = range(8765, 8776)) -> str | None:
    for port in ports:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.15) as sock:
                sock.sendall(b"GET /api/campaigns HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
                head = sock.recv(400)
                if b"fast-kernel" in head or b'"campaigns"' in head:
                    return f"http://127.0.0.1:{port}/"
        except OSError:
            continue
    return None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve(text: str, root: Path | None = None, cwd: Path | None = None) -> dict[str, Any]:
    sentence = " ".join(text.split())
    lower = sentence.lower()
    root = Path(root).resolve() if root else find_root(cwd)
    campaigns_dir = root / "campaigns"
    # only real campaigns count (never the examples/ or templates/ copies of GOAL.md + candidate/)
    existing: dict[str, Campaign] = {}
    if campaigns_dir.is_dir():
        for c in Campaign.discover_all(campaigns_dir, max_depth=1):
            if c.exists and c.root.parent == campaigns_dir:
                existing.setdefault(c.goal.model, c)
    plan: dict[str, Any] = {"text": sentence, "root": str(root), "campaigns_dir": str(campaigns_dir), "action": "optimize"}

    if STOP_RE.search(lower) and re.search(r"optimi|campaign|loop|experiment|agent|kernel", lower):
        plan["action"] = "stop"
    elif STATUS_RE.search(lower) and not re.search(r"optimi[sz]e\b", lower):
        plan["action"] = "status"

    # model: explicit campaign directory > path to a model file > keyword rules
    model: str | None = None
    custom_path: str | None = None
    for name, campaign in existing.items():
        if campaign.root.name.lower() in lower or f"campaigns/{campaign.root.name}".lower() in lower:
            model = name
            plan["campaign"] = str(campaign.root)
            break
    if model is None:
        path_match = PATH_RE.search(sentence)
        if path_match and not re.search(r"\b(mimi|lfm|yolo|codec|liquid)\b", lower):
            custom_path = path_match.group(1)
            model = "custom"
    if model is None:
        for name, pattern, _display in MODEL_RULES:
            if re.search(pattern, lower):
                model = name
                break
    if model is None:
        if len(existing) == 1:
            model = next(iter(existing))            # a single campaign is the unambiguous target ("continue", "stop", "status")
        elif plan["action"] in ("stop", "status") and existing:
            plan["campaigns"] = sorted(str(c.root) for c in existing.values())
            plan["steps"] = [f"cd {c} && uv run fast-kernel {'loop stop && uv run fast-kernel stop' if plan['action'] == 'stop' else 'status'}"
                             for c in plan["campaigns"]]
            return plan
        elif plan["action"] in ("stop", "status"):
            plan["campaigns"] = []
            plan["steps"] = []
            plan["hint"] = "no campaign exists under campaigns/ yet"
            return plan
        else:
            plan["action"] = "unknown"
            plan["known_models"] = [rule[0] for rule in MODEL_RULES] + ["custom"]
            plan["hint"] = ("Name a model: Mimi codec, LFM2.5, LFM2 audio, YOLO, or a path to a PyTorch model file "
                            "(e.g. 'Optimize the PyTorch model in ./my_model.py').")
            plan["campaigns"] = sorted(str(c.root) for c in existing.values())
            return plan
    plan["model"] = model
    plan["display"] = next((d for n, _p, d in MODEL_RULES if n == model), "custom torch module" if model == "custom" else model)
    if custom_path:
        plan["custom_path"] = custom_path
        slug = re.sub(r"[^a-z0-9]+", "-", Path(custom_path).stem.lower()).strip("-") or "custom"
        plan["campaign_name"] = slug
    campaign_root = Path(plan.get("campaign") or (existing[model].root if model in existing else campaigns_dir / plan.get("campaign_name", model)))
    plan["campaign"] = str(campaign_root)
    campaign = Campaign(campaign_root)
    plan["exists"] = campaign.exists

    # state on disk (read-only: resolving must never create files)
    incumbent = campaign.load_incumbent() if campaign.exists else None
    experiments = campaign.store.list_experiments() if (campaign.exists and campaign.db_path.exists()) else []
    plan["experiments"] = len(experiments)
    plan["has_baseline"] = any(e.get("number") == 0 for e in experiments)
    plan["incumbent"] = incumbent.to_dict() if incumbent and incumbent.number >= 0 else None
    caps = read_json(campaign.capabilities_path, {}) if campaign.exists else {}
    plan["probed"] = bool(caps.get("backends"))
    plan["loop_active"] = campaign.has_flag("loop.active") if campaign.exists else False
    plan["dashboard_url"] = _dashboard_url()
    missing = [extra for module, extra in EXTRAS.get(model or "", []) if not _module_available(module)]
    plan["missing_extras"] = missing
    plan["torch_available"] = _module_available("torch")

    # remaining steps (all commands are run from the repository root; the campaign path is absolute)
    steps: list[str] = []
    if plan["action"] == "stop":
        steps = [f"cd {campaign_root} && uv run fast-kernel loop stop", f"cd {campaign_root} && uv run fast-kernel stop"]
    elif plan["action"] == "status":
        steps = [f"cd {campaign_root} && uv run fast-kernel status && uv run fast-kernel history -n 10"]
    else:
        if not plan["torch_available"]:
            steps.append(f"cd {root} && uv sync --extra cuda")
        for extra in missing:
            steps.append(f"cd {root} && uv sync --extra cuda --extra {extra}")
        if not campaign.exists:
            init = f"cd {root} && uv run fast-kernel init {model}" + (f" --name {plan['campaign_name']}" if custom_path else "")
            steps.append(init)
            if custom_path:
                steps.append(f"edit {campaign_root}/spec.py so build_model() loads {custom_path} (see /fk-add-model)")
        if not plan["probed"]:
            steps.append(f"cd {campaign_root} && uv run fast-kernel probe")
        if not plan["has_baseline"]:
            steps.append(f"cd {campaign_root} && uv run fast-kernel baseline")
        if not plan["dashboard_url"]:
            steps.append(f"cd {root} && (uv run fast-kernel dashboard --root campaigns > .fast-kernel-dashboard.log 2>&1 &) ; sleep 1; cat .fast-kernel-dashboard.log")
        if not plan["loop_active"]:
            steps.append(f"cd {campaign_root} && uv run fast-kernel loop start")
        steps.append(f"cd {campaign_root} && uv run fast-kernel status --brief && uv run fast-kernel ideas")
        steps.append("run experiments (/fk-experiment procedure) until the user says stop")
    plan["steps"] = steps
    return plan

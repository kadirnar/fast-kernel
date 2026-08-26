"""fast-kernel command line interface."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__, knowledge, results
from .campaign import Campaign
from .frontmatter import render_frontmatter, split_frontmatter
from .util import fmt, read_json, run, which, write_json

TEMPLATES = Path(__file__).parent / "templates" / "models"


def _campaign(args) -> Campaign:
    root = getattr(args, "campaign", None) or os.environ.get("FAST_KERNEL_CAMPAIGN")
    campaign = Campaign(Path(root)) if root else Campaign.find()
    if campaign is None or not campaign.exists:
        sys.exit("error: no campaign here. Run `fast-kernel init <model>` or pass --campaign <dir>.")
    return campaign


def _set_nested(data: dict[str, Any], dotted: str, value: str) -> None:
    from .frontmatter import parse_scalar
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise SystemExit(f"cannot set {dotted}: {key} is not a mapping")
    node[keys[-1]] = parse_scalar(value)


# ---- commands --------------------------------------------------------------------------------
def cmd_init(args) -> None:
    model = args.model.lower()
    template = TEMPLATES / model
    if not template.exists():
        template = TEMPLATES / "custom"
    dest = Path(args.dir or (Path("campaigns") / (args.name or model))).resolve()
    if dest.exists() and any(dest.iterdir()) and not args.force:
        sys.exit(f"error: {dest} exists and is not empty (use --force to overwrite the template files)")
    dest.mkdir(parents=True, exist_ok=True)
    for src in template.rglob("*"):
        if src.is_dir() or "__pycache__" in src.parts:
            continue
        rel = src.relative_to(template)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    goal_path = dest / "GOAL.md"
    data, body = split_frontmatter(goal_path.read_text(encoding="utf-8"))
    data.setdefault("model", model if (TEMPLATES / model).exists() else "custom")
    if args.precision:
        data.setdefault("gates", {})["precision"] = args.precision
    for item in args.set or []:
        key, _, value = item.partition("=")
        _set_nested(data, key.strip(), value.strip())
    goal_path.write_text(render_frontmatter(data, body), encoding="utf-8")
    campaign = Campaign(dest)
    results.ensure_header(campaign.results_path)
    knowledge.ensure(campaign.knowledge_path)
    campaign.state_dir.mkdir(exist_ok=True)
    if not args.no_git:
        campaign.ensure_git()
    campaign.store.event("campaign.created", model=data.get("model"), root=str(dest))
    print(f"campaign ready: {dest}\n  model: {data.get('model')}\n  next: cd {dest} && fast-kernel probe && fast-kernel baseline")
    print(f'  or in Claude Code, just type: "Optimize the {data.get("model")} model."')


def cmd_probe(args) -> None:
    from .backends.base import device_capabilities, env_summary, probe_all
    from .backends.cuda_cpp import ensure_cuda_home
    ensure_cuda_home()
    campaign = Campaign.find() if not args.campaign else Campaign(Path(args.campaign))
    print("probing device ...", flush=True)
    device = device_capabilities(microbench=not args.no_microbench)
    print(f"  {device.get('name')} sm_{str(device.get('compute_capability', '')).replace('.', '')} {device.get('sm_count')} SMs "
          f"{device.get('total_memory_gb')} GB | torch {device.get('torch')} cuda {device.get('cuda')}")
    if device.get("measured_bandwidth_gbs"):
        print(f"  measured: {device['measured_bandwidth_gbs']} GB/s, bf16 {device.get('measured_bf16_tflops')} TFLOPS, "
              f"fp32 {device.get('measured_fp32_tflops')} TFLOPS, launch latency {device.get('launch_latency_us')} us")
    print("probing backends ...", flush=True)
    backends = probe_all(compile_test=not args.no_compile)
    for name, info in backends.items():
        state = "READY" if info.get("compiled") else ("importable" if info.get("available") else "missing")
        print(f"  {name:14s} {state:10s} v{info.get('version') or '?':10s} {('- ' + str(info.get('error'))[:110]) if info.get('error') else ''}")
        if info.get("fix"):
            print(f"  {'':14s} fix: {info['fix']}")
    payload = {"device_info": device, "backends": backends, "env": env_summary()}
    out = campaign.capabilities_path if campaign and campaign.exists else Path("capabilities.json")
    write_json(out, payload)
    print(f"written {out}")


def cmd_profile(args) -> None:
    campaign = _campaign(args)
    from .models.spec import load_spec
    from .profiling.plan import write_plan
    out = campaign.state_dir / "profile"
    out.mkdir(parents=True, exist_ok=True)
    history = campaign.store.list_experiments()
    hist_path = out / "history.json"
    write_json(hist_path, [{k: e.get(k) for k in ("number", "status", "target", "techniques")} for e in history])
    argv = [sys.executable, "-m", "fastkernel.harness.run", "--campaign", str(campaign.root), "--out", str(out), "--profile-only",
            "--history", str(hist_path)]
    if args.workload:
        argv += ["--workloads", args.workload]
    print("profiling current candidate ...", flush=True)
    result = run(argv, cwd=campaign.root, timeout=campaign.goal.bench.timeout_seconds, env={"PYTHONUNBUFFERED": "1"})
    (out / "run.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if not result.ok:
        print(result.stdout[-3000:], result.stderr[-3000:])
        sys.exit("profile failed (see above)")
    prof = read_json(out / "profile.json", {}) or {}
    if prof.get("error") or not prof.get("targets"):
        sys.exit(f"profile produced no targets: {prof.get('error', '')[:800]}")
    caps = read_json(campaign.capabilities_path, {}) or {}
    try:
        spec_notes = load_spec(campaign.root, campaign.goal.model, campaign.goal.model_args, campaign.goal.gates).notes
    except Exception:  # noqa: BLE001
        spec_notes = ""
    inc = campaign.load_incumbent()
    write_plan(campaign.root, profile=prof, targets=prof["targets"], device=caps.get("device_info", {}), workload=prof.get("workload", ""),
               experiment=inc.number if inc.number >= 0 else None, spec_notes=spec_notes, backends=caps.get("backends"))
    campaign.store.event("profile.updated", kernel_count=prof.get("kernel_count"), wall_ms=prof.get("wall_ms"),
                         gpu_busy_ratio=prof.get("gpu_busy_ratio"), top=prof["targets"][0]["title"])
    print(f"wall {fmt(prof.get('wall_ms'))} ms | GPU busy {fmt((prof.get('gpu_busy_ratio') or 0) * 100, 3)}% | {prof.get('kernel_count')} kernels | "
          f"{'LAUNCH BOUND' if prof.get('launch_bound') else 'GPU bound'}")
    _print_targets(prof["targets"][: args.top])
    print(f"written {campaign.plan_path} and {campaign.hotspots_path}")


def _refresh_statuses(campaign: Campaign, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """hotspots.json is rewritten only on keep; apply the technique matrix from the live history."""
    from .profiling.rank import technique_matrix
    matrix = technique_matrix(campaign.store.list_experiments())
    order = {"untried": 0, "accepted": 1, "crash": 2, "rejected": 3}
    for target in targets:
        techs = target.get("techniques") or []
        for tech in techs:
            tech["status"] = matrix.get((target["id"], tech["id"]), tech.get("status", "untried"))
        techs.sort(key=lambda t: (order.get(t["status"], 0), t.get("tier", 0), -t.get("expected_speedup", 1.0)))
        tried = [t for t in techs if t["status"] != "untried"]
        target["attempts"] = len(tried)
        target["accepted"] = sum(1 for t in tried if t["status"] == "accepted")
    return targets


def _print_targets(targets: list[dict[str, Any]]) -> None:
    print(f"{'#':>2} {'target':44s} {'share':>6} {'gain':>6} {'bound':8s} tried  next technique")
    for t in targets:
        nxt = next((x for x in t.get("techniques", []) if x["status"] == "untried"), None)
        print(f"{t['rank']:>2} {t['title'][:44]:44s} {t['fraction'] * 100:5.1f}% {t['amdahl_gain'] * 100:5.1f}% {t['boundness']:8s} "
              f"{t.get('attempts', 0):>5}  {(nxt or {}).get('id', '-')}  [{t['id']}]")


def cmd_baseline(args) -> None:
    campaign = _campaign(args)
    from .harness.evaluate import run_experiment
    record = run_experiment(campaign, args.message or "baseline: unmodified reference path", baseline=True, force=args.force,
                            repeats=args.repeats, warmup=args.warmup, profile=not args.no_profile)
    sys.exit(0 if record and record["status"] == "baseline" else 1)


def cmd_eval(args) -> None:
    campaign = _campaign(args)
    from .harness.evaluate import run_experiment
    techniques = [t for item in (args.technique or []) for t in item.split(",") if t]
    record = run_experiment(campaign, args.message, techniques=techniques, target=args.target, notes=args.notes or "",
                            simpler=args.simpler, force=args.force, profile=None if not args.no_profile else False, repeats=args.repeats,
                            warmup=args.warmup, workloads=args.workloads.split(",") if args.workloads else None)
    if record is None:
        sys.exit(1)
    sys.exit(0 if record["status"] in ("keep", "baseline") else 3)


def cmd_status(args) -> None:
    campaign = _campaign(args)
    summary = campaign.summary()
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return
    inc = summary["incumbent"]
    exps = campaign.store.list_experiments()
    last = exps[-1] if exps else None
    print(f"campaign {summary['name']} ({summary['model']}) -- {summary['objective'][:90]}")
    print(f"  experiments: {summary['experiments']} {summary['counts']} | loop: {'ACTIVE' if summary['loop_active'] else 'off'}"
          f"{' | PAUSED' if summary['paused'] else ''} | branch {summary['branch']} @ {summary['head']}")
    print(f"  incumbent: #{inc.get('number')} {summary['target_metric']}={fmt(inc.get('value'))} "
          f"(baseline {fmt(summary.get('baseline_value'))}, speedup {fmt(summary.get('speedup_vs_baseline'), 3)}x, noise floor "
          f"{fmt((inc.get('noise_floor') or 0) * 100, 3)}%, threshold {fmt(max(summary['min_improvement'], inc.get('noise_floor') or 0) * 100, 3)}%)")
    if last:
        print(f"  last: #{last['number']} [{last['status']}] {last.get('description', '')[:80]} -> {last.get('reason', '')[:100]}")
    hotspots = read_json(campaign.hotspots_path, {}) or {}
    if hotspots.get("targets") and not args.brief:
        print("  top targets:")
        _print_targets(_refresh_statuses(campaign, hotspots["targets"])[:5])
    if not args.brief:
        for line in knowledge.read_insights(campaign.knowledge_path, 5):
            print(f"  insight: {line[:140]}")


def cmd_history(args) -> None:
    campaign = _campaign(args)
    exps = campaign.store.list_experiments(limit=args.n)
    if args.json:
        print(json.dumps(exps, indent=2, default=str))
        return
    print(f"{'#':>4} {'status':9s} {'value':>10} {'delta':>8} {'speedup':>8} {'kern':>5} {'dur':>5}  description")
    for e in exps:
        delta = e.get("improvement")
        print(f"{e['number']:>4} {e['status']:9s} {fmt(e.get('primary_value')):>10} {(f'{delta * 100:+.1f}%' if isinstance(delta, (int, float)) else '-'):>8} "
              f"{fmt(e.get('speedup_vs_baseline'), 3):>7}x {str(e.get('kernel_count') or '-'):>5} {str(e.get('duration_s') or '-'):>5}  "
              f"{e.get('description', '')[:70]}")


def cmd_show(args) -> None:
    campaign = _campaign(args)
    from .dashboard.data import experiment_detail
    detail = experiment_detail(campaign, args.number)
    if not detail:
        sys.exit(f"no experiment #{args.number}")
    if args.json:
        detail.pop("patch", None)
        print(json.dumps(detail, indent=2, default=str))
        return
    c = detail["compact"]
    print(f"#{c['number']} [{c['status']}] {c['description']}\n  reason: {c.get('reason')}\n  value: {fmt(c.get('primary_value'))} "
          f"delta {fmt((c.get('improvement') or 0) * 100, 3)}% speedup {fmt(c.get('speedup_vs_baseline'), 3)}x kernels {c.get('kernel_count')}")
    for name, stats in (detail.get("metrics") or {}).items():
        if isinstance(stats, dict) and "median_ms" in stats:
            print(f"  {name}: median {fmt(stats['median_ms'])} ms min {fmt(stats.get('min_ms'))} p90 {fmt(stats.get('p90_ms'))}")
    gates = detail.get("gates") or {}
    print(f"  gates: {gates.get('summary')}")
    for chk in gates.get("failed_checks") or []:
        print(f"    x {chk['name']}: {chk['detail'][:160]}")
    if args.patch and detail.get("patch"):
        print(detail["patch"])
    if args.log:
        print(detail.get("log_tail", ""))


def cmd_ideas(args) -> None:
    campaign = _campaign(args)
    hotspots = read_json(campaign.hotspots_path, {}) or {}
    targets = _refresh_statuses(campaign, hotspots.get("targets") or [])
    if not targets:
        sys.exit("no hotspots yet: run `fast-kernel profile` (or `fast-kernel baseline`) first")
    if args.target:
        targets = [t for t in targets if t["id"] == args.target or t["class"] == args.target]
    print(f"untried ideas ranked by Amdahl gain (workload {hotspots.get('workload')}, after exp #{hotspots.get('experiment')}):")
    shown = 0
    for t in targets:
        untried = [x for x in t.get("techniques", []) if x["status"] == "untried"]
        if not untried:
            continue
        print(f"\n[{t['rank']}] {t['title']}  id={t['id']}  share {t['fraction'] * 100:.1f}%  gain {t['amdahl_gain'] * 100:.1f}%")
        if t.get("hint"):
            print(f"     hint: {t['hint'][:200]}")
        for x in untried[: args.per_target]:
            print(f"     - {x['id']:22s} tier {x['tier']} ~{x['expected_speedup']:.1f}x risk {x['risk']:6s} via {', '.join(x['backends'])} -> /{x['skill']}")
        tried = [x for x in t.get("techniques", []) if x["status"] != "untried"]
        if tried:
            print("     tried: " + ", ".join(f"{x['id']}={x['status']}" for x in tried))
        shown += 1
        if shown >= args.limit:
            break
    for line in knowledge.read_insights(campaign.knowledge_path, 6):
        print(f"insight: {line[:160]}")


def cmd_note(args) -> None:
    campaign = _campaign(args)
    tags = [t for item in (args.tags or []) for t in item.split(",") if t]
    exps = campaign.store.list_experiments(limit=1)
    knowledge.add_note(campaign.knowledge_path, args.text, tags, experiment=exps[-1]["number"] if exps else None)
    campaign.store.event("note", text=args.text[:500], tags=tags)
    print("noted")


def cmd_dashboard(args) -> None:
    from .dashboard.server import serve
    root = Path(args.root) if args.root else (Campaign.find().root.parent if Campaign.find() else Path.cwd())
    serve(root, port=args.port, host=args.host, open_browser=args.open)


def cmd_report(args) -> None:
    campaign = _campaign(args)
    from .dashboard.report import build_report
    out = build_report(campaign, Path(args.out or (campaign.root / "report.html")))
    print(f"report written: {out}")


def cmd_loop(args) -> None:
    campaign = _campaign(args)
    if args.action == "start":
        campaign.set_flag("loop.active", "manual")
        campaign.clear_flag("stop")
        campaign.store.event("loop.started", mode="claude-code")
        print("loop flag set: the Claude Code Stop hook will keep the session iterating (/fk-experiment) until `fast-kernel loop stop`.\n"
              "Alternatives: `/loop /fk-experiment` (self-paced wake-ups) or `fast-kernel auto` (headless).")
    elif args.action == "stop":
        campaign.clear_flag("loop.active")
        campaign.set_flag("stop")
        campaign.store.event("loop.stopped", mode="claude-code")
        print("loop flag cleared; the current experiment finishes, then the agent stops.")
    else:
        print(f"loop active: {campaign.has_flag('loop.active')}, paused: {campaign.has_flag('paused')}, stop: {campaign.has_flag('stop')}")


def cmd_flag(args) -> None:
    campaign = _campaign(args)
    if args.cmd == "pause":
        campaign.set_flag("paused")
    elif args.cmd == "resume":
        campaign.clear_flag("paused")
        campaign.clear_flag("stop")
    elif args.cmd == "stop":
        campaign.set_flag("stop")
        campaign.clear_flag("loop.active")
    campaign.store.event("control", action=args.cmd, source="cli")
    print(f"{args.cmd}: ok")


def cmd_auto(args) -> None:
    campaign = _campaign(args)
    from .agents.driver import run_auto
    run_auto(campaign, iterations=args.iterations, model=args.model, max_turns=args.max_turns, permission_mode=args.permission_mode,
             agents=args.agents, worker_iterations=args.worker_iterations)


def cmd_worker(args) -> None:
    campaign = _campaign(args)
    from .agents.worker import run_worker
    run_worker(campaign, args.name, iterations=args.iterations, model=args.model, max_turns=args.max_turns, permission_mode=args.permission_mode)


def cmd_inbox(args) -> None:
    campaign = _campaign(args)
    from .agents.driver import process_inbox
    processed = process_inbox(campaign)
    print(f"processed {len(processed)} proposals: " + ", ".join(f"{p['meta'].get('worker')}:{p['status']}" for p in processed))


def cmd_doctor(args) -> None:
    from .backends.cuda_cpp import find_nvcc
    print(f"fast-kernel {__version__} | python {sys.version.split()[0]} | {sys.executable}")
    checks = []
    try:
        import torch
        checks.append(("torch", f"{torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}"
                       + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "")))
    except ImportError:
        checks.append(("torch", "MISSING -> uv sync --extra cuda"))
    for mod, extra in (("triton", "cuda"), ("transformers", "cuda"), ("tilelang", "tilelang"), ("cutlass", "cute"), ("kernels", "hub"),
                       ("ultralytics", "yolo"), ("liquid_audio", "audio"), ("claude_agent_sdk", "agent")):
        try:
            m = __import__(mod)
            checks.append((mod, f"{getattr(m, '__version__', 'ok')}"))
        except Exception:  # noqa: BLE001
            checks.append((mod, f"missing (optional) -> uv sync --extra {extra}"))
    nvcc, home = find_nvcc()
    checks.append(("nvcc", f"{nvcc} (CUDA_HOME={home})" if nvcc else "missing -> `fast-kernel toolchain install --cuda 13.3` (pip wheels) or a CUDA toolkit"))
    from .backends.toolchain import list_toolchains
    tcs = list_toolchains()
    checks.append(("toolchains", ", ".join(f"cuda-{v}" for v, _ in tcs) if tcs else "none (optional: fast-kernel toolchain install --cuda 13.3 when nvcc rejects the host gcc)"))
    checks.append(("git", run(["git", "--version"]).stdout.strip() or "MISSING"))
    checks.append(("claude", (run([which("claude") or "claude", "--version"]).stdout.strip() or "missing") if which("claude") else "missing -> npm i -g @anthropic-ai/claude-code"))
    checks.append(("uv", run(["uv", "--version"]).stdout.strip() if which("uv") else "missing (optional)"))
    for name, value in checks:
        print(f"  {name:16s} {value}")
    campaign = Campaign.find()
    if campaign and campaign.exists:
        print(f"  campaign         {campaign.root} ({campaign.goal.model}), {campaign.store.next_experiment_number()} experiments")
    else:
        print("  campaign         none here (fast-kernel init <model>)")


def cmd_resolve(args) -> None:
    from .resolve import resolve
    plan = resolve(" ".join(args.text), root=Path(args.root) if args.root else None)
    if args.json:
        print(json.dumps(plan, indent=2, default=str))
        return
    print(f"action: {plan['action']}")
    if plan["action"] == "unknown":
        print(f"  {plan['hint']}")
        return
    print(f"model: {plan.get('model')} ({plan.get('display')})\ncampaign: {plan.get('campaign')} ({'exists, ' + str(plan.get('experiments')) + ' experiments' if plan.get('exists') else 'will be created'})")
    if plan.get("custom_path"):
        print(f"model file: {plan['custom_path']}")
    if plan.get("incumbent"):
        print(f"incumbent: #{plan['incumbent'].get('number')} value={plan['incumbent'].get('value')}")
    if plan.get("dashboard_url"):
        print(f"dashboard: {plan['dashboard_url']}")
    if plan.get("missing_extras"):
        print(f"missing extras: {', '.join(plan['missing_extras'])}")
    print("steps:")
    for i, step in enumerate(plan["steps"], 1):
        print(f"  {i}. {step}")


def cmd_toolchain(args) -> None:
    from .backends.toolchain import install_cuda_toolchain, list_toolchains, remove_toolchain, toolchain_root
    if args.action == "install":
        home = install_cuda_toolchain(args.cuda)
        print(f"CUDA {args.cuda} toolchain installed at {home}\n  it is now preferred by fast-kernel probe / TileLang / CUDA C++ (override with FAST_KERNEL_CUDA_HOME)")
    elif args.action == "remove":
        print("removed" if remove_toolchain(args.cuda) else "nothing to remove")
    else:
        items = list_toolchains()
        print(f"toolchains under {toolchain_root()}:" if items else f"no toolchains under {toolchain_root()} (fast-kernel toolchain install --cuda 13.3)")
        for version, home in items:
            print(f"  cuda-{version}: {home}")


def cmd_models(args) -> None:
    from .models.registry import BUILTIN
    seen = set()
    for _name, (module, cls) in BUILTIN.items():
        if cls in seen:
            continue
        seen.add(cls)
        aliases = [k for k, v in BUILTIN.items() if v[1] == cls]
        print(f"  {', '.join(aliases):28s} {module}.{cls}   template: {'yes' if (TEMPLATES / aliases[0]).exists() else 'custom'}")


def cmd_templates(args) -> None:
    root = Path(__file__).parent / "backends" / "templates"
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("__"):
            continue
        doc = path.read_text(encoding="utf-8").split('"""')[1].strip().splitlines()[0] if '"""' in path.read_text(encoding="utf-8") else ""
        print(f"  {path.name:32s} {doc[:90]}\n{'':34s}{path}")


# ---- parser ----------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fast-kernel", description="Autoresearch for model inference: profile, hypothesise, "
                                     "eval, keep/revert -- forever, with every experiment on a live graph.")
    parser.add_argument("--version", action="version", version=f"fast-kernel {__version__}")
    parser.add_argument("--campaign", "-C", help="campaign directory (default: search upwards from cwd)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="scaffold a campaign for a model (mimi | lfm25 | lfm-audio | yolo | custom)")
    p.add_argument("model")
    p.add_argument("--dir", help="destination directory (default campaigns/<model>)")
    p.add_argument("--name")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override GOAL.md frontmatter, e.g. model_args.variant=350m")
    p.add_argument("--precision", choices=["strict", "tolerant"])
    p.add_argument("--no-git", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("probe", help="detect GPU + measure roofline + probe kernel backends -> capabilities.json")
    p.add_argument("--no-compile", action="store_true", help="skip compile tests")
    p.add_argument("--no-microbench", action="store_true")
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("profile", help="trace the current candidate, rank hotspots (Amdahl), write PLAN.md + hotspots.json")
    p.add_argument("--workload")
    p.add_argument("--top", type=int, default=12)
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("baseline", help="experiment #0: measure the unmodified reference path + noise floor")
    p.add_argument("-m", "--message")
    p.add_argument("--repeats", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("eval", help="evaluate the current candidate/ tree: gates -> bench -> profile -> keep/revert")
    p.add_argument("-m", "--message", required=True, help="one-line hypothesis")
    p.add_argument("--technique", action="append", help="playbook technique id(s), comma separated")
    p.add_argument("--target", help="hotspot target id from PLAN.md")
    p.add_argument("--notes")
    p.add_argument("--simpler", action="store_true", help="keep if not slower beyond threshold and the code is simpler")
    p.add_argument("--force", action="store_true", help="evaluate even if candidate/ is unchanged")
    p.add_argument("--no-profile", action="store_true")
    p.add_argument("--repeats", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--workloads")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("status", help="campaign summary")
    p.add_argument("--json", action="store_true")
    p.add_argument("--brief", action="store_true")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("history", help="experiment ledger")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_history)

    p = sub.add_parser("show", help="details of one experiment")
    p.add_argument("number", type=int)
    p.add_argument("--patch", action="store_true")
    p.add_argument("--log", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("ideas", help="untried target x technique pairs ranked by expected end-to-end gain")
    p.add_argument("--target")
    p.add_argument("--limit", type=int, default=6)
    p.add_argument("--per-target", type=int, default=4)
    p.set_defaults(fn=cmd_ideas)

    p = sub.add_parser("note", help="append an insight to KNOWLEDGE.md")
    p.add_argument("text")
    p.add_argument("--tags", action="append")
    p.set_defaults(fn=cmd_note)

    p = sub.add_parser("dashboard", help="live dashboard (SSE) for every campaign under --root")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--root")
    p.add_argument("--open", action="store_true")
    p.set_defaults(fn=cmd_dashboard)

    p = sub.add_parser("report", help="export a self-contained HTML report")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("loop", help="start/stop the Claude Code Stop-hook loop flag")
    p.add_argument("action", choices=["start", "stop", "status"])
    p.set_defaults(fn=cmd_loop)
    for name in ("pause", "resume", "stop"):
        p = sub.add_parser(name, help=f"{name} the running loop / workers")
        p.set_defaults(fn=cmd_flag)

    p = sub.add_parser("auto", help="headless endless loop with `claude -p` (optionally N parallel worktree workers)")
    p.add_argument("--iterations", type=int)
    p.add_argument("--agents", type=int, default=0, help="parallel workers (0 = single agent loop)")
    p.add_argument("--worker-iterations", type=int)
    p.add_argument("--model")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--permission-mode", default="acceptEdits")
    p.set_defaults(fn=cmd_auto)

    p = sub.add_parser("worker", help="run one parallel worker (own worktree + hotspot lease)")
    p.add_argument("--name", required=True)
    p.add_argument("--iterations", type=int)
    p.add_argument("--model")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--permission-mode", default="acceptEdits")
    p.set_defaults(fn=cmd_worker)

    p = sub.add_parser("inbox", help="evaluate worker proposals on top of the incumbent")
    p.set_defaults(fn=cmd_inbox)

    p = sub.add_parser("resolve", help='map a sentence ("Optimize the Mimi codec model.") to model, campaign folder and remaining steps')
    p.add_argument("text", nargs="+")
    p.add_argument("--root", help="repository root (default: found by walking up from cwd)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("toolchain", help="self-contained CUDA toolchains from pip wheels (nvcc/cccl/crt/nvvm/runtime)")
    p.add_argument("action", choices=["install", "list", "remove"])
    p.add_argument("--cuda", default="13.3", help="CUDA minor version, e.g. 13.3")
    p.set_defaults(fn=cmd_toolchain)

    p = sub.add_parser("doctor", help="environment check with fix hints")
    p.set_defaults(fn=cmd_doctor)
    p = sub.add_parser("models", help="list built-in model specs")
    p.set_defaults(fn=cmd_models)
    p = sub.add_parser("templates", help="list starter kernel templates")
    p.set_defaults(fn=cmd_templates)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

/* fast-kernel dashboard: live (SSE) and static (embedded data) modes share this file. */
(() => {
  "use strict";
  const STATIC_DATA = window.__FK_DATA__ || null;
  const STATIC = !!STATIC_DATA;
  const $ = (id) => document.getElementById(id);
  const state = {
    campaigns: [], name: null, data: null, details: {}, events: [], lastEventId: 0, es: null,
    logScale: false, selected: null, filter: "", hover: null, refreshTimer: null,
  };
  const STATUS_VAR = { keep: "--good", bank: "--good", baseline: "--accent", remeasure: "--other", discard: "--other", crash: "--critical", error: "--warning", running: "--accent" };
  const STATUS_ICON = { keep: "✓", bank: "▣", baseline: "○", remeasure: "=", discard: "–", crash: "✕", error: "!", running: "…" };
  const SLOTS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"];

  // ---------------------------------------------------------------- utils
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const statusColor = (s) => css(STATUS_VAR[s] || "--other");
  function fmt(v, digits = 3) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    if (typeof v !== "number") return String(v);
    const a = Math.abs(v);
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 100) return v.toFixed(1);
    if (a >= 10) return v.toFixed(2);
    if (a >= 1) return v.toFixed(3);
    if (a === 0) return "0";
    return v.toPrecision(digits);
  }
  const pct = (v, d = 1) => (v === null || v === undefined) ? "–" : `${(v * 100).toFixed(d)}%`;
  const signedPct = (v) => (v === null || v === undefined) ? "–" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
  const ago = (iso) => {
    if (!iso) return "";
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  };
  const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; };
  const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); return node; };
  const metricName = () => (state.data && state.data.summary.target_metric) || "latency_ms";
  const minimize = () => !state.data || String(state.data.summary.direction || "minimize").startsWith("min");

  // ---------------------------------------------------------------- tooltip
  const tip = $("tooltip");
  function showTip(x, y, rows) {
    clear(tip);
    rows.forEach((r) => {
      if (typeof r === "string") { tip.appendChild(el("div", "k", r)); return; }
      const row = el("div", "row");
      if (r.color) { const i = el("i"); i.style.background = r.color; row.appendChild(i); }
      if (r.value !== undefined) row.appendChild(el("span", "v", r.value));
      if (r.label) row.appendChild(el("span", "k", r.label));
      tip.appendChild(row);
    });
    tip.style.display = "block";
    const rect = tip.getBoundingClientRect();
    const left = Math.min(x + 14, window.innerWidth - rect.width - 8);
    const top = Math.min(y + 14, window.innerHeight - rect.height - 8);
    tip.style.left = `${Math.max(4, left)}px`;
    tip.style.top = `${Math.max(4, top)}px`;
  }
  const hideTip = () => { tip.style.display = "none"; };

  // ---------------------------------------------------------------- data
  async function api(path, options) {
    const res = await fetch(path, options);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  }
  async function loadCampaigns() {
    if (STATIC) {
      state.campaigns = [{ name: STATIC_DATA.name }];
      fillSelect();
      return selectCampaign(STATIC_DATA.name);
    }
    const payload = await api("/api/campaigns");
    state.campaigns = payload.campaigns.length ? payload.campaigns : payload.names.map((n) => ({ name: n }));
    fillSelect();
    const wanted = decodeURIComponent(location.hash.slice(1)) || (state.campaigns[0] && state.campaigns[0].name);
    if (wanted) await selectCampaign(wanted);
  }
  function fillSelect() {
    const sel = clear($("campaign-select"));
    state.campaigns.forEach((c) => {
      const o = el("option", null, c.model ? `${c.name} · ${c.model} · ${c.experiments || 0} exp` : c.name);
      o.value = c.name;
      sel.appendChild(o);
    });
    sel.value = state.name || sel.value;
  }
  async function selectCampaign(name) {
    state.name = name;
    $("campaign-select").value = name;
    if (!STATIC) location.hash = encodeURIComponent(name);
    closeStream();
    state.events = [];
    state.details = {};
    if (STATIC) {
      state.data = STATIC_DATA.state;
      state.details = STATIC_DATA.details || {};
      state.events = STATIC_DATA.events || [];
      state.lastEventId = state.events.length ? state.events[state.events.length - 1].id : 0;
      $("stream-state").textContent = "static report";
      $("controls").classList.add("hidden");
      $("note-form").classList.add("hidden");
      renderAll();
      return;
    }
    await refreshState();
    const initial = await api(`/api/c/${encodeURIComponent(name)}/events?after=${Math.max(0, state.lastEventId - 200)}&limit=200`);
    state.events = initial.events;
    renderEvents();
    openStream();
  }
  async function refreshState() {
    if (STATIC || !state.name) return;
    try {
      const data = await api(`/api/c/${encodeURIComponent(state.name)}/state`);
      state.data = data;
      if (!state.lastEventId) state.lastEventId = data.last_event_id || 0;
      renderAll();
    } catch (err) {
      console.error(err);
    }
  }
  const scheduleRefresh = () => {
    if (state.refreshTimer) return;
    state.refreshTimer = setTimeout(() => { state.refreshTimer = null; refreshState(); }, 800);
  };
  function openStream() {
    if (STATIC) return;
    const es = new EventSource(`/api/c/${encodeURIComponent(state.name)}/stream?after=${state.lastEventId}`);
    state.es = es;
    es.addEventListener("hello", () => setStream("live", true));
    es.addEventListener("heartbeat", () => setStream("live", true));
    es.addEventListener("events", (e) => {
      const payload = JSON.parse(e.data);
      let structural = false;
      payload.events.forEach((ev) => {
        state.events.push(ev);
        state.lastEventId = Math.max(state.lastEventId, ev.id);
        if (!/^agent\.(text|tool)$/.test(ev.kind)) structural = true;
        if (ev.kind === "agent" || ev.kind.startsWith("agent.")) updateAgentFromEvent(ev);
      });
      if (state.events.length > 400) state.events = state.events.slice(-400);
      renderEvents();
      if (structural) scheduleRefresh();
    });
    es.onerror = () => setStream("reconnecting", false);
  }
  function closeStream() { if (state.es) { state.es.close(); state.es = null; } }
  function setStream(text, live) {
    $("stream-state").textContent = text;
    $("stream-pill").classList.toggle("live", live);
  }

  // ---------------------------------------------------------------- derived series
  function experiments() { return (state.data && state.data.experiments) || []; }
  function incumbentSeries() {
    // incumbent value after each experiment (keeps and baseline move it)
    const exps = experiments();
    let inc = null, incExp = null;
    return exps.map((e) => {
      if ((e.status === "keep" || e.status === "baseline") && e.primary_value != null) { inc = e.primary_value; incExp = e; }
      return { number: e.number, value: inc, speedup: incExp ? (incExp.speedup_vs_baseline || (incExp.number === 0 ? 1 : null)) : null,
               kernels: incExp ? incExp.kernel_count : null, exp: e };
    });
  }

  // ---------------------------------------------------------------- rendering
  function renderAll() {
    if (!state.data) return;
    renderHeader();
    renderKpis();
    renderExpChart();
    renderSpeedupChart();
    renderKernelChart();
    renderShare();
    renderTargets();
    renderLineage();
    renderTable();
    renderAgents();
    renderInsights();
    renderEvents();
    renderFooter();
  }
  function renderHeader() {
    const s = state.data.summary;
    $("objective").textContent = `${s.model} · ${s.objective}`;
    const pill = $("run-pill");
    pill.classList.remove("live", "paused");
    let text = "idle";
    const running = (state.data.agents || []).some((a) => a.state === "running");
    if (s.paused) { text = "paused"; pill.classList.add("paused"); }
    else if (s.loop_active || running) { text = s.loop_active ? "loop active" : "running"; pill.classList.add("live"); }
    $("run-state").textContent = text;
    $("btn-loop").textContent = s.loop_active ? "loop is on" : "start loop";
    $("btn-loop").disabled = !!s.loop_active;
    $("btn-pause").disabled = !!s.paused;
    $("btn-resume").disabled = !s.paused;
    $("th-metric").textContent = metricName();
  }
  function renderKpis() {
    const s = state.data.summary;
    const exps = experiments();
    const inc = s.incumbent || {};
    const base = exps.find((e) => e.number === 0);
    const incExp = exps.find((e) => e.number === inc.number) || base;
    const counts = s.counts || {};
    const wrap = clear($("kpis"));
    const tile = (label, value, delta, deltaClass, spark) => {
      const t = el("div", "tile");
      t.appendChild(el("div", "label", label));
      t.appendChild(el("div", "value", value));
      if (delta) t.appendChild(el("div", `delta ${deltaClass || ""}`, delta));
      if (spark) { const c = el("canvas"); t.appendChild(c); requestAnimationFrame(() => drawSpark(c, spark)); }
      wrap.appendChild(t);
    };
    const unit = metricName() === "latency_ms" ? " ms" : "";
    const baseV = s.baseline_value, incV = inc.value;
    const rel = (baseV && incV) ? (minimize() ? (incV - baseV) / baseV : (incV - baseV) / baseV) : null;
    const good = rel != null && (minimize() ? rel < 0 : rel > 0);
    tile(`incumbent ${metricName()}`, incV != null ? fmt(incV) + unit : "–", rel != null ? `${signedPct(rel)} vs baseline ${fmt(baseV)}${unit}` : "no baseline yet",
         rel == null ? "" : (good ? "up" : "down"), incumbentSeries().map((p) => p.value));
    tile("speedup vs baseline", s.speedup_vs_baseline ? `${s.speedup_vs_baseline.toFixed(2)}×` : "–",
         incExp ? `incumbent #${inc.number} · ${incExp.description ? incExp.description.slice(0, 40) : ""}` : "");
    tile("experiments", String(exps.length), `✓ ${counts.keep || 0} keep · – ${counts.discard || 0} discard · ✕ ${counts.crash || 0} crash${counts.error ? ` · ! ${counts.error} error` : ""}`);
    const k0 = base && base.kernel_count, k1 = incExp && incExp.kernel_count;
    tile("kernel launches / call", k1 != null ? String(k1) : "–", k0 != null && k1 != null && k0 !== k1 ? `${k1 < k0 ? "−" : "+"}${Math.abs(k1 - k0)} vs baseline ${k0}` : (k0 != null ? `baseline ${k0}` : ""),
         k0 != null && k1 != null ? (k1 < k0 ? "up" : (k1 > k0 ? "down" : "")) : "", incumbentSeries().map((p) => p.kernels));
    const busy = incExp && incExp.gpu_busy_ratio;
    tile("GPU busy (incumbent)", busy != null ? pct(busy, 0) : "–", busy != null ? (busy < 0.6 ? "launch / overhead bound" : "GPU bound") : "");
    const thr = incExp && (incExp.tokens_per_s != null ? `${fmt(incExp.tokens_per_s)} tok/s` : incExp.fps != null ? `${fmt(incExp.fps)} fps` : incExp.rtf != null ? `rtf ${fmt(incExp.rtf)}` : null);
    if (thr) tile("throughput", thr, incExp.rtf != null ? `${fmt(1 / incExp.rtf, 3)}× realtime` : "");
    const agents = state.data.agents || [];
    tile("agents", String(agents.filter((a) => a.state === "running").length), agents.length ? agents.map((a) => `${a.name}: ${a.state}`).join(" · ") : "none registered");
  }
  function drawSpark(canvas, values) {
    const vals = values.filter((v) => v != null);
    if (vals.length < 2) return;
    const { ctx, w, h } = setupCanvas(canvas, 26);
    const min = Math.min(...vals), max = Math.max(...vals);
    const x = (i) => 2 + (i / (vals.length - 1)) * (w - 4);
    const y = (v) => max === min ? h / 2 : 2 + (1 - (v - min) / (max - min)) * (h - 4);
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    vals.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
    ctx.fillStyle = css("--accent");
    ctx.beginPath(); ctx.arc(x(vals.length - 1), y(vals[vals.length - 1]), 3, 0, Math.PI * 2); ctx.fill();
  }

  // ---- canvas helpers
  function setupCanvas(canvas, height) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, canvas.clientWidth || canvas.parentElement.clientWidth || 600);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, height);
    return { ctx, w, h: height };
  }
  function niceTicks(min, max, count = 5) {
    if (!(max > min)) { max = min + 1; }
    const span = max - min;
    const step0 = span / Math.max(1, count);
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const norm = step0 / mag;
    const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }
  function logTicks(min, max) {
    const ticks = [];
    const lo = Math.floor(Math.log10(min)), hi = Math.ceil(Math.log10(max));
    for (let e = lo; e <= hi; e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v); }
    return ticks;
  }
  function axes(ctx, w, h, pad, yTicks, yScale, xLabels, xScale) {
    ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1; ctx.fillStyle = css("--muted"); ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    yTicks.forEach((t) => {
      const y = Math.round(yScale(t)) + 0.5;
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(fmt(t), pad.l - 6, y);
    });
    ctx.strokeStyle = css("--axis");
    ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    xLabels.forEach(({ x, text }) => ctx.fillText(text, xScale(x), h - pad.b + 6));
  }
  function ring(ctx, x, y, r, color) {
    ctx.beginPath(); ctx.arc(x, y, r + 2, 0, Math.PI * 2); ctx.fillStyle = css("--surface"); ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
  }

  // ---- chart: every experiment
  const expChart = { points: [], hover: null };
  function renderExpChart() {
    const canvas = $("chart-exp");
    const exps = experiments();
    const valid = exps.filter((e) => e.primary_value != null);
    $("chart-exp-empty").style.display = exps.length ? "none" : "block";
    const { ctx, w, h } = setupCanvas(canvas, 300);
    expChart.points = [];
    if (!exps.length) return;
    const pad = { l: 58, r: 16, t: 14, b: 30 };
    const values = valid.map((e) => e.primary_value);
    const inc = incumbentSeries();
    let min = Math.min(...values), max = Math.max(...values);
    if (state.logScale) { min = Math.max(min, 1e-6); }
    if (min === max) { min *= 0.9; max *= 1.1; }
    const padV = (max - min) * 0.08;
    const lo = state.logScale ? min / 1.15 : Math.max(0, min - padV), hi = state.logScale ? max * 1.15 : max + padV;
    const n = exps.length;
    const xScale = (i) => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (w - pad.l - pad.r);
    const yScale = state.logScale
      ? (v) => pad.t + (1 - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (h - pad.t - pad.b)
      : (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    const ticks = state.logScale ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    const every = Math.max(1, Math.ceil(n / Math.max(1, Math.floor((w - pad.l - pad.r) / 44))));
    const xLabels = exps.map((e, i) => ({ x: i, text: `#${e.number}` })).filter((_, i) => i % every === 0 || i === n - 1);
    axes(ctx, w, h, pad, ticks, yScale, xLabels, xScale);
    // incumbent step line (single series -> slot 1)
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.beginPath();
    let started = false;
    inc.forEach((p, i) => {
      if (p.value == null) return;
      const x = xScale(i), y = yScale(p.value);
      if (!started) { ctx.moveTo(x, y); started = true; }
      else { ctx.lineTo(x, yScale(inc[i - 1].value != null ? inc[i - 1].value : p.value)); ctx.lineTo(x, y); }
    });
    ctx.stroke();
    // points by status
    exps.forEach((e, i) => {
      if (e.primary_value == null) {
        if (e.status === "crash" || e.status === "error" || e.status === "running") {
          const x = xScale(i), y = h - pad.b - 8;
          ring(ctx, x, y, 4, statusColor(e.status));
          expChart.points.push({ x, y, exp: e });
        }
        return;
      }
      const x = xScale(i), y = yScale(e.primary_value);
      const r = expChart.hover === e.number ? 6 : 4.5;
      ring(ctx, x, y, r, statusColor(e.status));
      expChart.points.push({ x, y, exp: e });
    });
    // selective direct label: the incumbent's last value
    const last = [...inc].reverse().find((p) => p.value != null);
    if (last) {
      ctx.fillStyle = css("--ink-2"); ctx.font = "600 11px system-ui, sans-serif"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
      ctx.fillText(`${fmt(last.value)}${metricName() === "latency_ms" ? " ms" : ""}`, w - pad.r, yScale(last.value) - 8);
    }
    canvas.onpointermove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
      let best = null, bestD = 24;
      expChart.points.forEach((p) => { const d = Math.hypot(p.x - px, p.y - py); if (d < bestD) { bestD = d; best = p; } });
      if (!best) { if (expChart.hover != null) { expChart.hover = null; renderExpChart(); } hideTip(); return; }
      if (expChart.hover !== best.exp.number) { expChart.hover = best.exp.number; renderExpChart(); }
      const e = best.exp;
      showTip(ev.clientX, ev.clientY, [
        { value: `#${e.number} ${STATUS_ICON[e.status] || ""} ${e.status}`, color: statusColor(e.status) },
        { value: e.primary_value != null ? `${fmt(e.primary_value)} ${metricName()}` : "no metric", label: e.improvement != null ? `Δ ${signedPct(e.improvement)}` : "" },
        { value: e.speedup_vs_baseline ? `${e.speedup_vs_baseline.toFixed(2)}×` : "", label: e.speedup_vs_baseline ? "vs baseline" : "" },
        { value: e.kernel_count != null ? `${e.kernel_count} kernels` : "", label: e.gpu_busy_ratio != null ? `GPU busy ${pct(e.gpu_busy_ratio, 0)}` : "" },
        (e.description || "").slice(0, 140),
        e.reason ? e.reason.slice(0, 140) : "",
      ].filter((r) => r && (typeof r === "string" ? r.length : (r.value || r.label))));
    };
    canvas.onpointerleave = () => { hideTip(); if (expChart.hover != null) { expChart.hover = null; renderExpChart(); } };
    canvas.onclick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
      let best = null, bestD = 24;
      expChart.points.forEach((p) => { const d = Math.hypot(p.x - px, p.y - py); if (d < bestD) { bestD = d; best = p; } });
      if (best) openDrawer(best.exp.number);
    };
  }
  function stepChart(canvas, height, series, unitLabel, opts = {}) {
    const { ctx, w, h } = setupCanvas(canvas, height);
    const pts = series.map((v, i) => ({ i, v })).filter((p) => p.v != null);
    if (pts.length < 1) { ctx.fillStyle = css("--muted"); ctx.font = "12px system-ui"; ctx.fillText("waiting for data", 12, height / 2); return; }
    const pad = { l: 50, r: 14, t: 10, b: 22 };
    const vals = pts.map((p) => p.v);
    let min = opts.zero ? 0 : Math.min(...vals), max = Math.max(...vals);
    if (min === max) { max = min + (min || 1) * 0.5; }
    const padV = (max - min) * 0.1;
    const lo = opts.zero ? 0 : min - padV, hi = max + padV;
    const n = series.length;
    const xScale = (i) => pad.l + (n === 1 ? 0.5 : i / (n - 1)) * (w - pad.l - pad.r);
    const yScale = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    const every = Math.max(1, Math.ceil(n / Math.max(1, Math.floor((w - pad.l - pad.r) / 44))));
    axes(ctx, w, h, pad, niceTicks(lo, hi, 4), yScale, series.map((_, i) => ({ x: i, text: `#${i}` })).filter((_, i) => i % every === 0 || i === n - 1), xScale);
    ctx.strokeStyle = css("--accent"); ctx.lineWidth = 2; ctx.lineJoin = "round";
    ctx.beginPath();
    pts.forEach((p, k) => { const x = xScale(p.i), y = yScale(p.v); if (!k) ctx.moveTo(x, y); else { ctx.lineTo(x, yScale(pts[k - 1].v)); ctx.lineTo(x, y); } });
    ctx.stroke();
    const last = pts[pts.length - 1];
    ring(ctx, xScale(last.i), yScale(last.v), 4, css("--accent"));
    ctx.fillStyle = css("--ink-2"); ctx.font = "600 11px system-ui"; ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText(`${fmt(last.v)}${unitLabel}`, w - pad.r, yScale(last.v) - 8);
    canvas.onpointermove = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      let best = null, bestD = 30;
      pts.forEach((p) => { const d = Math.abs(xScale(p.i) - px); if (d < bestD) { bestD = d; best = p; } });
      if (!best) { hideTip(); return; }
      showTip(ev.clientX, ev.clientY, [{ value: `${fmt(best.v)}${unitLabel}`, label: `after experiment #${best.i}`, color: css("--accent") }]);
    };
    canvas.onpointerleave = hideTip;
  }
  function renderSpeedupChart() { stepChart($("chart-speedup"), 200, incumbentSeries().map((p) => p.speedup), "×", { zero: false }); }
  function renderKernelChart() { stepChart($("chart-kernels"), 120, incumbentSeries().map((p) => p.kernels), " kernels", { zero: true }); }

  // ---- where the time goes (stacked horizontal bars, categories = fixed slots)
  function renderShare() {
    const wrap = clear($("share-bars"));
    const legend = clear($("share-legend"));
    const base = state.data.baseline_targets || [];
    const cur = (state.data.hotspots && state.data.hotspots.targets) || [];
    const rows = [];
    const toShares = (targets) => {
      const groups = targets.filter((t) => t.category !== "launch-bound");
      const total = groups.reduce((a, t) => a + (t.fraction || 0), 0) || 1;
      const byCat = {};
      groups.forEach((t) => { byCat[t.category] = (byCat[t.category] || 0) + (t.fraction || 0) / total; });
      return byCat;
    };
    if (base.length) rows.push({ label: "baseline #0", shares: toShares(base), info: base.find((t) => t.category === "launch-bound") });
    if (cur.length) rows.push({ label: `incumbent #${state.data.hotspots.experiment != null ? state.data.hotspots.experiment : "?"}`, shares: toShares(cur), info: cur.find((t) => t.category === "launch-bound"), summary: state.data.hotspots.summary });
    if (!rows.length) { wrap.appendChild(el("div", "empty", "No profile yet — run fast-kernel profile.")); return; }
    // fixed slot order: first appearance, biggest first in the baseline; tail folded into Other
    const order = [];
    rows.forEach((r) => Object.entries(r.shares).sort((a, b) => b[1] - a[1]).forEach(([c]) => { if (!order.includes(c)) order.push(c); }));
    const cats = order.slice(0, 7);
    const slot = (c) => cats.includes(c) ? css(SLOTS[cats.indexOf(c)]) : css("--other");
    rows.forEach((r) => {
      const row = el("div", "bar-row");
      row.appendChild(el("div", "label", r.label));
      const track = el("div", "track");
      const entries = Object.entries(r.shares).sort((a, b) => cats.indexOf(a[0]) === -1 ? 1 : cats.indexOf(b[0]) === -1 ? -1 : cats.indexOf(a[0]) - cats.indexOf(b[0]));
      const other = entries.filter(([c]) => !cats.includes(c)).reduce((a, [, v]) => a + v, 0);
      const segs = entries.filter(([c]) => cats.includes(c));
      if (other > 0) segs.push(["other", other]);
      segs.forEach(([c, v]) => {
        const seg = el("div", "seg");
        seg.style.width = `${Math.max(0.6, v * 100)}%`;
        seg.style.background = c === "other" ? css("--other") : slot(c);
        if (v > 0.11) seg.appendChild(el("span", null, `${c} ${pct(v, 0)}`));
        seg.onpointermove = (ev) => showTip(ev.clientX, ev.clientY, [{ value: pct(v, 1), label: `${c} · ${r.label}`, color: seg.style.background }]);
        seg.onpointerleave = hideTip;
        track.appendChild(seg);
      });
      row.appendChild(track);
      const s = r.summary || {};
      row.appendChild(el("div", "total", s.wall_ms != null ? `${fmt(s.wall_ms)} ms` : (r.info ? `${pct(1 - r.info.fraction, 0)} busy` : "")));
      wrap.appendChild(row);
      if (r.info) wrap.appendChild(el("div", "muted", `${r.label}: GPU idle ${pct(r.info.fraction, 0)} of wall time (launch / overhead bound)`));
    });
    cats.forEach((c) => { const k = el("span", "key"); const sw = el("span", "swatch"); sw.style.background = slot(c); k.appendChild(sw); k.appendChild(el("span", null, c)); legend.appendChild(k); });
    if (order.length > cats.length) { const k = el("span", "key"); const sw = el("span", "swatch"); sw.style.background = css("--other"); k.appendChild(sw); k.appendChild(el("span", null, "other")); legend.appendChild(k); }
  }
  function renderTargets() {
    const tbody = clear($("targets-table").querySelector("tbody"));
    const targets = (state.data.hotspots && state.data.hotspots.targets) || [];
    const h = state.data.hotspots || {};
    $("targets-sub").textContent = targets.length ? `after experiment #${h.experiment != null ? h.experiment : "?"} · workload ${h.workload || ""} · ${h.summary && h.summary.kernel_count != null ? h.summary.kernel_count + " launches" : ""}` : "run fast-kernel profile";
    targets.slice(0, 12).forEach((t) => {
      const tr = el("tr");
      tr.appendChild(el("td", "num", String(t.rank)));
      const td = el("td"); td.appendChild(el("b", null, t.class)); td.appendChild(el("div", "muted", `${t.category} · ${t.instance_count || 1} inst · ${t.attempts || 0} tried`)); tr.appendChild(td);
      tr.appendChild(el("td", null, t.boundness));
      tr.appendChild(el("td", "num", pct(t.fraction, 1)));
      tr.appendChild(el("td", "num", t.sol_efficiency != null ? pct(t.sol_efficiency, 0) : "–"));
      tr.appendChild(el("td", "num", t.headroom != null ? pct(t.headroom, 1) : "–"));
      tr.onpointermove = (ev) => showTip(ev.clientX, ev.clientY, [{ value: t.title }, t.hint || "", `id ${t.id} · ${t.kernel_count} kernels`]);
      tr.onpointerleave = hideTip;
      tbody.appendChild(tr);
    });
  }

  // ---- lineage (SVG)
  function renderLineage() {
    const wrap = clear($("lineage"));
    const exps = experiments();
    if (!exps.length) { wrap.appendChild(el("div", "empty", "No experiments yet.")); return; }
    const NS = "http://www.w3.org/2000/svg";
    const gap = 46, width = Math.max(600, exps.length * gap + 60), height = 190, trunk = 95;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", width); svg.setAttribute("height", height); svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const pos = {};
    let side = 1;
    exps.forEach((e, i) => {
      // banked experiments are committed and built on, so they sit on the trunk like a keep
      const onTrunk = e.status === "keep" || e.status === "baseline" || e.status === "bank";
      const y = onTrunk ? trunk : (trunk + side * 58);
      if (!onTrunk) side = -side;
      pos[e.number] = { x: 30 + i * gap, y };
    });
    exps.forEach((e) => {
      if (e.parent == null || !pos[e.parent]) return;
      const a = pos[e.parent], b = pos[e.number];
      const path = document.createElementNS(NS, "path");
      const mx = (a.x + b.x) / 2;
      path.setAttribute("d", `M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}`);
      if (e.status === "keep" || e.status === "bank") path.setAttribute("stroke", css("--accent"));
      if (e.status === "keep") path.setAttribute("stroke", css("--accent"));
      svg.appendChild(path);
    });
    exps.forEach((e) => {
      const p = pos[e.number];
      const g = document.createElementNS(NS, "g");
      g.setAttribute("class", "node");
      const halo = document.createElementNS(NS, "circle");
      halo.setAttribute("cx", p.x); halo.setAttribute("cy", p.y); halo.setAttribute("r", 14); halo.setAttribute("fill", "transparent");
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("cx", p.x); c.setAttribute("cy", p.y); c.setAttribute("r", e.status === "keep" || e.status === "baseline" ? 8 : (e.status === "bank" ? 7 : 6));
      c.setAttribute("fill", statusColor(e.status)); c.setAttribute("stroke", css("--surface")); c.setAttribute("stroke-width", 2);
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", p.x); t.setAttribute("y", p.y + (p.y >= trunk ? 22 : -14)); t.setAttribute("text-anchor", "middle");
      t.textContent = `#${e.number}`;
      g.appendChild(halo); g.appendChild(c); g.appendChild(t);
      g.onpointermove = (ev) => showTip(ev.clientX, ev.clientY, [
        { value: `#${e.number} ${STATUS_ICON[e.status] || ""} ${e.status}`, color: statusColor(e.status) },
        { value: e.primary_value != null ? fmt(e.primary_value) : "", label: e.speedup_vs_baseline ? `${e.speedup_vs_baseline.toFixed(2)}× vs baseline` : "" },
        (e.description || "").slice(0, 120)]);
      g.onpointerleave = hideTip;
      g.onclick = () => openDrawer(e.number);
      svg.appendChild(g);
    });
    wrap.appendChild(svg);
    wrap.scrollLeft = wrap.scrollWidth;
  }

  // ---- table
  function renderTable() {
    const tbody = clear($("exp-table").querySelector("tbody"));
    const q = state.filter.toLowerCase();
    [...experiments()].reverse().filter((e) => !q || `${e.number} ${e.status} ${e.description} ${(e.techniques || []).join(" ")} ${e.agent || ""}`.toLowerCase().includes(q)).forEach((e) => {
      const tr = el("tr");
      if (state.selected === e.number) tr.classList.add("active");
      tr.appendChild(el("td", "num", String(e.number)));
      const st = el("span", `status ${e.status}`); st.appendChild(el("i")); st.appendChild(el("span", null, `${STATUS_ICON[e.status] || ""} ${e.status}`));
      const tdS = el("td"); tdS.appendChild(st); tr.appendChild(tdS);
      const tdD = el("td"); tdD.appendChild(el("div", null, e.description || "")); if (e.techniques && e.techniques.length) tdD.appendChild(el("div", "muted mono", e.techniques.join(", ") + (e.target ? ` → ${e.target}` : ""))); tr.appendChild(tdD);
      tr.appendChild(el("td", "num", e.primary_value != null ? fmt(e.primary_value) : "–"));
      tr.appendChild(el("td", "num", e.improvement != null ? signedPct(e.improvement) : "–"));
      tr.appendChild(el("td", "num", e.speedup_vs_baseline ? `${e.speedup_vs_baseline.toFixed(2)}×` : "–"));
      tr.appendChild(el("td", "num", e.kernel_count != null ? String(e.kernel_count) : "–"));
      tr.appendChild(el("td", null, e.gates_summary ? (e.gates_passed ? "pass" : `FAIL: ${(e.failed_checks || []).slice(0, 2).join(", ")}`) : (e.reason || "").slice(0, 40)));
      tr.appendChild(el("td", "num", e.duration_s != null ? `${Math.round(e.duration_s)}s` : "–"));
      tr.appendChild(el("td", "muted", e.agent || ""));
      tr.onclick = () => openDrawer(e.number);
      tbody.appendChild(tr);
    });
  }

  // ---- agents / insights / events
  function updateAgentFromEvent(ev) {
    if (!state.data) return;
    const p = ev.payload || {};
    const name = p.name || p.agent;
    if (!name) return;
    const agents = state.data.agents || (state.data.agents = []);
    let a = agents.find((x) => x.name === name);
    if (!a) { a = { name, state: "running", detail: "", updated_at: ev.ts }; agents.push(a); }
    if (ev.kind === "agent") { a.state = p.state; a.detail = p.detail || ""; }
    else if (ev.kind === "agent.tool") { a.state = "running"; a.detail = `${p.tool}: ${p.detail || ""}`; }
    else if (ev.kind === "agent.text") { a.detail = (p.text || "").slice(0, 160); }
    else if (ev.kind === "agent.result" || ev.kind === "agent.iteration") { a.state = "idle"; a.detail = p.result ? String(p.result).slice(0, 160) : `iteration done`; }
    a.updated_at = ev.ts;
    renderAgents();
    renderHeader();
  }
  function renderAgents() {
    const wrap = clear($("agents"));
    const agents = (state.data && state.data.agents) || [];
    if (!agents.length) { wrap.appendChild(el("div", "empty", "No agent activity recorded yet (fast-kernel auto / eval register here).")); return; }
    agents.sort((a, b) => (a.state === "running" ? -1 : 1) - (b.state === "running" ? -1 : 1) || a.name.localeCompare(b.name)).forEach((a) => {
      const row = el("div", `agent ${a.state}`);
      row.appendChild(el("i"));
      const body = el("div");
      const head = el("div"); head.appendChild(el("b", null, a.name)); head.appendChild(el("span", "muted", ` · ${a.state} `)); head.appendChild(el("time", null, ago(a.updated_at)));
      body.appendChild(head);
      body.appendChild(el("small", null, a.detail || ""));
      row.appendChild(body);
      wrap.appendChild(row);
    });
    const leases = (state.data && state.data.leases || []).filter((l) => l.state === "active");
    if (leases.length) wrap.appendChild(el("div", "muted", `leases: ${leases.map((l) => `${l.worker}→${l.target_id}`).join(", ")}`));
  }
  function renderInsights() {
    const ul = clear($("insights"));
    const items = (state.data && state.data.insights) || [];
    if (!items.length) { ul.appendChild(el("li", "muted", "no insights yet — `fast-kernel note \"...\"`")); return; }
    items.slice(-8).reverse().forEach((t) => ul.appendChild(el("li", null, t)));
  }
  function eventSummary(ev) {
    const p = ev.payload || {};
    switch (ev.kind) {
      case "experiment.started": return `#${p.number} started — ${p.description || ""}`;
      case "experiment.finished": return `#${p.number} ${p.status}${p.value != null ? ` · ${fmt(p.value)}` : ""}${p.improvement != null ? ` (${signedPct(p.improvement)})` : ""} — ${p.reason || ""}`;
      case "incumbent.promoted": return `incumbent → #${p.number} (${p.commit}) ${p.speedup_vs_baseline ? `${p.speedup_vs_baseline.toFixed(2)}× vs baseline` : ""}`;
      case "agent": return `${p.name}: ${p.state} ${p.detail || ""}`;
      case "agent.tool": return `${p.agent} · ${p.tool}: ${p.detail || ""}`;
      case "agent.text": return `${p.agent}: ${p.text || ""}`;
      case "agent.result": return `${p.agent} finished (${p.num_turns} turns${p.cost_usd != null ? `, $${(+p.cost_usd).toFixed(3)}` : ""})`;
      case "note": return `note: ${p.text || ""}`;
      case "profile.updated": return `profile: ${p.kernel_count} kernels, GPU busy ${pct(p.gpu_busy_ratio, 0)}, top ${p.top || ""}`;
      case "control": return `${p.action} (${p.source})`;
      case "lease": return `${p.worker} ${p.state} ${p.target}`;
      case "inbox.submitted": return `${p.worker} proposed: ${p.description || ""}`;
      default: return Object.entries(p).slice(0, 4).map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v).slice(0, 60) : String(v).slice(0, 80)}`).join(" ");
    }
  }
  function renderEvents() {
    const wrap = $("events");
    const atBottom = wrap.scrollTop + wrap.clientHeight >= wrap.scrollHeight - 20;
    clear(wrap);
    state.events.slice(-250).forEach((ev) => {
      const p = ev.payload || {};
      const row = el("div", `ev ${p.status || ""}`);
      row.appendChild(el("time", null, new Date(ev.ts).toLocaleTimeString()));
      row.appendChild(el("b", null, ev.kind));
      row.appendChild(el("span", null, eventSummary(ev)));
      wrap.appendChild(row);
    });
    if (atBottom || STATIC) wrap.scrollTop = wrap.scrollHeight;
    $("events-sub").textContent = STATIC ? `${state.events.length} events (embedded)` : `live over SSE · ${state.events.length} events buffered`;
  }
  function renderFooter() {
    const d = (state.data.capabilities && state.data.capabilities.device) || {};
    $("device").textContent = d.name ? `${d.name} (sm_${String(d.compute_capability || "").replace(".", "")}, ${d.sm_count} SMs, ${d.measured_bandwidth_gbs || "?"} GB/s, ${d.measured_bf16_tflops || "?"} bf16 TFLOPS, launch ${d.launch_latency_us || "?"} µs)` : "not probed";
    const b = (state.data.capabilities && state.data.capabilities.backends) || {};
    $("backends").textContent = Object.keys(b).length ? Object.entries(b).map(([k, v]) => `${k}:${v.compiled ? "ready" : v.available ? "importable" : "missing"}`).join(" · ") : "not probed";
  }

  // ---- drawer
  let drawerTab = "overview";
  async function openDrawer(number) {
    state.selected = number;
    renderTable();
    let detail = state.details[number];
    if (!detail && !STATIC) {
      try { detail = await api(`/api/c/${encodeURIComponent(state.name)}/experiments/${number}`); state.details[number] = detail; }
      catch (err) { detail = { compact: { number, description: `failed to load: ${err}` } }; }
    }
    if (!detail) return;
    const c = detail.compact || detail;
    $("drawer-title").textContent = `#${c.number} ${STATUS_ICON[c.status] || ""} ${c.status} — ${c.description || ""}`;
    $("drawer").classList.add("open");
    renderDrawerBody(detail);
  }
  function renderDrawerBody(detail) {
    const body = clear($("drawer-body"));
    const c = detail.compact || detail;
    document.querySelectorAll("#drawer-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === drawerTab));
    const section = (title) => { const s = el("section"); s.appendChild(el("h4", null, title)); body.appendChild(s); return s; };
    const kv = (pairs) => { const g = el("div", "kv"); pairs.forEach(([k, v]) => { if (v === undefined || v === null || v === "") return; const d = el("div"); d.appendChild(el("div", "label", k)); d.appendChild(el("div", "val", String(v))); g.appendChild(d); }); return g; };
    if (drawerTab === "overview") {
      section("verdict").appendChild(kv([["status", c.status], ["reason", c.reason], [metricName(), c.primary_value != null ? fmt(c.primary_value) : null], ["Δ vs incumbent", c.improvement != null ? signedPct(c.improvement) : null],
        ["threshold", c.threshold != null ? pct(c.threshold, 2) : null], ["speedup vs baseline", c.speedup_vs_baseline ? `${c.speedup_vs_baseline.toFixed(3)}×` : null], ["kernels / call", c.kernel_count], ["GPU busy", c.gpu_busy_ratio != null ? pct(c.gpu_busy_ratio, 0) : null],
        ["wall ms", c.wall_ms != null ? fmt(c.wall_ms) : null], ["peak VRAM MB", c.peak_vram_mb != null ? fmt(c.peak_vram_mb) : null], ["duration", c.duration_s != null ? `${c.duration_s}s` : null], ["agent", c.agent], ["techniques", (c.techniques || []).join(", ")], ["target", c.target], ["parent", c.parent], ["commit", c.commit], ["patch lines", c.patch_lines]]));
      const metrics = detail.metrics || {};
      const rows = Object.entries(metrics).filter(([, v]) => v && typeof v === "object" && v.median_ms != null);
      if (rows.length) {
        const s = section("workloads");
        const t = el("table"); const thead = el("thead"); const trh = el("tr");
        ["workload", "median ms", "min ms", "p90 ms", "std", "peak VRAM MB", "derived"].forEach((h, i) => trh.appendChild(el("th", i ? "num" : "", h)));
        thead.appendChild(trh); t.appendChild(thead);
        const tb = el("tbody");
        rows.forEach(([name, v]) => { const tr = el("tr"); tr.appendChild(el("td", null, name)); [v.median_ms, v.min_ms, v.p90_ms, v.std_ms, v.peak_vram_mb].forEach((x) => tr.appendChild(el("td", "num", x != null ? fmt(x) : "–")));
          tr.appendChild(el("td", "num", ["rtf", "tokens_per_s", "fps", "audio_x_realtime"].filter((k) => v[k] != null).map((k) => `${k} ${fmt(v[k])}`).join(" · "))); tb.appendChild(tr); });
        t.appendChild(tb); s.appendChild(t);
      }
      if (detail.candidate_report && Object.keys(detail.candidate_report).length) section("candidate report").appendChild(el("pre", null, JSON.stringify(detail.candidate_report, null, 2)));
      if (detail.candidate_logs && detail.candidate_logs.length) section("candidate logs").appendChild(el("pre", null, detail.candidate_logs.join("\n")));
      if (detail.notes) section("notes").appendChild(el("pre", null, detail.notes));
    } else if (drawerTab === "gates") {
      const g = detail.gates || {};
      const s = section(`gates — ${g.summary || "not run"}`);
      Object.entries(g.stages || {}).forEach(([name, st]) => { const d = el("div", "check"); d.appendChild(el("span", st.skipped ? "" : st.passed ? "ok" : "fail", st.skipped ? "·" : st.passed ? "✓" : "✕")); d.appendChild(el("span", null, name)); d.appendChild(el("code", null, st.skipped ? "skipped" : `${st.checks} checks`)); s.appendChild(d); });
      const checks = g.all_checks || [];
      if (checks.length) {
        const s2 = section("checks");
        checks.forEach((ch) => { const d = el("div", "check"); d.appendChild(el("span", ch.passed ? "ok" : "fail", ch.passed ? "✓" : "✕")); const mid = el("span"); mid.appendChild(el("span", null, ch.name)); mid.appendChild(el("div", "muted", ch.detail || "")); d.appendChild(mid); d.appendChild(el("code", null, ch.value != null ? `${fmt(ch.value)}${ch.threshold != null ? ` / ${fmt(ch.threshold)}` : ""}` : "")); s2.appendChild(d); });
      }
    } else if (drawerTab === "profile") {
      const p = detail.profile || {};
      section("profile").appendChild(kv([["kernels", p.kernel_count], ["wall ms", p.wall_ms != null ? fmt(p.wall_ms) : null], ["GPU busy ms", p.gpu_busy_ms != null ? fmt(p.gpu_busy_ms) : null], ["busy ratio", p.gpu_busy_ratio != null ? pct(p.gpu_busy_ratio, 0) : null]]));
      if (p.targets && p.targets.length) {
        const s = section("targets"); const t = el("table"); const tb = el("tbody");
        const trh = el("tr"); ["#", "target", "bound", "share", "SOL", "headroom", "kernels"].forEach((h, i) => trh.appendChild(el("th", i >= 3 && i <= 5 ? "num" : "", h))); const th = el("thead"); th.appendChild(trh); t.appendChild(th);
        p.targets.forEach((x) => { const tr = el("tr"); tr.appendChild(el("td", null, String(x.rank))); tr.appendChild(el("td", null, x.title)); tr.appendChild(el("td", null, x.boundness)); tr.appendChild(el("td", "num", pct(x.fraction))); tr.appendChild(el("td", "num", x.sol_efficiency != null ? pct(x.sol_efficiency, 0) : "–")); tr.appendChild(el("td", "num", x.headroom != null ? pct(x.headroom) : "–")); tr.appendChild(el("td", "num", String(x.kernel_count))); tb.appendChild(tr); });
        t.appendChild(tb); s.appendChild(t);
      }
      if (p.kernels && p.kernels.length) {
        const s = section("top kernels"); const t = el("table"); const tb = el("tbody");
        p.kernels.forEach((k) => { const tr = el("tr"); tr.appendChild(el("td", "mono", k.name.slice(0, 80))); tr.appendChild(el("td", "num", String(k.count))); tr.appendChild(el("td", "num", `${fmt(k.gpu_us)} µs`)); tr.appendChild(el("td", null, k.category)); tb.appendChild(tr); });
        t.appendChild(tb); s.appendChild(t);
      }
    } else if (drawerTab === "patch") {
      section("patch (candidate/ vs incumbent)").appendChild(el("pre", null, detail.patch || "(no diff — baseline or unchanged)"));
    } else if (drawerTab === "log") {
      section("run.log tail").appendChild(el("pre", null, detail.log_tail || "(no log)"));
    }
  }
  document.querySelectorAll("#drawer-tabs button").forEach((b) => b.onclick = () => { drawerTab = b.dataset.tab; if (state.selected != null) openDrawer(state.selected); });
  $("drawer-close").onclick = () => $("drawer").classList.remove("open");
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("drawer").classList.remove("open"); });

  // ---- controls
  async function control(action, extra = {}) {
    if (STATIC) return;
    try { await api(`/api/c/${encodeURIComponent(state.name)}/control`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action, ...extra }) }); scheduleRefresh(); }
    catch (err) { alert(`control failed: ${err}`); }
  }
  $("btn-loop").onclick = () => control("start-loop");
  $("btn-pause").onclick = () => control("pause");
  $("btn-resume").onclick = () => control("resume");
  $("btn-stop").onclick = () => { if (confirm("Stop the loop and workers after the current experiment?")) control("stop"); };
  $("note-form").onsubmit = (e) => { e.preventDefault(); const text = $("note-text").value.trim(); if (!text) return; control("note", { text }); $("note-text").value = ""; };
  $("btn-log").onclick = () => { state.logScale = !state.logScale; $("btn-log").classList.toggle("active", state.logScale); renderExpChart(); };
  $("filter").oninput = (e) => { state.filter = e.target.value; renderTable(); };
  $("campaign-select").onchange = (e) => selectCampaign(e.target.value);
  $("btn-theme").onclick = () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : cur === "light" ? "" : "dark";
    if (next) document.documentElement.setAttribute("data-theme", next); else document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("fk-theme", next); } catch (e) { /* ignore */ }
    renderAll();
  };
  try { const t = localStorage.getItem("fk-theme"); if (t) document.documentElement.setAttribute("data-theme", t); } catch (e) { /* ignore */ }
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 120); });
  if (!STATIC) setInterval(() => { if (state.data) { renderAgents(); } }, 15000);

  loadCampaigns().catch((err) => { console.error(err); $("stream-state").textContent = `error: ${err.message}`; });
})();

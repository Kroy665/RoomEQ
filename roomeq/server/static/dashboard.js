// RoomEQ dashboard: live engine state, response charts, filter editor, presets, measure / auto-tune.
"use strict";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const F_MIN = 20;
const F_MAX = 20000;
const X_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000];

const ui = {
  state: null,
  curves: null,
  curvesKey: "",
  filtersDirty: false,
  draft: [],
  volumeTimer: 0,
  volumeEditing: false,
  presetsLoaded: "",
};

// ------------------------------------------------------------------ helpers

async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status}`);
  return data;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function fmtHz(f) {
  return f >= 1000 ? `${+(f / 1000).toFixed(f >= 10000 ? 0 : 1)}k` : `${Math.round(f)}`;
}

function fmtDb(v, digits = 1) {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
}

// ------------------------------------------------------------------ charts

function niceDomain(values, minSpan) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo)) return [-10, 10];
  lo = Math.floor((lo - 2) / 5) * 5;
  hi = Math.ceil((hi + 2) / 5) * 5;
  if (hi - lo < minSpan) {
    const mid = (hi + lo) / 2;
    lo = Math.floor((mid - minSpan / 2) / 5) * 5;
    hi = lo + minSpan;
  }
  return [lo, hi];
}

/**
 * series: [{key,label,color,values,dash,width}]
 * opts: {yMin, yMax, bands, zeroLine, directLabels, height}
 */
function drawChart(container, freqs, series, opts) {
  container.textContent = "";
  const W = container.clientWidth || 600;
  const H = container.clientHeight || 300;
  const direct = opts.directLabels && W >= 640;          // narrow screens rely on the legend
  const m = { l: 44, r: direct ? 92 : 16, t: 22, b: 26 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none" }, container);
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const lx0 = Math.log10(F_MIN);
  const lx1 = Math.log10(F_MAX);
  const x = (f) => m.l + ((Math.log10(f) - lx0) / (lx1 - lx0)) * pw;
  const y = (v) => m.t + (1 - (v - opts.yMin) / (opts.yMax - opts.yMin)) * ph;

  for (const b of opts.bands || []) {
    const r = el("rect", { x: x(b.from), y: m.t, width: x(b.to) - x(b.from), height: ph, class: "band" }, svg);
    const bw = x(b.to) - x(b.from);
    const label = [b.label, b.short].find((l) => l && l.length * 6.2 + 8 <= bw);
    if (label) {
      const t = el("text", { x: (x(b.from) + x(b.to)) / 2, y: m.t + 12, "text-anchor": "middle" }, svg);
      t.textContent = label;
    }
    r.setAttribute("fill-opacity", b.strength ?? 1);
  }
  const step = (opts.yMax - opts.yMin) > 40 ? 10 : 5;
  for (let v = opts.yMin; v <= opts.yMax + 1e-9; v += step) {
    el("line", { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: v === 0 && opts.zeroLine ? "zero" : "gridline" }, svg);
    const t = el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, svg);
    t.textContent = v > 0 ? `+${v}` : `${v}`;
  }
  for (const f of X_TICKS) {
    el("line", { x1: x(f), x2: x(f), y1: m.t, y2: m.t + ph, class: "gridline" }, svg);
    const t = el("text", { x: x(f), y: H - 8, "text-anchor": "middle" }, svg);
    t.textContent = fmtHz(f);
  }
  const unit = el("text", { x: m.l - 6, y: m.t - 10, "text-anchor": "end" }, svg);
  unit.textContent = "dB";

  const clip = el("clipPath", { id: `clip-${container.id}` }, el("defs", {}, svg));
  el("rect", { x: m.l, y: m.t, width: pw, height: ph }, clip);
  const plot = el("g", { "clip-path": `url(#clip-${container.id})` }, svg);

  const drawn = [];
  for (const s of series) {
    if (!s.values) continue;
    let d = "";
    let pen = false;
    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      const v = s.values[i];
      if (f < F_MIN || f > F_MAX || !Number.isFinite(v)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(f).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    }
    el("path", {
      d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(s.dash ? { "stroke-dasharray": s.dash } : {}),
    }, plot);
    drawn.push(s);
  }

  // direct labels at the right edge, nudged apart so they never overlap
  if (direct) {
    const at = 12000;
    const idx = nearestIndex(freqs, at);
    const labels = drawn
      .filter((s) => Number.isFinite(s.values[idx]))
      .map((s) => ({ s, yy: Math.min(Math.max(y(s.values[idx]), m.t + 6), m.t + ph - 2) }))
      .sort((a, b) => a.yy - b.yy);
    for (let i = 1; i < labels.length; i++) {
      if (labels[i].yy - labels[i - 1].yy < 14) labels[i].yy = labels[i - 1].yy + 14;
    }
    for (const l of labels) {
      el("line", { x1: m.l + pw + 3, x2: m.l + pw + 11, y1: l.yy - 4, y2: l.yy - 4, stroke: l.s.color, "stroke-width": 2,
        ...(l.s.dash ? { "stroke-dasharray": "3 2" } : {}) }, svg);
      const t = el("text", { x: m.l + pw + 14, y: l.yy, class: "direct" }, svg);
      t.textContent = l.s.short || l.s.label;
    }
  }

  // crosshair + one tooltip listing every series at the hovered frequency
  const cross = el("line", { y1: m.t, y2: m.t + ph, class: "crosshair", visibility: "hidden" }, svg);
  const dots = drawn.map((s) => el("circle", { r: 4, fill: s.color, stroke: css("--card"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
  const tip = $("tooltip");
  const hide = () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
    tip.hidden = true;
  };
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const f = 10 ** (lx0 + ((px - m.l) / pw) * (lx1 - lx0));
    const i = nearestIndex(freqs, f);
    const fx = x(freqs[i]);
    cross.setAttribute("x1", fx);
    cross.setAttribute("x2", fx);
    cross.setAttribute("visibility", "visible");
    tip.textContent = "";
    const head = document.createElement("div");
    head.className = "t-head";
    head.textContent = `${freqs[i] >= 1000 ? (freqs[i] / 1000).toFixed(2) + " kHz" : freqs[i].toFixed(0) + " Hz"}`;
    tip.appendChild(head);
    drawn.forEach((s, k) => {
      const v = s.values[i];
      if (!Number.isFinite(v)) {
        dots[k].setAttribute("visibility", "hidden");
        return;
      }
      dots[k].setAttribute("cx", fx);
      dots[k].setAttribute("cy", y(Math.min(Math.max(v, opts.yMin), opts.yMax)));
      dots[k].setAttribute("visibility", "visible");
      const row = document.createElement("div");
      row.className = "t-row";
      const name = document.createElement("span");
      const sw = document.createElement("span");
      sw.className = "sw";
      sw.style.background = s.color;
      name.appendChild(sw);
      name.appendChild(document.createTextNode(s.label));
      const val = document.createElement("b");
      val.textContent = `${fmtDb(v)} dB`;
      row.appendChild(name);
      row.appendChild(val);
      tip.appendChild(row);
    });
    tip.hidden = false;
    const tx = ev.clientX + 16;
    const tw = tip.offsetWidth;
    tip.style.left = `${tx + tw > window.innerWidth - 8 ? ev.clientX - tw - 16 : tx}px`;
    tip.style.top = `${ev.clientY + 12}px`;
  });
}

function nearestIndex(freqs, f) {
  let lo = 0;
  let hi = freqs.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (freqs[mid] < f) lo = mid;
    else hi = mid;
  }
  return Math.abs(Math.log(freqs[lo] / f)) < Math.abs(Math.log(freqs[hi] / f)) ? lo : hi;
}

function responseSeries(c) {
  const out = [];
  if (c.before) out.push({ key: "before", label: "Before (measured)", short: "Before", color: css("--series-before"), values: c.before });
  if (c.after) {
    out.push({ key: "after", label: "After (measured)", short: "After", color: css("--series-after"), values: c.after });
  } else if (c.predicted && c.before) {
    out.push({ key: "pred", label: "After (predicted)", short: "Predicted", color: css("--series-after"), values: c.predicted, dash: "6 4" });
  }
  if (c.target) out.push({ key: "target", label: "Target", short: "Target", color: css("--target"), values: c.target, dash: "2 4", width: 1.5 });
  return out;
}

function renderCharts() {
  const c = ui.curves;
  if (!c) return;
  const series = responseSeries(c);
  const legend = $("legend");
  legend.textContent = "";
  for (const s of series) {
    const k = document.createElement("span");
    k.className = "key";
    const sw = document.createElementNS(SVGNS, "svg");
    sw.setAttribute("width", "22");
    sw.setAttribute("height", "6");
    el("line", { x1: 1, x2: 21, y1: 3, y2: 3, stroke: s.color, "stroke-width": s.width || 2, ...(s.dash ? { "stroke-dasharray": s.dash } : {}) }, sw);
    k.appendChild(sw);
    k.appendChild(document.createTextNode(s.label));
    legend.appendChild(k);
  }
  const all = series.flatMap((s) => s.values.filter((_, i) => c.freqs[i] >= 25 && c.freqs[i] <= 16000));
  const [yMin, yMax] = niceDomain(all, 30);
  const bands = [
    { from: F_MIN, to: 500, label: "full correction", short: "full", strength: 1 },
    { from: 500, to: 4000, label: "gentle only", short: "gentle", strength: 0.45 },
    { from: 4000, to: F_MAX, label: "no correction (phone mic unreliable)", short: "none", strength: 0 },
  ];
  if (series.length) {
    drawChart($("chart-resp"), c.freqs, series, { yMin, yMax, bands, directLabels: true });
  } else {
    // empty state: the real axes and zones, with a clear next step in the middle
    const host = $("chart-resp");
    drawChart(host, c.freqs, [], { yMin: -20, yMax: 15, bands });
    const svg = host.querySelector("svg");
    const vb = svg.viewBox.baseVal;
    const t1 = el("text", { x: vb.width / 2, y: vb.height / 2 - 6, "text-anchor": "middle", class: "empty-title" }, svg);
    t1.textContent = "No measurement yet";
    const t2 = el("text", { x: vb.width / 2, y: vb.height / 2 + 16, "text-anchor": "middle", class: "empty-sub" }, svg);
    t2.textContent = "Run Measure or Auto-tune below, with your phone at the listening position.";
  }
  const eqVals = c.eq;
  const [eMin, eMax] = niceDomain(eqVals, 15);
  drawChart($("chart-eq"), c.freqs, [{ key: "eq", label: "EQ", color: css("--series-eq"), values: eqVals }],
    { yMin: Math.min(eMin, -5), yMax: Math.max(eMax, 5), zeroLine: true });

  const parts = [];
  if (c.rms_before != null) parts.push(`RMS error vs target: before ${c.rms_before.toFixed(1)} dB`);
  if (c.rms_after != null && c.after) parts.push(`after ${c.rms_after.toFixed(1)} dB (measured)`);
  else if (c.rms_predicted != null) parts.push(`after ${c.rms_predicted.toFixed(1)} dB (predicted)`);
  $("rms-line").textContent = parts.join(" → ");
  renderTable(c, series);
}

function renderTable(c, series) {
  const host = $("data-table");
  host.textContent = "";
  const table = document.createElement("table");
  const hr = table.insertRow();
  for (const h of ["Hz", ...series.map((s) => s.label), "EQ"]) {
    const th = document.createElement("th");
    th.textContent = h;
    hr.appendChild(th);
  }
  const marks = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000];
  for (const f of marks) {
    const i = nearestIndex(c.freqs, f);
    const r = table.insertRow();
    r.insertCell().textContent = fmtHz(f);
    for (const s of series) r.insertCell().textContent = Number.isFinite(s.values[i]) ? s.values[i].toFixed(1) : "";
    r.insertCell().textContent = c.eq[i].toFixed(1);
  }
  host.appendChild(table);
}

// ------------------------------------------------------------------ filters

function zoneOf(f) {
  return f <= 500 ? "full" : f <= 4000 ? "gentle" : "none";
}

function renderFilters(filters) {
  const body = $("filter-rows");
  body.textContent = "";
  filters.forEach((f, i) => {
    const tr = body.insertRow();
    tr.className = `zone-${zoneOf(f.freq)}`;
    tr.insertCell().textContent = String(i + 1);
    const typeCell = tr.insertCell();
    const sel = document.createElement("select");
    for (const [v, t] of [["peak", "Peak"], ["lowshelf", "Low shelf"], ["highshelf", "High shelf"]]) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = t;
      sel.appendChild(o);
    }
    sel.value = f.type;
    sel.addEventListener("change", () => edit(i, "type", sel.value));
    typeCell.appendChild(sel);
    for (const [key, stepv, digits] of [["freq", 1, 1], ["gain_db", 0.1, 1], ["q", 0.05, 2]]) {
      const td = tr.insertCell();
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = String(stepv);
      inp.value = Number(f[key]).toFixed(digits);
      inp.addEventListener("input", () => edit(i, key, parseFloat(inp.value)));
      td.appendChild(inp);
    }
    const x = tr.insertCell();
    const b = document.createElement("button");
    b.className = "x ghost";
    b.type = "button";
    b.textContent = "×";
    b.title = "Remove filter";
    b.addEventListener("click", () => {
      ui.draft.splice(i, 1);
      markDirty();
      renderFilters(ui.draft);
    });
    x.appendChild(b);
  });
}

function edit(i, key, value) {
  ui.draft[i] = { ...ui.draft[i], [key]: value };
  markDirty();
}

function markDirty() {
  ui.filtersDirty = true;
  $("apply-filters").hidden = false;
  $("revert-filters").hidden = false;
}

function clearDirty() {
  ui.filtersDirty = false;
  $("apply-filters").hidden = true;
  $("revert-filters").hidden = true;
  $("filter-error").hidden = true;
}

// ------------------------------------------------------------------ state rendering

function renderState(s) {
  const e = s.engine;
  $("preset-name").textContent = `preset: ${s.preset}`;
  $("demo-pill").hidden = !s.demo;
  const pill = $("eq-pill");
  pill.className = "pill";
  if (s.panicked) {
    pill.textContent = "PANIC: EQ bypassed, −20 dB";
    pill.classList.add("crit");
  } else if (s.bypassed) {
    pill.textContent = "EQ bypassed";
    pill.classList.add("warn");
  } else if (s.filters.length) {
    pill.textContent = "✓ EQ on";
    pill.classList.add("good");
  } else {
    pill.textContent = "flat (no EQ)";
  }
  $("panic").classList.toggle("active", s.panicked);
  $("panic-label").textContent = s.panicked ? "RECOVER" : "PANIC";
  $("eq-on").classList.toggle("on", !s.bypassed && !s.panicked);
  $("eq-off").classList.toggle("on", s.bypassed || s.panicked);
  if (!ui.volumeEditing) {
    $("volume").value = s.volume_db;
    $("volume-val").textContent = `${s.volume_db.toFixed(1)} dB`;
  }
  meter("m-in", "s-in", e.peak_in_dbfs);
  meter("m-out", "s-out", e.peak_out_dbfs);
  $("s-gr").textContent = e.limiter_gr_db > 0.05 ? `−${e.limiter_gr_db.toFixed(1)} dB` : "idle";
  $("s-lat").textContent = `${s.latency_ms.toFixed(0)} ms`;
  $("s-drift").textContent = `${e.drift_ppm >= 0 ? "+" : "−"}${Math.abs(e.drift_ppm).toFixed(0)} ppm`;
  const g = e.underruns + e.overruns + e.faults;
  $("s-glitch").textContent = String(g);
  $("preamp").textContent = s.filters.length ? `preamp ${fmtDb(s.preamp_db)} dB (automatic)` : "";

  if (!ui.filtersDirty) {
    ui.draft = s.filters.map((f) => ({ ...f }));
    const key = JSON.stringify(ui.draft);
    if (key !== ui.lastFiltersKey) {
      ui.lastFiltersKey = key;
      renderFilters(ui.draft);
    }
  }

  // phone + jobs
  const ph = $("phone-status");
  if (ph.dataset.key !== `${s.phone.connected}|${s.phone_url}`) {
    ph.dataset.key = `${s.phone.connected}|${s.phone_url}`;
    ph.textContent = "";
    if (s.phone.connected) {
      const p = document.createElement("span");
      p.className = "pill good";
      p.textContent = `✓ Phone connected · ${Math.round(s.phone.sample_rate)} Hz · ${s.phone.transport}`;
      ph.appendChild(p);
    } else {
      const img = document.createElement("img");
      img.src = `/api/qr.svg?t=${Date.now()}`;
      img.alt = "QR code for the phone measurement page";
      const box = document.createElement("div");
      const t = document.createElement("p");
      t.textContent = "Phone not connected. Scan with your iPhone, then tap “Start microphone”:";
      const code = document.createElement("code");
      code.textContent = s.phone_url;
      box.appendChild(t);
      box.appendChild(code);
      ph.appendChild(img);
      ph.appendChild(box);
    }
  }
  renderJob(s.job);

  const notes = $("notes");
  notes.textContent = "";
  const list = [...s.notes];
  if (!s.calibrated && !list.some((n) => n.includes("calibration"))) {
    list.unshift("No microphone calibration loaded: the phone mic is treated as flat, so accuracy is reduced above ~4 kHz and below ~40 Hz.");
  }
  for (const n of list) {
    const li = document.createElement("li");
    li.textContent = n;
    notes.appendChild(li);
  }
  $("limits").textContent = s.limits;

  const ck = `${s.curves_version}|${JSON.stringify(s.filters)}|${s.bypassed}|${s.panicked}`;
  if (ck !== ui.curvesKey) {
    ui.curvesKey = ck;
    api("/api/curves").then((c) => {
      ui.curves = c;
      renderCharts();
    });
  }
}

function meter(barId, textId, db) {
  const bar = $(barId);
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  bar.style.width = `${pct}%`;
  bar.classList.toggle("hot", db > -1.5);
  $(textId).textContent = db < -100 ? "silence" : `${db.toFixed(1)} dBFS`;
}

function renderJob(job) {
  const busy = job && (job.state === "running" || job.state === "waiting");
  $("start-measure").disabled = busy;
  $("start-autotune").disabled = busy;
  $("start-verify").disabled = busy;
  if (!job) {
    $("job").hidden = true;
    return;
  }
  $("job").hidden = false;
  const st = $("job-state");
  st.className = "pill";
  const names = { running: "Running", waiting: "Waiting for you", done: "✓ Done", failed: "Failed", cancelled: "Cancelled" };
  const kinds = { autotune: "Auto-tune", measure: "Measure", verify: "Verify" };
  st.textContent = `${kinds[job.kind] || job.kind}: ${names[job.state] || job.state}`;
  if (job.state === "done") st.classList.add("good");
  if (job.state === "failed") st.classList.add("crit");
  if (job.state === "waiting") st.classList.add("warn");
  $("job-cancel").hidden = !busy;
  $("job-prompt").hidden = !job.prompt;
  $("job-prompt-text").textContent = job.prompt || "";
  const log = $("job-log");
  const text = job.lines.join("\n");
  if (log.textContent !== text) {
    const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = text;
    if (atEnd) log.scrollTop = log.scrollHeight;
  }
  const res = $("job-result");
  res.textContent = "";
  if (job.result) {
    const p = document.createElement("p");
    p.textContent = job.result.summary;
    p.style.whiteSpace = "pre-line";
    res.appendChild(p);
  }
  if (job.error) {
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = job.error;
    res.appendChild(p);
  }
}

async function refreshPresets() {
  const p = await api("/api/presets");
  const key = JSON.stringify(p);
  if (key === ui.presetsLoaded) return;
  ui.presetsLoaded = key;
  const sel = $("preset-select");
  sel.textContent = "";
  for (const item of p.presets) {
    const o = document.createElement("option");
    o.value = item.id;
    o.textContent = item.name;
    sel.appendChild(o);
  }
  const cur = p.presets.find((item) => item.name === p.current);
  if (cur) sel.value = cur.id;
}

async function poll() {
  try {
    const s = await api("/api/state");
    ui.state = s;
    renderState(s);
  } catch (_) {
    $("eq-pill").textContent = "engine not reachable";
    $("eq-pill").className = "pill crit";
  }
  setTimeout(poll, 500);
}

// ------------------------------------------------------------------ wiring

// ------------------------------------------------------------------ theme (auto / light / dark)

const THEMES = ["auto", "light", "dark"];

function applyTheme(t) {
  if (t === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
  $("theme").textContent = t === "auto" ? "Theme: auto" : t === "light" ? "Theme: light" : "Theme: dark";
  if (ui.curves) renderCharts();                           // chart colours come from CSS tokens
}

function initTheme() {
  const q = new URLSearchParams(location.search).get("theme");
  let t = THEMES.includes(q) ? q : "auto";
  if (!q) {
    try { t = localStorage.getItem("roomeq-theme") || "auto"; } catch (_) { /* private mode */ }
  }
  applyTheme(THEMES.includes(t) ? t : "auto");
  $("theme").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || "auto";
    const next = THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length];
    try { localStorage.setItem("roomeq-theme", next); } catch (_) { /* ignore */ }
    applyTheme(next);
  });
}

function wire() {
  $("panic").addEventListener("click", () => api("/api/panic", {}));
  $("eq-on").addEventListener("click", () => api("/api/bypass", { on: false }));
  $("eq-off").addEventListener("click", () => api("/api/bypass", { on: true }));
  const vol = $("volume");
  vol.addEventListener("input", () => {
    ui.volumeEditing = true;
    $("volume-val").textContent = `${parseFloat(vol.value).toFixed(1)} dB`;
    clearTimeout(ui.volumeTimer);
    ui.volumeTimer = setTimeout(() => {
      api("/api/volume", { db: parseFloat(vol.value) }).finally(() => { ui.volumeEditing = false; });
    }, 60);
  });
  $("add-filter").addEventListener("click", () => {
    ui.draft.push({ type: "peak", freq: 100, gain_db: -3, q: 2 });
    markDirty();
    renderFilters(ui.draft);
  });
  $("revert-filters").addEventListener("click", () => {
    clearDirty();
    ui.lastFiltersKey = "";
  });
  $("apply-filters").addEventListener("click", async () => {
    try {
      await api("/api/eq", { filters: ui.draft });
      clearDirty();
      ui.lastFiltersKey = "";
    } catch (err) {
      $("filter-error").hidden = false;
      $("filter-error").textContent = `Not applied: ${err.message}`;
    }
  });
  $("preset-load").addEventListener("click", async () => {
    const sel = $("preset-select");
    const name = sel.value;
    if (!name) return;
    await api("/api/presets/load", { name });
    clearDirty();
    $("preset-msg").textContent = `Loaded “${sel.options[sel.selectedIndex].textContent}”.`;
  });
  $("preset-save").addEventListener("click", async () => {
    const name = $("preset-name-input").value.trim();
    if (!name) {
      $("preset-msg").textContent = "Type a name first.";
      return;
    }
    const r = await api("/api/presets/save", { name });
    $("preset-msg").textContent = `Saved to ${r.path}`;
    ui.presetsLoaded = "";
    refreshPresets();
  });
  const start = (kind) => api("/api/job/start", {
    kind,
    positions: parseInt($("opt-positions").value, 10),
    repeats: parseInt($("opt-repeats").value, 10),
    iterations: parseInt($("opt-iterations").value, 10),
  }).catch((err) => alert(err.message));
  $("start-measure").addEventListener("click", () => start("measure"));
  $("start-autotune").addEventListener("click", () => start("autotune"));
  $("start-verify").addEventListener("click", () => start("verify"));
  $("job-continue").addEventListener("click", () => api("/api/job/continue", {}));
  $("job-cancel").addEventListener("click", () => api("/api/job/cancel", {}));

  document.addEventListener("keydown", (ev) => {
    if (ev.target.closest("input, select, textarea")) return;
    if (ev.code === "Space") {
      ev.preventDefault();
      api("/api/panic", {});
    } else if (ev.key === "b" || ev.key === "B") {
      api("/api/bypass", { on: !(ui.state && ui.state.bypassed) });
    }
  });
  new ResizeObserver(() => renderCharts()).observe($("chart-resp"));
}

initTheme();
wire();
refreshPresets();
setInterval(refreshPresets, 5000);
poll();

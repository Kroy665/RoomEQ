// RoomEQ phone recorder: raw mic PCM -> Mac, plus measurement guidance from the Mac.
"use strict";

const $ = (id) => document.getElementById(id);
const HEADER_BYTES = 8; // uint32 first-sample index, uint32 stream id (little endian)

const state = {
  ctx: null,
  stream: null,
  settings: {},
  streamId: 0,
  sent: 0,
  ws: null,
  wsOpened: false,
  http: false,
  pending: [],
  lastMsgId: 0,
  wakeLock: null,
};

function newStreamId() {
  return (Math.random() * 0xffffffff) >>> 0;
}

function setConn(text, ok) {
  $("conn").textContent = text;
  $("conn").classList.toggle("ok", !!ok);
}

function showStatus(text) {
  $("status-card").hidden = !text;
  $("status").textContent = text || "";
}

function hello() {
  return {
    type: "hello",
    sampleRate: state.ctx.sampleRate,
    streamId: state.streamId,
    userAgent: navigator.userAgent,
    settings: state.settings,
  };
}

// ------------------------------------------------------------------ transport

function frame(samples, startIndex) {
  const buf = new ArrayBuffer(HEADER_BYTES + samples.length * 4);
  const dv = new DataView(buf);
  dv.setUint32(0, startIndex >>> 0, true);
  dv.setUint32(4, state.streamId, true);
  new Float32Array(buf, HEADER_BYTES).set(samples);
  return buf;
}

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/phone`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  state.wsOpened = false;
  ws.onopen = () => {
    state.wsOpened = true;
    // a fresh stream after every (re)connect keeps sample indices consistent on the Mac
    state.streamId = newStreamId();
    state.sent = 0;
    ws.send(JSON.stringify(hello()));
    setConn("connected", true);
    $("transport").textContent = "WebSocket";
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onclose = () => {
    if (!state.wsOpened) {
      startHttpFallback();
    } else {
      setConn("reconnecting…", false);
      setTimeout(connectWs, 1000);
    }
  };
}

async function post(path, body, type) {
  const r = await fetch(path, { method: "POST", body, headers: { "Content-Type": type } });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
}

function sendControl(msg) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(msg));
  } else if (state.http) {
    post("/api/phone/control", JSON.stringify(msg), "application/json").catch(() => {});
  }
}

function startHttpFallback() {
  if (state.http) return;
  state.http = true;
  state.ws = null;
  state.streamId = newStreamId();
  state.sent = 0;
  $("transport").textContent = "HTTP fallback";
  post("/api/phone/control", JSON.stringify(hello()), "application/json")
    .then(() => setConn("connected (HTTP)", true))
    .catch(() => setConn("cannot reach Mac", false));
  setInterval(flushHttp, 250);
  setInterval(pollMessages, 700);
}

function flushHttp() {
  if (!state.pending.length) return;
  const chunks = state.pending;
  state.pending = [];
  const total = chunks.reduce((n, c) => n + c.samples.length, 0);
  const all = new Float32Array(total);
  let o = 0;
  for (const c of chunks) {
    all.set(c.samples, o);
    o += c.samples.length;
  }
  post("/api/phone/audio", frame(all, chunks[0].start), "application/octet-stream").catch(() => {
    setConn("cannot reach Mac", false);
  });
}

async function pollMessages() {
  try {
    const r = await fetch(`/api/phone/messages?after=${state.lastMsgId}`);
    for (const m of await r.json()) handle(m);
  } catch (_) {
    /* keep polling */
  }
}

function onChunk(samples) {
  const start = state.sent;
  state.sent += samples.length;
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(frame(samples, start));
  } else if (state.http) {
    state.pending.push({ samples, start });
  }
  updateMeter(samples);
}

// ------------------------------------------------------------------ UI

let meterPeak = 0;
let meterTimer = 0;
function updateMeter(samples) {
  let p = 0;
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i]);
    if (a > p) p = a;
  }
  meterPeak = Math.max(p, meterPeak * 0.85);
  const now = performance.now();
  if (now - meterTimer < 60) return;
  meterTimer = now;
  const db = 20 * Math.log10(Math.max(meterPeak, 1e-6));
  $("level-text").textContent = `${db.toFixed(0)} dBFS`;
  const fill = $("meter-fill");
  fill.style.width = `${Math.max(0, Math.min(100, ((db + 70) / 70) * 100))}%`;
  fill.classList.toggle("hot", db > -3);
}

function handle(m) {
  if (m.id) state.lastMsgId = Math.max(state.lastMsgId, m.id);
  switch (m.type) {
    case "position":
      $("instruction").hidden = false;
      $("step").textContent = `Position ${m.index + 1} of ${m.total}`;
      $("instruction-title").textContent = m.title || "Move the phone";
      $("instruction-text").textContent = m.text || "";
      $("ready").hidden = false;
      $("ready").disabled = false;
      showStatus("");
      break;
    case "status":
      showStatus(m.text);
      if (m.busy) $("ready").hidden = true;
      break;
    case "result":
      $("instruction").hidden = true;
      $("result").hidden = false;
      $("result-text").textContent = m.text || "";
      $("result-warnings").innerHTML = "";
      for (const w of m.warnings || []) {
        const li = document.createElement("li");
        li.textContent = w;
        $("result-warnings").appendChild(li);
      }
      showStatus(m.done ? "Done. You can close this page." : "");
      break;
  }
}

function describeProcessing(s) {
  const on = ["echoCancellation", "noiseSuppression", "autoGainControl"].filter((k) => s[k] === true);
  if (!on.length) return "";
  return `This browser kept ${on.join(", ")} on. The measurement may be less accurate.`;
}

async function requestWakeLock() {
  try {
    if ("wakeLock" in navigator) state.wakeLock = await navigator.wakeLock.request("screen");
  } catch (_) {
    /* not fatal */
  }
}

async function start() {
  $("start").disabled = true;
  $("start-error").hidden = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
      video: false,
    });
    const AC = window.AudioContext || window.webkitAudioContext;
    state.ctx = new AC({ latencyHint: "playback" });
    await state.ctx.resume();
    const src = state.ctx.createMediaStreamSource(state.stream);
    const sink = state.ctx.createGain();
    sink.gain.value = 0; // keep the graph running without playing anything
    sink.connect(state.ctx.destination);

    if (state.ctx.audioWorklet) {
      await state.ctx.audioWorklet.addModule("/static/recorder-worklet.js");
      const node = new AudioWorkletNode(state.ctx, "roomeq-recorder", {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 1,
        channelCountMode: "explicit",
      });
      node.port.onmessage = (e) => onChunk(e.data);
      src.connect(node);
      node.connect(sink);
    } else {
      const sp = state.ctx.createScriptProcessor(4096, 1, 1);
      sp.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(sp);
      sp.connect(sink);
    }

    const track = state.stream.getAudioTracks()[0];
    state.settings = track.getSettings ? track.getSettings() : {};
    const warn = describeProcessing(state.settings);
    $("processing").hidden = !warn;
    $("processing").textContent = warn;
    $("rate").textContent = `${state.ctx.sampleRate} Hz`;
    $("start-card").hidden = true;
    $("live").hidden = false;
    showStatus("Connected. Follow the instructions on your Mac.");
    connectWs();
    requestWakeLock();
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") {
        requestWakeLock();
        state.ctx.resume();
      }
    });
  } catch (err) {
    $("start").disabled = false;
    $("start-error").hidden = false;
    $("start-error").textContent = `Could not start the microphone: ${err.message || err}`;
  }
}

async function init() {
  $("start").addEventListener("click", start);
  $("ready").addEventListener("click", () => {
    $("ready").disabled = true;
    sendControl({ type: "ready" });
    showStatus("Starting…");
  });
  if (!window.isSecureContext) {
    $("start").disabled = true;
    try {
      const info = await (await fetch("/api/info")).json();
      const secure = info.https && info.https.length ? info.https[0] : null;
      if (secure) {
        $("https-link").href = secure;
        // already trusted? then go straight to the secure page
        const ok = await fetch(`${secure}/api/ping`, { mode: "cors", cache: "no-store" })
          .then((r) => r.ok)
          .catch(() => false);
        if (ok) {
          location.replace(secure);
          return;
        }
      }
    } catch (_) {
      /* fall through to the setup instructions */
    }
    $("insecure").hidden = false;
  }
}

init();

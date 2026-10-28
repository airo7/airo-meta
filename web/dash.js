const $ = (id) => document.getElementById(id);
const ICONS = {
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4m0 3v.5"/>',
  hypothesis: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.7m0 3v.5"/>',
  prediction: '<circle cx="12" cy="12" r="6.5"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
  result: '<rect x="2" y="8" width="20" height="8" rx="1"/><path d="M6 8v3m4-3v4m4-4v3m4-3v4"/>',
  conclusion: '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>',
  fallback: '<path d="M4 6h16M4 12h10M4 18h13"/>',
};
const KIND = { alert: "Surprise", hypothesis: "Hypothesis", prediction: "Prediction", result: "Result", conclusion: "Conclusion", fallback: "Scripted fallback" };
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const decimals = (v) => (v == null ? 0 : (String(+v.toPrecision(3)).split(".")[1] || "").length);
const el = (tag, cls, html = "") => Object.assign(document.createElement(tag), { className: cls, innerHTML: html });

let state = null, loaded = false, msgAt = 0;
const trail = [];

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    for (const id of ["hero", "video"]) $(id).src = `/${id === "hero" ? "view" : "video"}?${Date.now()}`;
    setTimeout(() => $("hero").naturalWidth || document.body.classList.add("mapview"), 2500);
  };
  ws.onmessage = (e) => update(JSON.parse(e.data));
  ws.onclose = () => {
    $("status").className = "chip offline";
    $("status").textContent = "Reconnecting";
    setTimeout(connect, 1000);
  };
}

function update(s) {
  if (!s.floor) return;
  const resized = !state || state.floor.w !== s.floor.w || state.floor.h !== s.floor.h;
  state = s;
  msgAt = performance.now();
  if (resized) fit();
  const last = trail.at(-1);
  if (s.pose && (!last || Math.hypot(s.pose.x - last[0], s.pose.y - last[1]) > 0.005)) {
    trail.push([s.pose.x, s.pose.y]);
    if (trail.length > 80) trail.shift();
  }
  const status = s.status || "idle", surprised = /surpris/i.test(status);
  $("status").textContent = status[0].toUpperCase() + status.slice(1);
  $("status").className = `chip ${surprised ? "surprised" : /learn/i.test(status) ? "relearning" : ""}`;
  document.body.classList.toggle("surprised", surprised);
  $("mode").textContent = { sim: "Simulation", real: "Live hardware" }[s.mode] || "";
  $("mode").hidden = !s.mode;
  $("command").textContent = s.command ? s.command[0].toUpperCase() + s.command.slice(1) : "";
  $("clock").textContent = clock(s.t || 0);
  $("floorsize").textContent = `${s.floor.w.toFixed(1)} by ${s.floor.h.toFixed(1)} m`;
  renderBody(s.body);
  renderLog(s.notebook || []);
  renderMeter(s.surprise);
  renderRelearn(s.relearn, s.body);
  loaded = true;
}

const body = { n: null, names: "", prev: null, last: null, shown: [], target: [], dec: [] };
function renderBody(b) {
  if (!b) return;
  const names = b.params.map((p) => p.name).join("|");
  if (names !== body.names) {
    body.names = names;
    body.shown = b.params.map((p) => p.value);
    $("params").replaceChildren(...b.params.map((p) => el("div", "param",
      `<div class="row"><span class="name">${p.name}</span><span class="delta"></span><span class="v"></span><span class="u">${p.unit || ""}</span><span class="k"></span></div>
       <div class="bar"><div class="fill"></div><i class="tick prev"></i><i class="tick key"></i></div>`)));
  }
  if (b.n !== body.n) {
    if (body.n !== null) {
      body.prev = body.last;
      const card = $("body");
      card.classList.remove("reveal");
      void card.offsetWidth;
      card.classList.add("reveal");
      clearTimeout(body.timer);
      body.timer = setTimeout(() => card.classList.remove("reveal"), 2600);
    }
    body.n = b.n;
    $("bodyn").innerHTML = `<span>Body #${b.n}</span>`;
  }
  body.last = b.params.map((p) => p.value);
  $("prevkey").hidden = !body.prev;
  body.target = body.last;
  $("bodyname").textContent = b.name || "";
  $("relearned").innerHTML = b.relearned_s == null ? "" : `<b>${b.relearned_s.toFixed(1)} s</b>to relearn`;
  b.params.forEach((p, i) => {
    const row = $("params").children[i], prev = body.prev?.[i];
    const pct = (v) => `${Math.max(0, Math.min(100, ((v - p.lo) / (p.hi - p.lo)) * 100))}%`;
    body.dec[i] = Math.max(decimals(p.value), decimals(p.key));
    row.querySelector(".k").textContent = p.key == null ? "" : `key ${p.key.toFixed(body.dec[i])}`;
    row.querySelector(".fill").style.width = pct(p.value);
    row.querySelector(".tick.key").style.left = p.key == null ? "-9px" : pct(p.key);
    row.querySelector(".tick.prev").style.display = prev == null ? "none" : "";
    if (prev != null) row.querySelector(".tick.prev").style.left = pct(prev);
    const d = prev ? ((p.value - prev) / Math.abs(prev)) * 100 : 0;
    row.querySelector(".delta").textContent = Math.abs(d) < 1 ? "" : `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(0)}%`;
  });
}

const log = { keys: [], cur: null, exp: 0, typing: [] };
function renderLog(entries) {
  const key = (e) => `${e.t}|${e.kind}|${e.text}`;
  if (entries.length < log.keys.length || log.keys.some((k, i) => k !== key(entries[i]))) {
    $("cards").replaceChildren();
    Object.assign(log, { keys: [], cur: null, exp: 0, typing: [] });
  }
  for (const e of entries.slice(log.keys.length)) {
    addEntry(e, loaded && !matchMedia("(prefers-reduced-motion: reduce)").matches);
    log.keys.push(key(e));
  }
  $("expcount").textContent = log.exp ? `${log.exp} experiment${log.exp > 1 ? "s" : ""}` : "";
}

function addEntry(e, animate) {
  const c = log.cur;
  if (!c || e.kind === "hypothesis" || e.kind === "alert" || c.classList.contains("alert") || c.done) {
    const alert = e.kind === "alert";
    log.cur = el("article", `card${alert ? " alert" : ""}`,
      `<header><h3>${alert ? "Surprise" : `Experiment ${++log.exp}`}</h3><span class="state ${alert ? "" : "running"}">${alert ? clock(e.t) : "Running"}</span></header>`);
    $("cards").append(log.cur);
  }
  const step = el("div", `step ${e.kind}`,
    `<svg viewBox="0 0 24 24">${ICONS[e.kind] || ICONS.fallback}</svg><div><div class="kind">${KIND[e.kind] || e.kind}</div><p></p></div>`);
  log.cur.append(step);
  const p = step.querySelector("p");
  if (e.measured != null && e.predicted != null) p.after(gauge(e));
  if (animate) {
    step.classList.add("typing");
    log.typing.push({ step, p, text: e.text, t0: performance.now() });
  } else p.textContent = e.text;
  if (e.kind === "conclusion" && !log.cur.classList.contains("alert")) {
    log.cur.done = true;
    const st = log.cur.querySelector(".state");
    st.className = "state";
    st.textContent = `Concluded ${clock(e.t)}`;
  }
  $("cards").scrollTo({ top: $("cards").scrollHeight, behavior: animate ? "smooth" : "auto" });
}

function gauge({ predicted: m, sd = 0, measured: x, unit = "" }) {
  let lo = Math.min(m - 3 * sd, x), hi = Math.max(m + 3 * sd, x);
  const pad = (hi - lo) * 0.1 || 1;
  lo -= pad; hi += pad;
  const X = (v) => `${((v - lo) / (hi - lo)) * 100}%`;
  const g = el("div", "gauge",
    `<div class="track"></div><div class="band"></div><div class="mean"></div><div class="meas"></div>
     <span class="pl" style="left:0;transform:none">Predicted ${m}${unit} ± ${sd}</span><span class="ml" style="left:auto;right:0;transform:none">Measured ${x}${unit}</span>`);
  Object.assign(g.querySelector(".band").style, { left: X(m - 2 * sd), width: `calc(${X(m + 2 * sd)} - ${X(m - 2 * sd)})` });
  g.querySelector(".mean").style.left = X(m);
  g.querySelector(".meas").style.left = X(x);
  return g;
}

function renderMeter(s) {
  if (!s) return;
  const vals = s.values, thr = s.threshold, last = vals.at(-1) ?? 0;
  $("sval").textContent = last.toFixed(1);
  const [g, w, h] = sizeCanvas($("spark"));
  const top = Math.max(thr * 1.8, ...vals) * 1.08, n = Math.max(vals.length, 120);
  const X = (i) => w - (vals.length - 1 - i) * (w / (n - 1)), Y = (v) => h - 3 - (v / top) * (h - 10), ty = Y(thr);
  const area = (stroke, fill) => {
    g.beginPath();
    vals.forEach((v, i) => g[i ? "lineTo" : "moveTo"](X(i), Y(v)));
    g.strokeStyle = stroke; g.lineWidth = 1.8; g.lineJoin = "round"; g.stroke();
    g.lineTo(X(vals.length - 1), h); g.lineTo(X(0), h); g.closePath();
    g.fillStyle = fill; g.fill();
  };
  if (vals.length > 1) {
    const blue = g.createLinearGradient(0, ty, 0, h);
    blue.addColorStop(0, "#8fb4ff40"); blue.addColorStop(1, "#8fb4ff00");
    g.save(); g.beginPath(); g.rect(0, ty, w, h); g.clip(); area("#8fb4ff", blue); g.restore();
    g.save(); g.beginPath(); g.rect(0, 0, w, ty); g.clip(); g.shadowColor = "#ff5a5f"; g.shadowBlur = 14; area("#ff5a5f", "#ff5a5f40"); g.restore();
  }
  g.setLineDash([3, 5]); g.strokeStyle = "#ff5a5f99"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, ty); g.lineTo(w, ty); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#ff8a8e"; g.font = "500 11px Archivo, system-ui, sans-serif";
  g.fillText(`Relearn threshold ${thr.toFixed(1)}`, 2, ty - 6);
}

let relearnKey = "";
function renderRelearn(r, b) {
  $("relearn").classList.toggle("hidden", !r);
  if (!r) return (relearnKey = "");
  const at = { changed: 0, testing: 1, done: 2 }[r.phase] ?? 0;
  $("relearn").classList.toggle("changed", r.phase === "changed");
  [...$("steps").children].forEach((li, i) => {
    li.classList.toggle("on", i <= at);
    li.classList.toggle("now", i === at && r.phase !== "done");
  });
  const key = `${r.phase}|${r.step}|${b?.n}`;
  if (key !== relearnKey) {
    relearnKey = key;
    $("rdetail").innerHTML = {
      changed: "Reality left the cloud. Pausing to run experiments.",
      testing: `Probe ${r.step} of ${r.steps}: <b></b><span class="timer"></span>`,
      done: `<span class="big">Body #${b?.n ?? ""}</span>Relearned in <b>${(r.elapsed_s ?? 0).toFixed(1)} s</b>`,
    }[r.phase] || "";
    if (r.phase === "testing") $("rdetail").querySelector("b").textContent = r.probe || "";
  }
  const timer = $("rdetail").querySelector(".timer");
  if (timer) timer.textContent = `${(r.elapsed_s ?? 0).toFixed(1)} s`;
}

let phaseKey = "";
function drawTimeline(now) {
  const phases = state.episode?.phases || [];
  $("timeline").hidden = !phases.length;
  const key = phases.map((p) => p.t + p.kind).join();
  if (key !== phaseKey) {
    phaseKey = key;
    $("bands").replaceChildren(...phases.map((p) => Object.assign(el("div", `band ${p.kind}`), { textContent: p.label })));
    for (const d of $("bands").children) d.fit = d.scrollWidth;
  }
  const span = 45, x0 = now - span * 0.88, W = $("bands").clientWidth;
  [...$("bands").children].forEach((d, i) => {
    const a = Math.max(phases[i].t, x0), z = Math.min(phases[i + 1]?.t ?? now, now);
    d.style.display = z <= x0 ? "none" : "";
    d.style.left = `${((a - x0) / span) * W + 2}px`;
    const width = Math.max(((z - a) / span) * W - 4, 3);
    d.style.width = `${width}px`;
    d.classList.toggle("narrow", width < d.fit);
  });
}

function frame() {
  requestAnimationFrame(frame);
  if (!state) return;
  drawMap($("overlay"), state, trail);
  drawTimeline((state.t || 0) + Math.min((performance.now() - msgAt) / 1000, 0.3));
  [...$("params").children].forEach((row, i) => {
    body.shown[i] += ((body.target[i] ?? 0) - body.shown[i]) * 0.07;
    row.querySelector(".v").textContent = body.shown[i].toFixed(body.dec[i] ?? 2);
  });
  for (const t of log.typing) {
    const n = Math.floor(((performance.now() - t.t0) / 1000) * 80);
    t.p.textContent = t.text.slice(0, n);
    if (n >= t.text.length) t.step.classList.remove("typing");
  }
  if (log.typing.length) $("cards").scrollTop = $("cards").scrollHeight;
  log.typing = log.typing.filter((t) => t.step.classList.contains("typing"));
}

function fit() {
  if (!state) return;
  const box = $("mapbox"), big = document.body.classList.contains("mapview");
  const s = Math.min(box.clientWidth / state.floor.w, big ? box.clientHeight / state.floor.h : Infinity);
  Object.assign($("stage").style, { width: `${state.floor.w * s}px`, height: `${state.floor.h * s}px` });
}

const TOGGLES = { b: "body", n: "log", m: "map", s: "dock", t: "timeline" };
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return e.key === "Escape" && e.target.blur();
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (TOGGLES[k]) $(TOGGLES[k]).classList.toggle("hidden");
  else if (k === "h") document.body.classList.toggle("clean");
  else if (k === "v") document.body.classList.toggle("mapview");
  else if (k === "?") $("help").classList.toggle("hidden");
  else if (k === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (k === "/") { e.preventDefault(); $("cmdtext").focus(); }
});

$("cmdform").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("cmdtext").value.trim();
  if (!text) return;
  fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  $("cmdtext").value = "";
  $("cmdtext").blur();
});
new ResizeObserver(fit).observe($("mapbox"));
connect();
frame();

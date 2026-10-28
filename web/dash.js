const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const C = { belief: css("--belief"), reality: css("--reality"), alarm: css("--alarm"), text: css("--text"), dim: css("--dim") };
const POLY_COLORS = ["#b8a4ff", "#5fd4c4", "#ff8fb1", "#9ad66f", "#f2d06b"];
const ICONS = {
  alert: '<path d="M12 3 2 20h20z"/><path d="M12 10v4m0 3v.5"/>',
  hypothesis: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.7m0 3v.5"/>',
  prediction: '<circle cx="12" cy="12" r="6.5"/><path d="M12 2v5m0 10v5M2 12h5m10 0h5"/>',
  result: '<rect x="2" y="8" width="20" height="8" rx="1"/><path d="M6 8v3m4-3v4m4-4v3m4-3v4"/>',
  conclusion: '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 3 3 5-6"/>',
  fallback: '<path d="M4 6h16M4 12h10M4 18h13"/>',
};
const LABELS = { alert: "Surprise", hypothesis: "Hypothesis", prediction: "Prediction", result: "Result", conclusion: "Conclusion", fallback: "Fallback" };

let state = null;
const trail = [];

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => ($("video").src = `/video?${Date.now()}`);
  ws.onmessage = (e) => update(JSON.parse(e.data));
  ws.onclose = () => {
    $("status").className = "offline";
    $("status").textContent = "Reconnecting";
    setTimeout(connect, 1000);
  };
}

function update(s) {
  if (!s.floor) return;
  const resized = !state || state.floor.w !== s.floor.w || state.floor.h !== s.floor.h;
  state = s;
  if (resized) fit();
  if (s.pose) {
    trail.push([s.pose.x, s.pose.y]);
    if (trail.length > 60) trail.shift();
  }
  const status = s.status || "idle";
  $("status").textContent = status[0].toUpperCase() + status.slice(1);
  $("status").className = /surpris/i.test(status) ? "surprised" : /learn/i.test(status) ? "relearning" : "";
  $("stage").classList.toggle("alarm", /surpris/i.test(status));
  $("lastcmd").innerHTML = s.command ? `Last command <q></q>` : "";
  if (s.command) $("lastcmd").querySelector("q").textContent = s.command;
  renderBody(s.body);
  renderNotebook(s.notebook || []);
  renderSpark(s.surprise);
}

const decimals = (v) => (v == null ? 0 : (String(+v.toPrecision(3)).split(".")[1] || "").length);

let paramNames = "", bodyN = null;
function renderBody(b) {
  if (!b) return;
  if (b.n !== bodyN) {
    if (bodyN !== null) {
      $("bodyn").classList.add("fresh");
      setTimeout(() => $("bodyn").classList.remove("fresh"), 2500);
    }
    bodyN = b.n;
    $("bodyn").textContent = `Body #${b.n}`;
  }
  $("bodyname").textContent = b.name || "";
  $("relearned").innerHTML = b.relearned_s == null ? "" : `<b>${b.relearned_s.toFixed(1)} s</b>to relearn`;
  const names = b.params.map((p) => p.name).join("|");
  if (names !== paramNames) {
    paramNames = names;
    $("params").innerHTML = b.params
      .map((p) => `<div class="param"><span>${p.name}</span><span><span class="v"></span> <span class="k"></span></span>
        <div class="bar"><div class="fill"></div><div class="key"></div></div></div>`)
      .join("");
  }
  b.params.forEach((p, i) => {
    const el = $("params").children[i];
    const pct = (v) => `${Math.max(0, Math.min(100, ((v - p.lo) / (p.hi - p.lo)) * 100))}%`;
    const d = Math.max(decimals(p.value), decimals(p.key));
    el.querySelector(".v").textContent = `${p.value.toFixed(d)} ${p.unit || ""}`;
    el.querySelector(".k").textContent = p.key == null ? "" : `key ${p.key.toFixed(d)}`;
    el.querySelector(".fill").style.width = pct(p.value);
    el.querySelector(".key").style.left = p.key == null ? "-9px" : pct(p.key);
  });
}

let shown = [];
function renderNotebook(entries) {
  const key = (e) => `${e.t}|${e.kind}|${e.text}`;
  if (entries.length < shown.length || shown.some((k, i) => k !== key(entries[i]))) {
    $("entries").innerHTML = "";
    shown = [];
  }
  for (const e of entries.slice(shown.length)) {
    const div = document.createElement("div");
    div.className = `entry ${e.kind}`;
    const m = Math.floor(e.t / 60), sec = String(Math.floor(e.t % 60)).padStart(2, "0");
    div.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[e.kind] || ICONS.fallback}</svg>
      <p><b>${LABELS[e.kind] || e.kind}</b></p><time>${m}:${sec}</time>`;
    div.querySelector("p").append(e.text);
    $("entries").append(div);
    shown.push(key(e));
  }
  $("entries").scrollTop = $("entries").scrollHeight;
}

function sizeCanvas(cv, w, h) {
  const dpr = devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return g;
}

function renderSpark(s) {
  if (!s) return;
  const vals = s.values, thr = s.threshold, last = vals.at(-1) ?? 0;
  $("sval").textContent = last.toFixed(1);
  $("sval").classList.toggle("hot", last > thr);
  const cv = $("sparkline"), w = cv.clientWidth, h = cv.clientHeight;
  const g = sizeCanvas(cv, w, h);
  const top = Math.max(thr * 1.8, ...vals) * 1.08;
  const n = Math.max(vals.length, 120);
  const X = (i) => w - (vals.length - 1 - i) * (w / (n - 1)), Y = (v) => h - 4 - (v / top) * (h - 8);
  const ty = Y(thr);

  const line = () => {
    g.beginPath();
    vals.forEach((v, i) => (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))));
  };
  const area = (color, fill) => {
    line();
    g.strokeStyle = color; g.lineWidth = 1.6; g.lineJoin = "round"; g.stroke();
    g.lineTo(X(vals.length - 1), h); g.lineTo(X(0), h); g.closePath();
    g.fillStyle = fill; g.fill();
  };
  if (vals.length > 1) {
    g.save(); g.beginPath(); g.rect(0, ty, w, h); g.clip(); area(C.belief, "#8fb4ff18"); g.restore();
    g.save(); g.beginPath(); g.rect(0, 0, w, ty); g.clip(); area(C.alarm, "#ff5f5f30"); g.restore();
  }
  g.setLineDash([4, 5]); g.strokeStyle = "#ff5f5f99"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, ty); g.lineTo(w, ty); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#ff5f5fcc"; g.font = "12px Archivo, system-ui, sans-serif"; g.textAlign = "left";
  g.fillText(`relearn threshold ${thr.toFixed(1)}`, 4, ty - 6);
}

function fit() {
  if (!state) return;
  const box = $("floor"), { w, h } = state.floor;
  const aw = box.clientWidth - 52, ah = box.clientHeight - 150;
  const s = Math.min(aw / w, ah / h);
  Object.assign($("stage").style, { width: `${w * s}px`, height: `${h * s}px` });
}

function drawFloor() {
  requestAnimationFrame(drawFloor);
  if (!state) return;
  const cv = $("overlay"), cw = cv.clientWidth, ch = cv.clientHeight;
  const g = sizeCanvas(cv, cw, ch);
  const { w, h } = state.floor, k = cw / w;
  const P = ([x, y]) => [x * k, (h - y) * k];
  const poly = (pts, close) => {
    g.beginPath();
    pts.forEach((p, i) => g[i ? "lineTo" : "moveTo"](...P(p)));
    if (close) g.closePath();
  };

  g.strokeStyle = "#ffffff0d"; g.lineWidth = 1;
  for (let x = 0.5; x < w; x += 0.5) { g.beginPath(); g.moveTo(x * k, 0); g.lineTo(x * k, ch); g.stroke(); }
  for (let y = 0.5; y < h; y += 0.5) { g.beginPath(); g.moveTo(0, (h - y) * k); g.lineTo(cw, (h - y) * k); g.stroke(); }

  g.font = "500 12px Archivo, system-ui, sans-serif";
  (state.polygons || []).forEach((p, i) => {
    const col = POLY_COLORS[i % POLY_COLORS.length];
    poly(p.points, true);
    g.fillStyle = col + "1f"; g.fill();
    g.strokeStyle = col + "cc"; g.lineWidth = 1.5; g.stroke();
    const [lx, top] = p.points.map(P).reduce((a, b) => (b[1] < a[1] ? b : a));
    const ly = Math.max(top, 26);
    const tw = g.measureText(p.label).width;
    g.fillStyle = "#0d1422d9"; g.beginPath(); g.roundRect(lx - 2, ly - 24, tw + 14, 19, 4); g.fill();
    g.fillStyle = col; g.fillText(p.label, lx + 5, ly - 10);
  });

  g.save();
  g.globalCompositeOperation = "lighter";
  g.strokeStyle = "#8fb4ff30"; g.lineWidth = 2; g.lineCap = "round"; g.lineJoin = "round";
  (state.cloud || []).forEach((c) => { poly(c); g.stroke(); });
  g.restore();

  const path = state.path || [];
  if (path.length > 1) {
    poly(path);
    g.setLineDash([9, 8]); g.strokeStyle = "#dde4efcc"; g.lineWidth = 2; g.stroke(); g.setLineDash([]);
    const [gx, gy] = P(path.at(-1));
    g.beginPath(); g.arc(gx, gy, 6, 0, 7); g.strokeStyle = C.text; g.stroke();
    g.beginPath(); g.arc(gx, gy, 2, 0, 7); g.fillStyle = C.text; g.fill();
  }

  for (let i = 1; i < trail.length; i++) {
    g.beginPath(); g.moveTo(...P(trail[i - 1])); g.lineTo(...P(trail[i]));
    g.strokeStyle = `rgba(255,181,71,${(i / trail.length) * 0.7})`; g.lineWidth = 3; g.stroke();
  }

  const pose = state.pose;
  if (pose) {
    const c = Math.cos(pose.theta), s = Math.sin(pose.theta);
    const at = (a, b) => [pose.x + c * a - s * b, pose.y + s * a + c * b];
    const [cx, cy] = P([pose.x, pose.y]);
    g.save();
    g.shadowColor = C.reality; g.shadowBlur = 8;
    g.strokeStyle = C.reality; g.fillStyle = C.reality; g.lineWidth = 2;
    g.beginPath(); g.arc(cx, cy, 0.15 * k, 0, 7); g.stroke();
    poly([at(0, 0), at(0.18, 0)]); g.stroke();
    poly([at(0.25, 0), at(0.17, 0.045), at(0.17, -0.045)], true); g.fill();
    g.restore();
  }
}

$("cmdform").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("cmdtext").value.trim();
  if (!text) return;
  fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
  $("cmdtext").value = "";
});
addEventListener("resize", fit);
connect();
drawFloor();

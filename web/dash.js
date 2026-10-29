const $ = (id) => document.getElementById(id);
const el = (tag, cls, html = "") => Object.assign(document.createElement(tag), { className: cls, innerHTML: html });
const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const minus = (s) => String(s).replace(/(^|[\s(])-(?=\d)/g, "$1−");
const setText = (node, v) => node.textContent !== v && (node.textContent = v);
const calm = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const place = (key) => (key.length <= 3 ? key.toUpperCase() : cap(key));
const ACCENT = "#0a84ff";
const AXES = { "Gain left": "Gain L", "Gain right": "Gain R", "Dead zone": "Dead", "Turn efficiency": "Turn", Delay: "Delay", "Wheel lag": "Lag", Drift: "Drift" };
const ICON = {
  stack: '<svg viewBox="0 0 16 16"><path d="M8 2 2 5l6 3 6-3zM2 8l6 3 6-3M2 11l6 3 6-3"/></svg>',
  check: '<svg viewBox="0 0 16 16"><path d="m3.5 8.5 3 3 6-7"/></svg>',
  zone: '<svg viewBox="0 0 16 16"><path d="M8 2.5 1.8 13.5h12.4zM8 7v3m0 1.8v.2"/></svg>',
};
const LABELS = { "Gain left": "Left gain", "Gain right": "Right gain", "Dead zone": "Dead zone", "Turn efficiency": "Turning", Delay: "Delay", "Wheel lag": "Lag", Drift: "Drift" };

const app = { state: null, sel: null, loaded: false, want: null, notes: {}, phases: {}, zones: new Set(), heard: new Set(), recallAt: {}, radar: null };

// Today's single-robot Mission.state() still works: it becomes a fleet of one.
function normalise(s) {
  if (s.robots) return s;
  if (!s.floor) return null;
  const robot = { id: "airo", name: "AIRO", color: ACCENT, real: s.mode === "real", memory: [], ...s };
  return { t: s.t, mode: s.mode, site: { name: "", floor: s.floor, stations: { red: null, blue: null }, zones: [] },
    selected: "airo", robots: [robot], models: [], sim: s.sim, legacy: true };
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    $("conn").classList.add("on");
    $("hero").src = `/view?${Date.now()}`;
    $("video").src = `/video?${Date.now()}`;
  };
  ws.onmessage = (e) => update(JSON.parse(e.data));
  ws.onclose = () => {
    $("conn").classList.remove("on");
    setTimeout(connect, 1000);
  };
}

function update(raw) {
  const s = normalise(raw);
  if (!s) return;
  const resized = !app.state || app.state.site.floor.w !== s.site.floor.w || app.state.site.floor.h !== s.site.floor.h;
  if (app.want && (s.selected === app.want.id || performance.now() > app.want.until)) app.want = null;
  s.selected = app.want?.id || s.selected;
  app.state = s;
  app.sel = s.robots.find((r) => r.id === s.selected) || s.robots[0];
  if (resized) sizeMap();
  setText($("site"), s.site.name || "");
  setText($("mode"), { sim: "Simulation", real: "Live hardware" }[s.mode] || "");
  $("mode").className = s.mode || "";
  document.body.classList.toggle("real", s.mode === "real");
  setText($("clock"), clock(s.t || 0));
  setText($("mapname"), s.site.name ? `${s.site.name} floor` : "Floor");
  events(s);
  renderFleet(s);
  renderNow(app.sel);
  stations(s);
  renderMemory(app.sel);
  renderFeed(s);
  renderNotebook(app.sel);
  renderSim(s.sim || {});
  for (const b of document.querySelectorAll("[data-busy]")) b.classList.toggle("on", b.dataset.busy === s.site?.busy);
  app.loaded = true;
}

function select(id) {
  if (!app.state || app.state.legacy || id === app.sel?.id) return;
  app.want = { id, until: performance.now() + 2000 };
  post("/cmd", { text: `select ${id.replace("-", " ")}` });
  update({ ...app.state, selected: id });
}

/* Toasts for the moments a viewer should not miss: a recall, a confirmed relearn, a zone a robot found. */
function events(s) {
  for (const r of s.robots) {
    const seen = app.notes[r.id] ?? 0, notes = r.notebook || [];
    if (app.loaded && notes.length > seen) {
      for (const e of notes.slice(seen)) {
        if (e.kind !== "recall") continue;
        app.recallAt[r.id] = { at: performance.now(), e };
        toast(`Recalled Body #${e.n} · ${e.name}${e.recall_s != null ? ` · ${e.recall_s.toFixed(1)} s` : ""}`, r);
      }
    }
    app.notes[r.id] = notes.length;
    const phase = r.relearn?.phase || null;
    if (app.loaded && phase === "done" && app.phases[r.id] !== "done") {
      toast(`Learned Body #${r.body?.n} · ${r.body?.name || "new body"} · ${(r.relearn.elapsed_s ?? 0).toFixed(1)} s`, r, ACCENT);
    }
    app.phases[r.id] = phase;
  }
  for (const z of s.site.zones || []) {
    const key = `${z.label}|${z.t}`;
    if (app.zones.has(key)) continue;
    app.zones.add(key);
    const who = s.robots.find((r) => r.id === z.found_by);
    if (app.loaded) toast(`${z.label || "Zone"} found${who ? ` by ${who.name}` : ""} · the fleet routes around it`, who, "#ff9f0a", ICON.zone);
  }
  for (const m of s.models || []) {
    if (m.kind !== "transcribe" || app.heard.has(m.id)) continue;
    app.heard.add(m.id);
    if (app.loaded && m.ok) heard(m.output);
  }
}

function toast(text, robot, tone = "#30d158", icon = ICON.check) {
  $("hint").classList.remove("show");
  const t = el("div", "toast mat", `<span class="badge">${icon}</span><span>${esc(text)}</span>`);
  t.style.setProperty("--tone", tone);
  if (robot) t.append(el("small", "", `<i class="dot" style="--c:${robot.color}"></i>${esc(robot.name)}`));
  $("toasts").prepend(t);
  while ($("toasts").children.length > 3) $("toasts").lastChild.remove();
  setTimeout(() => {
    t.classList.add("out");
    setTimeout(() => t.remove(), 400);
  }, 4200);
}

function heard(text) {
  const ask = $("ask");
  $("heard").textContent = `“${text}”`;
  ask.classList.add("heard");
  clearTimeout(ask.timer);
  ask.timer = setTimeout(() => ask.classList.remove("heard"), 5000);
}

/* Fleet rail */
function confidence(r) {
  const s = r.surprise;
  if (!s?.values?.length) return 1;
  const last = s.values.slice(-3).sort((a, b) => a - b), v = last[Math.floor(last.length / 2)];
  return 1 - clamp((v - 0.5) / ((s.threshold || 3) - 0.5), 0, 1);
}
const alerted = (r) => /surpris/i.test(r.status || "") || ["changed", "testing"].includes(r.relearn?.phase);

function statusLine(r) {
  const st = r.status || "idle", m = /\bto\s+(?:the\s+)?(\w+)\s*$/i.exec(r.command || "");
  return m && /deliver|driv|rerout|go/i.test(st) ? `${cap(st)} to ${place(m[1])}` : cap(st);
}

function robotCard(r) {
  const a = el("article", "robot mat", `
    <div class="row">
      <span class="ring"><svg viewBox="0 0 34 34"><circle class="track" cx="17" cy="17" r="14"/><circle class="val" cx="17" cy="17" r="14"/></svg><i class="dot"></i></span>
      <span class="who"><b></b><span></span></span>
      <svg class="spark" viewBox="0 0 64 26" preserveAspectRatio="none"><path class="area"/><line class="thr" x1="0" x2="64"/><path class="line"/></svg>
    </div>
    <div class="sub"><b></b><span class="bname"></span><span class="mem">${ICON.stack}<span></span></span></div>
    <div class="more"><div><div class="sig"><svg class="radar" viewBox="-79 -70 158 140"></svg><ul class="deltas"></ul></div></div></div>`);
  a.dataset.id = r.id;
  a.addEventListener("click", () => select(r.id));
  return a;
}

function renderFleet(s) {
  const box = $("fleet"), ids = s.robots.map((r) => r.id);
  for (const c of [...box.children]) if (!ids.includes(c.dataset.id)) c.remove();
  s.robots.forEach((r, i) => {
    let c = box.querySelector(`[data-id="${r.id}"]`);
    if (!c) c = robotCard(r);
    if (box.children[i] !== c) box.insertBefore(c, box.children[i] || null);
    const sel = r.id === app.sel.id, conf = confidence(r);
    c.style.setProperty("--c", r.color || ACCENT);
    c.classList.toggle("selected", sel);
    c.classList.toggle("alert", alerted(r));
    c.title = `Confidence ${Math.round(conf * 100)}%`;
    c.querySelector(".val").style.strokeDashoffset = `${88 * (1 - conf)}`;
    setText(c.querySelector(".who b"), r.name || r.id);
    setText(c.querySelector(".who span"), statusLine(r));
    setText(c.querySelector(".sub b"), r.body ? `Body #${r.body.n}` : "No body yet");
    setText(c.querySelector(".bname"), r.body?.name || "");
    const known = r.memory?.length || 0;
    setText(c.querySelector(".mem span"), String(known));
    c.querySelector(".mem").title = `${known} known ${known === 1 ? "body" : "bodies"}`;
    spark(c.querySelector(".spark"), r.surprise);
    if (sel) radar(c, r);
  });
}

function spark(svg, s) {
  if (!s?.values?.length) return;
  const vals = s.values.slice(-48), thr = s.threshold || 3, top = Math.max(thr * 1.5, ...vals);
  const X = (i) => (i / 47) * 64 + (48 - vals.length) * (64 / 47), Y = (v) => 25 - (v / top) * 23;
  const d = vals.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
  svg.querySelector(".line").setAttribute("d", d);
  svg.querySelector(".area").setAttribute("d", `${d}L64 26L${X(0).toFixed(1)} 26Z`);
  const thrLine = svg.querySelector(".thr");
  thrLine.setAttribute("y1", Y(thr));
  thrLine.setAttribute("y2", Y(thr));
}

const bareOf = (r) => {
  const mem = [...(r.memory || [])].sort((a, b) => a.n - b.n)[0];
  return mem?.params || (r.body?.params?.every((p) => p.key != null) ? r.body.params.map((p) => ({ ...p, value: p.key })) : null);
};
const norm = (p, v = p.value) => clamp((v - p.lo) / (p.hi - p.lo || 1), 0.04, 1);

function radar(card, r) {
  const params = r.body?.params;
  if (!params?.length) return;
  const svg = card.querySelector(".radar"), n = params.length, R = 44, bare = bareOf(r);
  const at = (i, k) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
    return [Math.cos(a) * R * k, Math.sin(a) * R * k];
  };
  const key = `${r.id}|${params.map((p) => p.name).join()}|${!!bare}`;
  if (svg.dataset.key !== key) {
    svg.dataset.key = key;
    const ring = (k) => params.map((_, i) => at(i, k).join(",")).join(" ");
    svg.innerHTML = `<polygon class="grid" points="${ring(1)}"/><polygon class="grid" points="${ring(0.5)}"/>` +
      params.map((p, i) => {
        const [x, y] = at(i, 1), [lx, ly] = at(i, 1.24), anchor = Math.abs(lx) < 6 ? "middle" : lx > 0 ? "start" : "end";
        return `<line class="grid" x1="0" y1="0" x2="${x}" y2="${y}"/><text x="${lx}" y="${ly + 3.5}" text-anchor="${anchor}">${esc(AXES[p.name] || p.name.split(" ")[0])}</text>`;
      }).join("") + `<polygon class="bare"/><polygon class="cur"/>`;
    if (app.radar?.id !== r.id) app.radar = { id: r.id, shown: params.map((p) => norm(p)) };
  }
  app.radar.target = params.map((p) => norm(p));
  app.radar.svg = svg;
  app.radar.at = at;
  svg.querySelector(".bare").setAttribute("points", bare ? bare.map((p, i) => at(i, norm(p)).join(",")).join(" ") : "");
  deltas(card.querySelector(".deltas"), params, bare, r);
}

function fmtDelta(p, b) {
  const d = p.value - b.value;
  if (p.lo >= 0 && Math.abs(b.value) > 1e-9) return `${d >= 0 ? "+" : "−"}${Math.abs((d / Math.abs(b.value)) * 100).toFixed(0)}%`;
  return `${d >= 0 ? "+" : "−"}${+Math.abs(d).toPrecision(2)} ${p.unit || ""}`.trim();
}

function deltas(ul, params, bare, r) {
  let rows;
  if (!bare) rows = `<li><span>No reference body yet</span></li>`;
  else {
    const top = params.map((p, i) => ({ p, b: bare[i], k: Math.abs(norm(p) - norm(bare[i])) }))
      .filter((d) => d.k > 0.015).sort((a, b) => b.k - a.k).slice(0, 3);
    rows = top.length ? top.map(({ p, b }) => `<li><b>${fmtDelta(p, b)}</b><span>${esc(LABELS[p.name] || p.name)}</span></li>`).join("")
      : `<li><b>Bare</b><span>Matches the first body</span></li>`;
    rows += `<li class="key"><i></i>${r.memory?.length ? "Bare body" : "Hand-measured"}</li>`;
  }
  if (ul.dataset.v !== rows) {
    ul.dataset.v = rows;
    ul.innerHTML = rows;
  }
}

function stations(s) {
  const box = $("menu-send"), keys = Object.keys(s.site.stations || {}), key = keys.join();
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.innerHTML = keys.map((k) => `<button type="button" data-cmd="deliver to ${k}">${place(k)}</button>`).join("");
}

/* Now line */
function renderNow(r) {
  const now = $("now"), text = r.now || (r.command ? `${cap(r.command)}.` : statusLine(r));
  now.style.setProperty("--c", r.color || ACCENT);
  now.querySelector(".dot").style.setProperty("--c", r.color || ACCENT);
  setText($("nowname"), r.name || r.id);
  setText($("nowstatus"), statusLine(r));
  const p = $("nowtext");
  if (p.textContent !== text) {
    p.textContent = text;
    p.classList.remove("swap");
    void p.offsetWidth;
    if (app.loaded) p.classList.add("swap");
  }
  now.classList.toggle("alert", alerted(r));
  $("cmdtext").placeholder = `Tell ${r.name || "AIRO"} what to do`;

  const rl = r.relearn, rc = app.recallAt[r.id], recalled = rc && performance.now() - rc.at < 4500 && !rl;
  now.classList.toggle("working", !!rl || !!recalled);
  const [a, b, c] = $("steps").children;
  if (recalled) {
    setText(a, "Noticed");
    setText(b, "Recalled from memory");
    setText(c, `Body #${rc.e.n} · ${rc.e.recall_s?.toFixed(1) ?? "–"} s`);
    for (const li of [a, b, c]) li.className = "on done";
  } else if (rl) {
    const at = { changed: 0, testing: 1, done: 2 }[rl.phase] ?? 0;
    setText(b, rl.phase === "testing" ? `Testing ${rl.step || 1} of ${rl.steps || "?"}${rl.probe ? ` · ${rl.probe}` : ""}` : "Testing");
    setText(c, rl.phase === "done" ? `Body #${r.body?.n} confirmed` : "Confirmed");
    [a, b, c].forEach((li, i) => (li.className = `${i <= at ? "on" : ""} ${i === at && rl.phase !== "done" ? "now" : ""} ${rl.phase === "done" ? "done" : ""}`));
  }
  setText($("nowside"), rl ? `${(rl.elapsed_s ?? 0).toFixed(1)} s` : "");
  const notes = r.notebook || [], since = rl ? [...notes].reverse().find((e) => e.kind === "alert")?.t ?? -Infinity : Infinity;
  const entry = recalled ? rc.e : rl ? [...notes].reverse().find((e) => e.predicted != null && e.t >= since) : null;
  const g = $("nowgauge"), gk = entry ? `${r.id}|${entry.t}|${entry.measured}` : "";
  if (g.dataset.key !== gk) {
    g.dataset.key = gk;
    g.replaceChildren(...(entry ? [gauge(entry)] : []));
  }
}

/* Memory shelf */
function placeholder(m, color) {
  if (!m.params?.length) return "";
  const n = m.params.length, pts = m.params.map((p, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n, k = norm(p) * 22;
    return `${(Math.cos(a) * k).toFixed(1)},${(Math.sin(a) * k).toFixed(1)}`;
  }).join(" ");
  return `<svg viewBox="-28 -28 56 56"><circle r="22" stroke="rgba(255,255,255,.1)" stroke-width="1"/><polygon points="${pts}" fill="${color}40" stroke="${color}" stroke-width="1.4"/></svg>`;
}

function renderMemory(r) {
  const box = $("tiles"), mem = r.memory || [], names = Object.fromEntries(app.state.robots.map((x) => [x.id, x.name]));
  $("memory").hidden = !mem.length;
  setText($("memwho"), `${r.name} knows ${mem.length} ${mem.length === 1 ? "body" : "bodies"}`);
  const fresh = box.dataset.robot !== r.id;
  if (fresh) {
    box.replaceChildren();
    box.dataset.robot = r.id;
  }
  const ns = mem.map((m) => String(m.n));
  for (const t of [...box.children]) if (!ns.includes(t.dataset.n)) t.remove();
  mem.forEach((m, i) => {
    let t = box.querySelector(`[data-n="${m.n}"]`);
    if (!t) {
      t = el("div", "tile", `<div class="pic"></div><b></b><small></small>`);
      t.dataset.n = m.n;
      if (fresh || !app.loaded) t.style.animation = "none";
    }
    if (box.children[i] !== t) box.insertBefore(t, box.children[i] || null);
    t.style.setProperty("--c", r.color || ACCENT);
    t.classList.toggle("active", !!m.active);
    t.classList.toggle("shared", !!m.from);
    const pic = t.querySelector(".pic"), src = m.portrait || "";
    if (pic.dataset.src !== src || !pic.firstChild) {
      pic.dataset.src = src;
      pic.innerHTML = src ? `<img src="${esc(src)}" alt="">` : placeholder(m, r.color || ACCENT);
    }
    const label = `#${m.n} <span>${esc(m.name || "")}</span>`;
    if (t.querySelector("b").innerHTML !== label) t.querySelector("b").innerHTML = label;
    const note = m.from ? `From ${names[m.from] || m.from}${m.verified ? "" : ", untested"}`
      : m.recalls ? `Recalled ${m.recalls}×` : m.cause || "Learned here";
    setText(t.querySelector("small"), note);
    t.title = [m.cause, m.from && `Shared by ${names[m.from] || m.from}`, m.verified ? "Verified on this robot" : "Not yet verified here"].filter(Boolean).join("\n");
  });
  const rc = app.recallAt[r.id];
  if (rc && !rc.shown) {
    rc.shown = true;
    const t = box.querySelector(`[data-n="${rc.e.n}"]`);
    if (t) {
      t.classList.remove("recalled");
      void t.offsetWidth;
      t.classList.add("recalled");
      t.scrollIntoView({ inline: "nearest", block: "nearest", behavior: calm() ? "auto" : "smooth" });
    }
  }
}

/* Floor map: the corner tile, or full screen with V */
function sizeMap() {
  const f = app.state.site.floor, rail = $("rail").getBoundingClientRect().width || 312;
  document.documentElement.style.setProperty("--map-h", `${Math.round(Math.min(innerHeight < 960 ? 190 : 260, ((rail - 20) * f.h) / f.w + 42))}px`);
}

function mapRect() {
  const stage = $("stage"), f = app.state.site.floor, big = document.body.classList.contains("mapview");
  const W = stage.clientWidth, H = stage.clientHeight;
  let box = { x: 0, y: 0, w: W, h: H };
  if (big) {
    const side = $("side").getBoundingClientRect(), rail = $("rail").getBoundingClientRect(), now = $("now").getBoundingClientRect();
    box = { x: rail.right + 32, y: rail.top, w: side.left - rail.right - 64, h: now.top - rail.top - 32 };
  }
  const k = Math.min(box.w / f.w, box.h / f.h);
  return { x: box.x + (box.w - f.w * k) / 2, y: box.y + (box.h - f.h * k) / 2, k };
}

$("plan").addEventListener("click", (e) => {
  if (!app.state) return;
  const r = mapRect(), b = $("plan").getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, f = app.state.site.floor;
  const hit = app.state.robots.filter((o) => o.pose).map((o) => ({ o, d: Math.hypot(r.x + o.pose.x * r.k - x, r.y + (f.h - o.pose.y) * r.k - y) }))
    .sort((a, b) => a.d - b.d)[0];
  if (hit && hit.d < 28) select(hit.o.id);
  else if (!document.body.classList.contains("mapview")) toggleMap();
});

function toggleMap() {
  document.body.classList.toggle("mapview");
}
$("hero").addEventListener("click", () => document.body.classList.contains("mapview") && toggleMap());

function frame() {
  requestAnimationFrame(frame);
  if (!app.state) return;
  const rect = mapRect(), f = app.state.site.floor;
  Object.assign($("video").style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${f.w * rect.k}px`, height: `${f.h * rect.k}px` });
  drawPlan($("plan"), app.state, app.sel.id, rect);
  const rd = app.radar;
  if (rd?.svg?.isConnected && rd.target) {
    rd.shown = rd.target.map((v, i) => (rd.shown[i] ?? v) + (v - (rd.shown[i] ?? v)) * (calm() ? 1 : 0.09));
    rd.svg.querySelector(".cur").setAttribute("points", rd.shown.map((k, i) => rd.at(i, k).join(",")).join(" "));
  }
}

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return e.key === "Escape" && e.target.blur();
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "v") toggleMap();
  else if (k === "h") document.body.classList.toggle("clean");
  else if (k === "n") tab($("notebook").hidden ? "notebook" : "models");
  else if (k === "r" && app.state) {
    const ids = app.state.robots.map((r) => r.id);
    select(ids[(ids.indexOf(app.sel.id) + 1) % ids.length]);
  } else if (k === "?") $("help").hidden = !$("help").hidden;
  else if (k === "escape") $("help").hidden = true;
  else if (k === "f") document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  else if (k === "/") {
    e.preventDefault();
    $("ask").classList.remove("heard");
    $("cmdtext").focus();
  }
});

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("cmdtext").value.trim();
  if (!text) return;
  post("/cmd", { text });
  $("cmdtext").value = "";
  $("cmdtext").blur();
});
$("ask").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b?.dataset.menu) return menu(b.dataset.menu);
  if (b?.dataset.cmd) return menu(null), post("/cmd", { text: b.dataset.cmd });
  if (e.target.closest("#acts, .menu")) return;
  $("ask").classList.remove("heard");
  $("cmdtext").focus();
});
new ResizeObserver(() => app.state && sizeMap()).observe($("rail"));
// feed.js and view.js load after this file; state must not arrive before they have.
addEventListener("DOMContentLoaded", () => {
  connect();
  frame();
});

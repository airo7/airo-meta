const GLYPH = {
  segment: '<svg viewBox="0 0 20 20"><path d="M3 7V4.5A1.5 1.5 0 0 1 4.5 3H7M13 3h2.5A1.5 1.5 0 0 1 17 4.5V7M17 13v2.5a1.5 1.5 0 0 1-1.5 1.5H13M7 17H4.5A1.5 1.5 0 0 1 3 15.5V13"/><path d="M7.5 8.5c1-2 4.5-2.2 5.3-.2.7 1.8-.6 2.4-.2 3.8.3 1.4-1.6 2.4-3.4 1.6-1.7-.7-2.6-3.3-1.7-5.2z"/></svg>',
  describe: '<svg viewBox="0 0 20 20"><path d="M1.8 10S5 4.5 10 4.5 18.2 10 18.2 10 15 15.5 10 15.5 1.8 10 1.8 10z"/><circle cx="10" cy="10" r="2.6"/></svg>',
  wonder: '<svg viewBox="0 0 20 20"><path d="M8.5 2.5 10 7l4.5 1.5L10 10l-1.5 4.5L7 10 2.5 8.5 7 7zM15 12.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z"/></svg>',
  transcribe: '<svg viewBox="0 0 20 20"><path d="M2.5 9v2M5.5 6.5v7M8.5 3.5v13M11.5 6v8M14.5 8v4M17.5 9v2"/></svg>',
  image: '<svg viewBox="0 0 20 20"><rect x="2.5" y="3.5" width="15" height="13" rx="2.5"/><circle cx="7" cy="8" r="1.5"/><path d="m3 15 4.5-4.5 3 3 2-2L17 15.5"/></svg>',
};
const MODEL_NAME = { segment: "SAM", describe: "Muse Spark", wonder: "Muse Wonder", transcribe: "Muse Voice", image: "Muse Image" };
const NOTE_ICON = {
  alert: '<path d="M10 3 2 17h16zM10 8.5v3.5m0 2.2v.3"/>',
  hypothesis: '<circle cx="10" cy="10" r="7.5"/><path d="M8 8a2 2 0 1 1 2.8 1.9c-.5.2-.8.6-.8 1.1v.5m0 2.3v.2"/>',
  prediction: '<circle cx="10" cy="10" r="5.5"/><path d="M10 1.5v4m0 9v4M1.5 10h4m9 0h4"/>',
  result: '<rect x="2" y="7" width="16" height="6" rx="1"/><path d="M5.5 7v2.5m3-2.5v3.5m3-3.5v2.5m3-2.5v3.5"/>',
  conclusion: '<circle cx="10" cy="10" r="7.5"/><path d="m6.8 10.3 2.3 2.3 4.2-5"/>',
  recall: '<path d="M3.5 10a6.5 6.5 0 1 0 2-4.7M3.5 3.5v3.3h3.3"/><path d="M10 6.5V10l2.5 1.5"/>',
  fallback: '<path d="M4 6h12M4 10h8M4 14h10"/>',
};
const NOTE_KIND = { alert: "Surprise", hypothesis: "Hypothesis", prediction: "Prediction", result: "Result", conclusion: "Conclusion", recall: "Recall", fallback: "Scripted fallback" };
const feed = { filter: "", cards: new Map() };

function ago(t) {
  const now = t > 1e9 ? Date.now() / 1000 : app.state?.t ?? t, d = Math.max(0, now - t);
  return d < 5 ? "now" : d < 60 ? `${Math.floor(d)} s ago` : d < 3600 ? `${Math.floor(d / 60)} min ago` : `${Math.floor(d / 3600)} h ago`;
}
const latency = (ms) => (ms == null ? "" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

function words(v) {
  if (v == null || v === "") return "";
  if (typeof v === "string") {
    try {
      const j = JSON.parse(v);
      if (j && typeof j === "object") return words(j);
    } catch {}
    return v;
  }
  if (Array.isArray(v)) return v.map(words).join(", ");
  if (typeof v === "object") return Object.entries(v).map(([k, x]) => (typeof x === "number" ? `${k} ${x}` : words(x))).join(" ");
  return String(v);
}

function tools(list) {
  if (!list?.length) return "";
  return `<div class="tools">${list.map((t) => {
    const args = words(t.args), res = t.result == null ? "…" : words(t.result);
    return `<span class="pill" title="${esc(`${t.name} ${args} → ${res}`)}"><b>${esc(t.name)}</b>${args ? ` ${esc(args)}` : ""} <i>→</i> ${esc(minus(res))}</span>`;
  }).join("")}</div>`;
}

function cardHtml(m) {
  const r = app.state.robots.find((x) => x.id === m.robot), ver = /(\d+(?:\.\d+)+)$/.exec(m.model || "")?.[1] || "";
  const who = r ? `<i class="dot" style="--c:${r.color}"></i>${esc(r.name)}` : esc(m.robot || "Fleet");
  const img = m.image ? esc(m.image) : "";
  let body;
  if (!m.ok) body = `<p class="in">${esc(m.input)}</p><p class="failmsg">${esc(m.output || "No answer")}</p>`;
  else if (m.kind === "segment") body = `<p class="in">Segment <q>${esc(m.input)}</q></p>${img ? `<img class="shot" src="${img}" alt="">` : ""}<p class="out">${esc(m.output)}</p>`;
  else if (m.kind === "describe") body = `<p class="in">${esc(m.input)}</p>${img ? `<div class="split"><img src="${img}" alt=""><p class="quote">${esc(m.output)}</p></div>` : `<p class="quote">${esc(m.output)}</p>`}`;
  else if (m.kind === "transcribe") body = `<p class="quote">${esc(m.output)}</p>`;
  else if (m.kind === "image") body = `<p class="in">${esc(m.input)}</p>${img ? `<img class="shot square" src="${img}" alt="">` : ""}<p class="out">${esc(m.output)}</p>`;
  else body = `<p class="in">${esc(m.input)}</p><p class="out">${esc(minus(m.output))}</p><div class="tl">${tools(m.tools)}</div>`;
  return `<article><header><span class="glyph">${GLYPH[m.kind] || GLYPH.wonder}</span>
    <span class="title"><b>${esc(MODEL_NAME[m.kind] || m.model)}<small>${esc(ver)}</small></b><span>${who} · <time></time></span></span>
    <span class="lat">${m.ok ? "" : "Failed · "}${latency(m.ms)}<i></i></span></header>${body}</article>`;
}

function renderFeed(s) {
  const box = $("feed"), list = s.models || [], ids = new Set(list.map((m) => m.id));
  for (const [id, c] of feed.cards) if (!ids.has(id)) (c.remove(), feed.cards.delete(id));
  for (const m of list) {
    let c = feed.cards.get(m.id);
    if (!c) {
      c = el("div", `card${m.ok ? "" : " failed"}`, `<div>${cardHtml(m)}</div>`);
      c.dataset.kind = m.kind;
      feed.cards.set(m.id, c);
      box.prepend(c);
      if (app.loaded && !calm()) {
        c.classList.add("enter", "fresh");
        requestAnimationFrame(() => requestAnimationFrame(() => c.classList.remove("enter")));
        setTimeout(() => c.classList.remove("fresh"), 2400);
      }
      c.toolsKey = JSON.stringify(m.tools);
    } else if (m.tools && JSON.stringify(m.tools) !== c.toolsKey) {
      c.toolsKey = JSON.stringify(m.tools);
      const tl = c.querySelector(".tl");
      if (tl) tl.innerHTML = tools(m.tools);
    }
    c.classList.toggle("hide", !!feed.filter && m.kind !== feed.filter);
    setText(c.querySelector("time"), ago(m.t));
  }
  const shown = list.filter((m) => !feed.filter || m.kind === feed.filter).length;
  let empty = box.querySelector(".empty");
  if (!shown && !empty) box.append((empty = el("p", "empty", "")));
  if (empty) shown ? empty.remove() : setText(empty, list.length ? "No calls to this model yet." : "Calls to Meta's models appear here as the robots see, think and listen.");
}

$("filter").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  feed.filter = b.dataset.kind;
  for (const x of $("filter").children) x.classList.toggle("on", x === b);
  if (app.state) renderFeed(app.state);
});

function tab(name) {
  for (const b of $("tabs").children) b.classList.toggle("on", b.dataset.tab === name);
  $("feed").hidden = name !== "models";
  $("filter").hidden = name !== "models";
  $("notebook").hidden = name !== "notebook";
  if (name === "notebook") $("notebook").scrollTop = $("notebook").scrollHeight;
}
$("tabs").addEventListener("click", (e) => e.target.dataset.tab && tab(e.target.dataset.tab));

/* Lab notebook: hypothesis, prediction, result, conclusion, grouped into experiments. */
const nb = { robot: null, keys: [], cur: null, exp: 0 };

function renderNotebook(r) {
  const entries = r.notebook || [], key = (e) => `${e.t}|${e.kind}|${e.text}`;
  if (nb.robot !== r.id || entries.length < nb.keys.length || nb.keys.some((k, i) => k !== key(entries[i]))) {
    $("notebook").replaceChildren();
    Object.assign(nb, { robot: r.id, keys: [], cur: null, exp: 0 });
  }
  const animate = nb.keys.length > 0 && !calm();
  for (const e of entries.slice(nb.keys.length)) {
    addNote(e, animate);
    nb.keys.push(key(e));
  }
  if (!entries.length && !$("notebook").firstChild) $("notebook").append(el("p", "empty", "Experiments appear here when the robot tests its body."));
}

function addNote(e, animate) {
  $("notebook").querySelector(".empty")?.remove();
  const c = nb.cur, own = e.kind === "alert" || e.kind === "recall";
  if (!c || own || e.kind === "hypothesis" || c.single || c.done) {
    const title = e.kind === "alert" ? "Surprise" : e.kind === "recall" ? `Recalled Body #${e.n} · ${e.name}` : `Experiment ${++nb.exp}`;
    const state = e.kind === "recall" && e.recall_s != null ? `${e.recall_s.toFixed(1)} s` : clock(e.t);
    nb.cur = el("section", `exp ${own ? e.kind : ""}`, `<header><h3>${esc(title)}</h3><span class="state">${state}</span></header>`);
    nb.cur.single = own;
    $("notebook").append(nb.cur);
  }
  const label = nb.cur.single ? "" : `<div class="kind">${NOTE_KIND[e.kind] || cap(e.kind)}</div>`;
  const step = el("div", `step ${e.kind}`, `<svg viewBox="0 0 20 20">${NOTE_ICON[e.kind] || NOTE_ICON.fallback}</svg><div>${label}<p>${esc(e.text)}</p></div>`);
  if (!animate) step.style.animation = "none";
  if (e.predicted != null && e.measured != null) step.lastChild.append(gauge(e));
  nb.cur.append(step);
  if (e.kind === "conclusion") nb.cur.done = true;
  $("notebook").scrollTo({ top: $("notebook").scrollHeight, behavior: animate ? "smooth" : "auto" });
}

/* Predicted mean with its ±2σ band; the measured dot turns orange when it lands outside. */
function gauge({ predicted: m, sd = 0, measured: x, unit = "" }) {
  let lo = Math.min(m - 3 * sd, x ?? m), hi = Math.max(m + 3 * sd, x ?? m);
  const pad = (hi - lo) * 0.12 || 1;
  lo -= pad;
  hi += pad;
  const X = (v) => `${((v - lo) / (hi - lo)) * 100}%`, u = unit || "";
  const g = el("div", "gauge", `<div class="track"></div><div class="band"></div><div class="mean"></div>
    <span>Predicted <b>${minus(m)}${esc(u)}</b>${sd ? ` ± ${sd}` : ""}</span>${x != null ? `<span class="r">Measured <b>${minus(x)}${esc(u)}</b></span>` : ""}`);
  Object.assign(g.querySelector(".band").style, { left: X(m - 2 * sd), width: `calc(${X(m + 2 * sd)} - ${X(m - 2 * sd)})` });
  g.querySelector(".mean").style.left = X(m);
  if (x != null) {
    const dot = el("div", `meas${Math.abs(x - m) > 2 * sd ? " off" : ""}`);
    dot.style.left = X(x);
    g.append(dot);
  }
  return g;
}

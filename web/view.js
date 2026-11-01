const VIEWS = ["chase", "iso", "top", "floor", "deck"];
const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M7 5h3.2v14H7zM13.8 5H17v14h-3.2z"/></svg>';
const SPEEDS = { 0.25: "¼×", 0.5: "½×", 1: "1×", 2: "2×" };
const cam = { live: false, busy: false, orbit: [0, 0], pan: [0, 0], zoom: 1, queue: [], hinted: false, parts: "" };
const hero = $("hero");

function renderSim(sim, real) {
  cam.live = !!sim.camera;
  $("toolbar").hidden = !cam.live;
  document.body.classList.toggle("interactive", cam.live);
  if (!cam.live) return real && hint("Click the floor to send the car · scroll to zoom, drag to pan");
  for (const b of $("views").children) b.classList.toggle("on", b.dataset.cam === sim.preset);
  const parts = (sim.parts || []).join();
  if (parts !== cam.parts) {
    cam.parts = parts;
    $("menu-parts").innerHTML = (sim.parts || []).map((p) => `<button data-cam="${p}">${cap(p)}</button>`).join("");
  }
  for (const b of $("menu-parts").children) b.classList.toggle("on", b.dataset.cam === sim.preset);
  hint();
  const play = $("play");
  if (play.dataset.cmd !== (sim.paused ? "resume" : "pause") || !play.firstChild) {
    play.innerHTML = sim.paused ? ICON_PLAY : ICON_PAUSE;
    play.dataset.cmd = sim.paused ? "resume" : "pause";
    play.title = sim.paused ? "Resume (Space)" : "Pause (Space)";
  }
  setText($("speed"), SPEEDS[sim.speed] || `${sim.speed}×`);
  for (const b of $("menu-speed").children) b.classList.toggle("on", +b.dataset.cmd.split(" ")[1] === sim.speed);
  const changes = (sim.changes || []).map((c) => String(c).toLowerCase());
  for (const b of $("menu-changes").querySelectorAll("[data-on]")) b.classList.toggle("on", changes.some((c) => c.includes(b.dataset.on)));
}

const dismissHint = () => $("hint").classList.remove("show");

function hint(text) {
  if (cam.hinted) return;
  cam.hinted = true;
  say(text);
}

function say(text, ms = 6000) {
  const h = $("hint");
  if (text) h.textContent = text;
  h.classList.add("show");
  clearTimeout(h.timer);
  h.timer = setTimeout(dismissHint, ms);
}
const interactive = () => cam.live && !document.body.classList.contains("mapview");

function menu(name) {
  for (const b of document.querySelectorAll("[data-menu]")) {
    const open = b.dataset.menu === name && !b.classList.contains("open");
    b.classList.toggle("open", open);
    const m = $(`menu-${b.dataset.menu}`);
    m.classList.toggle("open", open);
    if (open) m.style.left = `${b.offsetLeft + b.offsetWidth / 2 - m.offsetWidth / 2}px`;
  }
}

function camera(action, then) {
  cam.queue.push([action, then]);
  flush();
}

function next() {
  if (cam.queue.length) return cam.queue.shift();
  const take = (key, rest) => {
    const a = { [key]: cam[key] };
    cam[key] = rest;
    return [a];
  };
  if (cam.orbit[0] || cam.orbit[1]) return take("orbit", [0, 0]);
  if (cam.pan[0] || cam.pan[1]) return take("pan", [0, 0]);
  if (cam.zoom !== 1) return take("zoom", 1);
}

// One request in flight: input piling up during a slow render is merged into the next request.
async function flush() {
  if (cam.busy) return;
  const job = next();
  if (!job) return;
  cam.busy = true;
  try {
    const r = await (await post("/camera", job[0])).json();
    job[1]?.(r);
  } catch {}
  cam.busy = false;
  flush();
}

// The server renders a fixed aspect and #hero crops it with object-fit: cover.
function frameBox() {
  return cover(hero.getBoundingClientRect(), hero.naturalWidth / hero.naturalHeight || 16 / 9);
}

// Live camera on hardware: the phone frame drawn at lens.s screen px per frame px, its corner at (lens.x, lens.y).
const lens = { mode: "fit", s: 0, x: 0, y: 0, to: null, zoom: 1, fit: 1, box: null };
const live = () => app.state?.mode === "real";

function freeArea() {
  const c = document.body.classList, clean = c.contains("clean"), edge = (id, side) => $(id).getBoundingClientRect()[side];
  const l = clean || c.contains("railfold") ? 0 : edge("rail", "right"), r = clean || c.contains("sidefold") ? innerWidth : edge("side", "left");
  const t = clean ? 0 : edge("top", "bottom"), b = clean ? innerHeight : edge("now", "top"), m = 20;
  return { x: l + m, y: t + m, w: r - l - 2 * m, h: b - t - 2 * m };
}

function fitLens(cam, size, a) {
  let pts = [[0, 0], size];
  if (cam) {
    const { w, h } = app.state.site.floor, m = 0.14, p = app.sel?.pose;
    pts = [[-m, -m], [w + m, -m], [w + m, h + m], [-m, h + m], ...(p ? [[p.x, p.y]] : [])].map((q) => homography(cam.to_image, q));
  }
  const us = pts.map((p) => p[0]), vs = pts.map((p) => p[1]);
  const u0 = Math.min(...us), u1 = Math.max(...us), v0 = Math.min(...vs), v1 = Math.max(...vs);
  const s = Math.min(a.w / (u1 - u0), a.h / (v1 - v0));
  return { s, x: a.x + a.w / 2 - (s * (u0 + u1)) / 2, y: a.y + a.h / 2 - (s * (v0 + v1)) / 2 };
}

// The AR overlay and click-to-go map through this same box.
function liveBox() {
  const cam = app.state.camera, size = cam?.size || [hero.naturalWidth || 16, hero.naturalHeight || 9], p = app.sel?.pose;
  const a = freeArea(), fit = fitLens(cam, size, a), cx = a.x + a.w / 2, cy = a.y + a.h / 2;
  lens.fit = fit.s;
  let goal = lens.to;
  if (lens.mode === "fit") goal = fit;
  else if (lens.mode === "follow" && cam && p) {
    const [u, v] = homography(cam.to_image, [p.x, p.y]);
    goal = { s: lens.zoom, x: cx - lens.zoom * u, y: cy - lens.zoom * v };
  }
  if (!lens.s) Object.assign(lens, fit);
  else if (goal) {
    // ease the frame point under the area's centre linearly and the scale geometrically, so zooms don't swoop
    const k = calm() ? 1 : 0.16, u = (cx - lens.x) / lens.s, v = (cy - lens.y) / lens.s;
    const s = lens.s * (goal.s / lens.s) ** k;
    lens.x = cx - s * (u + ((cx - goal.x) / goal.s - u) * k);
    lens.y = cy - s * (v + ((cy - goal.y) / goal.s - v) * k);
    lens.s = s;
    if (lens.to && Math.abs(goal.x - lens.x) + Math.abs(goal.y - lens.y) + Math.abs(goal.s / s - 1) * 100 < 0.5) lens.to = null;
  }
  for (const b of $("lens").children) b.classList.toggle("on", b.dataset.lens === lens.mode);
  $("lens").querySelector("[data-lens=follow]").disabled = !(cam && p);
  const box = (lens.box = { x: lens.x, y: lens.y, w: size[0] * lens.s, h: size[1] * lens.s });
  Object.assign(hero.style, { left: `${box.x}px`, top: `${box.y}px`, width: `${box.w}px`, height: `${box.h}px` });
  if (hero.complete && hero.naturalWidth) $("ambient").getContext("2d").drawImage(hero, 0, 0, 32, 18);
  return box;
}

const zoomRange = (s) => clamp(s, lens.fit * 0.5, lens.fit * 12);

function zoomAt(f, x, y, smooth) {
  if (lens.mode === "follow") return void (lens.zoom = zoomRange(lens.zoom * f));
  const from = (smooth && lens.to) || lens, s = zoomRange(from.s * f), k = s / from.s;
  const to = { s, x: x - (x - from.x) * k, y: y - (y - from.y) * k };
  lens.mode = "free";
  lens.to = smooth ? to : null;
  if (!smooth) Object.assign(lens, to);
}

function zoomStep(f) {
  const a = freeArea();
  zoomAt(f, a.x + a.w / 2, a.y + a.h / 2, true);
}

function follow(on = lens.mode !== "follow") {
  if (!on) return void (lens.mode = "free");
  if (!app.state?.camera || !app.sel?.pose) return;
  lens.mode = "follow";
  lens.to = null;
  lens.zoom = zoomRange(Math.max(lens.s, lens.fit * 2.5));
}

function onCar(e) {
  const cam = app.state?.camera, p = app.sel?.pose;
  if (!cam || !p || !lens.box) return false;
  const [x, y] = floorToScreen(cam, lens.box, [p.x, p.y]), [x1, y1] = floorToScreen(cam, lens.box, [p.x + 0.15, p.y]);
  return Math.hypot(e.clientX - x, e.clientY - y) < Math.max(24, Math.hypot(x1 - x, y1 - y));
}

$("lens").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  b.blur();
  dismissHint();
  ({ in: () => zoomStep(1.6), out: () => zoomStep(1 / 1.6), fit: () => ((lens.mode = "fit"), (lens.to = null)), follow: () => follow() })[b.dataset.lens]();
});

// On hardware a click on the phone frame sends the selected car to that floor point.
function goTo(e) {
  const s = app.state, cam = s.camera, { w, h } = s.site.floor, m = cam.inset;
  const [x, y] = screenToFloor(cam, lens.box, e.clientX, e.clientY);
  const at = `(${x.toFixed(2)}, ${y.toFixed(2)})`, inside = x >= m && x <= w - m && y >= m && y <= h - m;
  dismissHint();
  app.pin = { xy: [x, y], at: performance.now(), inside };
  if (!inside) toast(`${at} is outside the floor inset`, null, "#ff453a", ICON.zone);
  else if (s.dry) toast(`Dry run: not driving to ${at}`, app.sel, "#ff9f0a", ICON.zone);
  post("/cmd", { text: `go to ${x.toFixed(2)} ${y.toFixed(2)}` });
}

function pick(e, focus) {
  const f = frameBox(), x = e.clientX, y = e.clientY;
  camera({ pick: [(x - f.x) / f.w, (y - f.y) / f.h], focus }, (r) => showTag(x, y, r.part));
}

function showTag(x, y, part) {
  const tag = $("tag");
  clearTimeout(tag.timer);
  if (!part) return tag.classList.remove("show");
  tag.querySelector("b").textContent = part.name;
  tag.querySelector("span").textContent = part.detail || "";
  Object.assign(tag.style, { left: `${x}px`, top: `${y}px` });
  tag.classList.toggle("flip", x > innerWidth - 300);
  tag.classList.add("show");
  tag.timer = setTimeout(() => tag.classList.remove("show"), 4000);
}

// In real mode a zoomed-out frame leaves bare page around it, which pans and zooms too.
const onView = (e) => e.target === hero || (live() && (e.target === document.body || e.target === document.documentElement));
let drag = null, dragged = false;
addEventListener("pointerdown", (e) => {
  if (!onView(e) || !(interactive() || live())) return;
  dismissHint();
  menu(null);
  $("tag").classList.remove("show");
  drag = { x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, pan: e.button > 0 || e.shiftKey || live(), moved: false };
  hero.setPointerCapture(e.pointerId);
});
hero.addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (!drag.moved && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 4) return;
  drag.moved = true;
  document.body.classList.add("grabbing");
  const f = frameBox(), dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (live()) {
    lens.mode = "free";
    lens.to = null;
    lens.x += dx;
    lens.y += dy;
    return;
  }
  if (drag.pan) {
    cam.pan[0] += dx / f.w;
    cam.pan[1] += dy / f.h;
  } else {
    // MuJoCo's own viewer convention: a full-height drag turns 180 degrees, azimuth -= dx.
    cam.orbit[0] -= (dx / f.h) * 180;
    cam.orbit[1] -= (dy / f.h) * 180;
  }
  flush();
});
const endDrag = () => {
  dragged = !!drag?.moved;
  drag = null;
  document.body.classList.remove("grabbing");
};
hero.addEventListener("pointerup", endDrag);
hero.addEventListener("pointercancel", endDrag);
addEventListener("click", (e) => {
  if (!onView(e) || dragged || e.detail !== 1) return;
  if (interactive()) pick(e, false);
  else if (live() && app.state.camera && !onCar(e)) goTo(e);
});
addEventListener("dblclick", (e) => {
  if (!onView(e)) return;
  if (interactive()) pick(e, true);
  else if (live() && onCar(e)) follow(true);
});
hero.addEventListener("contextmenu", (e) => interactive() && e.preventDefault());
addEventListener("wheel", (e) => {
  if (!onView(e) || !(interactive() || live())) return;
  e.preventDefault();
  dismissHint();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  // Trackpad pinch arrives as ctrl+wheel with much smaller deltas.
  if (live()) return zoomAt(Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX, e.clientY);
  cam.zoom = Math.min(2, Math.max(0.5, cam.zoom * Math.exp(dy * (e.ctrlKey ? 0.01 : 0.0015))));
  flush();
}, { passive: false });

$("toolbar").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  b.blur();
  if (b.dataset.menu) return menu(b.dataset.menu);
  if (b.dataset.cam) camera({ preset: b.dataset.cam });
  else if (b.dataset.cmd) post("/cmd", { text: b.dataset.cmd });
  if (b.closest(".menu")) menu(null);
});
addEventListener("pointerdown", (e) => !e.target.closest("#toolbar, #acts, .menu") && menu(null));

addEventListener("keydown", (e) => {
  if (e.key === "Escape") menu(null);
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey) return;
  if (live()) {
    if (e.key === "0") (lens.mode = "fit"), (lens.to = null);
    else if (e.key === "+" || e.key === "=") zoomStep(1.6);
    else if (e.key === "-") zoomStep(1 / 1.6);
  }
  if (!cam.live) return;
  if (e.key >= "1" && e.key <= "5") camera({ preset: VIEWS[e.key - 1] });
  else if (e.key === " ") {
    e.preventDefault();
    post("/cmd", { text: $("play").dataset.cmd });
  }
});

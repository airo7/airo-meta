const VIEWS = ["chase", "iso", "top", "floor"];
const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>';
const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M7 5h3v14H7zM14 5h3v14h-3z"/></svg>';
const cam = { live: false, busy: false, orbit: [0, 0], pan: [0, 0], zoom: 1, queue: [], hinted: false };
const hero = $("hero");

function renderSim(sim) {
  cam.live = !!sim.camera;
  $("viewctl").hidden = !cam.live;
  document.body.classList.toggle("interactive", cam.live);
  if (!cam.live) return;
  if (!cam.hinted) {
    cam.hinted = true;
    $("hint").classList.add("show");
    setTimeout(dismissHint, 6000);
  }
  const play = $("play");
  if (play.classList.contains("paused") !== !!sim.paused || !play.firstChild) {
    play.classList.toggle("paused", !!sim.paused);
    play.innerHTML = sim.paused ? ICON_PLAY : ICON_PAUSE;
    play.dataset.cmd = sim.paused ? "resume" : "pause";
  }
  for (const b of $("speeds").children) b.classList.toggle("on", +b.dataset.cmd.split(" ")[1] === sim.speed);
  const changes = (sim.changes || []).map((c) => String(c).toLowerCase());
  for (const b of $("changes").children) b.classList.toggle("on", changes.some((c) => c.includes(b.dataset.cmd)));
}

const dismissHint = () => $("hint").classList.remove("show");
const interactive = () => cam.live && !document.body.classList.contains("mapview");

function camera(action, then) {
  cam.queue.push([action, then]);
  flush();
}

function next() {
  if (cam.queue.length) return cam.queue.shift();
  const take = (key, rest) => { const a = { [key]: cam[key] }; cam[key] = rest; return [a]; };
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
    if (r.camera) for (const b of $("cambar").querySelectorAll("[data-cam]")) b.classList.toggle("on", b.dataset.cam === r.camera.preset);
    job[1]?.(r);
  } catch {}
  cam.busy = false;
  flush();
}

// The server renders a fixed aspect and #hero crops it with object-fit: cover.
function frameBox() {
  const r = hero.getBoundingClientRect(), ar = hero.naturalWidth / hero.naturalHeight || 16 / 9;
  const w = Math.max(r.width, r.height * ar), h = w / ar;
  return { x: r.left + (r.width - w) / 2, y: r.top + (r.height - h) / 2, w, h };
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

let drag = null, dragged = false;
hero.addEventListener("pointerdown", (e) => {
  if (!interactive()) return;
  dismissHint();
  $("tag").classList.remove("show");
  drag = { x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, pan: e.button > 0 || e.shiftKey, moved: false };
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
hero.addEventListener("click", (e) => interactive() && !dragged && e.detail === 1 && pick(e, false));
hero.addEventListener("dblclick", (e) => interactive() && pick(e, true));
hero.addEventListener("contextmenu", (e) => interactive() && e.preventDefault());
hero.addEventListener("wheel", (e) => {
  if (!interactive()) return;
  e.preventDefault();
  dismissHint();
  const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  // Trackpad pinch arrives as ctrl+wheel with much smaller deltas.
  cam.zoom = Math.min(2, Math.max(0.5, cam.zoom * Math.exp(dy * (e.ctrlKey ? 0.01 : 0.0015))));
  flush();
}, { passive: false });

$("viewctl").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  b?.blur();
  if (b?.dataset.cam) camera({ preset: b.dataset.cam });
  else if (b?.dataset.cmd) post("/cmd", { text: b.dataset.cmd });
});

addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.metaKey || e.ctrlKey || e.altKey || !cam.live) return;
  if (e.key >= "1" && e.key <= "4") camera({ preset: VIEWS[e.key - 1] });
  else if (e.key === " ") {
    e.preventDefault();
    post("/cmd", { text: $("play").dataset.cmd });
  }
});

function sizeCanvas(cv) {
  const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return g;
}

const alpha = (hex, a) => `${hex}${Math.round(a * 255).toString(16).padStart(2, "0")}`;

// Plan view of the site in floor metres (x right, y up), fitted into rect {x, y, k px per metre}.
function drawPlan(cv, state, selected, rect) {
  const g = sizeCanvas(cv), { w, h } = state.site.floor, k = rect.k, big = k > 120;
  const P = ([x, y]) => [rect.x + x * k, rect.y + (h - y) * k];
  const trace = (pts, close) => {
    g.beginPath();
    pts.forEach((p, i) => g[i ? "lineTo" : "moveTo"](...P(p)));
    if (close) g.closePath();
  };
  const font = (px, weight = 500) => (g.font = `${weight} ${px}px -apple-system, BlinkMacSystemFont, system-ui, sans-serif`);
  const fs = big ? 13 : 10.5;

  g.beginPath();
  g.roundRect(rect.x, rect.y, w * k, h * k, big ? 14 : 8);
  g.fillStyle = state.mode === "real" ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.045)";
  g.fill();
  g.strokeStyle = "rgba(255,255,255,0.12)";
  g.lineWidth = 1;
  g.stroke();
  g.save();
  g.clip();
  g.strokeStyle = "rgba(255,255,255,0.045)";
  for (let x = 1; x < w; x++) (g.beginPath(), g.moveTo(...P([x, 0])), g.lineTo(...P([x, h])), g.stroke());
  for (let y = 1; y < h; y++) (g.beginPath(), g.moveTo(...P([0, y])), g.lineTo(...P([w, y])), g.stroke());
  g.restore();

  const sel = state.robots.find((r) => r.id === selected);
  for (const p of sel?.polygons || []) {
    trace(p.points, true);
    g.fillStyle = "rgba(255,255,255,0.08)";
    g.fill();
  }

  for (const z of state.site.zones || []) {
    const [x0, y0, x1, y1] = z.polygon.map(P).reduce(([a, b, c, d], [x, y]) => [Math.min(a, x), Math.min(b, y), Math.max(c, x), Math.max(d, y)], [1e9, 1e9, -1e9, -1e9]);
    g.save();
    trace(z.polygon, true);
    g.fillStyle = "rgba(255,159,10,0.12)";
    g.fill();
    g.clip();
    g.strokeStyle = "rgba(255,159,10,0.6)";
    g.lineWidth = 1;
    for (let i = x0 - (y1 - y0); i < x1; i += big ? 7 : 4.5) (g.beginPath(), g.moveTo(i, y1), g.lineTo(i + (y1 - y0), y0), g.stroke());
    g.restore();
    trace(z.polygon, true);
    g.strokeStyle = "rgba(255,159,10,0.85)";
    g.lineWidth = 1.2;
    g.stroke();
    font(fs, 600);
    g.fillStyle = "#ff9f0a";
    g.textAlign = "center";
    g.fillText(z.label || z.kind, (x0 + x1) / 2, y1 + fs + 3);
  }

  font(fs);
  for (const [name, at] of Object.entries(state.site.stations || {})) {
    if (!at) continue;
    const [x, y] = P(at), s = big ? 14 : 9, left = at[0] > w / 2;
    g.beginPath();
    g.roundRect(x - s / 2, y - s / 2, s, s, 2.5);
    g.strokeStyle = "rgba(235,235,245,0.6)";
    g.lineWidth = 1.2;
    g.stroke();
    g.fillStyle = "rgba(235,235,245,0.62)";
    g.textAlign = left ? "right" : "left";
    g.fillText(name.length <= 3 ? name.toUpperCase() : name[0].toUpperCase() + name.slice(1), x + (left ? -s : s), y + (at[1] > h / 2 ? fs + s * 0.4 : -s * 0.6));
  }
  g.textAlign = "left";

  for (const m of state.site.movers || []) {
    if (m.pose[0] < 0 || m.pose[1] < 0 || m.pose[0] > w || m.pose[1] > h) continue;
    const [cx, cy] = P(m.pose), s = big ? 7 : 4.5;
    g.save();
    g.translate(cx, cy);
    g.rotate(-m.pose[2]);
    g.beginPath();
    if (m.kind === "forklift") g.roundRect(-s * 1.6, -s, s * 3.2, s * 2, 2);
    else g.arc(0, 0, s, 0, 7);
    g.fillStyle = "rgba(235,235,245,0.85)";
    g.strokeStyle = "rgba(0,0,0,0.45)";
    g.lineWidth = 1.2;
    g.fill();
    g.stroke();
    g.restore();
    if (big) {
      font(11, 500);
      g.fillStyle = "rgba(235,235,245,0.6)";
      g.fillText(m.name || (m.kind === "human" ? "Worker" : "Forklift"), cx + s * 1.8, cy + 4);
    }
  }

  const robots = [...state.robots].sort((a, b) => (a.id === selected) - (b.id === selected));
  for (const r of robots) {
    const col = r.color || "#0a84ff", on = r.id === selected;
    g.lineCap = g.lineJoin = "round";
    g.strokeStyle = alpha(col, on ? 0.22 : 0.12);
    g.lineWidth = 1;
    for (const c of r.cloud || []) (trace(c), g.stroke());
    if (r.path?.length > 1) {
      trace(r.path);
      g.setLineDash(big ? [6, 6] : [3, 4]);
      g.strokeStyle = alpha(col, on ? 0.85 : 0.4);
      g.lineWidth = on ? 1.6 : 1.1;
      g.stroke();
      g.setLineDash([]);
    }
    if (!r.pose) continue;
    const [cx, cy] = P([r.pose.x, r.pose.y]), s = big ? 13 : on ? 8 : 6.5;
    car(g, cx, cy, -r.pose.theta, s, col, on);
    if (big) {
      font(13, 600);
      g.fillStyle = "rgba(255,255,255,0.92)";
      g.fillText(r.name || r.id, cx + s * 1.6, cy - s * 0.9);
    }
  }
}

function car(g, cx, cy, angle, s, col, halo) {
  if (halo) {
    g.beginPath();
    g.arc(cx, cy, s * 1.9, 0, 7);
    g.fillStyle = alpha(col, 0.2);
    g.fill();
  }
  g.save();
  g.translate(cx, cy);
  g.rotate(angle);
  g.beginPath();
  g.moveTo(s * 1.15, 0);
  g.lineTo(-s * 0.8, s * 0.78);
  g.lineTo(-s * 0.4, 0);
  g.lineTo(-s * 0.8, -s * 0.78);
  g.closePath();
  g.fillStyle = col;
  g.strokeStyle = "rgba(0,0,0,0.45)";
  g.lineWidth = 1.2;
  g.fill();
  g.stroke();
  g.restore();
}

// The sim hero fills its rect with object-fit: cover; the image's box on screen.
function cover(r, aspect) {
  const w = Math.max(r.width, r.height * aspect), h = w / aspect;
  return { x: r.left + (r.width - w) / 2, y: r.top + (r.height - h) / 2, w, h };
}

function homography(M, [x, y]) {
  const d = M[2][0] * x + M[2][1] * y + M[2][2];
  return [(M[0][0] * x + M[0][1] * y + M[0][2]) / d, (M[1][0] * x + M[1][1] * y + M[1][2]) / d];
}

// cam is the state's camera: to_image and to_floor map floor metres and phone frame pixels (size) both ways.
function floorToScreen(cam, box, p) {
  const [u, v] = homography(cam.to_image, p);
  return [box.x + (u * box.w) / cam.size[0], box.y + (v * box.h) / cam.size[1]];
}

function screenToFloor(cam, box, x, y) {
  return homography(cam.to_floor, [((x - box.x) * cam.size[0]) / box.w, ((y - box.y) * cam.size[1]) / box.h]);
}

// Over the live phone frame: the floor inset, and the selected car's ghost cloud, plan, goal and heading.
// box is the phone frame's rect on screen, which the canvas covers 1:1.
function drawAR(cv, state, sel, pin, box) {
  const g = sizeCanvas(cv), cam = state.camera;
  if (!cam) return;
  const P = (p) => floorToScreen(cam, box, p), col = sel.color || "#0a84ff", { w, h } = state.site.floor, m = cam.inset;
  const trace = (pts, close) => {
    g.beginPath();
    pts.forEach((p, i) => g[i ? "lineTo" : "moveTo"](...P(p)));
    if (close) g.closePath();
  };
  const ring = (xy, c, r, a) => {
    const [x, y] = P(xy);
    g.beginPath();
    g.arc(x, y, r, 0, 7);
    g.strokeStyle = alpha(c, a);
    g.lineWidth = 2;
    g.stroke();
    g.beginPath();
    g.arc(x, y, 3, 0, 7);
    g.fillStyle = alpha(c, a);
    g.fill();
  };
  g.lineCap = g.lineJoin = "round";
  g.shadowColor = "rgba(0,0,0,0.4)";
  g.shadowBlur = 4;
  trace([[m, m], [w - m, m], [w - m, h - m], [m, h - m]], true);
  g.strokeStyle = "rgba(255,255,255,0.5)";
  g.lineWidth = 1.2;
  g.stroke();
  for (const p of sel.polygons || []) {
    trace(p.points, true);
    g.fillStyle = "rgba(255,159,10,0.18)";
    g.fill();
    g.strokeStyle = "rgba(255,159,10,0.9)";
    g.lineWidth = 2;
    g.stroke();
  }
  g.strokeStyle = alpha(col, 0.3);
  g.lineWidth = 1;
  for (const c of sel.cloud || []) (trace(c), g.stroke());
  if (sel.path?.length > 1) {
    trace(sel.path);
    g.setLineDash([6, 6]);
    g.strokeStyle = alpha(col, 0.9);
    g.lineWidth = 2;
    g.stroke();
    g.setLineDash([]);
  }
  if (sel.goal) ring(sel.goal, col, 10, 1);
  const age = pin ? (performance.now() - pin.at) / 1800 : 1;
  if (age < 1) ring(pin.xy, pin.inside ? col : "#ff453a", 6 + 12 * age, 1 - age);
  if (sel.pose) {
    const { x, y, theta } = sel.pose, c = Math.cos(theta), s = Math.sin(theta);
    trace(Array.from({ length: 32 }, (_, i) => [x + 0.13 * Math.cos(i * Math.PI / 16), y + 0.13 * Math.sin(i * Math.PI / 16)]), true);
    g.fillStyle = alpha(col, 0.14);
    g.fill();
    g.strokeStyle = alpha(col, 0.9);
    g.lineWidth = 2;
    g.stroke();
    const [ax, ay] = P([x + 0.18 * c, y + 0.18 * s]), [bx, by] = P([x + 0.28 * c, y + 0.28 * s]);
    car(g, ax, ay, Math.atan2(by - ay, bx - ax), 7, col, false);
  }
}

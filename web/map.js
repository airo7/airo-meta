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
    const [cx, cy] = P([r.pose.x, r.pose.y]), s = big ? 13 : on ? 8 : 6.5, t = -r.pose.theta;
    if (on) {
      g.beginPath();
      g.arc(cx, cy, s * 1.9, 0, 7);
      g.fillStyle = alpha(col, 0.2);
      g.fill();
    }
    g.save();
    g.translate(cx, cy);
    g.rotate(t);
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
    if (big) {
      font(13, 600);
      g.fillStyle = "rgba(255,255,255,0.92)";
      g.fillText(r.name || r.id, cx + s * 1.6, cy - s * 0.9);
    }
  }
}

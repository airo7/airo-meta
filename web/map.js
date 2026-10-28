const POLY_COLORS = ["#b8a4ff", "#5fd4c4", "#ff8fb1", "#9ad66f", "#f2d06b"];

function sizeCanvas(cv) {
  const w = cv.clientWidth, h = cv.clientHeight, dpr = devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return [g, w, h];
}

function fitStage(stage, box, floor) {
  const s = Math.min(box.clientWidth / floor.w, box.clientHeight / floor.h);
  stage.style.width = `${floor.w * s}px`;
  stage.style.height = `${floor.h * s}px`;
}

function drawMap(cv, state, trail) {
  const [g, cw, ch] = sizeCanvas(cv);
  const { w, h } = state.floor, k = cw / w;
  const P = ([x, y]) => [x * k, (h - y) * k];
  const poly = (pts, close) => {
    g.beginPath();
    pts.forEach((p, i) => g[i ? "lineTo" : "moveTo"](...P(p)));
    if (close) g.closePath();
  };

  g.strokeStyle = "#ffffff10"; g.lineWidth = 1;
  for (let x = 0.5; x < w; x += 0.5) { g.beginPath(); g.moveTo(x * k, 0); g.lineTo(x * k, ch); g.stroke(); }
  for (let y = 0.5; y < h; y += 0.5) { g.beginPath(); g.moveTo(0, (h - y) * k); g.lineTo(cw, (h - y) * k); g.stroke(); }

  const fs = Math.max(10, Math.min(13, k * 0.055));
  g.font = `500 ${fs}px Archivo, system-ui, sans-serif`;
  (state.polygons || []).forEach((p, i) => {
    const col = POLY_COLORS[i % POLY_COLORS.length];
    poly(p.points, true);
    g.fillStyle = col + "22"; g.fill();
    g.strokeStyle = col + "cc"; g.lineWidth = 1.3; g.stroke();
    const [x, top] = p.points.map(P).reduce((a, b) => (b[1] < a[1] ? b : a));
    const ly = Math.max(top, fs * 2), tw = g.measureText(p.label).width, lx = Math.min(x, cw - tw - 12);
    g.fillStyle = "#070c18d9"; g.beginPath(); g.roundRect(lx - 2, ly - fs * 1.9, tw + 12, fs * 1.6, 4); g.fill();
    g.fillStyle = col; g.fillText(p.label, lx + 4, ly - fs * 0.75);
  });

  g.save();
  g.globalCompositeOperation = "lighter";
  g.strokeStyle = "#8fb4ff38"; g.lineWidth = 1.6; g.lineCap = "round"; g.lineJoin = "round";
  (state.cloud || []).forEach((c) => { poly(c); g.stroke(); });
  g.restore();

  const path = state.path || [];
  if (path.length > 1) {
    poly(path);
    g.setLineDash([7, 6]); g.strokeStyle = "#e6ecf5cc"; g.lineWidth = 1.6; g.stroke(); g.setLineDash([]);
    const [gx, gy] = P(path.at(-1));
    g.beginPath(); g.arc(gx, gy, 5, 0, 7); g.strokeStyle = "#e6ecf5"; g.stroke();
  }

  for (let i = 1; i < trail.length; i++) {
    poly([trail[i - 1], trail[i]]);
    g.strokeStyle = `rgba(255,181,71,${(i / trail.length) * 0.8})`; g.lineWidth = 2.5; g.stroke();
  }

  const pose = state.pose;
  if (!pose) return;
  const c = Math.cos(pose.theta), s = Math.sin(pose.theta);
  const at = (a, b) => [pose.x + c * a - s * b, pose.y + s * a + c * b];
  const [cx, cy] = P([pose.x, pose.y]);
  g.save();
  g.shadowColor = "#ffb547"; g.shadowBlur = 10;
  g.strokeStyle = g.fillStyle = "#ffb547"; g.lineWidth = 2;
  g.beginPath(); g.arc(cx, cy, 0.15 * k, 0, 7); g.stroke();
  poly([at(0.27, 0), at(0.17, 0.055), at(0.17, -0.055)], true); g.fill();
  g.restore();
}

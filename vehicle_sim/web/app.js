import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// ------------------------------------------------------------------ 상수
const STATE = {
  idle: ["대기", "#94a3b8"], training: ["학습", "#3b82f6"], uploading: ["전송", "#f59e0b"],
  waiting: ["대기열", "#eab308"], queued: ["보관", "#a855f7"], skipped: ["불참", "#64748b"],
  dead: ["끊김", "#ef4444"], excluded: ["영구 제외", "#7f1d1d"], absent: ["맵 밖", "#334155"],
};
const PCOL = { 1: "#ef4444", 2: "#f97316", 3: "#94a3b8", 4: "#eab308", 5: "#a855f7", 6: "#06b6d4", 7: "#ec4899", 8: "#22c55e", 9: "#dc2626" };
const FOCUS = { // 유스케이스 → 맵 위치 (유스케이스 표에서 클릭하면 카메라 이동)
  "1-2": "service", "1-4": "school", "1-5": "home", "2-1": [1000, 300], "2-2": [1450, 300], "2-4": [600, 450],
  "2-5": [1100, 900], "2-6": "tunnel", "3-3": "gas", "3-4": "security", "4-2": "congest", "4-5": "school",
  "5-3": "shadow", "5-4": "home", "6-2": "service", "8-3": "gate_in", "8-4": "weather", "9-1": "datacenter", "9-2": "datacenter",
};
const $ = (s) => document.querySelector(s);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
const fmtT = (t) => new Date(t * 1000).toLocaleTimeString("ko-KR", { hour12: false });

let STATIC = null, CFG = null, LAST = null, evOffset = 0, selected = 1;
const caseCount = {}, caseLast = {}, caseVeh = {};
let lastRoundShown = -1;

// ------------------------------------------------------------------ 3D 장면
const holder = $("#map3d");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
holder.appendChild(renderer.domElement);
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.position = "absolute";
labelRenderer.domElement.style.top = "0";
labelRenderer.domElement.style.pointerEvents = "none";
holder.appendChild(labelRenderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b1222");
scene.fog = new THREE.Fog("#0b1222", 2600, 4200);
const camera = new THREE.PerspectiveCamera(38, 1, 5, 9000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;
scene.add(new THREE.HemisphereLight("#cfe0ff", "#1a2433", 1.1));
const sun = new THREE.DirectionalLight("#ffffff", 1.2);
sun.position.set(-600, 1400, 800);
scene.add(sun);

const W2 = 1000, H2 = 700;  // 맵 중심
const P = (x, z, y = 0) => new THREE.Vector3(x - W2, y, z - H2);
const layers = { cells: [], labels: [], beams: [] };

function label(text, cls, pos, layer = "labels") {
  const d = el("div", "lbl " + (cls || ""), text);
  const o = new CSS2DObject(d);
  o.position.copy(pos);
  scene.add(o);
  layers[layer].push(o);
  return o;
}
function flat(geom, color, opacity, y, x, z, layer) {
  const m = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide }));
  m.rotation.x = -Math.PI / 2;
  m.position.copy(P(x, z, y));
  scene.add(m);
  if (layer) layers[layer].push(m);
  return m;
}
function box(w, h, d, color, x, z, y = 0, opts = {}) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), new THREE.MeshStandardMaterial({ color, roughness: 0.8, ...opts }));
  m.position.copy(P(x, z, y + h / 2));
  scene.add(m);
  return m;
}
const ZONE = {}, PLACE = {}, CELL = {};
let rain = null, weatherGroup = null;

function buildScene(S) {
  const [MW, MH] = S.map.size;
  flat(new THREE.PlaneGeometry(MW + 300, MH + 300), "#10231c", 1, -1, MW / 2, MH / 2);
  const grid = new THREE.GridHelper(2400, 24, "#1c3a2e", "#16302a");
  grid.position.y = -0.5;
  scene.add(grid);
  // 도로
  for (const rd of S.map.roads) {
    for (let i = 0; i < rd.pts.length - 1; i++) {
      const [ax, az] = rd.pts[i], [bx, bz] = rd.pts[i + 1];
      const len = Math.hypot(bx - ax, bz - az);
      const m = new THREE.Mesh(new THREE.BoxGeometry(len + rd.width, 1, rd.width),
        new THREE.MeshStandardMaterial({ color: rd.id === "highway" ? "#3a4256" : "#2c3344", roughness: 1 }));
      m.position.copy(P((ax + bx) / 2, (az + bz) / 2, 0.5));
      m.rotation.y = -Math.atan2(bz - az, bx - ax);
      scene.add(m);
      if (rd.id === "highway") {
        for (let s = 0; s < len; s += 40) {
          const t = s / len;
          const dash = box(18, 0.4, 1.6, "#d6d3a8", ax + (bx - ax) * t, az + (bz - az) * t, 1);
          dash.rotation.y = m.rotation.y;
        }
      }
    }
    const mid = rd.pts[Math.floor((rd.pts.length - 1) / 2)];
    label(`${rd.name} · ${rd.speed} km/h`, "", P(mid[0], mid[1] + 26, 4));
  }
  // 셀(기지국)
  for (const c of S.map.cells) {
    const [x, z] = c.pos;
    const disc = flat(new THREE.CircleGeometry(c.r, 72), c.color, 0.07, 1.2, x, z, "cells");
    const ring = flat(new THREE.RingGeometry(c.r - 4, c.r, 96), c.color, 0.6, 1.4, x, z, "cells");
    const tower = new THREE.Group();
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(2.5, 5, 80, 8), new THREE.MeshStandardMaterial({ color: "#cbd5e1" }));
    mast.position.y = 40;
    const head = new THREE.Mesh(new THREE.SphereGeometry(7, 16, 12), new THREE.MeshStandardMaterial({ color: c.color, emissive: c.color, emissiveIntensity: 0.8 }));
    head.position.y = 82;
    tower.add(mast, head);
    tower.position.copy(P(x, z));
    scene.add(tower);
    CELL[c.id] = { ...c, top: P(x, z, 84), disc, ring };
    label(`📡 ${c.name} · 반경 ${c.r} m`, "cell", P(x, z, 110));
  }
  // 권역 경계
  const bx = S.map.region_border_x;
  const g = new THREE.BufferGeometry().setFromPoints([P(bx, -60, 3), P(bx, MH + 60, 3)]);
  const line = new THREE.Line(g, new THREE.LineDashedMaterial({ color: "#fbbf24", dashSize: 24, gapSize: 16 }));
  line.computeLineDistances();
  scene.add(line);
  label("W 권역 ◀ 권역 경계 ▶ E 권역", "zone", P(bx, -40, 10));
  // 구역
  for (const z of S.map.zones) {
    ZONE[z.id] = z;
    if (z.kind === "tunnel") {
      const [x0, z0, x1, z1] = z.rect;
      const t = box(x1 - x0, 34, z1 - z0 + 30, "#64748b", (x0 + x1) / 2, (z0 + z1) / 2, 0, { transparent: true, opacity: 0.82 });
      box(14, 40, z1 - z0 + 40, "#334155", x0, (z0 + z1) / 2);
      box(14, 40, z1 - z0 + 40, "#334155", x1, (z0 + z1) / 2);
      label("🚇 터널 (무선 단절)", "zone", P((x0 + x1) / 2, z0 - 40, 50));
    } else if (z.kind === "shadow") {
      const [x0, z0, x1, z1] = z.rect;
      for (let x = x0 + 18; x < x1 - 10; x += 42) {
        for (const [za, zb] of [[z0 + 10, 1182], [1222, z1 - 4]]) {
          for (let zz = za + 14; zz < zb; zz += 40) {
            const h = 50 + ((x * 7 + zz * 3) % 120);
            box(30, h, 28, "#334155", x, zz, 0, { emissive: "#1e293b" });
          }
        }
      }
      flat(new THREE.PlaneGeometry(x1 - x0, z1 - z0), "#64748b", 0.12, 1.6, (x0 + x1) / 2, (z0 + z1) / 2);
      label("🏢 빌딩 숲 (음영지역 · Gilbert)", "zone", P((x0 + x1) / 2, z0 - 20, 170));
    } else if (z.rect) {
      const [x0, z0, x1, z1] = z.rect;
      const col = { congestion: "#f97316", school: "#facc15", security: "#ef4444" }[z.kind] || "#94a3b8";
      flat(new THREE.PlaneGeometry(x1 - x0, z1 - z0), col, 0.22, 1.8, (x0 + x1) / 2, (z0 + z1) / 2);
      const pts = [P(x0, z0, 3), P(x1, z0, 3), P(x1, z1, 3), P(x0, z1, 3), P(x0, z0, 3)];
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col })));
      if (z.kind === "security") for (let i = 0; i < 4; i++) box(4, 22, 4, "#ef4444", [x0, x1, x1, x0][i], [z0, z0, z1, z1][i]);
      const icon = { congestion: "🚦", school: "🏫", security: "🔒" }[z.kind] || "";
      label(`${icon} ${z.name}${z.speed ? ` · ${z.speed} km/h` : ""}`, "zone", P((x0 + x1) / 2, z0 - 16, 30));
    } else if (z.circle) {
      const [x, zz, r] = z.circle;
      weatherGroup = new THREE.Group();
      const disc = new THREE.Mesh(new THREE.CircleGeometry(r, 64), new THREE.MeshBasicMaterial({ color: "#38bdf8", transparent: true, opacity: 0.1, depthWrite: false }));
      disc.rotation.x = -Math.PI / 2;
      disc.position.copy(P(x, zz, 2));
      const cloud = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.9, r * 0.9, 8, 48), new THREE.MeshBasicMaterial({ color: "#94a3b8", transparent: true, opacity: 0.16, depthWrite: false }));
      cloud.position.copy(P(x, zz, 220));
      const N = 900, pos = new Float32Array(N * 3);
      for (let i = 0; i < N; i++) {
        const a = Math.random() * Math.PI * 2, rr = Math.sqrt(Math.random()) * r * 0.9;
        pos[i * 3] = x - W2 + Math.cos(a) * rr; pos[i * 3 + 1] = Math.random() * 215; pos[i * 3 + 2] = zz - H2 + Math.sin(a) * rr;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      rain = new THREE.Points(geo, new THREE.PointsMaterial({ color: "#7dd3fc", size: 3, transparent: true, opacity: 0.7 }));
      weatherGroup.add(disc, cloud, rain);
      scene.add(weatherGroup);
      weatherGroup.userData.label = label("🌧 강우 구역 (대역폭↓ · 데이터 분포 변화)", "zone", P(x, zz - r + 10, 240));
    }
  }
  // 장소
  for (const p of S.map.places) {
    PLACE[p.id] = p;
    const [x, z] = p.pos;
    if (p.kind === "home") {
      for (let i = -1; i <= 1; i++) {
        box(34, 22, 30, "#e2e8f0", x + i * 60, z + 52);
        const roof = new THREE.Mesh(new THREE.ConeGeometry(28, 18, 4), new THREE.MeshStandardMaterial({ color: "#b45309" }));
        roof.position.copy(P(x + i * 60, z + 52, 31));
        roof.rotation.y = Math.PI / 4;
        scene.add(roof);
      }
      flat(new THREE.PlaneGeometry(190, 26), "#475569", 0.6, 1.9, x, z + 22);
      label("🏠 집·주차장 (시동 꺼짐)", "place", P(x, z + 90, 50));
    } else if (p.kind === "gas") {
      box(70, 4, 46, "#16a34a", x - 70, z, 26);
      for (const [dx, dz] of [[-30, -18], [30, -18], [-30, 18], [30, 18]]) box(3, 26, 3, "#d1d5db", x - 70 + dx, z + dz);
      box(8, 14, 8, "#f8fafc", x - 82, z); box(8, 14, 8, "#f8fafc", x - 58, z);
      label("⛽ 주유소·충전소", "place", P(x - 70, z - 30, 46));
    } else if (p.kind === "service") {
      box(90, 34, 50, "#1d4ed8", x, z - 52);
      label("🔧 서비스센터 (OTA 설치·재부팅)", "place", P(x, z - 90, 50));
    } else if (p.kind === "server") {
      for (let i = 0; i < 4; i++) box(18, 40, 26, "#0f172a", x - 36 + i * 24, z, 0, { emissive: "#1e3a8a", emissiveIntensity: 0.6 });
      const glow = new THREE.Mesh(new THREE.SphereGeometry(10, 16, 12), new THREE.MeshBasicMaterial({ color: "#60a5fa" }));
      glow.position.copy(P(x, z, 60));
      scene.add(glow);
      PLACE[p.id].glow = glow;
      PLACE[p.id].top = P(x, z, 60);
      PLACE[p.id].label = label("🖥 FL 집계 서버 (SASAFL)", "place", P(x, z - 30, 80));
    } else if (p.kind === "gate") {
      box(6, 40, 6, "#a3a3a3", x + (x < 10 ? 12 : -12), 300 - 26);
      box(6, 40, 6, "#a3a3a3", x + (x < 10 ? 12 : -12), 300 + 26);
      box(6, 6, 58, "#a3a3a3", x + (x < 10 ? 12 : -12), 300, 40);
      label(x < 10 ? "➡ 진입 게이트 (새 차량 합류)" : "이탈 게이트 ➡", "place", P(x + (x < 10 ? 70 : -60), 260, 50));
    }
  }
  fitView();
}

function fitView() {
  // 맵 전체(2000×1400 m + 여유)가 화면 가로·세로에 모두 들어가도록 거리를 계산한다
  const v = THREE.MathUtils.degToRad(camera.fov) / 2;
  const h = Math.atan(Math.tan(v) * camera.aspect);
  const d = Math.max(1120 / Math.tan(h), 820 / Math.tan(v)) * 1.02;
  const dir = new THREE.Vector3(0, 0.80, 0.60).normalize();
  controls.target.set(0, 0, 30);
  camera.position.copy(controls.target).addScaledVector(dir, d);
  controls.update();
}

// ------------------------------------------------------------------ 차량
const CARS = {};
const carGeo = new THREE.BoxGeometry(38, 14, 18);
const cabGeo = new THREE.BoxGeometry(18, 10, 15);
function makeCar(vid) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: "#94a3b8", emissive: "#000000", roughness: 0.5 });
  const body = new THREE.Mesh(carGeo, mat);
  body.position.y = 9;
  const cab = new THREE.Mesh(cabGeo, new THREE.MeshStandardMaterial({ color: "#0f172a" }));
  cab.position.set(-2, 19, 0);
  body.userData.vid = vid;
  g.add(body, cab);
  const ring = new THREE.Mesh(new THREE.RingGeometry(26, 32, 32), new THREE.MeshBasicMaterial({ color: "#ffffff", side: THREE.DoubleSide }));
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 2;
  ring.visible = false;
  g.add(ring);
  const d = el("div", "vlbl", `V${vid}`);
  const lo = new CSS2DObject(d);
  lo.position.set(0, 46, 0);
  g.add(lo);
  scene.add(g);
  const beamGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const beam = new THREE.Line(beamGeo, new THREE.LineDashedMaterial({ color: "#f59e0b", dashSize: 10, gapSize: 8 }));
  beam.visible = false;
  scene.add(beam);
  layers.beams.push(beam);
  return (CARS[vid] = { g, body, mat, ring, d, lo, beam, target: null, heading: 0, status: "idle" });
}

function carStatus(vid) {
  const w = LAST?.world?.vehicles?.[vid] || {};
  const c = LAST?.clients?.[vid] || {};
  if (w.present === false) return "absent";
  return c.status || "idle";
}

function updateCars() {
  const K = CFG.fl.K;
  for (let vid = 1; vid <= K; vid++) {
    const car = CARS[vid] || makeCar(vid);
    const w = LAST?.world?.vehicles?.[vid];
    const c = LAST?.clients?.[vid] || {};
    if (!w) { car.g.visible = false; continue; }
    const st = carStatus(vid);
    car.status = st;
    car.g.visible = w.present !== false;
    car.lo.visible = car.g.visible && $("#optLabels").checked;
    car.target = P(w.pos[0], w.pos[1]);
    car.heading = -w.heading;
    if (!car.init) { car.g.position.copy(car.target); car.init = true; }
    const [name, col] = STATE[st] || STATE.idle;
    car.mat.color.set(col);
    car.mat.emissive.set(st === "uploading" ? "#7c2d12" : "#000000");
    car.ring.visible = vid === selected;
    const badges = [w.engine_off ? "🅿" : "", w.charging ? "⚡" : "", w.ota ? "🔧" : "", w.zones?.tunnel ? "🚇" : "",
      w.zones?.security ? "🔒" : "", !w.link_up && w.present ? "📵" : "", c.queue ? `📦${c.queue}` : "",
      (c.energy ?? 999) < (CFG.energy.min_energy_j) ? "🪫" : ""].join("");
    car.d.className = "vlbl" + (vid === selected ? " sel" : "");
    car.d.style.background = col;
    car.d.innerHTML = `V${vid} <small>${name}${c.version !== undefined ? ` · v${c.version}` : ""}</small> ${badges}`;
    // 업로드 선: 차량 → 서빙 기지국
    const cell = w.cell && CELL[w.cell];
    car.beam.visible = $("#optBeams").checked && st === "uploading" && !!cell;
    if (car.beam.visible) {
      const pos = car.beam.geometry.attributes.position;
      pos.setXYZ(0, car.g.position.x, 22, car.g.position.z);
      pos.setXYZ(1, cell.top.x, cell.top.y, cell.top.z);
      pos.needsUpdate = true;
      car.beam.computeLineDistances();
    }
  }
}

function floatAt(vid, text, color) {
  const car = CARS[vid];
  if (!car) return;
  const d = el("div", "float", text);
  d.style.background = color;
  const o = new CSS2DObject(d);
  o.position.set(0, 70, 0);
  car.g.add(o);
  setTimeout(() => car.g.remove(o), 2600);
}
const flashes = [];
function flashArrival(vid) {
  const w = LAST?.world?.vehicles?.[vid];
  const dc = PLACE.datacenter;
  if (!w || !dc) return;
  const cell = w.cell && CELL[w.cell];
  const pts = [P(w.pos[0], w.pos[1], 22)];
  if (cell) pts.push(cell.top);
  pts.push(dc.top);
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: "#22c55e", transparent: true, opacity: 1 }));
  scene.add(line);
  flashes.push({ line, t0: performance.now() });
  dc.glow.material.color.set("#22c55e");
  setTimeout(() => dc.glow.material.color.set("#60a5fa"), 600);
}

// ------------------------------------------------------------------ 렌더 루프
let fitted = false;
function resize() {
  const r = holder.getBoundingClientRect();
  if (!r.width) return;
  renderer.setSize(r.width, r.height);
  labelRenderer.setSize(r.width, r.height);
  camera.aspect = r.width / Math.max(1, r.height);
  camera.updateProjectionMatrix();
  if (!fitted) { fitted = true; fitView(); }
}
window.addEventListener("resize", resize);
let lastFrame = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  for (const car of Object.values(CARS)) {
    if (!car.target) continue;
    car.g.position.lerp(car.target, Math.min(1, dt * 6));
    let dh = car.heading - car.g.rotation.y;
    dh = Math.atan2(Math.sin(dh), Math.cos(dh));
    car.g.rotation.y += dh * Math.min(1, dt * 8);
    if (car.beam.visible) car.beam.material.dashOffset -= dt * 60;
  }
  if (rain && weatherGroup.visible) {
    const p = rain.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) { let y = p.getY(i) - dt * 160; if (y < 0) y = 215; p.setY(i, y); }
    p.needsUpdate = true;
  }
  for (let i = flashes.length - 1; i >= 0; i--) {
    const f = flashes[i], a = 1 - (now - f.t0) / 1400;
    if (a <= 0) { scene.remove(f.line); flashes.splice(i, 1); } else f.line.material.opacity = a;
  }
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
  requestAnimationFrame(frame);
}

// 클릭으로 차량 선택 / 호버 정보
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
function pick(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  const hit = ray.intersectObjects(Object.values(CARS).map((c) => c.body))[0];
  return hit ? hit.object.userData.vid : null;
}
renderer.domElement.addEventListener("click", (ev) => { const v = pick(ev); if (v) selectVehicle(v); });
renderer.domElement.addEventListener("mousemove", (ev) => {
  const v = pick(ev), tip = $("#tooltip");
  if (!v) { tip.style.display = "none"; return; }
  const w = LAST?.world?.vehicles?.[v] || {}, c = LAST?.clients?.[v] || {};
  tip.innerHTML = `<b>V${v}</b> · ${STATE[carStatus(v)]?.[0]} — ${c.detail || ""}<br>
    ${w.speed_kmh ?? "-"} km/h · 셀 ${w.cell || "없음"} · ${w.link_up ? "연결됨" : "연결 없음"} · 권역 ${w.region}<br>
    에너지 ${c.energy ?? "-"} J · 새 데이터 ${c.new_data ?? "-"} · 모델 v${c.version ?? "-"} (서버 v${c.server_version ?? "-"})`;
  tip.style.display = "block";
  tip.style.left = ev.clientX + 14 + "px";
  tip.style.top = ev.clientY + 10 + "px";
});

function focusOn(key) {
  let x, z;
  if (Array.isArray(key)) [x, z] = key;
  else if (PLACE[key]) [x, z] = PLACE[key].pos;
  else if (ZONE[key]) {
    const zz = ZONE[key];
    if (zz.rect) { x = (zz.rect[0] + zz.rect[2]) / 2; z = (zz.rect[1] + zz.rect[3]) / 2; } else [x, z] = zz.circle;
  } else return;
  document.querySelector('[data-tab="map"]').click();
  controls.target.copy(P(x, z));
  camera.position.copy(P(x - 120, z + 620, 560));
}

// ------------------------------------------------------------------ 탭
document.querySelectorAll(".tabs").forEach((nav) => nav.addEventListener("click", (e) => {
  const b = e.target.closest(".tab");
  if (!b) return;
  nav.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  const panes = nav.parentElement.querySelectorAll(":scope > .tabpane");
  panes.forEach((p) => p.classList.toggle("active", p.id === "pane-" + b.dataset.tab));
  if (b.dataset.tab === "map") resize();
}));

// ------------------------------------------------------------------ 범례
function buildLegends() {
  $("#legendState").innerHTML = "<b style='grid-column:1/-1'>차량 색 = 연합학습 상태</b>" +
    Object.entries(STATE).map(([k, [n, c]]) => `<span><i style="background:${c}"></i>${n}</span>`).join("") +
    "<span style='grid-column:1/-1' class='muted'>🅿주차 ⚡충전 🔧OTA 🚇터널 🔒보안 📵연결 없음 📦보관 🪫에너지↓</span>";
  $("#legendPlaces").innerHTML = "<b style='grid-column:1/-1'>사건 색 = 패턴 (서버가 받은 모습)</b>" +
    Object.entries(STATIC.patterns).map(([k, n]) => `<span><i style="background:${PCOL[k]}"></i>P${k} ${n}</span>`).join("");
}

// ------------------------------------------------------------------ 설정 폼
const VFIELDS = [
  ["성능·중단", [
    ["ecu_factor", "ECU 속도 배수 (비우면 1~2.2 무작위)", "num", 1, 3, 0.1, "4-1"],
    ["crash_prob", "계산 도중 중단 확률", "num", 0, 0.5, 0.01, "1-1"],
    ["safety_preempt_prob", "보호구역 안전 기능 선점 확률", "num", 0, 1, 0.05, "1-4"],
    ["adas_slowdown", "보호구역 연산 점유 (추가 배수)", "num", 0, 5, 0.1, "4-5"]]],
  ["참여 조건", [
    ["min_data", "학습 최소 새 데이터 (개)", "num", 0, 2000, 8, "3-2"],
    ["data_rate_per_s", "주행 중 데이터 수집 (개/초)", "num", 0, 100, 1, "3-2"],
    ["charging_only", "충전 중일 때만 참여", "bool", 0, 0, 0, "3-3"],
    ["energy_gate", "에너지 기준 확인 (끄면 방전까지 학습)", "bool", 0, 0, 0, "3-1 · 1-3"],
    ["leave", "맵에서 이탈 (라벨 쏠림 차량이면 8-2)", "bool", 0, 0, 0, "8-2 · 8-3"]]],
  ["전송", [
    ["store_forward", "끊기면 보관 후 한꺼번에 전송", "bool", 0, 0, 0, "5-2 · 5-3 · 5-4"],
    ["retransmit", "패킷 오류 시 재전송", "bool", 0, 0, 0, "4-3"],
    ["dup_prob", "수신 확인 손실(중복 전송) 확률", "num", 0, 1, 0.05, "9-3"]]],
  ["내용 오염", [
    ["lr_override", "학습률 덮어쓰기 (비우면 정상)", "num", 0, 1, 0.001, "7-1"],
    ["label_noise", "라벨 오류 비율", "num", 0, 1, 0.05, "7-2", true],
    ["weight_noise", "가중치 노이즈 (센서 고장)", "num", 0, 5, 0.1, "7-4"],
    ["backdoor_boost", "업데이트 증폭 (백도어)", "num", 0, 50, 1, "7-3"],
    ["clock_skew", "시계 오차 → 버전 오차", "num", -10, 10, 1, "7-5"],
    ["drift_noise", "강우 구역 입력 노이즈", "num", 0, 2, 0.1, "8-4"]]],
];
const GFIELDS = [
  ["연합학습 (SASAFL)", "fl", [
    ["K", "차량 수 K", "num", 2, 16, 1, "", true], ["dataset", "데이터셋", "sel:MNIST,FashionMNIST,CIFAR10", 0, 0, 0, "", true],
    ["C", "최소 참여 비율 C", "num", 0.1, 1, 0.1, "", true], ["tau", "lag tolerance τ", "num", 1, 20, 1, "", true],
    ["epochs", "로컬 epoch", "num", 1, 5, 1, "", true], ["lr", "학습률", "num", 0.0001, 0.1, 0.001, "", true],
    ["split", "데이터 분할", "sel:gaussian,uniform,dirichlet", 0, 0, 0, "8-2", true], ["alpha", "Dirichlet α", "num", 0.01, 10, 0.05, "8-2", true],
    ["rejoin", "끊긴 차량 복귀 허용 (끄면 원본 SASAFL처럼 영구 제외)", "bool", 0, 0, 0, ""]]],
  ["통신", "net", [
    ["bandwidth_mbps", "상향 대역폭 (Mbps)", "num", 1, 200, 1, "4-4"], ["fragments", "업로드 조각 수", "num", 1, 50, 1, "2-x"],
    ["frag_fail", "조각당 패킷 오류 확률", "num", 0, 0.5, 0.01, "2-3"], ["doppler_kmh", "도플러 기준 속도 (km/h)", "num", 30, 150, 5, "2-2"],
    ["doppler_fail", "고속 조각 실패 확률", "num", 0, 1, 0.05, "2-2"], ["cell_capacity", "셀당 동시 업로드 수", "num", 1, 10, 1, "4-2"],
    ["tdma_groups", "TDMA 그룹 수 G (1=끔)", "num", 1, 10, 1, "5-1"], ["tdma_slot_s", "TDMA 슬롯 길이 (초)", "num", 0.5, 20, 0.5, "5-1"],
    ["retx_timeout_s", "재전송 대기 (초)", "num", 1, 30, 1, "4-3"], ["retx_total_s", "재전송 전체 제한 (초)", "num", 1, 120, 1, "4-3"],
    ["gilbert_p", "음영지역 좋음 유지 확률 p", "num", 0, 1, 0.05, "5-3"], ["gilbert_q", "음영지역 나쁨 유지 확률 q", "num", 0, 1, 0.05, "5-3"],
    ["region_instability", "권역 경계 통과 시 끊김 확률", "num", 0, 1, 0.05, "2-5"], ["handover_drops", "핸드오버 시 업로드 끊김", "bool", 0, 0, 0, "2-4"],
    ["weather_bw_mult", "강우 구역 대역폭 배수", "num", 0.05, 1, 0.05, "4-4"], ["edge_quality", "보류 기준 통신 품질", "num", 0, 1, 0.05, "5-2"]]],
  ["에너지", "energy", [
    ["budget_j", "에너지 예산 (J)", "num", 100, 5000, 50, "1-3", true], ["round_j_min", "라운드당 소모 최소 (J)", "num", 0, 200, 1, "1-3"],
    ["round_j_max", "라운드당 소모 최대 (J)", "num", 0, 200, 1, "1-3"], ["min_energy_j", "참여 최소 에너지 (J)", "num", 0, 2000, 10, "3-1"],
    ["charge_rate_jps", "충전 속도 (J/초)", "num", 1, 500, 5, "1-3"], ["upload_j", "업로드 소모 (J)", "num", 0, 50, 1, "2-7"]]],
  ["주행 환경", "world", [
    ["availability_mode", "참여 가능 비율 주기", "sel:off,cycle", 0, 0, 0, "8-1"], ["availability_min", "가용률 최저", "num", 0, 1, 0.05, "8-1"],
    ["availability_max", "가용률 최고", "num", 0, 1, 0.05, "8-1"], ["availability_period_s", "가용률 주기 (초)", "num", 30, 3600, 30, "8-1"],
    ["weather_on", "강우 구역 켜기", "bool", 0, 0, 0, "8-4"], ["visitor_gap_mean_s", "통과 차량 재진입 평균 (초, 포아송)", "num", 5, 600, 5, "8-3"],
    ["ota_prob", "서비스센터 OTA 확률", "num", 0, 1, 0.05, "1-2 · 6-2"], ["speed_scale", "주행 속도 배율", "num", 0.1, 5, 0.1, ""]]],
];

function inputFor(def, value, onchange) {
  const [key, name, type, min, max, step, uc, restart] = def;
  const f = el("div", "field");
  f.appendChild(el("label", "", `${name}${uc ? `<span class="uc">${uc}</span>` : ""}${restart ? '<span class="restart">재시작</span>' : ""}`));
  let inp;
  if (type === "bool") {
    inp = el("input"); inp.type = "checkbox"; inp.checked = !!value;
    inp.onchange = () => onchange(inp.checked);
  } else if (type.startsWith("sel:")) {
    inp = el("select");
    for (const o of type.slice(4).split(",")) inp.appendChild(new Option(o, o, false, o === value));
    inp.onchange = () => onchange(inp.value);
  } else {
    inp = el("input"); inp.type = "number"; inp.min = min; inp.max = max; inp.step = step;
    inp.value = value === null || value === undefined ? "" : value;
    inp.onchange = () => onchange(inp.value === "" ? null : Number(inp.value));
  }
  inp.dataset.key = key;
  f.appendChild(inp);
  return f;
}
async function postCfg(patch) {
  const r = await fetch("/api/config", { method: "POST", body: JSON.stringify(patch) });
  CFG = (await r.json()).config;
}
function buildGlobalForm() {
  const box = $("#globalForm");
  box.innerHTML = "";
  for (const [title, sec, defs] of GFIELDS) {
    box.appendChild(el("div", "fgroup", title));
    for (const d of defs) box.appendChild(inputFor(d, CFG[sec][d[0]], (v) => postCfg({ [sec]: { [d[0]]: v } })));
  }
}
function buildVehForm() {
  const box = $("#vehForm");
  box.innerHTML = "";
  const all = el("div", "field");
  all.innerHTML = `<label><b>모든 차량에 적용</b><span class="uc">기본값을 바꿈</span></label>`;
  const cb = el("input"); cb.type = "checkbox"; cb.id = "applyAll";
  all.appendChild(cb);
  box.appendChild(all);
  const cur = { ...CFG.vehicle_default, ...(CFG.vehicles[String(selected)] || {}) };
  for (const [title, defs] of VFIELDS) {
    box.appendChild(el("div", "fgroup", title));
    for (const d of defs) box.appendChild(inputFor(d, cur[d[0]], (v) => {
      if ($("#applyAll").checked) postCfg({ vehicle_default: { [d[0]]: v } });
      else postCfg({ vehicles: { [String(selected)]: { [d[0]]: v } } });
    }));
  }
}
function selectVehicle(v) {
  selected = v;
  buildVehForm();
  renderVehPicker();
  renderVehInfo();
}
function renderVehPicker() {
  const K = CFG.fl.K;
  for (const [id, target] of [["#vehPicker", "pick"], ["#vehGrid", "grid"]]) {
    const box = $(id);
    if (box.children.length !== K) {
      box.innerHTML = "";
      for (let v = 1; v <= K; v++) {
        const b = el("div", "vchip");
        b.onclick = () => selectVehicle(v);
        box.appendChild(b);
      }
    }
    [...box.children].forEach((b, i) => {
      const v = i + 1, st = carStatus(v), [n, c] = STATE[st] || STATE.idle;
      b.style.background = c;
      b.classList.toggle("sel", v === selected);
      b.innerHTML = `V${v}<small>${n}</small>`;
    });
  }
}
function renderVehInfo() {
  const w = LAST?.world?.vehicles?.[selected] || {}, c = LAST?.clients?.[selected] || {};
  const route = STATIC.routes[w.route]?.name || "-";
  const zones = Object.entries(w.zones || {}).filter(([, b]) => b).map(([k]) =>
    ({ tunnel: "터널", shadow: "음영", congestion: "혼잡", school: "보호구역", security: "보안", weather: "강우" }[k])).join(", ");
  const hist = c.label_hist || [];
  const mx = Math.max(1, ...hist);
  const cn = c.counters || {};
  $("#vehInfo").innerHTML = `
    <div class="row"><b>V${selected} · ${route}</b><span>${STATE[carStatus(selected)]?.[0]} — ${c.detail || ""}</span></div>
    <div class="row"><span>속도 ${w.speed_kmh ?? "-"} km/h · 셀 ${w.cell || "범위 밖"} · 품질 ${w.quality ?? "-"}</span><span>권역 ${w.region || "-"}</span></div>
    <div class="row"><span>구역: ${zones || "없음"}</span><span>${w.link_up ? "🟢 연결" : "🔴 연결 없음"}${w.server_ok === false && w.link_up ? " (서버 차단)" : ""}</span></div>
    <div class="row"><span>에너지 ${c.energy ?? "-"} J · 새 데이터 ${c.new_data ?? "-"} · ECU ×${c.ecu ?? "-"}</span><span>모델 v${c.version ?? "-"} / 서버 v${c.server_version ?? "-"}</span></div>
    <div class="row"><span>학습 ${cn.trained ?? 0} · 전송 ${cn.sent ?? 0} · 폐기 ${cn.dropped ?? 0} · 중단 ${cn.aborted ?? 0} · 불참 ${cn.skipped ?? 0} · 보관 ${cn.queued ?? 0}</span><span>낭비 ${c.wasted_mb ?? 0} MB</span></div>
    <div class="muted small">라벨 분포 (데이터 ${c.n_data ?? "-"}개, 최다 라벨 비중 ${Math.round((c.label_skew || 0) * 100)}%)</div>
    <div class="bars">${hist.map((h, i) => `<div title="${i}: ${h}" style="height:${(h / mx) * 100}%"></div>`).join("")}</div>`;
}

// ------------------------------------------------------------------ FL 상태 패널
function renderFL() {
  const s = LAST.server || {};
  const hist = s.history || [];
  const last = hist[hist.length - 1];
  $("#kRound").textContent = s.round ?? "-";
  $("#kAcc").textContent = last ? last.acc.toFixed(1) + "%" : "-";
  $("#kQuota").textContent = s.quota ?? "-";
  $("#kBuf").textContent = s.cached !== undefined ? `${s.cached}/${s.quota}` : "-";
  const wsec = s.phase === "waiting" ? s.waiting_s || 0 : 0;
  $("#waitFill").style.width = Math.min(100, (wsec / 60) * 100) + "%";
  $("#waitText").textContent = s.phase === "waiting" ? `quota ${s.quota}개 모일 때까지 대기 ${wsec.toFixed(0)}초 (버퍼 ${s.cached}개${wsec > 60 ? " · ⚠ 9-2 무한 대기 위험" : ""})` : "집계 중";
  $("#flPhase").textContent = { waiting: "업데이트 대기", aggregated: "집계 완료", start: "시작" }[s.phase] || "-";
  drawChart(hist);
  // 새 라운드를 위에 하나씩
  const feed = $("#roundFeed");
  for (const r of hist) {
    if (r.round <= lastRoundShown || !r.picked) continue;
    lastRoundShown = r.round;
    if (!r.picked.length) continue;
    const card = el("div", "round");
    card.innerHTML = `<div class="round-head"><b>${r.round}라운드</b><span>정확도 ${r.acc}% · quota ${r.quota}→${r.next_quota ?? "-"} · ${r.dur ?? "-"}초</span></div>`;
    const cars = el("div", "round-cars");
    r.picked.forEach((p, i) => {
      const chip = el("span", "pcar" + (p.staleness > 0 ? " stale" : "") + ((r.dups || []).includes(p.vid) ? " dup" : ""),
        `V${p.vid} · staleness ${p.staleness} · ${Math.round(p.weight * 100)}%`);
      chip.style.background = STATE.training[1];
      chip.style.animationDelay = i * 140 + "ms";
      chip.title = `모델 버전 v${p.version}, 집계 비중 ${p.weight}`;
      cars.appendChild(chip);
    });
    card.appendChild(cars);
    feed.prepend(card);
    while (feed.children.length > 60) feed.lastChild.remove();
  }
}
function drawChart(hist) {
  const cv = $("#accChart"), ctx = cv.getContext("2d");
  const W = (cv.width = cv.clientWidth * devicePixelRatio), H = (cv.height = 110 * devicePixelRatio);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#263252"; ctx.lineWidth = 1;
  for (let y = 0; y <= 4; y++) { ctx.beginPath(); ctx.moveTo(0, (H * y) / 4); ctx.lineTo(W, (H * y) / 4); ctx.stroke(); }
  ctx.fillStyle = "#8b97b3"; ctx.font = `${10 * devicePixelRatio}px sans-serif`;
  ctx.fillText("정확도 % (라운드별)", 6 * devicePixelRatio, 12 * devicePixelRatio);
  if (hist.length < 2) return;
  const n = hist.length - 1;
  ctx.strokeStyle = "#4f8cff"; ctx.lineWidth = 2 * devicePixelRatio; ctx.beginPath();
  hist.forEach((h, i) => { const x = (i / n) * W, y = H - (h.acc / 100) * H; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

// ------------------------------------------------------------------ 사건
function evLine(e) {
  const p = e.pattern;
  const d = el("div", "ev");
  d.style.borderLeftColor = p ? PCOL[p] : "#263252";
  const tag = p ? `<span class="pt" style="background:${PCOL[p]}">${e.example}</span>` :
    `<span class="pt" style="background:#334155">${{ round: "집계", arrive: "도착", upload: "전송", world: "환경", control: "제어", fl: "FL" }[e.kind] || e.kind}</span>`;
  d.innerHTML = `${tag}${e.vid ? `<b>V${e.vid}</b> ` : ""}${e.msg}<span class="tm">${fmtT(e.t)}</span>`;
  return d;
}
let logFilter = null;
const ALL_EVENTS = [];
function handleEvents(evs) {
  const feed = $("#eventFeed"), log = $("#logList");
  for (const e of evs) {
    if (e.example) {
      caseCount[e.example] = (caseCount[e.example] || 0) + 1;
      caseLast[e.example] = e.t;
      if (e.vid) (caseVeh[e.example] = caseVeh[e.example] || new Set()).add(e.vid);
    }
    if (e.kind === "arrive") flashArrival(e.vid);
    if (e.example && e.vid) floatAt(e.vid, `${e.example} ${shortName(e.example)}`, PCOL[e.pattern]);
    ALL_EVENTS.push(e);
    if (ALL_EVENTS.length > 3000) ALL_EVENTS.shift();
    if (e.kind !== "world") {            // 주행 환경 사건(셀 진입 등)은 '사건 기록' 탭에만
      feed.prepend(evLine(e));
      while (feed.children.length > 120) feed.lastChild.remove();
    }
    if (!logFilter || e.pattern === logFilter) {
      log.prepend(evLine(e));
      while (log.children.length > 800) log.lastChild.remove();
    }
    if (e.example && e.pattern) toast(e);
  }
  if (evs.some((e) => e.example)) renderCases();
}
function shortName(ex) {
  const u = STATIC.usecases.find((u) => u[0] === ex);
  return u ? u[1].split(" (")[0] : "";
}
function toast(e) {
  const box = $("#toastBox");
  const t = el("div", "toast", `<b style="color:${PCOL[e.pattern]}">P${e.pattern} ${STATIC.patterns[e.pattern]} · ${e.example} ${shortName(e.example)}</b><br>${e.vid ? `V${e.vid} — ` : ""}${e.msg}`);
  t.style.borderLeftColor = PCOL[e.pattern];
  box.appendChild(t);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => t.remove(), 5000);
}

// ------------------------------------------------------------------ 유스케이스 표
function renderCases() {
  const box = $("#caseTable");
  const now = Date.now() / 1000;
  const total = Object.keys(caseCount).length;
  let html = `<p class="muted">39개 예시 중 이번 실행에서 <b style="color:#e6ebf5">${total}</b>개가 실제로 일어났습니다.
    행을 누르면 맵에서 그 장소로 이동합니다. 차량별 설정은 오른쪽 '차량 설정'에서 바꿉니다.</p>`;
  for (const [p, name] of Object.entries(STATIC.patterns)) {
    const rows = STATIC.usecases.filter((u) => u[0].startsWith(p + "-"));
    const sum = rows.reduce((a, u) => a + (caseCount[u[0]] || 0), 0);
    html += `<div class="ptitle"><span class="pt" style="background:${PCOL[p]}">P${p}</span>${name}<span class="muted small">${sum}회</span></div>
      <table class="cases"><tr><th style="width:44px">id</th><th>예시</th><th>근거</th><th>맵·설정에서</th><th style="width:56px">발생</th><th style="width:80px">차량</th></tr>`;
    for (const [id, ex, ev, where] of rows) {
      const n = caseCount[id] || 0, recent = caseLast[id] && now - caseLast[id] < 6;
      html += `<tr class="${recent ? "recent" : n ? "hit" : ""}" data-id="${id}"><td><b>${id}</b></td><td>${ex}</td><td class="muted">${ev}</td>
        <td>${where}</td><td class="cnt">${n || "-"}</td><td>${caseVeh[id] ? [...caseVeh[id]].sort((a, b) => a - b).map((v) => "V" + v).join(" ") : ""}</td></tr>`;
    }
    html += "</table>";
  }
  box.innerHTML = html;
  box.querySelectorAll("tr[data-id]").forEach((tr) => tr.onclick = () => FOCUS[tr.dataset.id] && focusOn(FOCUS[tr.dataset.id]));
}
function buildLogFilter() {
  const box = $("#logFilter");
  const mk = (p, text, col) => {
    const b = el("button", "btn small", text);
    if (col) b.style.borderColor = col;
    b.onclick = () => {
      logFilter = p;
      const log = $("#logList");
      log.innerHTML = "";
      for (const e of ALL_EVENTS.slice(-800)) if (!p || e.pattern === p) log.prepend(evLine(e));
    };
    box.appendChild(b);
  };
  mk(null, "전체");
  for (const [p, n] of Object.entries(STATIC.patterns)) mk(Number(p), `P${p} ${n}`, PCOL[p]);
}
function resetCounts() { for (const o of [caseCount, caseLast, caseVeh]) for (const k in o) delete o[k]; }

// ------------------------------------------------------------------ 실행 제어
$("#btnStart").onclick = async () => {
  await fetch("/api/control", { method: "POST", body: JSON.stringify({ action: "start" }) });
  evOffset = 0; lastRoundShown = -1; resetCounts(); ALL_EVENTS.length = 0;
  $("#roundFeed").innerHTML = ""; $("#eventFeed").innerHTML = ""; $("#logList").innerHTML = "";
  for (const k in CARS) { scene.remove(CARS[k].g); scene.remove(CARS[k].beam); delete CARS[k]; }
};
$("#btnStop").onclick = () => fetch("/api/control", { method: "POST", body: JSON.stringify({ action: "stop" }) });
$("#btnCrash").onclick = () => { if (confirm("집계 서버 프로세스를 강제 종료합니다 (9-1). SASAFL은 복구 경로가 없어 학습이 멈춥니다.")) fetch("/api/control", { method: "POST", body: JSON.stringify({ action: "server_crash" }) }); };
$("#btnReset").onclick = async () => { await fetch("/api/control", { method: "POST", body: JSON.stringify({ action: "reset_config" }) }); const r = await (await fetch("/api/state?ev=999999999")).json(); CFG = r.config; buildGlobalForm(); buildVehForm(); };
$("#btnView").onclick = fitView;
// 프리셋: 차량마다 다른 장애를 걸어 39개 예시가 한 맵에서 모두 일어나게 한다 (차량 번호는 K≥8 기준)
const PRESET_ALL = {
  net: { frag_fail: 0.03, doppler_fail: 0.25, region_instability: 0.5, cell_capacity: 1, tdma_groups: 2, tdma_slot_s: 3 },
  energy: { budget_j: 400, min_energy_j: 120 },
  world: { availability_mode: "cycle", availability_min: 0.5, availability_max: 1.0, availability_period_s: 300, ota_prob: 0.7, weather_on: true },
  vehicle_default: { crash_prob: 0.05 },
  vehicles: {
    // 오염(7-x)을 세게 걸면 global loss가 크게 올라 SASAFL quota가 K까지 오르고(→ 한 대만 빠져도 9-2 무한 대기),
    // 더 세면 loss NaN으로 서버가 죽는다(9-1). 데모가 오래 이어지도록 약하게 둔다.
    "1": { min_data: 300, data_rate_per_s: 3 },            // 3-2 데이터 부족
    "2": { store_forward: true, energy_gate: false },      // 5-2·5-3·5-4 보관 후 전송, 1-3·2-7 방전(충전소에서 회복)
    "3": { lr_override: 0.02 },                            // 7-1 학습률 오류
    "4": { store_forward: true, dup_prob: 0.4 },           // 통과 차량: 5-x, 9-3 중복
    "5": { label_noise: 0.3, ecu_factor: 2.2 },            // 7-2 라벨 오류(재시작), 4-1 느린 ECU
    "6": { backdoor_boost: 0.5, charging_only: true },     // 7-3 백도어, 3-3 충전 중만 (도심 순환: 충전소 지남)
    "7": { weight_noise: 0.05, clock_skew: 1 },            // 7-4 센서 고장, 7-5 시계 오차
    "8": { retransmit: true, dup_prob: 0.3 },              // 통과 차량: 2-2 도플러·4-3 재전송, 9-3
  },
};
const PRESET_CLEAN = {
  net: { frag_fail: 0.01, doppler_fail: 0.15, region_instability: 0.3, cell_capacity: 2, tdma_groups: 1 },
  energy: { budget_j: 1000, min_energy_j: 100 },
  world: { availability_mode: "off", ota_prob: 0.3 },
  vehicle_default: { crash_prob: 0.02 },
};
$("#presetAll").onclick = async () => {
  await postCfg(PRESET_ALL);
  $("#presetNote").textContent = "적용됨. 라벨 오류(V5)·에너지 예산은 다음 시작부터 반영됩니다 → ▶ 시작을 다시 누르세요. 9-1은 '서버 장애' 버튼, 8-2는 라벨이 쏠린 차량의 '맵에서 이탈'로 일으킵니다.";
  buildGlobalForm(); buildVehForm();
};
$("#presetClean").onclick = async () => {
  const clear = {};
  for (const k of Object.keys(CFG.vehicles || {})) clear[k] = null;
  await postCfg({ ...PRESET_CLEAN, vehicles: clear });
  $("#presetNote").textContent = "장애 설정을 껐습니다 (주행 환경에 따른 끊김은 그대로).";
  buildGlobalForm(); buildVehForm();
};
$("#optCells").onchange = (e) => layers.cells.forEach((m) => (m.visible = e.target.checked));
$("#optLabels").onchange = (e) => layers.labels.forEach((m) => (m.visible = e.target.checked));

function renderRun() {
  const procs = LAST.procs || {};
  const fl = procs.fl || "", chip = $("#runChip");
  const running = fl === "running";
  chip.className = "chip " + (running ? "run" : LAST.note ? "bad" : "");
  chip.textContent = running ? "학습 실행 중" : LAST.note ? "서버 장애" : fl ? `정지 (${fl})` : "정지";
  const el2 = LAST.world?.elapsed;
  $("#runInfo").textContent = (LAST.run ? `실행 ${LAST.run}` : "") + (el2 ? ` · ${Math.floor(el2 / 60)}분 ${Math.floor(el2 % 60)}초` : "") + (LAST.note ? ` · ${LAST.note}` : "");
  if (weatherGroup) { weatherGroup.visible = CFG.world.weather_on; weatherGroup.userData.label.visible = CFG.world.weather_on; }
}

async function poll() {
  try {
    const r = await (await fetch(`/api/state?ev=${evOffset}`)).json();
    LAST = r;
    CFG = r.config;
    evOffset = r.ev;
    renderRun();
    if (r.world?.vehicles) updateCars();
    renderVehPicker();
    renderVehInfo();
    renderFL();
    handleEvents(r.events || []);
  } catch (e) { console.error(e); }
  setTimeout(poll, 400);
}

(async function main() {
  STATIC = await (await fetch("/api/static")).json();
  const r = await (await fetch("/api/state?ev=0")).json();
  CFG = r.config;
  buildScene(STATIC);
  buildLegends();
  buildGlobalForm();
  buildVehForm();
  buildLogFilter();
  renderCases();
  resize();
  requestAnimationFrame(frame);
  poll();
})();

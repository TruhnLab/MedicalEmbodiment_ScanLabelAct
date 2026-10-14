// Task console (projects/navigation): the chosen room in 3D (mesh + labels, as in the mapping console's
// 3D view) with the live robot in it. The navigator (projects/navigation/navigator.py, started through
// the helper) finds the robot's position in the room and tracks it; this page shows it and sends the
// commands (room, localize, go to an object, stop).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { createLabels } from './labels3d.js';

const $ = (id) => document.getElementById(id);
const NAV = 'http://127.0.0.1:8767';
const HEADERS = { 'X-LanderPi-Console': '1', 'Content-Type': 'application/json' };
const POLL_MS = 300;
const ROBOT = { length: 0.30, width: 0.25, height: 0.14 };  // chassis (half diagonal 0.20 m, explore_core.py)
const LIDAR_Z = 0.09;  // m above the floor: base_link 0.054 + shell 0.018 + lidar 0.021 (tf_static)

const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private window */ } },
};

// ---------- 3D scene (like view3d.js: z-up world in a y-up scene, drawn only when something changed) ----------

const host = $('view3dHost');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080a);
const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
scene.add(new THREE.GridHelper(20, 20, 0x3a4150, 0x22262e));
scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.2));

// at most 30 draws per second and only after a change: the mesh kept the laptop's CPU busy otherwise
// (view3d.js, 2026-10-06)
const MAX_FPS = 30;
let dirty = true;
const redraw = () => { dirty = true; };
controls.addEventListener('change', redraw);
let lastTick = 0;
function frame(t) {
  requestAnimationFrame(frame);
  if (t - lastTick < 1000 / MAX_FPS - 2) return;
  lastTick = t;
  controls.update();
  if (!dirty) return;
  dirty = false;
  renderer.render(scene, camera);
  labels.frame();
}
function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  redraw();
}
new ResizeObserver(resize).observe(host);

const labels = createLabels({ world, camera, renderer, host, redraw });
labels.flyTo = (target, dist) => {
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, dist);
  redraw();
};

// ---------- the robot and its lidar scan ----------

// drawn on top of the mesh (no depth test): in the overview the 30 cm robot hid between the stools
const robot = new THREE.Group();
const robotMat = new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.9, depthTest: false });
const body = new THREE.Mesh(new THREE.BoxGeometry(ROBOT.length, ROBOT.width, ROBOT.height), robotMat);
body.position.z = ROBOT.height / 2;
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.06, 0.16, 16), robotMat);
nose.rotation.z = -Math.PI / 2;  // the cone points along +y; the robot's front is +x
nose.position.set(ROBOT.length / 2 + 0.08, 0, ROBOT.height / 2);
const ring = new THREE.Mesh(new THREE.RingGeometry(0.35, 0.42, 48), new THREE.MeshBasicMaterial(
  { color: 0xef4444, transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthTest: false }));
ring.position.z = 0.01;
robot.add(body, nose, ring);
robot.traverse((o) => { o.renderOrder = 10; });
robot.visible = false;
world.add(robot);

// lidar points in screen pixels, on top: they show at a glance whether the scan lies on the room's walls
const scanGeo = new THREE.BufferGeometry();
const scanPts = new THREE.Points(scanGeo, new THREE.PointsMaterial(
  { color: 0x7dd3fc, size: 3, sizeAttenuation: false, depthTest: false, transparent: true }));
scanPts.renderOrder = 9;
scanPts.visible = false;
world.add(scanPts);

// the drive (projects/navigation/goto.py): planned path (dashed) and the stop in front of the object
const pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial(
  { color: 0x22c55e, dashSize: 0.12, gapSize: 0.06, depthTest: false, transparent: true }));
pathLine.renderOrder = 8;
pathLine.visible = false;
const goalMark = new THREE.Group();
const goalMat = new THREE.MeshBasicMaterial({ color: 0x22c55e, depthTest: false, transparent: true, opacity: 0.8 });
const goalRing = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.2, 40), goalMat);
const goalArrow = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.14, 12), goalMat);
goalArrow.rotation.z = -Math.PI / 2;
goalArrow.position.set(0.2, 0, 0.02);
goalMark.add(goalRing, goalArrow);
goalMark.traverse((o) => { o.renderOrder = 8; });
goalMark.visible = false;
world.add(pathLine, goalMark);
let shownPath = '';

// finding things (live.py): the search area (orange outline on the floor) and the finds (yellow markers)
const areaLine = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial(
  { color: 0xf59e0b, depthTest: false, transparent: true }));
areaLine.renderOrder = 8;
areaLine.visible = false;
const finds = new THREE.Group();
world.add(areaLine, finds);
let shownArea = '';
let shownFinds = '';
const findMat = new THREE.MeshBasicMaterial({ color: 0xfacc15, depthTest: false, transparent: true, opacity: 0.95 });
const seenMat = new THREE.MeshBasicMaterial({ color: 0xc9ced6, depthTest: false, transparent: true, opacity: 0.85 });

// the detector's boxes over the camera picture (small and big, M): the relevant ones (what is looked for,
// confident, in the search area) thick yellow, found ones marked; the other candidates thin grey
const overlay = document.createElement('canvas');
overlay.className = 'det-overlay';
document.querySelector('.cam-view').append(overlay);

function drawOverlay(o) {
  const view = overlay.parentElement;
  const cw = view.clientWidth, ch = view.clientHeight, dpr = window.devicePixelRatio || 1;
  if (overlay.width !== Math.round(cw * dpr) || overlay.height !== Math.round(ch * dpr)) {
    overlay.width = Math.round(cw * dpr);
    overlay.height = Math.round(ch * dpr);
  }
  const ctx = overlay.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  if (!o || !o.boxes.length) return;
  // the picture is shown with object-fit: contain: scale and centre like it
  // shift_px: how far the picture moved sideways since the boxes' image (the robot turned meanwhile)
  const s = Math.min(cw / o.w, ch / o.h), ox = (cw - o.w * s) / 2 + (o.shift_px || 0) * s, oy = (ch - o.h * s) / 2;
  const fs = Math.max(10, Math.round(11 * Math.min(1.6, s)));  // the label grows a little with the picture
  ctx.font = `${fs}px Inter, system-ui, sans-serif`;
  for (const b of [...o.boxes].sort((a, c) => a.relevant - c.relevant)) {  // relevant ones on top
    const [x0, y0, x1, y1] = b.box.map((v, i) => (i % 2 ? oy + v * s : ox + v * s));
    if (b.outline?.length) {  // the detector's mask, tinted (user, 2026-10-09): one polygon per piece
      ctx.beginPath();
      for (const poly of b.outline) {
        poly.forEach(([x, y], i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, ox + x * s, oy + y * s));
        ctx.closePath();
      }
      ctx.setLineDash([]);
      ctx.fillStyle = b.relevant ? 'rgba(250, 204, 21, 0.3)' : 'rgba(231, 233, 238, 0.15)';
      ctx.fill();
      ctx.lineWidth = b.relevant ? 2 : 1;
      ctx.strokeStyle = b.relevant ? '#facc15' : 'rgba(231, 233, 238, 0.55)';
      ctx.stroke();
    }
    ctx.setLineDash(b.relevant ? [] : [4, 3]);
    ctx.lineWidth = b.relevant ? 3 : 1;
    ctx.strokeStyle = b.relevant ? '#facc15' : 'rgba(231, 233, 238, 0.55)';
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    // the image check's words (✓ fits, ✗ does not); none before it: the detector only knows the searched words
    const text = b.text ?? `${b.found ? '✓ ' : ''}${b.label} ${b.score.toFixed(2)}`;
    if (!text) continue;
    const tw = ctx.measureText(text).width + 8, th = fs + 5;
    ctx.fillStyle = b.relevant ? '#facc15' : 'rgba(10, 12, 16, 0.7)';
    ctx.fillRect(x0, Math.max(0, y0 - th), tw, th);
    ctx.fillStyle = b.relevant ? '#111' : '#e7e9ee';
    ctx.fillText(text, x0 + 4, Math.max(0, y0 - th) + fs);
  }
}

// The boxes come late anyway (detection ~0.2 s); polled with /state (300 ms) they waited another 0.15 s on
// average (user, 2026-10-09: "a bit of delay"). While the detector runs, /overlay alone every 100 ms.
const OVERLAY_POLL_MS = 50;
let overlayShown = 0;  // time of the overlay drawn: an older one (from /state) doesn't replace a newer one
function showOverlay(o) {
  if (o && o.time < overlayShown) return;
  overlayShown = o ? o.time : 0;
  drawOverlay(o);
}
async function pollOverlay() {
  if (state?.live?.images_per_s > 0) {
    try {
      showOverlay((await (await fetch(`${NAV}/overlay`, { cache: 'no-store' })).json()).overlay);
    } catch { /* the /state poll notices */ }
  }
  setTimeout(pollOverlay, OVERLAY_POLL_MS);
}
setTimeout(pollOverlay, OVERLAY_POLL_MS);  // after `state` below is declared

function showLive(live) {
  showOverlay(live?.overlay);
  const area = live?.looking_for?.length ? live.area : null;
  const aKey = JSON.stringify(area);
  if (aKey !== shownArea) {
    shownArea = aKey;
    areaLine.geometry.dispose();
    areaLine.geometry = new THREE.BufferGeometry().setFromPoints((area || []).map(([x, y]) => new THREE.Vector3(x, y, 0.02)));
  }
  areaLine.visible = !!area;
  const found = live?.found || [];
  // every confirmed sighting: finds in the search area yellow, those outside grey (numbered, for the chat)
  const seen = live?.seen || found.map((f) => ({ ...f, in_area: true }));
  const fKey = JSON.stringify(seen.map((f) => [f.id, f.position, f.in_area, f.verdict]));
  if (fKey !== shownFinds) {
    shownFinds = fKey;
    finds.clear();
    for (const f of seen) {
      const checking = f.verdict === 'pending' || f.verdict == null;
      const mat = f.in_area && !checking ? findMat : seenMat;  // grey until the image check said "fits"
      const m = new THREE.Mesh(new THREE.OctahedronGeometry(f.in_area ? 0.07 : 0.05), mat);
      m.position.set(f.position[0], f.position[1], Math.max(0.12, f.position[2] + 0.12));
      m.renderOrder = 9;
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.08, 0.11, 24), mat);
      ring.position.set(f.position[0], f.position[1], 0.015);
      ring.renderOrder = 9;
      finds.add(m, ring);
    }
  }
  const el = $('liveInfo');
  const looking = live?.looking_for?.length ? `Looking for ${live.looking_for.join(', ')}`
    + `${live.simulated ? ' (simulated camera)' : ` · ${live.images_per_s} images/s`}` : '';
  const at = (f) => `#${f.id ?? '?'} ${f.verdict === 'yes' && f.what ? f.what : f.label} at (${f.position[0].toFixed(2)}, ${f.position[1].toFixed(2)})`;
  const isChecking = (f) => f.verdict === 'pending';
  const found_ = seen.filter((f) => f.in_area && !isChecking(f)).map(at).join(', ');
  const checking = seen.filter(isChecking).map(at).join(', ');
  const other = seen.filter((f) => !f.in_area && !isChecking(f)).map(at).join(', ');
  el.textContent = [looking, found_ ? `found: ${found_}` : '', checking ? `checking: ${checking}` : '',
    other ? `also seen: ${other}` : '',
    live?.error ? `detection error: ${live.error}` : ''].filter(Boolean).join(' · ');
}

// ---------- rooms ----------

let rooms = [];
let room = null;           // the shown room
let mesh = null;
const loader = new PLYLoader();

function info(text) { $('view3dInfo').textContent = text; }
function progress(v) {  // null hidden, 'busy', or 0..1
  const bar = $('view3dProgress');
  bar.hidden = v === null;
  bar.classList.toggle('busy', v === 'busy');
  bar.firstElementChild.style.width = typeof v === 'number' ? `${(100 * v).toFixed(1)}%` : '';
}

async function refreshRooms() {
  try {
    rooms = (await (await fetch('/api/rooms', { cache: 'no-store' })).json()).rooms;
  } catch {
    info('Cannot list rooms: open the task console with start_task_console.bat.');
    return;
  }
  const sel = $('roomSelect');
  sel.innerHTML = '';
  for (const r of rooms) {
    const tags = [r.labels ? (r.labels_stale ? 'labels outdated' : `${r.objects ?? '?'} objects`) : 'no labels yet'];
    if (!r.mesh) tags.push('no 3D model');
    sel.append(new Option(`${r.title || r.name} · ${tags.join(', ')}`, r.name));
  }
  $('roomDeleteBtn').disabled = !rooms.length;
  if (!rooms.length) {
    room = null;
    info('No rooms yet: in the mapping console\'s 3D view, choose a labelled session and press "Use as room".');
    return;
  }
  // the navigator's room first (the robot is in it; another tab may have chosen it), then the last one here
  const keep = state?.room || room?.name || store.get('tasks.room');
  sel.value = rooms.some((r) => r.name === keep) ? keep : rooms[0].name;
  // tell the navigator if it has no room, or one that was removed meanwhile
  const send = !state?.room || !rooms.some((r) => r.name === state.room);
  if (!room || room.name !== sel.value) await loadRoom(rooms.find((r) => r.name === sel.value), send);
}

$('roomDeleteBtn').addEventListener('click', async () => {
  const r = rooms.find((x) => x.name === $('roomSelect').value);
  if (!r || !confirm(`Remove the room "${r.title || r.name}" from the task console?\n\n`
    + `The recording ${r.session} and its labels stay; "Use as room" in the mapping console brings it back.`)) return;
  try {
    const res = await (await fetch(`/api/room/delete/${r.name}`, { method: 'POST', headers: { 'X-LanderPi-Console': '1' } })).json();
    if (!res.ok) throw new Error(res.error);
  } catch (e) {
    info(`Could not remove the room: ${e.message}`);
    return;
  }
  if (room?.name === r.name) {
    room = null;
    labels.clear();
    if (mesh) { world.remove(mesh); mesh.geometry.dispose(); mesh = null; }
  }
  await refreshRooms();
});

$('roomSelect').addEventListener('change', () => loadRoom(rooms.find((r) => r.name === $('roomSelect').value), true));

async function fetchPly(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get('Content-Length')) || 0;
  const parts = [];
  let got = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    info(`Loading the room: ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ''} MB`);
    progress(total ? got / total : 'busy');
  }
  const buf = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { buf.set(p, at); at += p.length; }
  info('Loading the room: preparing for display…');
  progress('busy');
  await new Promise((r) => setTimeout(r, 30));
  const geo = loader.parse(buf.buffer);
  // camera colours are sRGB; three.js treats vertex colours as linear (view3d.js)
  const c = geo.getAttribute('color');
  if (c) {
    const a = c.array;
    for (let i = 0; i < a.length; i++) a[i] = a[i] <= 0.04045 ? a[i] / 12.92 : ((a[i] + 0.055) / 1.055) ** 2.4;
  }
  return geo;
}

// send: tell the navigator (the user chose this room); otherwise the page just follows the navigator
async function loadRoom(r, send) {
  if (!r) return;
  room = r;
  $('roomSelect').value = r.name;
  store.set('tasks.room', r.name);
  labels.clear();
  if (mesh) { world.remove(mesh); mesh.geometry.dispose(); mesh = null; }
  if (send) sendRoom();
  if (!r.mesh) { info(`${r.name}: the session has no 3D model.`); return; }
  try {
    const geo = await fetchPly(`/data/sessions/${r.session}/reconstruction/mesh_view.ply`);
    if (room !== r) return;  // another room was chosen meanwhile
    mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    world.add(mesh);
    progress(null);
    resetView();
    if (r.labels && !r.labels_stale) {
      await labels.load(r.session, `${r.session}:${r.labels_time}`, mesh);
      labels.show(showing === 'labels');
    }
    info(`${r.title || r.name}: ${r.labels && !r.labels_stale ? `${labels.count} objects. Click one (or its tag) for details.`
      : 'no labels for this 3D model yet (Label it in the mapping console).'}`);
  } catch (e) {
    progress(null);
    info(`Could not load the room: ${e.message}`);
  }
}

function resetView() {
  const box = new THREE.Box3().setFromObject(mesh || world);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.35 * size, 0.55 * size, 0.55 * size));
  camera.near = size / 1000;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  redraw();
}
$('resetViewBtn').addEventListener('click', resetView);

let showing = 'labels';
function setShow(what) {
  showing = what;
  $('showMesh').classList.toggle('active', what === 'mesh');
  $('showLabels').classList.toggle('active', what === 'labels');
  labels.show(what === 'labels');
}
$('showMesh').addEventListener('click', () => setShow('mesh'));
$('showLabels').addEventListener('click', () => setShow('labels'));
$('showScan').addEventListener('change', redraw);

// ---------- selected object: "Go to" (the navigator drives there, goto.py) ----------

labels.onSelect = () => showGo();

const DRIVING = ['planning', 'driving', 'docking', 'turning'];
const driving = () => DRIVING.includes(state?.drive?.state);

// ---------- pin & go: a point on the floor instead of an object ----------

let pinMode = false;
let pin = null;  // [x, y] in the map
const pinMark = new THREE.Group();
const pinMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b, depthTest: false, transparent: true, opacity: 0.9 });
const pinHead = new THREE.Mesh(new THREE.SphereGeometry(0.06, 16, 12), pinMat);
pinHead.position.z = 0.35;
const pinStem = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.3, 8), pinMat);
pinStem.rotation.x = Math.PI / 2;  // cylinders stand along y; the world is z-up
pinStem.position.z = 0.17;
const pinFoot = new THREE.Mesh(new THREE.RingGeometry(0.06, 0.09, 24), pinMat);
pinMark.add(pinHead, pinStem, pinFoot);
pinMark.traverse((o) => { o.renderOrder = 9; });
pinMark.visible = false;
world.add(pinMark);

function setPinMode(on) {
  pinMode = on;
  labels.clickable = !on;  // labels3d.js: no object selection while clicks place the pin
  $('pinBtn').classList.toggle('active', on);
  document.body.classList.toggle('pin-mode', on);
  if (!on) { pin = null; pinMark.visible = false; }
  redraw();
  showGo();
}
$('pinBtn').addEventListener('click', () => setPinMode(!pinMode));

// a click (not a drag) on the room in pin mode: the floor point under it. The event goes on to the view's
// OrbitControls: stopping it (to keep labels3d.js from selecting an object) left them in drag mode after
// every pin (user, 2026-10-07); labels.clickable does that now
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);  // the floor (z = 0) in the y-up scene
const pinRay = new THREE.Raycaster();
let pinDown = null;
renderer.domElement.addEventListener('pointerdown', (e) => { pinDown = [e.clientX, e.clientY]; }, true);
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!pinMode || !pinDown || Math.hypot(e.clientX - pinDown[0], e.clientY - pinDown[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  pinRay.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
  const hit = pinRay.ray.intersectPlane(floorPlane, new THREE.Vector3());
  if (!hit) return;
  const p = world.worldToLocal(hit.clone());
  pin = [p.x, p.y];
  pinMark.position.set(p.x, p.y, 0);
  pinMark.visible = true;
  redraw();
  showGo();
}, true);

// "Go to <object / the pin>"; "Cancel" while the robot drives. Needs Connect, Armed and a localized robot
// (the note says what is missing).
function goTarget() {
  if (pinMode) return pin ? { pin, label: 'the pin' } : null;
  const o = labels.selected;
  return o ? { id: o.id, label: o.label } : null;
}

// The button turns into Cancel whenever a drive starts, also one of the chat: a click meant for "Go to ..." just
// then, or Enter while the button still had the focus, cancelled the chat's drive (twice, 2026-10-09)
const CANCEL_ARM_MS = 800;
let cancelShownAt = 0;

function showGo() {
  const t = goTarget();
  const d = state?.drive;
  $('goBox').hidden = !t && !driving() && !(pinMode && pageOnline());
  const btn = $('goBtn');
  if (driving()) {
    if (!btn.classList.contains('danger')) cancelShownAt = Date.now();
    btn.textContent = `Cancel (to ${d.target.label})`;
    btn.classList.add('danger');
    btn.disabled = false;
    $('goNote').textContent = d.message;
    return;
  }
  btn.classList.remove('danger');
  if (!t) {
    btn.textContent = 'Go to the pin';
    btn.disabled = true;
    $('goNote').textContent = 'Click a spot on the floor.';
    return;
  }
  btn.textContent = `Go to ${t.label}`;
  const missing = !pageOnline() ? 'Connect first.' : !$('armed').checked ? 'Arm first (Armed, top right).'
    : state?.status !== 'localized' ? 'Localize first (⟳ Localize: one turn on the spot).' : '';
  btn.disabled = !!missing;
  const same = d && (t.pin ? d.target.id === null : d.target.id === t.id);
  $('goNote').textContent = missing || (same ? d.message : '');
}

$('goBtn').addEventListener('click', async () => {
  $('goBtn').blur();  // keys must not press it later
  if (driving()) {
    if (Date.now() - cancelShownAt < CANCEL_ARM_MS) return;  // it just turned into Cancel: the click meant Go
    nav('/stop', { reason: 'console: Cancel button' }).catch(() => {});
    return;
  }
  const t = goTarget();
  if (!t) return;
  try {
    await nav('/goto', t.pin ? { x: t.pin[0], y: t.pin[1] } : { object: t.id });
    if (t.pin) setPinMode(false);  // the pin is handed over (the green stop marker shows it); user, 2026-10-08
  } catch (e) {
    $('goNote').textContent = `Could not start: ${e.message}`;
  }
});

function showDrive(d) {
  const path = d && DRIVING.includes(d.state) ? d.path : [];
  const key = JSON.stringify(path);
  if (key !== shownPath) {
    shownPath = key;
    pathLine.geometry.dispose();
    pathLine.geometry = new THREE.BufferGeometry().setFromPoints(path.map(([x, y]) => new THREE.Vector3(x, y, 0.03)));
    pathLine.computeLineDistances();
  }
  pathLine.visible = path.length > 1;
  goalMark.visible = !!(d?.goal && DRIVING.concat(['arrived']).includes(d.state));
  if (goalMark.visible) {
    goalMark.position.set(d.goal[0], d.goal[1], 0.02);
    goalMark.rotation.z = d.goal[2];
  }
}

// ---------- navigator ----------

async function nav(path, body) {
  const r = await fetch(NAV + path, { method: 'POST', headers: HEADERS, body: JSON.stringify(body || {}) });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || r.status);
  return j;
}

// The navigator's room is the truth (one robot, one room): a page only sends a room when the user picks
// one, or when the navigator has none (just started). Re-sending whenever they differed made two open
// task consoles switch the navigator between their rooms twice a second (2026-10-07).
let sentRoom = null;  // the room sent last, until the navigator reports it
function sendRoom() {
  if (!room) return;
  sentRoom = room.name;
  nav('/room', { name: room.name }).catch(() => { sentRoom = null; /* not up yet: showState() retries */ });
}

const helperPost = (path) => fetch(path, { method: 'POST', headers: { 'X-LanderPi-Console': '1' } });
async function startNavigator() {
  try { await helperPost('/api/navigator/start'); } catch { /* poll() shows it */ }
}

// Connect (app.js, top bar) also starts localization: the navigator is started when this page is online
// and stopped when the user disconnects. Before, the navigator connected by itself in the background and
// it was unclear whether and when the robot was connected (user, 2026-10-07).
const pageOnline = () => $('status').dataset.state === 'ok';
let wasOnline = false;
new MutationObserver(() => {
  const st = $('status').dataset.state;
  if (st === 'ok' && !wasOnline) {
    wasOnline = true;
    startNavigator();
  } else if (st === 'off' && wasOnline) {  // Disconnect pressed (a lost link shows "Reconnecting…")
    wasOnline = false;
    helperPost('/api/navigator/stop').catch(() => {});
  }
  showState();
}).observe($('status'), { attributes: true, attributeFilter: ['data-state'] });

const STATUS = {  // navigator status -> [pill state, text]
  'no room': ['off', 'Localization: no room'], idle: ['off', 'Not localized'], turning: ['busy', 'Localizing: turning…'],
  searching: ['busy', 'Localizing…'], localized: ['ok', 'Localized'], uncertain: ['warn', 'Localization uncertain'],
  ambiguous: ['warn', 'Localization not sure'], failed: ['error', 'Localization failed'],
};

let state = null;
let offlineSince = null;
async function pollOnce() {
  try {
    state = await (await fetch(`${NAV}/state`, { cache: 'no-store' })).json();
    offlineSince = null;
  } catch {
    state = null;
    offlineSince ??= Date.now();
    // online but no navigator (e.g. stopped from another tab): start it again
    if (pageOnline() && Date.now() - offlineSince > 8000) { offlineSince = Date.now(); startNavigator(); }
  }
  showState();
}
async function poll() {
  await pollOnce();
  setTimeout(poll, POLL_MS);
}

function showState() {
  const pill = $('navStatus');
  const s = state;
  if (!s) {
    pill.dataset.state = pageOnline() ? 'busy' : 'off';
    pill.textContent = pageOnline() ? 'Localization starting…' : 'Localization off · Connect first';
  } else if (!s.connected) {
    pill.dataset.state = 'error';
    pill.textContent = 'Localization: robot not reachable';
  } else {
    const [st, text] = STATUS[s.status] || ['off', s.status];
    pill.dataset.state = st;
    pill.textContent = s.status === 'turning' && s.turned_deg !== null ? `Turning… ${s.turned_deg}°` : text;
  }
  if (s && sentRoom && s.room === sentRoom) sentRoom = null;
  if (s && room && !s.room && !sentRoom) sendRoom();  // the navigator was (re)started
  if (s?.room && room && s.room !== room.name && !sentRoom) {  // another tab chose a room: follow it
    const r = rooms.find((x) => x.name === s.room);
    if (r) loadRoom(r, false);
  }

  const msg = $('navMessage');
  msg.textContent = s ? s.message : '';
  msg.classList.toggle('warn', !!s && ['uncertain', 'ambiguous', 'failed'].includes(s.status));
  const sure = s && ['localized', 'turning', 'searching'].includes(s.status);
  $('navPose').textContent = s?.pose
    ? `x ${s.pose[0].toFixed(2)} m  y ${s.pose[1].toFixed(2)} m  heading ${(s.pose[2] * 180 / Math.PI).toFixed(0)}°`
      + `${s.score !== null ? `  fit ${s.score.toFixed(2)}` : ''}${s.cmd_topic !== '/controller/cmd_vel' ? `  (dry run: ${s.cmd_topic})` : ''}`
    : '';

  robot.visible = !!s?.pose;
  if (s?.pose) {
    robot.position.set(s.pose[0], s.pose[1], 0);
    robot.rotation.z = s.pose[2];
    robotMat.opacity = sure ? 0.9 : 0.4;  // unsure: shown faintly
    if ($('followRobot').checked) {
      const p = new THREE.Vector3(s.pose[0], s.pose[1], 0).applyMatrix4(world.matrixWorld);
      camera.position.add(p.clone().sub(controls.target));
      controls.target.copy(p);
    }
  }
  scanPts.visible = !!s?.scan && $('showScan').checked;
  if (s?.scan) {
    const a = new Float32Array(s.scan.length * 3);
    s.scan.forEach(([x, y], i) => { a[3 * i] = x; a[3 * i + 1] = y; a[3 * i + 2] = LIDAR_Z; });
    scanGeo.setAttribute('position', new THREE.BufferAttribute(a, 3));
  }
  const busy = s && ['turning', 'searching'].includes(s.status);
  $('localizeTurnBtn').disabled = !s?.connected || !room || busy || !$('armed').checked;
  showDrive(s?.drive);
  showLive(s?.live);
  showRecording(s);
  showChatLive();
  showGo();
  redraw();
}

async function localize() {
  try {
    await nav('/localize');
  } catch (e) {
    info(`Could not localize: ${e.message}`);
  }
}
$('localizeTurnBtn').addEventListener('click', localize);

// STOP, Space, un-arming and manual driving (app.js) pause automatic motion with this event, as they pause
// the mapping console's exploration: here they stop a localization turn
document.addEventListener('landerpi:pause-auto', (e) => {
  if (state && (['turning', 'searching'].includes(state.status) || driving())) {
    nav('/stop', { reason: `console: ${e.detail || 'pause'}` }).catch(() => {});
  }
});
$('armed').addEventListener('change', showState);  // the turn button needs Armed

// ---------- big view: room or camera (M, or a click on the small camera), like map / drive view ----------

function toggleBigView() {
  document.body.classList.toggle('task-camera');
  resize();
}
document.querySelector('.cam-view').addEventListener('click', () => {
  if (!document.body.classList.contains('task-camera')) toggleBigView();
});
const roomViewBtn = document.createElement('button');  // the way back from the big camera (besides M)
roomViewBtn.id = 'roomViewBtn';
roomViewBtn.className = 'chip';
roomViewBtn.title = 'Back to the room (M)';
roomViewBtn.textContent = '◱ Room view';
roomViewBtn.addEventListener('click', (e) => { e.stopPropagation(); toggleBigView(); });
document.querySelector('.cam-overlay').prepend(roomViewBtn);
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input:not([type]), input[type="text"], input[type="number"], textarea, select')) return;
  if (e.key.toLowerCase() === 'm' && !e.ctrlKey && !e.repeat) toggleBigView();
});

// ---------- chat (assistant.py in the navigator) ----------

const CHAT_INTRO = 'Ask me to drive somewhere, e.g. “drive to the cabinet near the sink” or “go to the door”. '
  + 'I need Connect, Armed and a localized robot to move.';
let chatBusy = false;
const CHAT_POLL_MS = 500;  // the steps while the chat works
let chatShown = -1;  // messages shown, to follow other tabs

const escHtml = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
// the model writes a little Markdown: **bold**, lists, line breaks
const md = (t) => escHtml(t).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>');

function chatBubble(m) {
  const div = document.createElement('div');
  if (m.role === 'tool') {
    const r = m.result;
    div.className = `chat-msg tool${r?.error ? ' bad' : ''}`;
    const out = !r ? ' …' : r.error ? ` → ${r.error}` : r.message ? ` → ${r.message}` : r.result ? ` → ${r.result}` : '';
    div.textContent = `⚙ ${m.text}${out}`;
  } else if (m.role === 'thought') {
    div.className = 'chat-msg thought';
    div.innerHTML = md(m.text);
  } else {
    div.className = `chat-msg ${m.role === 'user' ? 'user' : m.role === 'note' ? 'note' : 'bot'}`;
    div.innerHTML = md(m.text);
  }
  return div;
}

// Each answer's way there (the model's notes, its tool calls and their results) in a steps box: open and filling
// while the chat works, collapsed to "▸ n steps" once the answer is there (user, 2026-10-09: no feedback about
// what the robot intends). The final answer is the turn's last message of the model after its last tool call.
function turnsOf(messages) {
  const turns = [];
  for (const m of messages) {
    if (m.role === 'user' || m.role === 'note' || !turns.length) turns.push({ head: m.role === 'user' || m.role === 'note' ? m : null, rest: [] });
    else turns[turns.length - 1].rest.push(m);
    if (m.role === 'note') turns.push({ head: null, rest: [] });
  }
  return turns;
}

function stepsBox(steps, working) {
  const box = document.createElement('details');
  box.className = `chat-steps${working ? ' working' : ''}`;
  box.open = working;
  const sum = document.createElement('summary');
  const tools = steps.filter((m) => m.role === 'tool').length;
  sum.textContent = working ? `Working… ${tools ? `(${tools} step${tools > 1 ? 's' : ''})` : ''}`
    : `${tools} step${tools === 1 ? '' : 's'}`;
  box.append(sum);
  for (const m of steps) box.append(chatBubble(m.role === 'assistant' ? { ...m, role: 'thought' } : m));
  if (working) {
    const live = document.createElement('div');
    live.className = 'chat-msg live';
    live.id = 'chatLive';
    box.append(live);
  }
  return box;
}

function renderChat(messages, working = false) {
  const log = $('chatLog');
  const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  const openBoxes = new Set([...log.querySelectorAll('details.chat-steps[open]:not(.working)')].map((d) => d.dataset.turn));
  log.innerHTML = '';
  log.append(chatBubble({ role: 'assistant', text: CHAT_INTRO }));
  const turns = turnsOf(messages);
  turns.forEach((t, i) => {
    if (t.head) log.append(chatBubble(t.head));
    if (!t.rest.length && !(working && i === turns.length - 1)) return;
    const busy = working && i === turns.length - 1;
    const lastTool = t.rest.map((m) => m.role).lastIndexOf('tool');
    const finals = busy ? [] : t.rest.filter((m, k) => k > lastTool && m.role === 'assistant');
    const steps = t.rest.filter((m) => !finals.includes(m));
    if (steps.length || busy) {
      const box = stepsBox(steps, busy);
      box.dataset.turn = String(i);
      if (openBoxes.has(String(i))) box.open = true;
      log.append(box);
    }
    for (const m of finals) log.append(chatBubble(m));
  });
  chatShown = messages.length;
  showChatLive();
  if (atEnd || working) log.scrollTop = log.scrollHeight;
}

// while the chat works: what the robot does right now (the drive's or search's message), under the steps
function showChatLive() {
  const el = $('chatLive');
  if (!el) return;
  const d = state?.drive;
  const live = state?.live;
  const parts = [];
  if (d && DRIVING.includes(d.state)) parts.push(`🚗 ${d.message}`);
  // a tool's progress (manipulation's pick: "following the red cube: 12 cm to drive", "closing the gripper")
  const n = state?.last_note;
  if (n && n.kind === 'manipulation' && Date.now() / 1000 - n.time < 5) parts.push(`🦾 ${n.text}`);
  if (live?.looking_for?.length) parts.push(`👁 looking for ${live.looking_for.join(', ')}`);
  const checking = (live?.seen || []).filter((f) => f.verdict === 'pending').length;
  if (checking) parts.push(`🔍 checking ${checking} candidate${checking > 1 ? 's' : ''}`);
  el.textContent = parts.join(' · ') || 'thinking…';
}

async function loadChat() {
  try {
    const r = await (await fetch(`${NAV}/chat`, { cache: 'no-store' })).json();
    if (chatBusy || r.messages.length !== chatShown) renderChat(r.messages, chatBusy);
  } catch { /* navigator not running: showChatState() says so */ }
}

function showChatState() {
  const ok = pageOnline() && !!state;
  // the field stays usable while the robot works (only sending waits): disabled, typed keys went to the
  // page, where W A S D / Q E / Space drive and stop the robot (app.js)
  $('chatInput').disabled = !ok;
  $('chatSend').disabled = !ok || chatBusy;
  $('chatInput').placeholder = !pageOnline() ? 'Connect first' : !state ? 'Starting…'
    : chatBusy ? 'Working on it…' : 'Ask the robot…';
}

$('chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('chatInput').value.trim();
  if (!text || chatBusy) return;
  $('chatInput').value = '';
  const log = $('chatLog');
  log.append(chatBubble({ role: 'user', text }));
  log.append(stepsBox([], true));  // filled by the polls below while the model works
  showChatLive();
  log.scrollTop = log.scrollHeight;
  chatBusy = true;
  showChatState();
  const poll = setInterval(loadChat, CHAT_POLL_MS);
  let error = null;
  try {
    // waits until the model is done, including drives it started (their progress shows in the room view)
    const r = await (await fetch(`${NAV}/chat`, { method: 'POST', headers: HEADERS,
      body: JSON.stringify({ message: text, armed: $('armed').checked }) })).json();
    if (!r.ok) error = r.error;
  } catch (err) {
    error = `The navigator did not answer (${err.message}).`;
  }
  clearInterval(poll);
  chatBusy = false;
  await loadChat();  // the whole turn, its steps collapsed
  if (error) log.append(chatBubble({ role: 'note', text: error }));
  showChatState();
  log.scrollTop = log.scrollHeight;
  $('chatInput').focus();
});

$('chatResetBtn').addEventListener('click', async () => {
  try { await nav('/chat/reset'); } catch { /* not running */ }
  renderChat([]);
});

renderChat([]);
setInterval(() => { showChatState(); if (state) loadChat(); }, 2000);

resize();
requestAnimationFrame(frame);
(async () => {
  // the navigator's room decides which room to show, so ask a running navigator before choosing one
  for (let i = 0; i < 3 && !state; i++) {
    await pollOnce();
    if (!state) await new Promise((r) => setTimeout(r, 500));
  }
  await refreshRooms();
  poll();
})();

// ---------- recording (● Record) and the recorded runs ----------

function showRecording(s) {
  const btn = $('recordBtn');
  const rec = s?.recording;
  btn.classList.toggle('recording', !!rec);
  btn.disabled = !s?.connected;
  btn.textContent = rec ? `■ Stop recording ${Math.floor(rec.seconds / 60)}:${String(rec.seconds % 60).padStart(2, '0')}` : '● Record';
}

$('recordBtn').addEventListener('click', async () => {
  try {
    await nav(state?.recording ? '/record/stop' : '/record/start');
  } catch (e) {
    info(`Could not ${state?.recording ? 'stop' : 'start'} recording: ${e.message}`);
  }
});

const helperPostJson = (path) => fetch(path, { method: 'POST', headers: { 'X-LanderPi-Console': '1' } }).then((r) => r.json());
let runsTimer = null;

async function showRuns() {
  let runs;
  try {
    runs = (await (await fetch('/api/task_runs', { cache: 'no-store' })).json()).runs;
  } catch {
    $('runsInfo').textContent = 'The helper does not list runs (restart start_task_console.bat).';
    return;
  }
  const list = $('runList');
  list.innerHTML = '';
  const playing = $('runVideo').dataset.run;
  for (const r of runs) {
    const row = document.createElement('div');
    row.className = `run-row${r.name === playing ? ' playing' : ''}`;
    const when = new Date((r.start || 0) * 1000);
    const dur = r.duration_s ? `${Math.floor(r.duration_s / 60)}:${String(Math.round(r.duration_s % 60)).padStart(2, '0')}` : '?';
    row.innerHTML = `${r.thumb ? `<img src="/data/task_runs/${r.name}/thumb.jpg" alt="">` : '<div class="nothumb"></div>'}
      <div class="what"><b>${when.toLocaleString()}</b> · ${escHtml(r.room || '')} · ${dur} · ${r.size_mb ?? '?'} MB${r.keep ? ' · ★ kept' : ''}
        <div>${escHtml((r.requests || []).join(' / ') || 'no chat requests')}</div>
        <div>${(r.finds || []).length ? escHtml(r.finds.join(' / ')) : 'nothing found'}${r.video ? '' : ' · video being made…'}</div></div>
      <div class="acts">
        <button class="chip" data-a="play" title="Play" ${r.video ? '' : 'disabled'}>▶</button>
        <button class="chip" data-a="keep" title="${r.keep ? 'Unstar (allow deleting)' : 'Keep (star, protects from deleting)'}">${r.keep ? '★' : '☆'}</button>
        <button class="chip delete" data-a="delete" title="Delete this run" ${r.keep ? 'disabled' : ''}>🗑</button>
      </div>`;
    row.querySelector('[data-a="play"]').onclick = () => {
      const v = $('runVideo');
      v.hidden = false;
      v.dataset.run = r.name;
      v.src = `/data/task_runs/${r.name}/video.mp4`;
      v.play().catch(() => {});
      showRuns();
    };
    row.querySelector('[data-a="keep"]').onclick = async () => { await helperPostJson(`/api/task_runs/keep/${r.name}?on=${r.keep ? 0 : 1}`); showRuns(); };
    row.querySelector('[data-a="delete"]').onclick = async () => {
      if (!confirm(`Delete the run of ${when.toLocaleString()} (${r.size_mb ?? '?'} MB)?`)) return;
      const res = await helperPostJson(`/api/task_runs/delete/${r.name}`);
      if (!res.ok) alert(res.error);
      if ($('runVideo').dataset.run === r.name) { $('runVideo').removeAttribute('src'); $('runVideo').hidden = true; }
      showRuns();
    };
    list.append(row);
  }
  const total = runs.reduce((a, r) => a + (r.size_mb || 0), 0);
  $('runsInfo').textContent = runs.length ? `${runs.length} runs, ${total.toFixed(0)} MB (data/task_runs)` : 'No runs recorded yet: ● Record.';
}

$('runsBtn').addEventListener('click', () => {
  $('runsDialog').showModal();
  showRuns();
  clearInterval(runsTimer);
  runsTimer = setInterval(() => { if ($('runsDialog').open) showRuns(); else clearInterval(runsTimer); }, 4000);
});
$('runsDialog').addEventListener('close', () => { $('runVideo').pause(); });

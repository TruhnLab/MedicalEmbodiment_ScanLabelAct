// 3D view (projects/mapping, M3): shows the reconstruction of a recorded session (mesh or point
// cloud, plus the camera path) and starts reconstructions on the laptop via the helper. Labels
// (projects/scene_understanding, labels3d.js): the session's objects on the mesh, after Label.
// Uses $ and helperFetch from app.js.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { createLabels } from './labels3d.js';

const host = $('view3dHost');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio || 1);
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080a);
const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
// the glide after a drag is computed per update: 0.1 at 30 updates per second (MAX_FPS below) feels like
// the default 0.05 at 60, and ends twice as soon
controls.dampingFactor = 0.1;

// ROS maps are z-up, three.js is y-up: everything from the session goes into this rotated group
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);
const grid = new THREE.GridHelper(20, 20, 0x3a4150, 0x22262e);  // 1 m grid on the floor (z = 0)
scene.add(grid);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 2.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.0);
sun.position.set(3, 8, 5);
scene.add(sun);

// Drawn only when something changed (view moved, model loaded, labels changed), at most 30 times per
// second: drawing the ~700k-triangle mesh 60 times a second kept the laptop's CPU busy (no GPU) for
// as long as the 3D tab was open, even with nothing moving (2026-10-06).
const MAX_FPS = 30;
let dirty = true;
const redraw = () => { dirty = true; };
controls.addEventListener('change', redraw);  // also while the view still glides after a drag (damping)

const loader = new PLYLoader();
const labels = createLabels({ world, camera, renderer, host, redraw });
labels.flyTo = (target, dist) => {
  const dir = camera.position.clone().sub(controls.target).normalize();
  controls.target.copy(target);
  camera.position.copy(target).addScaledVector(dir, dist);
  redraw();
};
const state = { session: null, mesh: null, points: null, path: null, show: 'mesh' };
let sessions = [];

// ---------- rendering only while the 3D view is visible ----------

let running = false;
let lastTick = 0;
let lastDraw = 0;
function frame(t) {
  if (!document.body.classList.contains('mode-3d')) { running = false; return; }
  requestAnimationFrame(frame);
  if (t - lastTick < 1000 / MAX_FPS - 2) return;  // the browser calls ~60 times a second
  lastTick = t;
  controls.update();  // fires 'change' while the view moves
  // one draw per second anyway, in case a change was not reported
  if (!dirty && t - lastDraw < 1000) return;
  dirty = false;
  lastDraw = t;
  renderer.render(scene, camera);
  labels.frame();
}

function resize() {
  const w = host.clientWidth;
  const h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  redraw();
}
new ResizeObserver(resize).observe(host);

new MutationObserver(() => {
  if (document.body.classList.contains('mode-3d') && !running) {
    running = true;
    resize();
    refreshSessions();
    requestAnimationFrame(frame);
  }
}).observe(document.body, { attributes: true, attributeFilter: ['class'] });

// ---------- sessions ----------

// overall fraction of a reconstruction from reconstruct.py's progress.json (equal share per stage)
function fraction(p) {
  if (!p) return 0;
  return Math.min(1, (p.step - 1 + (p.total ? p.done / p.total : 0)) / p.steps);
}

// a dropdown entry: the session name, then small status indicators (muted, see map_view.css; browsers
// without the customizable <select> show them as plain text)
function sessionOption(s) {
  const dur = s.duration_s ? `${Math.round(s.duration_s)} s` : '?';
  const opt = new Option(`${s.name} (${dur}, ${s.keyframes} kf${s.slam ? '' : ', no SLAM'})`, s.name);
  const tags = [];
  if (s.reconstructing) tags.push([`reconstructing ${Math.round(100 * fraction(s.progress))} %`, 'busy']);
  else if (s.failed) tags.push(['reconstruction failed', 'bad']);
  else if (s.mesh) tags.push(['3D ready', '']);
  if (s.labeling) tags.push([`labelling ${Math.round(100 * fraction(s.label_progress))} %`, 'busy']);
  else if (s.label_failed) tags.push(['labelling failed', 'bad']);
  else if (s.labels && !s.labels_stale) tags.push(['Labels ready', '']);
  for (const [text, kind] of tags) {
    const span = document.createElement('span');
    span.className = `ind ${kind}`;
    span.textContent = ` · ${text}`;
    opt.append(span);
  }
  return opt;
}

// what the view shows, e.g. "<session>:<mesh time>"; a change (new session, a finished
// reconstruction) loads the model again
let shown = null;
let pollTimer = null;

async function refreshSessions() {
  clearTimeout(pollTimer);
  try {
    sessions = (await (await fetch('/api/sessions', { cache: 'no-store' })).json()).sessions;
  } catch {
    $('view3dInfo').textContent = 'Cannot list sessions: open the console with start_mapping_console.bat.';
    return;
  }
  const sel = $('sessionSelect');
  const keep = sel.value || state.session;
  sel.querySelectorAll('option').forEach((o) => o.remove());  // keeps the <button> of the dropdown
  for (const s of sessions) sel.append(sessionOption(s));
  if (!sessions.length) {
    $('view3dInfo').textContent = 'No recordings yet: record one in the Map view (● Record).';
    return;
  }
  sel.value = sessions.some((s) => s.name === keep) ? keep : sessions[0].name;
  const current = sessions.find((s) => s.name === sel.value);
  updateButton(current);
  if (current.reconstructing) {
    showReconstructing(current);
  } else if (current.failed) {
    if (shown !== `${current.name}:failed`) showStatus(current, 'failed');
  } else if (current.mesh) {
    if (shown !== `${current.name}:${current.mesh_time}`) load(current);
  } else if (shown !== `${current.name}:empty`) {
    showStatus(current, 'empty');
  }
  showLabelStatus(current);
  if (state.show === 'labels' && state.mesh && shown === `${current.name}:${current.mesh_time}`) await ensureLabels(current);
  // follow running jobs closely; otherwise check now and then for new recordings
  if (!document.body.classList.contains('mode-3d')) return;
  const busy = sessions.some((s) => s.reconstructing || s.labeling);
  pollTimer = setTimeout(refreshSessions, busy ? 1000 : 10000);
}

function updateButton(s) {
  $('deleteSessionBtn').disabled = s.reconstructing || s.labeling;
  const btn = $('reconstructBtn');
  // while a job runs, its own button stops it
  btn.disabled = s.labeling;
  btn.textContent = s.reconstructing ? 'Stop reconstruction' : s.mesh ? 'Reconstruct again' : 'Reconstruct';
  const lb = $('labelBtn');
  lb.disabled = s.reconstructing || (!s.labeling && !s.mesh_view);
  lb.textContent = s.labeling ? 'Stop labelling' : s.labels ? 'Label again' : 'Label';
  const usable = s.labels && !s.labels_stale && !s.reconstructing;
  $('show3dLabels').disabled = !usable;
  // a room for the task console (projects/navigation) needs the labels of the current reconstruction
  const rb = $('roomBtn');
  rb.disabled = !usable || s.labeling;
  rb.textContent = s.in_rooms ? 'Update room' : 'Use as room';
  if (!usable && state.show === 'labels') setShow('mesh');
}

// labelling progress, failure, or labels that belong to an older reconstruction
function showLabelStatus(s) {
  const el = $('labelStatus');
  const p = s.label_progress;
  let text = '';
  if (s.labeling) {
    text = p ? `Labelling: step ${p.step} of ${p.steps}, ${p.stage}${p.total ? ` ${p.done} / ${p.total}` : ''}`
      + `${p.detail ? ` (${p.detail})` : ''}` : 'Labelling: starting…';
  } else if (s.label_failed) {
    text = `Labelling failed: ${s.label_error || 'no message'} (full log: logs/label_${s.name}.log)`;
  } else if (s.labels_stale) {
    text = 'The labels are from an older reconstruction: press Label again.';
  }
  el.textContent = text;
  el.hidden = !text;
  el.classList.toggle('warn', !s.labeling);
  const bar = $('labelProgress');
  bar.hidden = !s.labeling;
  bar.firstElementChild.style.width = `${(100 * fraction(p)).toFixed(1)}%`;
}

// loads the labels of the shown session if they are not loaded yet (or were made again)
async function ensureLabels(s) {
  const key = `${s.name}:${s.labels_time}`;
  if (!s.labels || s.labels_stale || !state.mesh) return;
  try {
    if (labels.key !== key) await labels.load(s.name, key, state.mesh);
    if (state.show === 'labels') {
      $('view3dInfo').textContent = `${s.name} · ${labels.count} objects. Click an object (or its tag) for details, `
        + 'Esc to deselect.';
    }
  } catch (e) {
    $('view3dInfo').textContent = `Could not load the labels: ${e.message}`;
    setShow('mesh');
  }
}

function setProgress(value) {  // null: hidden, 'busy': indeterminate, else 0..1
  const bar = $('view3dProgress');
  bar.hidden = value === null;
  bar.classList.toggle('busy', value === 'busy');
  bar.firstElementChild.style.width = typeof value === 'number' ? `${(100 * value).toFixed(1)}%` : '';
}

function showReconstructing(s) {
  if (!shown || !shown.startsWith(`${s.name}:busy`)) {
    clearModel();
    state.session = s.name;
    shown = `${s.name}:busy`;
  }
  const p = s.progress;
  const count = p && p.total ? ` ${p.done} / ${p.total}` : '';
  $('view3dInfo').textContent = p
    ? `Reconstructing on the laptop: step ${p.step} of ${p.steps}, ${p.stage}${count}`
    : 'Reconstructing on the laptop: starting…';
  setProgress(p ? fraction(p) : 'busy');
}

function showStatus(s, what) {
  clearModel();
  state.session = s.name;
  shown = `${s.name}:${what}`;
  setProgress(null);
  $('view3dInfo').textContent = what === 'failed'
    ? `Reconstruction failed: ${s.error || 'no message'} (full log: logs/reconstruct_${s.name}.log)`
    : 'No 3D model yet: press Reconstruct.';
}

$('sessionSelect').addEventListener('change', refreshSessions);

// stops a running reconstruction or labelling (tools/robot_helper.py stop_job); what it wrote so far stays
async function stopJob(kind, name, what) {
  if (!confirm(`Stop the ${what} of ${name}?

What it wrote so far stays; run it again for a complete result.`)) return;
  try {
    const r = await helperFetch(`/api/${kind}/${name}/stop`, { method: 'POST' });
    if (!r.ok) throw new Error(r.error);
    $('view3dInfo').textContent = `Stopped the ${what} of ${name}.`;
    shown = null;
    refreshSessions();
  } catch (e) {
    $('view3dInfo').textContent = e.message === 'not found'
      ? 'Could not stop: the helper is older than this page. Close the helper window and start start_mapping_console.bat again.'
      : `Could not stop: ${e.message}`;
  }
}

$('labelBtn').addEventListener('click', async () => {
  const name = $('sessionSelect').value;
  if (!name) return;
  if (sessions.find((x) => x.name === name)?.labeling) return stopJob('label', name, 'labelling');
  try {
    const r = await helperFetch(`/api/label/${name}`, { method: 'POST' });
    if (!r.ok) throw new Error(r.error);
    refreshSessions();
  } catch (e) {
    $('view3dInfo').textContent = e.message === 'not found'
      ? 'Could not start: the helper is older than this page. Close the console window and start start_mapping_console.bat again.'
      : `Could not start labelling: ${e.message}`;
  }
});

// task console (projects/navigation/make_room.py): this session becomes (or renews) a room the robot can be
// put into; the title is what the task console's room list shows
$('roomBtn').addEventListener('click', async () => {
  const s = sessions.find((x) => x.name === $('sessionSelect').value);
  if (!s) return;
  const title = prompt(s.in_rooms
    ? `Update the room made from ${s.name} (its map and objects).\n\nName in the task console (empty: keep the current one):`
    : `Make ${s.name} a room for the task console.\n\nName in the task console (e.g. "Kitchen"; empty: the session name):`, '');
  if (title === null) return;
  try {
    const r = await helperFetch(`/api/room/make/${s.name}?title=${encodeURIComponent(title.trim())}`, { method: 'POST' });
    if (!r.ok) throw new Error(r.error);
    $('view3dInfo').textContent = `Room ready for the task console: ${r.message}`;
    refreshSessions();
  } catch (e) {
    $('view3dInfo').textContent = e.message === 'not found'
      ? 'Could not make the room: the helper is older than this page. Close the helper window and start start_mapping_console.bat again.'
      : `Could not make the room: ${e.message}`;
  }
});

$('reconstructBtn').addEventListener('click', async () => {
  const name = $('sessionSelect').value;
  if (!name) return;
  if (sessions.find((x) => x.name === name)?.reconstructing) return stopJob('reconstruct', name, 'reconstruction');
  try {
    const r = await helperFetch(`/api/reconstruct/${name}`, { method: 'POST' });
    if (!r.ok) throw new Error(r.error);
    shown = null;
    refreshSessions();
  } catch (e) {
    $('view3dInfo').textContent = `Could not start: ${e.message}`;
  }
});

$('deleteSessionBtn').addEventListener('click', async () => {
  const s = sessions.find((x) => x.name === $('sessionSelect').value);
  if (!s) return;
  const what = `${s.name} (${s.duration_s ? `${Math.round(s.duration_s)} s, ` : ''}${s.keyframes} keyframes`
    + `${s.mesh ? ', with its 3D model' : ''})`;
  if (!confirm(`Delete the recording ${what} from the laptop?\n\nThis cannot be undone.`)) return;
  let r;
  try {
    r = await helperFetch(`/api/delete-session/${s.name}`, { method: 'POST' });
  } catch (e) {
    r = { ok: false, error: e.message };
  }
  if (!r.ok) {
    $('view3dInfo').textContent = r.error === 'not found'
      ? 'Could not delete: the helper is older than this page. Close the console window and start start_mapping_console.bat again.'
      : `Could not delete ${s.name}: ${r.error}`;
    return;
  }
  // gone: drop it from the list right away and show the newest remaining recording
  sessions = sessions.filter((x) => x.name !== s.name);
  const sel = $('sessionSelect');
  sel.querySelector(`option[value="${s.name}"]`)?.remove();
  clearModel();
  shown = null;
  state.session = null;
  sel.value = sessions.length ? sessions[0].name : '';
  refreshSessions();
});

// ---------- loading the model ----------

function clearModel() {
  redraw();
  labels.clear();
  for (const key of ['mesh', 'points', 'path']) {
    if (state[key]) {
      world.remove(state[key]);
      state[key].geometry.dispose();
      state[key] = null;
    }
  }
}

// downloads a PLY with a progress bar (models are 100-200 MB), then parses it
async function fetchPly(url, label) {
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
    $('view3dInfo').textContent = `${label}: ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ''} MB`;
    setProgress(total ? got / total : 'busy');
  }
  const buffer = new Uint8Array(got);
  let at = 0;
  for (const p of parts) { buffer.set(p, at); at += p.length; }
  $('view3dInfo').textContent = `${label}: preparing for display…`;
  setProgress('busy');
  await new Promise((r) => setTimeout(r, 30));  // let the page show the message before parsing
  const geometry = loader.parse(buffer.buffer);
  srgbToLinear(geometry);
  return geometry;
}

// The colours come from camera images (sRGB). three.js treats vertex colours as linear and encodes
// the output to sRGB again, which washed the colours out; convert them to linear once on load.
function srgbToLinear(geometry) {
  const colours = geometry.getAttribute('color');
  if (!colours) return;
  const a = colours.array;
  for (let i = 0; i < a.length; i++) {
    const c = a[i];
    a[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  colours.needsUpdate = true;
}

async function load(s) {
  clearModel();
  state.session = s.name;
  const key = `${s.name}:${s.mesh_time}`;
  shown = key;
  const base = `/data/sessions/${s.name}/reconstruction`;
  try {
    // the point cloud is only loaded when "Points" is chosen (both files together are 150-400 MB)
    const [meshGeo, posesCsv] = await Promise.all([
      fetchPly(`${base}/${s.mesh_view ? 'mesh_view.ply' : 'mesh.ply'}`, 'Loading the mesh'),
      fetch(`${base}/poses.csv`, { cache: 'no-store' }).then((r) => (r.ok ? r.text() : '')),
    ]);
    if (shown !== key) return;  // another session was chosen meanwhile
    setProgress(null);
    // unlit: the photos already contain the room's lighting, so show their colours as they are
    state.mesh = new THREE.Mesh(meshGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
    state.path = pathLine(posesCsv);
    for (const key of ['mesh', 'path']) if (state[key]) world.add(state[key]);
    if (state.show === 'points') await loadPoints();
    applyVisibility();
    resetView();
    if (state.show === 'labels') await ensureLabels(s);
    const tris = meshGeo.index ? meshGeo.index.count / 3 : meshGeo.attributes.position.count / 3;
    $('view3dInfo').textContent = `${s.name} · mesh ${Math.round(tris).toLocaleString()} triangles · `
      + `${s.keyframes} keyframes${s.slam ? ', SLAM poses' : ', odometry poses only'}. `
      + 'Drag to rotate, right-drag to pan, wheel to zoom.';
  } catch (e) {
    setProgress(null);
    $('view3dInfo').textContent = `Could not load the model: ${e.message}`;
  }
}

// camera positions from poses.csv (4x4 row-major: T03, T13, T23 = position)
function pathLine(csv) {
  const rows = csv.trim().split(/\r?\n/).slice(1).map((l) => l.split(',').map(Number));
  if (rows.length < 2) return null;
  const pts = rows.map((r) => new THREE.Vector3(r[2 + 3], r[2 + 7], r[2 + 11]));
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xff4d4d }));
  line.userData.points = pts;
  return line;
}

function applyVisibility() {
  redraw();
  if (state.mesh) state.mesh.visible = state.show === 'mesh' || state.show === 'labels';
  labels.show(state.show === 'labels');
  if (state.points) state.points.visible = state.show === 'points';
  if (state.path) state.path.visible = $('show3dPath').checked;
}

// look at the whole model from above and behind the start of the path
function resetView() {
  const box = new THREE.Box3().setFromObject(state.mesh || state.points || world);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.35 * size, 0.55 * size, 0.55 * size));
  camera.near = size / 1000;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
  redraw();
}

async function loadPoints() {
  const s = sessions.find((x) => x.name === state.session);
  if (state.points || !s || !s.cloud) return;
  const info = $('view3dInfo').textContent;
  const key = shown;
  let geo;
  try {
    geo = await fetchPly(`/data/sessions/${s.name}/reconstruction/cloud.ply`, 'Loading the point cloud');
  } finally {
    setProgress(null);
  }
  if (shown !== key || state.points) return;
  state.points = new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.01, vertexColors: true }));
  world.add(state.points);
  redraw();
  $('view3dInfo').textContent = `${info.split('. Drag')[0]} · ${geo.attributes.position.count.toLocaleString()} points`;
}

async function setShow(what) {
  state.show = what;
  $('show3dMesh').classList.toggle('active', what === 'mesh');
  $('show3dPoints').classList.toggle('active', what === 'points');
  $('show3dLabels').classList.toggle('active', what === 'labels');
  applyVisibility();
  try {
    if (what === 'points') await loadPoints();
    if (what === 'labels') {
      const s = sessions.find((x) => x.name === state.session);
      if (s) await ensureLabels(s);
    }
  } catch (e) {
    setProgress(null);
    $('view3dInfo').textContent = `Could not load: ${e.message}`;
  }
  applyVisibility();
}

$('show3dMesh').addEventListener('click', () => setShow('mesh'));
$('show3dPoints').addEventListener('click', () => setShow('points'));
$('show3dLabels').addEventListener('click', () => setShow('labels'));
$('show3dPath').addEventListener('change', applyVisibility);
$('reset3dBtn').addEventListener('click', resetView);

if (document.body.classList.contains('mode-3d')) {
  running = true;
  resize();
  refreshSessions();
  requestAnimationFrame(frame);
}

// The labelled room in 3D: the robot's mesh of the kitchen (assets/data/room.bin, made by tools/export_room.py)
// in its photo colours, or with every named object in its own colour, an upright box and a tag. Click an object
// (or its tag) for its name and description; click empty space to go back to all objects.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (s, el = document) => el.querySelector(s);

// golden-angle hues: neighbouring ids get clearly different colours (as in the robot's own console)
function objectColour(id) {
  return new THREE.Color().setHSL(((id * 137.508) % 360) / 360, 0.72, 0.58);
}
const srgbToLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

// base: the site's root as seen from the page (tools/capture.html loads it from one folder down)
export async function initRoom(host, { capture = false, base = '' } = {}) {
  const [meta, buf] = await Promise.all([
    fetch(`${base}assets/data/room.json`).then((r) => r.json()),
    fetch(`${base}assets/data/room.bin`).then((r) => r.arrayBuffer()),
  ]);
  const V = meta.vertices, F = meta.faces;
  let off = 0;
  const index = new Uint32Array(buf, off, 3 * F); off += 12 * F;
  const q = new Uint16Array(buf, off, 3 * V); off += 6 * V;
  const rgb = new Uint8Array(buf, off, 3 * V); off += 3 * V;
  const obj = new Uint8Array(buf, off, V);

  // positions around the room's centre, floor at z = 0
  const lo = meta.box_min, hi = meta.box_max;
  const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
  const pos = new Float32Array(3 * V);
  for (let i = 0; i < V; i++) {
    pos[3 * i] = lo[0] + (q[3 * i] / 65535) * (hi[0] - lo[0]) - cx;
    pos[3 * i + 1] = lo[1] + (q[3 * i + 1] / 65535) * (hi[1] - lo[1]) - cy;
    pos[3 * i + 2] = lo[2] + (q[3 * i + 2] / 65535) * (hi[2] - lo[2]);
  }
  const lut = new Float32Array(256).map((_, i) => srgbToLinear(i / 255));
  const photo = new Float32Array(3 * V);
  for (let i = 0; i < 3 * V; i++) photo[i] = lut[rgb[i]];

  const objects = meta.objects.map((o, k) => ({ ...o, k, colour: objectColour(o.id) }));
  const linear = objects.map((o) => o.colour.clone().convertSRGBToLinear());

  // ---------- scene ----------
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: capture });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  host.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100);
  camera.up.set(0, 0, 1);

  const geo = new THREE.BufferGeometry();
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const colours = new THREE.BufferAttribute(photo.slice(), 3);
  geo.setAttribute('color', colours);
  geo.computeBoundingSphere();
  const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  scene.add(mesh);

  const boxes = new THREE.Group();
  for (const o of objects) {
    const z0 = Math.max(o.z_range[0], 0), z1 = o.z_range[1];
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(Math.max(o.size[0], 0.02), Math.max(o.size[1], 0.02), Math.max(z1 - z0, 0.02))),
      new THREE.LineBasicMaterial({ color: o.colour, transparent: true, opacity: 0.9 }));
    box.position.set(o.center[0] - cx, o.center[1] - cy, (z0 + z1) / 2);
    box.rotation.z = THREE.MathUtils.degToRad(o.yaw_deg);
    o.box = box;
    o.top = new THREE.Vector3(o.center[0] - cx, o.center[1] - cy, z1 + 0.04);
    o.area = o.size[0] * o.size[1];
    boxes.add(box);
  }
  scene.add(boxes);

  // ---------- colours: photo <-> objects, blended over time ----------
  let mix = 1, target = 1, selected = null;
  const objMode = new Float32Array(3 * V);
  function buildObjMode() {
    const dim = selected === null ? 0.92 : 0.6;
    for (let i = 0; i < V; i++) {
      const k = 3 * i, o = obj[i];
      const on = o !== 255 && (selected === null || selected === o);
      if (on) {
        const c = linear[o];
        objMode[k] = 0.35 * photo[k] + 0.65 * c.r;
        objMode[k + 1] = 0.35 * photo[k + 1] + 0.65 * c.g;
        objMode[k + 2] = 0.35 * photo[k + 2] + 0.65 * c.b;
      } else {
        const y = (0.2126 * photo[k] + 0.7152 * photo[k + 1] + 0.0722 * photo[k + 2]) * dim;
        objMode[k] = objMode[k + 1] = objMode[k + 2] = y;
      }
    }
  }
  function applyMix() {
    const a = colours.array;
    for (let i = 0; i < a.length; i++) a[i] = photo[i] + (objMode[i] - photo[i]) * mix;
    colours.needsUpdate = true;
    for (const o of objects) {
      o.box.visible = mix > 0.5 && (selected === null || selected === o.k);
      o.box.material.opacity = 0.9 * mix;
    }
  }
  buildObjMode();
  applyMix();

  // ---------- camera ----------
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.35);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 22;
  controls.maxPolarAngle = Math.PI * 0.46;
  controls.enableZoom = false;  // the page scrolls past; zoom only after a click into the view
  controls.autoRotate = !capture;
  controls.autoRotateSpeed = 0.55;
  const span = Math.hypot(hi[0] - lo[0], hi[1] - lo[1]);
  function setView(azimuth, elevation, dist) {
    camera.position.set(Math.cos(azimuth) * Math.cos(elevation) * dist, Math.sin(azimuth) * Math.cos(elevation) * dist,
                        0.35 + Math.sin(elevation) * dist);
    camera.lookAt(controls.target);
    controls.update();
  }
  setView(-2.2, 0.68, span * 0.74);

  let idle = null;
  controls.addEventListener('start', () => { controls.autoRotate = false; clearTimeout(idle); });
  controls.addEventListener('end', () => { clearTimeout(idle); idle = setTimeout(() => { controls.autoRotate = !capture; }, 9000); });
  renderer.domElement.addEventListener('pointerdown', () => { controls.enableZoom = true; });
  host.addEventListener('pointerleave', () => { controls.enableZoom = false; });

  // ---------- tags ----------
  const tagLayer = $('#roomTags', host);
  for (const o of objects) {
    const t = document.createElement('div');
    t.className = 'tag';
    t.textContent = o.label;
    t.style.setProperty('--c', `#${o.colour.getHexString()}`);
    t.addEventListener('click', (e) => { e.stopPropagation(); select(o.k); });
    tagLayer.append(t);
    o.tag = t;
  }
  const byArea = [...objects].sort((a, b) => b.area - a.area);
  const tmp = new THREE.Vector3();
  function placeTags() {
    const w = host.clientWidth, h = host.clientHeight, placed = [];
    const show = mix > 0.5;
    const order = selected === null ? byArea : [objects[selected]];
    for (const o of objects) if (!order.includes(o)) o.tag.style.opacity = '0';
    for (const o of order) {
      tmp.copy(o.top).project(camera);
      const x = (tmp.x * 0.5 + 0.5) * w, y = (-tmp.y * 0.5 + 0.5) * h;
      const tw = o.label.length * 6.6 + 20, th = 22;
      const r = [x - tw / 2, y - th, x + tw / 2, y];
      const free = !placed.some((p) => r[0] < p[2] && r[2] > p[0] && r[1] < p[3] && r[3] > p[1]);
      const ok = show && tmp.z < 1 && x > 0 && x < w && y > 50 && y < h && free;
      o.tag.style.opacity = ok ? '1' : '0';
      o.tag.style.pointerEvents = ok ? 'auto' : 'none';
      o.tag.classList.toggle('sel', selected === o.k);
      if (ok) { placed.push(r); o.tag.style.transform = `translate(${x}px, ${y}px) translate(-50%, -100%)`; }
    }
  }

  // ---------- selection ----------
  const info = $('#roomInfo', host);
  function select(k) {
    selected = k;
    buildObjMode();
    if (target === 0) setMode(1); else applyMix();
    if (k === null) { info.hidden = true; return; }
    const o = objects[k];
    info.style.setProperty('--c', `#${o.colour.getHexString()}`);
    $('span', info).textContent = o.label;
    $('p', info).textContent = o.description || '';
    info.hidden = false;
  }
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(mesh)[0];
    let k = null;
    if (hit) {
      const ids = [hit.face.a, hit.face.b, hit.face.c].map((v) => obj[v]).filter((v) => v !== 255);
      k = ids.length ? ids[0] : null;
    }
    select(k === selected ? null : k);
  });

  // ---------- mode buttons ----------
  const seg = $('.seg', host);
  function setMode(m) {
    target = m;
    for (const b of seg.querySelectorAll('button')) b.setAttribute('aria-pressed', String(Number(b.dataset.mode) === m));
    const on = seg.querySelector('button[aria-pressed="true"]'), pill = seg.querySelector('.pill');
    pill.style.width = `${on.offsetWidth}px`;
    pill.style.transform = `translateX(${on.offsetLeft - 4}px)`;
    if (m === 0) { selected = null; buildObjMode(); info.hidden = true; }
  }
  seg.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => setMode(Number(b.dataset.mode))));

  // ---------- size and drawing ----------
  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();

  let visible = true;
  new IntersectionObserver((e) => { visible = e[0].isIntersecting; }).observe(host);
  let last = performance.now();
  function frame(t) {
    const dt = Math.min(0.05, (t - last) / 1000);
    last = t;
    if (visible) {
      if (mix !== target) {
        mix = target > mix ? Math.min(target, mix + dt * 2.2) : Math.max(target, mix - dt * 2.2);
        applyMix();
      }
      controls.update(dt);
      renderer.render(scene, camera);
      placeTags();
    }
    if (!capture) requestAnimationFrame(frame);
  }
  $('#roomLoading', host)?.remove();
  requestAnimationFrame(frame);

  // for tools/capture_orbit.mjs: a fixed view and blend, drawn on request
  const api = {
    objects: objects.length,
    view(azimuth, elevation, distFactor, m) {
      mix = target = m;
      applyMix();
      setView(azimuth, elevation, span * distFactor);
      renderer.render(scene, camera);
      placeTags();
    },
  };
  if (capture) window.__room = api;
  return api;
}

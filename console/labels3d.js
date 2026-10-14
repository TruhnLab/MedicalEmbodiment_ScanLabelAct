// Labels in the 3D view (projects/scene_understanding): the objects of a labelled session on its mesh.
// Each object's surface gets its own colour (the rest of the room turns grey), with an upright box and
// a tag; the panel on the right lists the objects (with a filter) and shows the chosen one's
// description, state and keyframes with its mask. Click an object in 3D, on its tag or in the list.
// Data: <session>/scene/scene.json and vertex_objects.bin (an object id per vertex of mesh_view.ply).

import * as THREE from 'three';

const $ = (id) => document.getElementById(id);

// golden-angle hues: neighbouring ids get clearly different colours (same as export_viewer.py)
export function objectColour(id) {
  const h = ((id * 137.508) % 360) / 60, c = 200 / 255, x = c * (1 - Math.abs((h % 2) - 1)), m = 1 - c;
  const [r, g, b] = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][Math.floor(h) % 6];
  return new THREE.Color(r + m, g + m, b + m);
}
const css = (c) => `#${c.getHexString()}`;
const esc = (t) => String(t ?? '').replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));

const GROUPS = [['door', 'Doors'], ['furniture', 'Furniture'], ['appliance', 'Appliances'], ['fixture', 'Fixtures'],
                ['container', 'Containers'], ['floor_item', 'On the floor'], ['other', 'Other']];

// redraw: tells the 3D view that something changed (it only draws then)
export function createLabels({ world, camera, renderer, host, redraw = () => {} }) {
  const st = { key: null, session: null, data: null, mesh: null, photo: null, ids: null, on: false,
               selected: null, filter: '', boxes: new Map(), group: null };
  const tags = $('labelTags');
  const panel = $('labelPanel');

  function visible(o) {
    if (!st.filter) return true;
    const q = st.filter.toLowerCase();
    return `${o.id} ${o.label} ${o.description} ${o.category} ${o.state || ''}`.toLowerCase().includes(q);
  }

  function restorePhoto() {
    if (!st.mesh || !st.photo) return;
    const col = st.mesh.geometry.getAttribute('color');
    col.array.set(st.photo);
    col.needsUpdate = true;
    redraw();
  }

  function recolour() {
    if (!st.mesh || !st.on) return;
    const col = st.mesh.geometry.getAttribute('color');
    const out = col.array, photo = st.photo, ids = st.ids;
    const objs = new Map(st.data.objects.map((o) => [o.id, {
      c: objectColour(o.id).convertSRGBToLinear(),
      show: visible(o) && (st.selected === null || st.selected === o.id),
    }]));
    const dim = st.selected !== null || st.filter ? 0.45 : 0.8;
    for (let i = 0; i < ids.length; i++) {
      const k = 3 * i;
      const o = ids[i] >= 0 ? objs.get(ids[i]) : null;
      if (o && o.show) {
        out[k] = 0.5 * photo[k] + 0.5 * o.c.r;
        out[k + 1] = 0.5 * photo[k + 1] + 0.5 * o.c.g;
        out[k + 2] = 0.5 * photo[k + 2] + 0.5 * o.c.b;
      } else {
        // background (and objects filtered out): grey photo, so the objects stand out
        const y = (0.2126 * photo[k] + 0.7152 * photo[k + 1] + 0.0722 * photo[k + 2]) * dim;
        out[k] = out[k + 1] = out[k + 2] = y;
      }
    }
    col.needsUpdate = true;
    redraw();
  }

  function makeBox(o) {
    const [cx, cy] = o.center, [w, l] = o.size;
    const z0 = Math.max(o.z_range[0], 0), z1 = o.z_range[1];
    const geo = new THREE.EdgesGeometry(new THREE.BoxGeometry(Math.max(w, 0.02), Math.max(l, 0.02), Math.max(z1 - z0, 0.02)));
    const line = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: objectColour(o.id) }));
    line.position.set(cx, cy, (z0 + z1) / 2);
    line.rotation.z = THREE.MathUtils.degToRad(o.yaw_deg);  // cv2.minAreaRect: size[0] along the yaw
    line.userData.top = new THREE.Vector3(cx, cy, z1 + 0.05);
    const tag = document.createElement('div');
    tag.textContent = `${o.id} ${o.label}`;
    tag.style.borderLeftColor = css(objectColour(o.id));
    tag.onclick = () => select(o.id, true);
    tags.appendChild(tag);
    return { line, tag };
  }

  function updateOverlays() {
    redraw();  // also places the tags anew
    if (!st.data) return;
    for (const o of st.data.objects) {
      const b = st.boxes.get(o.id);
      const show = st.on && visible(o) && (st.selected === null || st.selected === o.id);
      b.line.visible = show;
      b.tag.style.display = show ? '' : 'none';
      b.tag.classList.toggle('sel', st.selected === o.id);
    }
  }

  function renderList() {
    const ul = $('labelList');
    ul.innerHTML = '';
    for (const [cat, title] of GROUPS) {
      const objs = st.data.objects.filter((o) => (o.category === cat || (cat === 'other' && !GROUPS.some(([c]) => c === o.category)))
        && visible(o)).sort((a, b) => a.label.localeCompare(b.label) || a.id - b.id);
      if (!objs.length) continue;
      const h = document.createElement('li');
      h.className = 'group';
      h.textContent = `${title} (${objs.length})`;
      ul.appendChild(h);
      for (const o of objs) {
        const li = document.createElement('li');
        li.innerHTML = `<span class="sw" style="background:${css(objectColour(o.id))}"></span>`
          + `<span>${esc(o.label)}</span><span class="n">#${o.id}${o.confidence === 'low' ? ' ?' : ''}</span>`;
        li.classList.toggle('sel', st.selected === o.id);
        li.onclick = () => select(st.selected === o.id ? null : o.id, true);
        ul.appendChild(li);
      }
    }
  }

  const fmt = (v) => Number(v).toFixed(2);
  function renderDetails() {
    const box = $('labelDetails');
    const o = st.data?.objects.find((x) => x.id === st.selected);
    box.hidden = !o;
    if (!o) return;
    const seen = (o.state_seen || []).slice(0, 4).map((s) => `${esc(s.state)} at ${new Date(s.stamp * 1000).toLocaleTimeString()}`);
    const base = `/data/sessions/${st.session}/scene/`;
    const imgs = (o.mask_views || o.crops || []).map((f) => `<img src="${base}${esc(f)}" alt="view of ${esc(o.label)}">`).join('');
    box.innerHTML = `
      <h3><span class="sw" style="background:${css(objectColour(o.id))}"></span>#${o.id} ${esc(o.label)}</h3>
      <p>${esc(o.description)}</p>
      ${o.confidence === 'low' ? '<p class="warn">Uncertain: its views may show different things.</p>' : ''}
      <dl>
        <dt>Category</dt><dd>${esc(o.category)}, ${esc(o.movable)}</dd>
        ${o.state ? `<dt>State</dt><dd>${esc(o.state)} when scanned${seen.length ? ` (${seen.join(', ')})` : ''}</dd>` : ''}
        <dt>Position</dt><dd>x ${fmt(o.center[0])}, y ${fmt(o.center[1])} m</dd>
        <dt>Size</dt><dd>${fmt(o.size[0])} × ${fmt(o.size[1])} m, ${fmt(o.size[2])} m high</dd>
        <dt>Seen in</dt><dd>${o.views.length} keyframes</dd>
      </dl>
      <div class="imgs">${imgs}</div>`;
    const srcs = [...box.querySelectorAll('img')].map((img) => img.src);
    box.querySelectorAll('img').forEach((img, i) => {
      img.onclick = () => { big.srcs = srcs; showBig(i); $('labelBig').showModal(); };
    });
  }

  function select(id, fly) {
    st.selected = id;
    renderList();
    renderDetails();
    recolour();
    updateOverlays();
    api.onSelect?.(id);  // the task console (tasks.js): "go there" for the selected object
    if (fly && id !== null) {
      const o = st.data.objects.find((x) => x.id === id);
      const target = new THREE.Vector3(...o.center).applyMatrix4(world.matrixWorld);
      const dist = Math.max(1.2, 2.5 * Math.hypot(...o.size));
      api.flyTo?.(target, dist);
    }
  }

  // enlarged view of one object's keyframes; the arrow buttons on the image go through them (not the
  // arrow keys: those move the arm in app.js)
  const big = { srcs: [], i: 0 };
  function showBig(i) {
    big.i = (i + big.srcs.length) % big.srcs.length;
    $('labelBig').querySelector('img').src = big.srcs[big.i];
    $('labelBigCount').textContent = `${big.i + 1} / ${big.srcs.length}`;
    $('labelBigPrev').hidden = $('labelBigNext').hidden = big.srcs.length < 2;
  }
  $('labelBigPrev').addEventListener('click', () => showBig(big.i - 1));
  $('labelBigNext').addEventListener('click', () => showBig(big.i + 1));

  // click (not drag) on the mesh: the object under the pointer
  const raycaster = new THREE.Raycaster();
  let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    // api.clickable = false: the page uses clicks for something else (the task console's pin & go)
    if (api.clickable === false) return;
    if (!st.on || !st.mesh || !down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1), camera);
    const hit = raycaster.intersectObject(st.mesh)[0];
    const id = hit ? [hit.face.a, hit.face.b, hit.face.c].map((v) => st.ids[v]).find((v) => v >= 0) : undefined;
    select(id === undefined || id === st.selected ? null : id, false);
  });
  $('labelFilter').addEventListener('input', (e) => {
    st.filter = e.target.value.trim();
    renderList(); recolour(); updateOverlays();
  });
  addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && st.on && st.selected !== null && !$('labelBig').open) select(null);
  });

  const tmp = new THREE.Vector3();
  const api = {
    get key() { return st.key; },
    get count() { return st.data ? st.data.objects.length : 0; },
    get selected() { return st.data?.objects.find((o) => o.id === st.selected) ?? null; },

    // key: "<session>:<labels time>"; mesh: the 3D view's mesh (mesh_view.ply, sRGB colours made linear)
    async load(session, key, mesh) {
      api.clear();
      const base = `/data/sessions/${session}/scene`;
      const [data, idsBuf] = await Promise.all([
        fetch(`${base}/scene.json`, { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(`scene.json: ${r.status}`); return r.json(); }),
        fetch(`${base}/vertex_objects.bin`, { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(`vertex_objects.bin: ${r.status}`); return r.arrayBuffer(); }),
      ]);
      const ids = new Int16Array(idsBuf);
      const col = mesh.geometry.getAttribute('color');
      if (ids.length !== col.count) throw new Error('the labels belong to another reconstruction: press Label again');
      Object.assign(st, { key, session, data, mesh, ids, photo: col.array.slice(), selected: null });
      st.group = new THREE.Group();
      for (const o of data.objects) {
        const b = makeBox(o);
        st.boxes.set(o.id, b);
        st.group.add(b.line);
      }
      world.add(st.group);
      redraw();
      renderList();
      renderDetails();
      api.show(st.on);
      return data.objects.length;
    },

    clear() {
      restorePhoto();
      if (st.group) {
        world.remove(st.group);
        st.group.traverse((x) => x.geometry?.dispose());
      }
      tags.innerHTML = '';
      st.boxes.clear();
      Object.assign(st, { key: null, session: null, data: null, mesh: null, photo: null, ids: null, group: null, selected: null });
      panel.hidden = true;
    },

    show(on) {
      st.on = on;
      panel.hidden = !(on && st.data);
      tags.hidden = !on;
      if (on) recolour(); else restorePhoto();
      updateOverlays();
    },

    // every frame: tags follow their boxes
    frame() {
      if (!st.on || !st.data) return;
      const w = host.clientWidth, h = host.clientHeight;
      for (const b of st.boxes.values()) {
        if (b.tag.style.display === 'none') continue;
        tmp.copy(b.line.userData.top).applyMatrix4(world.matrixWorld).project(camera);
        const off = tmp.z > 1 || Math.abs(tmp.x) > 1.1 || Math.abs(tmp.y) > 1.1;
        b.tag.style.visibility = off ? 'hidden' : 'visible';
        b.tag.style.left = `${(tmp.x + 1) / 2 * w}px`;
        b.tag.style.top = `${(1 - tmp.y) / 2 * h}px`;
      }
    },
  };
  return api;
}

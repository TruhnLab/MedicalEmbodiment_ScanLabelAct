'use strict';

// Map view (projects/mapping) inside the console: the live SLAM map from the offboard ROS
// container (rosbridge on ws://127.0.0.1:9091), with the camera as picture-in-picture. Driving
// keys keep working, so you can drive while watching the map grow. Uses $, settings and
// saveSettings from app.js; everything else stays in this scope.

(() => {
  const OFFBOARD_URL = 'ws://127.0.0.1:9091';  // offboard_ros container
  const RECORDER_STATUS = 'http://127.0.0.1:8766/status.json';
  const GROWTH_WINDOW_S = 60;

  let offboard = null;
  let map = null;          // { w, h, res, ox, oy, image (canvas), hash }
  let lastSaved = null;    // { hash, name } of the last saved map
  let robot = null;        // { x, y, yaw }
  let path = [];           // [[x, y], ...]
  let explore = null;      // exploration status from offboard_ros/explorer.py (JSON)
  let plan = [];           // the exploration's planned path [[x, y], ...]
  let coverage = null;     // { image (canvas), w, h, res, ox, oy }: surfaces the camera saw / not yet
  let obstacles = null;    // the same for obstacles only the depth camera sees (furniture), drawn always
  let exploreCmd = null;   // ROSLIB.Topic /explore/command
  let exploreStart = null;    // when the current exploration started (for the elapsed time)
  let exploreEvents = [];    // [{time, text}], newest first
  let autoRecording = false;  // the recorder was started by Explore (and is stopped when the room is done)
  let noGo = [];           // no-go lines [[x1, y1, x2, y2], ...] in map coordinates
  let noGoRestored = false; // taken over from the explorer's status once after loading the page
  let drawing = null;      // null, or { first: [x, y] | null } while drawing a line
  const HANDLE_PX = 7;     // radius of the draggable line ends
  const areaHistory = [];  // [time s, m²]
  const view = { cx: 0, cy: 0, scale: 60, follow: true };  // scale in px per metre
  const cv = $('mapCanvas');

  // ---------- view switch ----------

  // views: drive (camera), map (live SLAM map, camera as picture-in-picture), 3d (reconstructions,
  // view3d.js reacts to the body class)
  function setMode(mode) {
    if (!['drive', 'map', '3d'].includes(mode)) mode = 'drive';
    document.body.classList.toggle('mode-map', mode === 'map');
    document.body.classList.toggle('mode-3d', mode === '3d');
    $('viewDriveBtn').classList.toggle('active', mode === 'drive');
    $('viewMapBtn').classList.toggle('active', mode === 'map');
    $('view3dBtn').classList.toggle('active', mode === '3d');
    settings.view = mode;
    saveSettings();
    if (mode === 'map') draw();
  }

  $('viewDriveBtn').addEventListener('click', () => setMode('drive'));
  $('viewMapBtn').addEventListener('click', () => setMode('map'));
  $('view3dBtn').addEventListener('click', () => setMode('3d'));
  // clicking the picture-in-picture camera goes back to the drive view
  document.querySelector('.cam-view').addEventListener('click', (e) => {
    if (document.body.classList.contains('mode-map') && !e.target.closest('select')) setMode('drive');
  });
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input:not([type]), input[type="text"], input[type="number"], textarea')) return;
    if (e.key.toLowerCase() === 'm' && !e.ctrlKey && !e.repeat) {
      e.preventDefault();
      setMode(document.body.classList.contains('mode-map') ? 'drive' : 'map');
    }
  });

  // ---------- offboard ROS connection ----------

  function setSlamState(connected) {
    $('slamStatus').dataset.state = connected ? 'ok' : 'off';
    $('slamStatus').textContent = connected ? 'SLAM running' : 'SLAM offline';
    $('slamStartBtn').disabled = connected;
    $('slamResetBtn').disabled = !connected;
    $('slamStopBtn').disabled = !connected;
    $('saveMapBtn').disabled = !connected;
    if (!connected) explore = null;
    updateExplore();
  }

  function connectOffboard() {
    const r = new ROSLIB.Ros({ url: OFFBOARD_URL });
    r.on('connection', () => {
      offboard = r;
      setSlamState(true);
      new ROSLIB.Topic({ ros: r, name: '/map', messageType: 'nav_msgs/msg/OccupancyGrid', throttle_rate: 1000, queue_length: 1 })
        .subscribe(onMap);
      new ROSLIB.Topic({ ros: r, name: '/robot_pose', messageType: 'geometry_msgs/msg/PoseStamped', throttle_rate: 100, queue_length: 1 })
        .subscribe(onPose);
      new ROSLIB.Topic({ ros: r, name: '/robot_path', messageType: 'nav_msgs/msg/Path', throttle_rate: 1000, queue_length: 1 })
        .subscribe((msg) => { path = msg.poses.map((p) => [p.pose.position.x, p.pose.position.y]); });
      exploreCmd = new ROSLIB.Topic({ ros: r, name: '/explore/command', messageType: 'std_msgs/msg/String' });
      setTimeout(sendNoGo, 1000);
      new ROSLIB.Topic({ ros: r, name: '/explore/status', messageType: 'std_msgs/msg/String', queue_length: 1 })
        .subscribe((msg) => { onExploreStatus(JSON.parse(msg.data)); });
      new ROSLIB.Topic({ ros: r, name: '/explore/path', messageType: 'nav_msgs/msg/Path', queue_length: 1 })
        .subscribe((msg) => { plan = msg.poses.map((p) => [p.pose.position.x, p.pose.position.y]); draw(); });
      new ROSLIB.Topic({ ros: r, name: '/explore/coverage', messageType: 'nav_msgs/msg/OccupancyGrid', queue_length: 1 })
        .subscribe(onCoverage);
      new ROSLIB.Topic({ ros: r, name: '/explore/events', messageType: 'std_msgs/msg/String', queue_length: 20 })
        .subscribe((msg) => onExploreEvent(JSON.parse(msg.data)));
    });
    r.on('close', () => {
      if (offboard === r) offboard = null;
      setSlamState(false);
      setTimeout(connectOffboard, 3000);
    });
    r.on('error', () => {});  // 'close' follows and retries
  }

  function onMap(msg) {
    const { width: w, height: h, resolution: res, origin } = msg.info;
    const css = getComputedStyle(document.documentElement);
    const colour = (name) => {
      const n = parseInt(css.getPropertyValue(name).trim().slice(1), 16);
      return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
    };
    const [unknown, free, occ] = [colour('--surface-2'), colour('--map-free'), colour('--map-occupied')];
    const image = document.createElement('canvas');
    image.width = w;
    image.height = h;
    const ctx = image.getContext('2d');
    const img = ctx.createImageData(w, h);
    let freeCells = 0;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const v = msg.data[row * w + col];
        const o = ((h - 1 - row) * w + col) * 4;  // grid row 0 is the bottom, image row 0 the top
        const c = v < 0 ? unknown : v >= 50 ? occ : free;
        if (v >= 0 && v < 50) freeCells++;
        img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const first = map === null;
    map = { w, h, res, ox: origin.position.x, oy: origin.position.y, image, hash: mapHash(msg), data: msg.data };
    $('mapPlaceholder').style.display = 'none';

    const area = freeCells * res * res;
    const now = performance.now() / 1000;
    areaHistory.push([now, area]);
    while (areaHistory.length && now - areaHistory[0][0] > GROWTH_WINDOW_S) areaHistory.shift();
    $('mArea').textContent = `${area.toFixed(1)} m²`;
    const [t0, a0] = areaHistory[0];
    $('mGrowth').textContent = now - t0 > 5 ? `${((area - a0) / (now - t0) * 60).toFixed(2)} m²` : '–';
    if (first) view.needsZoom = true;  // set on the first draw: the canvas may be hidden right now
    draw();
  }

  // Fingerprint of the map content (FNV-1a over size, origin and cells): SLAM republishes the map
  // every 2 s even when nothing changed, so this tells whether a save would add anything.
  function mapHash(msg) {
    const { width, height, origin } = msg.info;
    let h = 2166136261;
    const add = (v) => { h ^= v & 0xff; h = Math.imul(h, 16777619); };
    [width, height, Math.round(origin.position.x * 1000), Math.round(origin.position.y * 1000)]
      .forEach((v) => { add(v); add(v >> 8); add(v >> 16); });
    for (const v of msg.data) add(v);
    return h >>> 0;
  }

  function onPose(msg) {
    const q = msg.pose.orientation;
    robot = {
      x: msg.pose.position.x,
      y: msg.pose.position.y,
      yaw: Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
    };
    let len = 0;
    for (let i = 1; i < path.length; i++) len += Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]);
    $('mPath').textContent = `${len.toFixed(1)} m`;
    draw();
  }

  // ---------- drawing ----------

  function toScreen(x, y, W, H) {
    return [W / 2 + (x - view.cx) * view.scale, H / 2 - (y - view.cy) * view.scale];
  }

  function fitMap() {
    if (!map) return;
    view.cx = map.ox + (map.w * map.res) / 2;
    view.cy = map.oy + (map.h * map.res) / 2;
    view.scale = 0.9 * Math.min(cv.clientWidth / (map.w * map.res), cv.clientHeight / (map.h * map.res));
    view.follow = false;
    $('mapFollow').checked = false;
    draw();
  }

  function draw() {
    if (!document.body.classList.contains('mode-map')) return;
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth;
    const H = cv.clientHeight;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr);
      cv.height = Math.round(H * dpr);
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    ctx.fillStyle = css.getPropertyValue('--surface-2');
    ctx.fillRect(0, 0, W, H);
    if (view.needsZoom && map) {
      // first map: zoom so ~the whole map (at least 4 m) fits the visible canvas
      view.scale = Math.min(120, 0.8 * Math.min(W, H) / Math.max(4, map.w * map.res, map.h * map.res));
      view.needsZoom = false;
    }
    if (view.follow && robot) {
      view.cx = robot.x;
      view.cy = robot.y;
    }

    if (map) {
      const [x0, y0] = toScreen(map.ox, map.oy + map.h * map.res, W, H);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(map.image, x0, y0, map.w * map.res * view.scale, map.h * map.res * view.scale);
    }

    // 1 m grid
    ctx.strokeStyle = css.getPropertyValue('--border');
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const left = view.cx - W / 2 / view.scale;
    const bottom = view.cy - H / 2 / view.scale;
    for (let x = Math.ceil(left); x <= left + W / view.scale; x++) { const [sx] = toScreen(x, 0, W, H); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); }
    for (let y = Math.ceil(bottom); y <= bottom + H / view.scale; y++) { const [, sy] = toScreen(0, y, W, H); ctx.moveTo(0, sy); ctx.lineTo(W, sy); }
    ctx.stroke();
    ctx.globalAlpha = 1;

    // camera coverage of the exploration: surfaces seen (green) / not seen yet (red)
    if (obstacles) {  // part of the map: also when no exploration runs
      const [x0, y0] = toScreen(obstacles.ox, obstacles.oy + obstacles.h * obstacles.res, W, H);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(obstacles.image, x0, y0, obstacles.w * obstacles.res * view.scale, obstacles.h * obstacles.res * view.scale);
    }
    if (coverage && explore && explore.state !== 'idle') {
      const [x0, y0] = toScreen(coverage.ox, coverage.oy + coverage.h * coverage.res, W, H);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(coverage.image, x0, y0, coverage.w * coverage.res * view.scale, coverage.h * coverage.res * view.scale);
    }
    // no-go lines (and the first end of one being drawn)
    ctx.strokeStyle = css.getPropertyValue('--danger');
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    for (const [x1, y1, x2, y2] of noGo) {
      const [a1, b1] = toScreen(x1, y1, W, H);
      const [a2, b2] = toScreen(x2, y2, W, H);
      // a thin dark outline first: the line stays crisp on top of the coverage overlay
      ctx.strokeStyle = css.getPropertyValue('--cam-bg');
      ctx.lineWidth = 7;
      ctx.beginPath();
      ctx.moveTo(a1, b1);
      ctx.lineTo(a2, b2);
      ctx.stroke();
      ctx.strokeStyle = css.getPropertyValue('--danger');
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(a1, b1);
      ctx.lineTo(a2, b2);
      ctx.stroke();
      // handles: drag to move an end
      ctx.fillStyle = css.getPropertyValue('--danger');
      for (const [hx, hy] of [[a1, b1], [a2, b2]]) {
        ctx.beginPath();
        ctx.arc(hx, hy, HANDLE_PX - 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    if (drawing && drawing.first) {
      const [a1, b1] = toScreen(...drawing.first, W, H);
      ctx.beginPath();
      ctx.arc(a1, b1, 5, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.lineCap = 'butt';

    // the exploration's planned path and goal
    if (plan.length > 1 && explore && ['exploring', 'paused'].includes(explore.state)) {
      ctx.strokeStyle = css.getPropertyValue('--text');  // planned path: neutral, unlike coverage and the driven path
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      plan.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y, W, H); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.stroke();
      ctx.setLineDash([]);
      const [gx, gy] = toScreen(...plan[plan.length - 1], W, H);
      ctx.beginPath();
      ctx.arc(gx, gy, 6, 0, 2 * Math.PI);
      ctx.stroke();
    }

    if (path.length > 1) {
      ctx.strokeStyle = css.getPropertyValue('--accent');
      ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y, W, H); i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
      ctx.stroke();
    }

    if (robot && explore && ['exploring', 'paused'].includes(explore.state)) {
      // what the camera sees: 74 deg wide, drawn 1.5 m long
      const [sx, sy] = toScreen(robot.x, robot.y, W, H);
      const len = 1.5 * view.scale;
      const half = (74 / 2) * Math.PI / 180;
      ctx.fillStyle = 'rgba(80, 160, 255, 0.18)';
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.arc(sx, sy, len, -robot.yaw - half, -robot.yaw + half);
      ctx.closePath();
      ctx.fill();
    }
    if (robot) {
      const [sx, sy] = toScreen(robot.x, robot.y, W, H);
      const size = Math.max(8, 0.18 * view.scale);
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(-robot.yaw);
      ctx.fillStyle = css.getPropertyValue('--robot');
      ctx.beginPath();
      ctx.moveTo(size, 0);
      ctx.lineTo(-size * 0.6, size * 0.55);
      ctx.lineTo(-size * 0.6, -size * 0.55);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  // ---------- pan & zoom ----------

  cv.addEventListener('wheel', (e) => {
    e.preventDefault();
    view.scale = Math.min(400, Math.max(5, view.scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    draw();
  }, { passive: false });
  let drag = null;
  let moving = null;  // { line: index, end: 0 | 1 } while dragging a no-go line's end

  // map coordinates of a pointer event
  function mapPoint(e) {
    const rect = cv.getBoundingClientRect();
    return [view.cx + (e.clientX - rect.left - cv.clientWidth / 2) / view.scale,
      view.cy - (e.clientY - rect.top - cv.clientHeight / 2) / view.scale];
  }

  // the no-go line end under the pointer: { line, end } or null
  function endAt(e) {
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    for (let i = noGo.length - 1; i >= 0; i--) {
      for (const end of [0, 1]) {
        const [sx, sy] = toScreen(noGo[i][2 * end], noGo[i][2 * end + 1], cv.clientWidth, cv.clientHeight);
        if (Math.hypot(sx - px, sy - py) <= HANDLE_PX + 3) return { line: i, end };
      }
    }
    return null;
  }

  // the no-go line under the pointer (within a few pixels): its index or -1
  function lineAt(e) {
    const rect = cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    for (let i = noGo.length - 1; i >= 0; i--) {
      const [ax, ay] = toScreen(noGo[i][0], noGo[i][1], cv.clientWidth, cv.clientHeight);
      const [bx, by] = toScreen(noGo[i][2], noGo[i][3], cv.clientWidth, cv.clientHeight);
      const len2 = (bx - ax) ** 2 + (by - ay) ** 2 || 1;
      const t = Math.max(0, Math.min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / len2));
      if (Math.hypot(ax + t * (bx - ax) - px, ay + t * (by - ay) - py) <= 7) return i;
    }
    return -1;
  }

  // right-click on a line removes just that line
  cv.addEventListener('contextmenu', (e) => {
    const i = lineAt(e);
    if (i < 0) return;
    e.preventDefault();
    noGo.splice(i, 1);
    sendNoGo();
    updateNoGo();
    toast('No-go line removed');
    draw();
  });

  cv.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;  // right button: see contextmenu
    if (!drawing) {
      const hit = endAt(e);
      if (hit) {
        moving = hit;
        cv.setPointerCapture(e.pointerId);
        return;
      }
    }
    if (drawing) {
      const [x, y] = mapPoint(e);
      if (!drawing.first) {
        drawing.first = [x, y];
        $('mHint').textContent = 'No-go line: now click its other end (Esc cancels).';
      } else {
        noGo.push([...drawing.first, x, y]);
        stopDrawing();
        sendNoGo();
        $('mHint').textContent = '';
        toast("No-go line set: the robot stays on the start's side of it");
      }
      draw();
      return;
    }
    drag = [e.clientX, e.clientY];
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener('pointermove', (e) => {
    if (moving) {
      const [x, y] = mapPoint(e);
      noGo[moving.line][2 * moving.end] = x;
      noGo[moving.line][2 * moving.end + 1] = y;
      draw();
      return;
    }
    if (!drawing) cv.style.cursor = endAt(e) ? 'move' : lineAt(e) >= 0 ? 'context-menu' : '';
    if (!drag) return;
    view.cx -= (e.clientX - drag[0]) / view.scale;
    view.cy += (e.clientY - drag[1]) / view.scale;
    drag = [e.clientX, e.clientY];
    view.follow = false;
    $('mapFollow').checked = false;
    draw();
  });
  cv.addEventListener('pointerup', () => {
    if (moving) {
      moving = null;
      sendNoGo();  // the explorer gets the line once it is in place
      toast('No-go line moved');
    }
    drag = null;
  });
  $('mapFollow').addEventListener('change', () => { view.follow = $('mapFollow').checked; draw(); });
  $('mapFitBtn').addEventListener('click', fitMap);
  new ResizeObserver(draw).observe(cv);

  // ---------- SLAM container control (the helper runs docker compose) ----------

  async function slam(action, busyText) {
    if (action === 'reset' && !confirm('Throw the current map away and start a new one?')) return;
    sendExplore('stop');  // a new map or no SLAM: the exploration ends for good
    if (action === 'reset') {
      noGo = [];  // they belong to the old map
      sendNoGo();
      updateNoGo();
    }
    $('mHint').textContent = busyText;
    try {
      const r = await helperFetch(`/api/slam/${action}`, { method: 'POST' });
      if (!r.ok) throw new Error(r.error);
      if (action !== 'stop') {
        map = null; robot = null; path = []; areaHistory.length = 0;
        $('mapPlaceholder').style.display = '';
        $('mapPlaceholder').textContent = 'SLAM starting… the map appears after the first scans.';
        draw();
      }
      $('mHint').textContent = action === 'stop' ? 'SLAM stopped.' : '';
    } catch (e) {
      $('mHint').textContent = `SLAM ${action} failed: ${e.message}`;
    }
  }

  $('slamStartBtn').addEventListener('click', () => slam('start', 'Starting SLAM…'));
  $('slamResetBtn').addEventListener('click', () => slam('reset', 'Starting a new map…'));
  $('slamStopBtn').addEventListener('click', () => slam('stop', 'Stopping SLAM…'));

  $('saveMapBtn').addEventListener('click', () => {
    if (!offboard || !map) return;
    if (lastSaved && lastSaved.hash === map.hash) {
      $('mHint').textContent = `No changes since the last save (${lastSaved.name}).`;
      return;
    }
    const hash = map.hash;
    const file = `map_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, '')}`;
    const btn = $('saveMapBtn');
    btn.disabled = true;  // one save at a time: no duplicates from double clicks
    $('mHint').textContent = 'Saving map…';
    const done = (text) => { $('mHint').textContent = text; btn.disabled = !offboard; };
    new ROSLIB.Service({ ros: offboard, name: '/slam_toolbox/save_map', serviceType: 'slam_toolbox/srv/SaveMap' })
      .callService({ name: { data: `/offboard_ros/maps/${file}` } }, (res) => {
        if (res.result === 0) lastSaved = { hash, name: file };
        done(res.result === 0 ? `Map saved: data/maps/${file}.pgm/.yaml` : `Saving failed (code ${res.result}).`);
      }, (err) => done(`Saving failed: ${err}`));
  });

  // ---------- recorder status ----------

  let recording = false;
  let recorderBusy = false;  // between clicking and the recorder reporting its new state
  let recorderRequest = 0;   // counts start/stop requests (for the "did not respond" timeout)
  let recorderStopRequested = false;  // the user (or the end of an exploration) asked it to stop

  async function pollRecorder() {
    const el = $('recStatus');
    let st = null;
    try {
      st = await (await fetch(RECORDER_STATUS, { cache: 'no-store' })).json();
    } catch {
      // recorder not running
    }
    const nowRecording = !!(st && st.recording);
    if (recording && !nowRecording && !recorderStopRequested) {
      // stopped by itself (battery, SLAM gone, size limit, ...): say so loudly, and don't keep
      // exploring without a recording
      const reason = (st && st.stop_reason) || 'the recorder ended';
      const exploring = explore && explore.state === 'exploring';
      if (exploring) document.dispatchEvent(new CustomEvent('landerpi:pause-auto', { detail: 'recording stopped' }));
      showBanner('warn', 'Recording stopped', `Reason: ${reason}.` + (exploring ? ' The exploration is paused.' : ''),
        /battery/i.test(reason) ? 'Charge the robot before the next run.' : '');
    }
    if (!nowRecording) recorderStopRequested = false;
    if (nowRecording !== recording) recorderBusy = false;
    recording = nowRecording;
    el.textContent = !st ? 'off' : recording ? `● ${st.keyframes} kf · ${st.rx_mb_s} MB/s` : 'finished';
    el.classList.toggle('rec', recording);
    if (st && st.hints && st.hints.length && recording) $('mHint').textContent = st.hints.join(' · ');
    const btn = $('recordBtn');
    btn.classList.toggle('recording', recording);
    btn.textContent = recording ? `■ Stop recording ${Math.floor(st.elapsed_s / 60)}:${String(Math.floor(st.elapsed_s % 60)).padStart(2, '0')}`
      : recorderBusy ? 'Starting…' : '● Record';
    btn.disabled = recorderBusy;
  }

  $('recordBtn').addEventListener('click', async () => {
    const action = recording ? 'stop' : 'start';
    if (action === 'stop') recorderStopRequested = true;
    if (action === 'start' && !offboard && !confirm('SLAM is not running, so the images will only get odometry poses. Record anyway?')) return;
    recorderBusy = true;
    $('recordBtn').disabled = true;
    $('recordBtn').textContent = action === 'start' ? 'Starting…' : 'Stopping…';
    try {
      const r = await helperFetch(`/api/recorder/${action}`, { method: 'POST' });
      if (!r.ok) throw new Error(r.error);
      $('mHint').textContent = action === 'start'
        ? 'Recorder starting (prepares the robot, ~10 s)…' : 'Recorder finishing: writing remaining keyframes…';
    } catch (e) {
      recorderBusy = false;
      $('mHint').textContent = `Recorder ${action} failed: ${e.message}`;
    }
    // give up waiting after 40 s, e.g. if the robot could not be prepared (see logs/recorder_*.log);
    // only for this request (a later start/stop has its own timer)
    const request = ++recorderRequest;
    setTimeout(() => {
      if (recorderBusy && request === recorderRequest) {
        recorderBusy = false;
        $('mHint').textContent = 'Recorder did not respond: see logs/recorder_*.log';
      }
    }, 40000);
  });

  // ---------- autonomous exploration (offboard_ros/explorer.py) ----------

  function sendExplore(cmd) {
    if (exploreCmd && explore && explore.state !== 'idle') exploreCmd.publish({ data: cmd });
  }

  function onExploreStatus(st) {
    const prev = explore && explore.state;
    explore = st;
    // after a page reload: the no-go lines live on in the explorer (only once: later, the console's
    // own edits are what counts, and a status sent just before an edit must not bring lines back)
    if (!noGoRestored) {
      noGoRestored = true;
      if (!noGo.length && Array.isArray(st.lines) && st.lines.length) {
        noGo = st.lines.map((l) => l.map(Number));
        updateNoGo();
        draw();
        toast(`${noGo.length} no-go line${noGo.length === 1 ? '' : 's'} restored from the exploration`);
      }
    }
    if (st.state === 'exploring' && !['exploring', 'paused'].includes(prev)) exploreStart = Date.now();
    updatePanel();
    if (prev === 'exploring' && st.state === 'done') {
      if (autoRecording && recording) $('recordBtn').click();  // stop the recording it started
      autoRecording = false;
      // after the click: its "recorder finishing" hint must not hide why the exploration ended
      setTimeout(() => { $('mHint').textContent = `Exploration finished: ${st.message}. The recording is being saved.`; }, 100);
      const s = Math.floor((Date.now() - (exploreStart || Date.now())) / 1000);
      // e.g. "room done: no unexplored areas left; the camera saw 79 % of the surfaces, 2.0 m could not
      // be reached; the rest are small pieces (< 40 cm)"
      const m = st.message.replace(/^room done: /, '');
      const why = `${m[0].toUpperCase()}${m.slice(1)}.`;
      showBanner('ok', 'Exploration finished', why,
        `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')} min · ${$('mPath').textContent} driven · ` +
        `${Math.round(100 * (st.coverage || 0))} % of the surfaces seen` + (st.given_up_m2 ? ` · ${st.given_up_m2} m² given up` : ''),
        true);
    }
    updateExplore();
  }

  // a message in the middle of the map that stays until dismissed, with a sound (the user is
  // often not looking at the screen when a run ends)
  function showBanner(kind, title, text, stats, offer3d = false) {
    $('runBanner').className = `run-banner chip ${kind}`;
    $('rbTitle').textContent = title;
    $('rbText').textContent = text;
    $('rbStats').textContent = stats || '';
    $('rb3d').hidden = !offer3d;
    $('runBanner').hidden = false;
    try {
      const ac = new AudioContext();
      [0, 0.18].forEach((t, i) => {
        const o = ac.createOscillator();
        const g = ac.createGain();
        o.frequency.value = kind === 'ok' ? (i ? 880 : 660) : 440;
        g.gain.setValueAtTime(0.15, ac.currentTime + t);
        g.gain.exponentialRampToValueAtTime(0.001, ac.currentTime + t + 0.16);
        o.connect(g).connect(ac.destination);
        o.start(ac.currentTime + t);
        o.stop(ac.currentTime + t + 0.17);
      });
    } catch {
      // no audio: the banner is enough
    }
  }
  $('rbClose').addEventListener('click', () => { $('runBanner').hidden = true; });
  $('rb3d').addEventListener('click', () => { $('runBanner').hidden = true; setMode('3d'); });

  function onExploreEvent(ev) {
    exploreEvents.unshift(ev);
    exploreEvents = exploreEvents.slice(0, 30);
    const ol = $('epEvents');
    ol.innerHTML = '';
    for (const e of exploreEvents) {
      const li = document.createElement('li');
      const t = new Date(e.time * 1000);
      li.innerHTML = `<span class="t">${t.toTimeString().slice(0, 8)}</span>`;
      li.append(document.createTextNode(e.text));
      ol.append(li);
    }
  }

  // the exploration panel: current activity, what comes next, progress (never overwritten)
  function updatePanel() {
    const st = explore;
    const panel = $('explorePanel');
    panel.hidden = !st || st.state === 'idle';
    if (panel.hidden) return;
    const titles = { exploring: 'Exploring', paused: 'Paused', done: 'Done', stopped: 'Stopped' };
    const mode = '';  // one exploration mode
    $('epState').textContent = `${titles[st.state] || st.state}${mode ? ` · ${mode}` : ''}`;
    const activity = {
      'looking around': 'Turning once on the spot to look around',
      frontiers: 'Exploring new areas',
      coverage: 'Visiting surfaces the camera has not seen well enough',
      returning: "Driving back to the start's side of a no-go line",
    };
    $('epActivity').textContent = st.state === 'exploring'
      ? (st.blocked ? 'Obstacle ahead: stopped, re-planning around it' : st.message && !/^exploring$/.test(st.message)
        ? st.message[0].toUpperCase() + st.message.slice(1) : activity[st.phase] || 'Planning')
      : st.state === 'paused' ? 'Paused. Resume continues where it stopped; STOP or Space paused it.'
        : st.message ? st.message[0].toUpperCase() + st.message.slice(1) : '';
    let next = '';
    if (st.state === 'exploring') {
      if (st.phase === 'looking around') next = 'Then: drive to the nearest unexplored area';
      else if (st.phase === 'frontiers') next = `Then: ${st.frontiers} unexplored area${st.frontiers === 1 ? '' : 's'} known, afterwards the remaining surfaces`;
      else if (st.phase === 'coverage') next = `Then: ${st.remaining} more viewpoint${st.remaining === 1 ? '' : 's'}, then a final check`;
      else if (st.phase === 'returning') next = "Then: continue exploring on the start's side";
    } else if (st.state === 'done') {
      next = 'The recording is saved: reconstruct it in the 3D view.';
    }
    $('epNext').textContent = next;
    $('epBar').style.width = `${Math.round(100 * (st.coverage || 0))}%`;
    const parts = [`${Math.round(100 * (st.coverage || 0))} % of the surfaces seen`
      + (st.coverage_visible != null ? ` (${Math.round(100 * st.coverage_visible)} % of what was in view)` : '')];
    if (st.given_up_m2) parts.push(`${st.given_up_m2} m² given up`);
    if (st.walls) parts.push(`${st.walls} no-go line${st.walls === 1 ? '' : 's'}`);
    $('epCounts').textContent = parts.join(' · ');
  }

  setInterval(() => {
    if (!exploreStart || !explore || !['exploring', 'paused'].includes(explore.state)) return;
    const s = Math.floor((Date.now() - exploreStart) / 1000);
    $('epTime').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }, 1000);

  // short notice for the user's own actions; fades out, never replaces the exploration status
  function toast(text) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = text;
    $('toasts').append(el);
    setTimeout(() => el.classList.add('out'), 2600);
    setTimeout(() => el.remove(), 3300);
  }

  function onCoverage(msg) {
    const { width: w, height: h, resolution: res, origin } = msg.info;
    const image = document.createElement('canvas');
    image.width = w;
    image.height = h;
    const ctx = image.getContext('2d');
    const img = ctx.createImageData(w, h);
    const obstImage = document.createElement('canvas');
    obstImage.width = w;
    obstImage.height = h;
    const obst = obstImage.getContext('2d').createImageData(w, h);
    // green seen, red not yet, orange given up; translucent so the map, the no-go lines and the
    // paths drawn on top stay readable (floor: Quick 3D's band, lighter than the thin walls)
    const ox = origin.position.x, oy = origin.position.y;
    const sameGrid = map && map.data && map.w === w && map.h === h
      && Math.abs(map.ox - ox) < 1e-6 && Math.abs(map.oy - oy) < 1e-6;
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const v = msg.data[row * w + col];
        if (v < 0) continue;
        if (v === 75) {  // furniture seen by the depth camera: like a wall, a bit muted (own layer)
          const o2 = ((h - 1 - row) * w + col) * 4;
          obst.data[o2] = 175; obst.data[o2 + 1] = 180; obst.data[o2 + 2] = 190; obst.data[o2 + 3] = 255;
          continue;
        }
        const floor = sameGrid && map.data[row * w + col] >= 0 && map.data[row * w + col] < 50;
        const o = ((h - 1 - row) * w + col) * 4;
        const [r, g, b] = v >= 100 ? [40, 200, 90] : v >= 50 ? [235, 70, 70] : [240, 150, 40];  // seen / not yet / given up
        img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = floor ? 110 : 180;
      }
    }
    ctx.putImageData(img, 0, 0);
    obstImage.getContext('2d').putImageData(obst, 0, 0);
    coverage = { image, w, h, res, ox: origin.position.x, oy: origin.position.y };
    obstacles = { image: obstImage, w, h, res, ox: origin.position.x, oy: origin.position.y };
    draw();
  }

  function updateExplore() {
    const btn = $('exploreBtn');
    const st = explore ? explore.state : null;
    btn.disabled = !offboard || !exploreCmd;
    btn.classList.toggle('running', st === 'exploring');
    btn.classList.toggle('paused', st === 'paused');
    btn.textContent = st === 'exploring' ? '❚❚ Pause' : st === 'paused' ? '▶ Resume' : '▶ Explore';
    const pct = explore ? `${Math.round(100 * explore.coverage)} % seen` : '';
    $('exStatus').textContent = !st ? 'off'
      : st === 'exploring' ? `${explore.blocked ? 'blocked' : explore.phase === 'returning' ? 'back to start side' : explore.phase || 'planning'} · ${pct}`
        : st === 'paused' ? `paused · ${pct}` : st === 'done' ? `done · ${pct}` : st === 'stopped' ? 'stopped' : 'ready';
  }

  $('exploreBtn').addEventListener('click', () => {
    const st = explore ? explore.state : 'idle';
    if (st === 'exploring') {
      exploreCmd.publish({ data: 'pause' });
      toast('Exploration paused');
      return;
    }
    if (!$('armed').checked) {
      $('mHint').textContent = 'Tick "Motion armed" first: the exploration drives the robot.';
      return;
    }
    if (st === 'paused') {
      exploreCmd.publish({ data: 'resume' });
      toast('Exploring again. STOP or Space pauses');
      return;
    }
    if (!confirm('Start exploring (up to 0.3 m/s)? The robot drives '
        + 'by itself and records the room.\n\nSTOP or Space pauses it at any time. Keep an eye on it for the first runs.')) return;
    if (!recording) {
      autoRecording = true;
      $('recordBtn').click();  // record the whole exploration
    }
    sendNoGo();
    exploreCmd.publish({ data: 'start' });  // one exploration mode (2026-10-06)
    exploreEvents = [];
    $('epEvents').innerHTML = '';
    toast('Exploration started. STOP or Space pauses');
  });

  // ---------- no-go lines ----------

  function sendNoGo() {
    if (exploreCmd) exploreCmd.publish({ data: `walls ${JSON.stringify(noGo.map((l) => l.map((v) => +v.toFixed(3))))}` });
  }

  function updateNoGo() {
    $('noGoClearBtn').hidden = !noGo.length;
    $('noGoClearBtn').textContent = `✕ ${noGo.length} line${noGo.length === 1 ? '' : 's'}`;
    $('noGoBtn').classList.toggle('drawing', !!drawing);
    cv.classList.toggle('drawing', !!drawing);
  }

  function stopDrawing() {
    drawing = null;
    updateNoGo();
  }

  $('noGoBtn').addEventListener('click', () => {
    if (drawing) {
      stopDrawing();
      $('mHint').textContent = '';
    } else {
      drawing = { first: null };
      updateNoGo();
      $('mHint').textContent = 'No-go line: click its first end on the map (Esc cancels).';
    }
    draw();
  });
  $('noGoClearBtn').addEventListener('click', () => {
    if (!confirm(`Remove all ${noGo.length} no-go lines?`)) return;
    noGo = [];
    sendNoGo();
    updateNoGo();
    draw();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && drawing) {
      stopDrawing();
      $('mHint').textContent = '';
      toast('No-go line cancelled');
      draw();
    }
  });


  // STOP, Space, manual driving or disarming pauses the exploration (app.js)
  document.addEventListener('landerpi:pause-auto', (e) => {
    if (explore && explore.state === 'exploring' && exploreCmd) {
      exploreCmd.publish({ data: 'pause' });
      toast(`Exploration paused (${e.detail})`);
    }
  });

  // ---------- start ----------

  setSlamState(false);
  setMode(location.hash === '#map' ? 'map' : location.hash === '#3d' ? '3d' : settings.view || 'drive');
  if (SERVED_BY_HELPER) connectOffboard();
  else $('mapPlaceholder').textContent = 'Open the console with start_mapping_console.bat to use the map.';
  pollRecorder();
  setInterval(pollRecorder, 1000);
})();

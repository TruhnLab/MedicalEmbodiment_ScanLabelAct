'use strict';

// LanderPi Console: talks to the robot through rosbridge (WebSocket) using roslibjs.

const $ = (id) => document.getElementById(id);

const DEFAULTS = {
  url: 'ws://192.168.149.1:9090',
  cameraTopic: '',
  cameraSource: 'mjpeg',  // like the factory app: fast; rosbridge is the fallback
  cmdVelTopic: '',
  scanTopic: '',
  chassis: 'mecanum',
  maxLin: 0.15,   // m/s
  maxAng: 0.5,    // rad/s
  fps: 5,         // only for the rosbridge video source; raw frames are ~0.8 MB each
  scanRange: 4,   // m
  armSpeed: 150,  // servo units (0-1000 range) per second
  batterySound: true,  // quiet laptop chime on battery warnings
  robotIndicators: true,  // battery status on the robot's RGB LEDs and OLED
  armInvert: {},  // servo id -> true to reverse its key direction
  // arm pose for mapping: camera straight ahead and tilted up from the start pose (tilt 100 looks
  // at the floor); P moves there, Shift+P saves the current pose instead
  surveyPose: { 1: 500, 2: 730, 3: 130, 4: 265, 5: 500 },
};

const SETTINGS_VERSION = 2;

// Preferred topic names, tried in order when nothing was chosen before.
const PREFER = {
  camera: [/rgb0\/image$/, /rgb.*compressed/, /color.*compressed/, /compressed$/, /rgb/, /color/],
  cmdVel: [/^\/controller\/cmd_vel$/, /^\/cmd_vel$/, /cmd_vel/],
  scan: [/^\/scan$/, /scan/],
};

const CAMERA_TYPES = ['sensor_msgs/CompressedImage', 'sensor_msgs/Image'];
const TWIST_TYPE = 'geometry_msgs/Twist';
const SCAN_TYPE = 'sensor_msgs/LaserScan';
const DEPTH_MAX_MM = 4000;
// The helper (tools/robot_helper.py) serves this page at http://127.0.0.1:8765/ and relays the
// robot's video; opened as a file, the browser blocks the video (Local Network Access).
const SERVED_BY_HELPER = /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(location.origin);
const HELPER_URL = SERVED_BY_HELPER ? location.origin : 'http://127.0.0.1:8765';
const BATTERY_TOPIC = '/ros_robot_controller/battery';  // UInt16, millivolts

// Topics verified on our LanderPi. Used when rosapi (topic discovery) is not available.
const KNOWN_TOPICS = {
  '/ascamera/camera_publisher/rgb0/image': 'sensor_msgs/msg/Image',
  '/ascamera/camera_publisher/depth0/image_raw': 'sensor_msgs/msg/Image',
  '/ascamera/camera_publisher/ir0/image': 'sensor_msgs/msg/Image',
  '/controller/cmd_vel': 'geometry_msgs/msg/Twist',
  '/cmd_vel': 'geometry_msgs/msg/Twist',
  '/scan': 'sensor_msgs/msg/LaserScan',
  '/scan_raw': 'sensor_msgs/msg/LaserScan',
  '/odom': 'nav_msgs/msg/Odometry',
  '/imu': 'sensor_msgs/msg/Imu',
  '/joint_states': 'sensor_msgs/msg/JointState',
  '/servo_controller': 'servo_controller_msgs/msg/ServosPosition',
  '/controller_manager/servo_states': 'servo_controller_msgs/msg/ServoStateList',
  '/ros_robot_controller/battery': 'std_msgs/msg/UInt16',
  '/ros_robot_controller/set_rgb': 'ros_robot_controller_msgs/msg/RGBStates',
  '/ros_robot_controller/set_oled': 'ros_robot_controller_msgs/msg/OLEDState',
};

let settings = loadSettings();
let ros = null;
let topicTypes = {};  // topic name -> message type
let cameraSub = null;
let mjpegView = null;  // MjpegView while the camera comes as MJPEG
let batterySub = null;
let scanSub = null;
let echoSub = null;
let cmdVelPub = null;
let servoPub = null;
let servoStateSub = null;

// ---------- settings ----------

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem('landerpi-console') || '{}');
    // v2: MJPEG became the default video source (rosbridge was the default before, because the
    // robot's video server looked broken; it only needed a restart)
    if ((saved.version || 1) < 2) delete saved.cameraSource;
    return { ...DEFAULTS, ...saved, version: SETTINGS_VERSION };
  } catch {
    return { ...DEFAULTS, version: SETTINGS_VERSION };
  }
}

function saveSettings() {
  try {
    localStorage.setItem('landerpi-console', JSON.stringify(settings));
  } catch {
    // storage unavailable (e.g. blocked for file://) - settings just won't persist
  }
}

// ---------- helpers ----------

const normType = (t) => (t || '').replace('/msg/', '/');

function log(msg, level = 'info') {
  const el = $('log');
  const line = document.createElement('div');
  line.className = level;
  line.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  el.prepend(line);
  while (el.childElementCount > 200) el.lastChild.remove();
  // the latest message is also shown on the dashboard, since the full log lives in the drawer
  const last = $('lastLog');
  last.textContent = msg;
  last.className = `lastlog ${level === 'info' ? 'muted' : level}`;
}

function setStatus(state, text) {
  const el = $('status');
  el.dataset.state = state;
  el.textContent = text;
  $('connectBtn').textContent = state === 'off' || state === 'error' ? 'Connect' : 'Disconnect';
}

// rosbridge (ROS 2) sends uint8[] as base64 strings; older setups may send plain arrays.
function toBase64(data) {
  if (typeof data === 'string') return data;
  let s = '';
  for (let i = 0; i < data.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, data.slice(i, i + 0x8000));
  }
  return btoa(s);
}

function toBytes(data) {
  if (typeof data !== 'string') return Uint8Array.from(data);
  const bin = atob(data);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Shortens long arrays/strings so large messages stay readable in the echo view.
function summarize(v) {
  if (typeof v === 'string' && v.length > 200) return `${v.slice(0, 200)}… (${v.length} chars)`;
  if (Array.isArray(v)) {
    const head = v.slice(0, 20).map(summarize);
    if (v.length > 20) head.push(`… ${v.length - 20} more`);
    return head;
  }
  if (v && typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v)) o[k] = summarize(v[k]);
    return o;
  }
  return v;
}

// ---------- connection ----------

// After a lost connection (robot rebooted, rosbridge restarted by the helper, Wi-Fi hiccup) the
// console reconnects by itself until the user presses Disconnect.
let keepConnected = false;
let reconnectTimer = null;

function connect() {
  disconnect();
  keepConnected = true;
  settings.url = $('url').value.trim();
  saveSettings();
  setStatus('connecting', 'Connecting…');

  const r = new ROSLIB.Ros({ url: settings.url });
  ros = r;
  r.on('connection', () => {
    if (r !== ros) return;
    setStatus('ok', 'Online');
    log(`Connected to ${settings.url}`);
    // Leave the URL field so W/A/S/D go to the drive controls instead of typing.
    if (document.activeElement) document.activeElement.blur();
    refreshTopics();
  });
  r.on('error', () => {
    if (r !== ros) return;
    setStatus('error', 'Error');
    log(`Cannot reach ${settings.url}. Is the laptop on the robot's Wi-Fi and is rosbridge running?`, 'error');
  });
  r.on('close', () => {
    if (r !== ros) return;
    teardown();
    if (keepConnected) {
      setStatus('connecting', 'Reconnecting…');
      log('Connection lost, reconnecting…', 'warn');
      reconnectTimer = setTimeout(connect, 3000);
    } else {
      setStatus('off', 'Offline');
      log('Connection closed', 'warn');
    }
  });
}

function disconnect() {
  keepConnected = false;
  clearTimeout(reconnectTimer);
  if (!ros) return;
  sendStop();
  clearRobotIndicators();
  const r = ros;
  ros = null;
  teardown();
  r.close();
  setStatus('off', 'Offline');
}

function teardown() {
  setArmed(false);
  pressed.clear();
  armKeys.clear();
  armButtons.clear();
  stopCamera();
  for (const sub of [batterySub, scanSub, echoSub, servoStateSub]) if (sub) sub.unsubscribe();
  for (const pub of [cmdVelPub, servoPub]) if (pub) pub.unadvertise();
  batterySub = scanSub = echoSub = servoStateSub = cmdVelPub = servoPub = null;
  showCamPlaceholder('Not connected');
  resetBattery();
}

function refreshTopics() {
  if (!ros) return;
  ros.getTopics((res) => {
    topicTypes = {};
    res.topics.forEach((t, i) => { topicTypes[t] = res.types[i]; });
    log(`Found ${res.topics.length} topics`);
    renderTopicTable();
    fillSelectors();
  }, (err) => {
    log(`Could not list topics (${err}). Using the known LanderPi topics instead. Rebooting the robot restores discovery.`, 'warn');
    topicTypes = { ...KNOWN_TOPICS };
    renderTopicTable();
    fillSelectors();
  });

  ros.getNodes((nodes) => {
    $('nodeList').textContent = nodes.slice().sort().join('\n') || '(none)';
  }, (err) => log(`Could not list nodes: ${err}`, 'error'));
}

// ---------- topic selection ----------

function topicsOfType(types) {
  return Object.keys(topicTypes).filter((n) => types.includes(normType(topicTypes[n]))).sort();
}

function pickTopic(names, current, prefer) {
  if (names.includes(current)) return current;
  for (const re of prefer) {
    const hit = names.find((n) => re.test(n));
    if (hit) return hit;
  }
  return '';
}

function fillSelect(sel, names, value) {
  sel.innerHTML = '';
  sel.append(new Option('— none —', ''));
  for (const n of names) sel.append(new Option(`${n}  [${normType(topicTypes[n]).split('/')[1]}]`, n));
  sel.value = value;
}

function fillSelectors() {
  const cams = topicsOfType(CAMERA_TYPES);
  fillSelect($('cameraTopic'), cams, pickTopic(cams, settings.cameraTopic, PREFER.camera));

  const scans = topicsOfType([SCAN_TYPE]);
  fillSelect($('scanTopic'), scans, pickTopic(scans, settings.scanTopic, PREFER.scan));

  const twists = topicsOfType([TWIST_TYPE]);
  $('twistTopics').innerHTML = '';
  for (const n of twists) $('twistTopics').append(new Option(n));
  if (!settings.cmdVelTopic) $('cmdVelTopic').value = pickTopic(twists, '', PREFER.cmdVel);

  startCamera();
  startScan();
  startBattery();
  setupCmdVel();
  startArm();
}

// ---------- battery ----------

// 2S Li-ion pack: 8.4 V full, ~7.0 V empty. The robot's own (loud) buzzer goes off around 7.0 V,
// so warn quietly well before that and stop driving motors/servos before it gets there.
const BATTERY_LOW = 7.4;
const BATTERY_CRITICAL = 7.1;
const BATTERY_HYSTERESIS = 0.1;  // must rise this far above a threshold to clear it
const BATTERY_AVG_MS = 10000;    // average out dips while motors/servos draw current

const batterySamples = [];  // [time, volts]
let batteryLevel = 'none';

function startBattery() {
  if (batterySub) batterySub.unsubscribe();
  batterySub = null;
  resetBattery();
  if (!ros || !topicTypes[BATTERY_TOPIC]) return;
  batterySub = new ROSLIB.Topic({
    ros, name: BATTERY_TOPIC, messageType: topicTypes[BATTERY_TOPIC], throttle_rate: 2000, queue_length: 1,
  });
  batterySub.subscribe((msg) => onBattery(msg.data / 1000));
  if (topicTypes[RGB_TOPIC]) rgbPub = new ROSLIB.Topic({ ros, name: RGB_TOPIC, messageType: 'ros_robot_controller_msgs/msg/RGBStates' });
  if (topicTypes[OLED_TOPIC]) oledPub = new ROSLIB.Topic({ ros, name: OLED_TOPIC, messageType: 'ros_robot_controller_msgs/msg/OLEDState' });
}

function resetBattery() {
  batterySamples.length = 0;
  batteryLevel = 'none';
  showBattery(null);
  showBanner(null);
  stopLedBlink();
  rgbPub = oledPub = null;
}

const batteryPercent = (v) => Math.round(Math.min(1, Math.max(0, (v - 7.0) / 1.4)) * 100);

function classifyBattery(v, prev) {
  const h = BATTERY_HYSTERESIS;
  if (v < BATTERY_CRITICAL || (prev === 'critical' && v < BATTERY_CRITICAL + h)) return 'critical';
  if (v < BATTERY_LOW || ((prev === 'low' || prev === 'critical') && v < BATTERY_LOW + h)) return 'low';
  return 'ok';
}

function onBattery(volts) {
  const now = performance.now();
  batterySamples.push([now, volts]);
  while (now - batterySamples[0][0] > BATTERY_AVG_MS) batterySamples.shift();
  const avg = batterySamples.reduce((sum, [, v]) => sum + v, 0) / batterySamples.length;

  const prev = batteryLevel;
  batteryLevel = classifyBattery(avg, prev);
  showBattery(avg);
  updateRobotIndicators(avg, batteryLevel !== prev);
  if (batteryLevel === prev) return;

  const pct = batteryPercent(avg);
  if (batteryLevel === 'critical') {
    if (document.body.classList.contains('is-armed')) emergencyStop();
    setArmed(false);
    showBanner(`Battery critical (${avg.toFixed(2)} V). Motion disarmed. Switch off and charge now, before the robot starts beeping.`, 'critical');
    log(`Battery critical (${avg.toFixed(2)} V): motion disarmed.`, 'error');
    chime(2);
  } else if (batteryLevel === 'low') {
    showBanner(`Battery low (${avg.toFixed(2)} V, ~${pct}%). Finish up and charge soon.`, 'low');
    // only chime when getting worse, not when recovering from critical
    if (prev !== 'critical') {
      log(`Battery low (${avg.toFixed(2)} V, ~${pct}%).`, 'warn');
      chime(1);
    }
  } else if (batteryLevel === 'ok') {
    showBanner(null);
  }
}

// ---------- battery status on the robot (RGB LEDs + OLED), a silent alternative to its buzzer ----------

const RGB_TOPIC = '/ros_robot_controller/set_rgb';
const OLED_TOPIC = '/ros_robot_controller/set_oled';
const LED_COLORS = { ok: [0, 40, 0], low: [140, 60, 0], critical: [160, 0, 0] };
const OLED_STATUS = { ok: 'Battery OK', low: 'LOW: charge soon', critical: 'CHARGE NOW!' };
const OLED_EVERY_MS = 10000;

let rgbPub = null;
let oledPub = null;
let ledBlink = null;
let lastOled = 0;

function setLeds([r, g, b]) {
  if (!rgbPub) return;
  rgbPub.publish({ states: [1, 2].map((index) => ({ index, red: r, green: g, blue: b })) });
}

function setOled(line1, line2) {
  if (!oledPub) return;
  oledPub.publish({ index: 1, text: line1 });
  oledPub.publish({ index: 2, text: line2 });
}

function stopLedBlink() {
  clearInterval(ledBlink);
  ledBlink = null;
}

function updateRobotIndicators(volts, levelChanged) {
  if (!settings.robotIndicators) return;
  if (levelChanged) {
    stopLedBlink();
    setLeds(LED_COLORS[batteryLevel]);
    if (batteryLevel === 'critical') {
      let on = true;
      ledBlink = setInterval(() => { on = !on; setLeds(on ? LED_COLORS.critical : [0, 0, 0]); }, 700);
    }
  }
  const now = performance.now();
  if (levelChanged || now - lastOled > OLED_EVERY_MS) {
    setOled(`Batt ${volts.toFixed(2)}V ${batteryPercent(volts)}%`, OLED_STATUS[batteryLevel]);
    lastOled = now;
  }
}

function clearRobotIndicators() {
  stopLedBlink();
  setLeds([0, 0, 0]);
  setOled('LanderPi', 'console offline');
}

function showBattery(volts) {
  const el = $('battery');
  el.dataset.level = volts == null ? 'none' : batteryLevel;
  el.textContent = volts == null ? '– V' : `${volts.toFixed(2)} V · ${batteryPercent(volts)}%`;
}

function showBanner(text, level) {
  const el = $('banner');
  el.textContent = text || '';
  el.dataset.level = level || '';
  el.hidden = !text;
}

// A short, quiet two-tone chime from the laptop instead of the robot's buzzer.
let audioCtx = null;

function chime(times) {
  if (!settings.batterySound) return;
  try {
    audioCtx = audioCtx || new AudioContext();
    const t0 = audioCtx.currentTime;
    for (let i = 0; i < times; i++) {
      [660, 880].forEach((freq, j) => {
        const start = t0 + i * 0.6 + j * 0.16;
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0, start);
        gain.gain.linearRampToValueAtTime(0.08, start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + 0.3);
        osc.connect(gain).connect(audioCtx.destination);
        osc.start(start);
        osc.stop(start + 0.32);
      });
    }
  } catch {
    // audio not available - the banner and log still show the warning
  }
}

function renderTopicTable() {
  const filter = $('topicFilter').value.trim().toLowerCase();
  const body = $('topicRows');
  body.innerHTML = '';
  for (const name of Object.keys(topicTypes).sort()) {
    const type = topicTypes[name];
    if (filter && !name.toLowerCase().includes(filter) && !type.toLowerCase().includes(filter)) continue;
    const tr = document.createElement('tr');
    tr.dataset.topic = name;
    if (echoSub && echoSub.name === name) tr.className = 'selected';
    tr.innerHTML = '<td></td><td></td>';
    tr.children[0].textContent = name;
    tr.children[1].textContent = type;
    tr.addEventListener('click', () => startEcho(name));
    body.append(tr);
  }
}

function startEcho(name) {
  if (echoSub) echoSub.unsubscribe();
  echoSub = null;
  if (!ros) return;
  const type = topicTypes[name];
  $('echoTitle').textContent = `${name}  [${type}]  (waiting for messages…)`;
  $('echoOut').textContent = '';
  echoSub = new ROSLIB.Topic({ ros, name, messageType: type, throttle_rate: 500, queue_length: 1 });
  echoSub.subscribe((msg) => {
    $('echoTitle').textContent = `${name}  [${type}]  ${new Date().toLocaleTimeString()}`;
    $('echoOut').textContent = JSON.stringify(summarize(msg), null, 2);
  });
  renderTopicTable();
}

// ---------- camera ----------

const cam = { frames: 0, lastFrame: 0, info: '', warned: false };

function showCamPlaceholder(text) {
  $('camImg').style.display = 'none';
  $('camCanvas').style.display = 'none';
  $('camPlaceholder').style.display = '';
  $('camPlaceholder').textContent = text;
  $('camStats').textContent = '–';
}

function stopCamera() {
  if (cameraSub) cameraSub.unsubscribe();
  cameraSub = null;
  if (mjpegView) mjpegView.stop();
  mjpegView = null;
  const img = $('camImg');
  img.onload = img.onerror = null;
  img.removeAttribute('src');
}

// Reads an MJPEG stream with fetch() and shows only the newest frame. An <img> would decode every
// frame on the main thread and fall behind whenever the page is busy; here frames that arrive
// while one is being drawn are simply dropped, and decoding happens off the main thread.
class MjpegView {
  constructor(url, canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.frames = 0;
    this.latest = null;       // { jpeg, stamp }
    this.busy = false;
    this.clockOffset = null;  // min(laptop time - robot stamp): delay beyond it is "extra"
    this.extraDelayMs = null;
    this.controller = new AbortController();
    this.done = this.read(url).catch(() => {});
  }

  stop() {
    this.controller.abort();
  }

  async read(url) {
    const res = await fetch(url, { signal: this.controller.signal, cache: 'no-store' });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      const joined = new Uint8Array(buf.length + value.length);
      joined.set(buf);
      joined.set(value, buf.length);
      buf = joined;
      // parts: headers (with Content-Length, X-Timestamp), blank line, JPEG bytes
      for (;;) {
        const end = indexOfHeaderEnd(buf);
        if (end < 0) break;
        const headers = decoder.decode(buf.subarray(0, end));
        const length = Number((/content-length:\s*(\d+)/i.exec(headers) || [])[1]);
        if (!length) { buf = buf.slice(end + 4); continue; }
        if (buf.length < end + 4 + length) break;
        const stamp = Number((/x-timestamp:\s*([\d.]+)/i.exec(headers) || [])[1]) || null;
        this.latest = { jpeg: buf.slice(end + 4, end + 4 + length), stamp };
        buf = buf.slice(end + 4 + length);
        this.schedule();
      }
    }
  }

  schedule() {
    if (this.busy) return;  // the frame being drawn is followed by whatever is newest then
    this.busy = true;
    requestAnimationFrame(() => this.draw());
  }

  async draw() {
    const frame = this.latest;
    this.latest = null;
    try {
      const bmp = await createImageBitmap(new Blob([frame.jpeg], { type: 'image/jpeg' }));
      if (this.canvas.width !== bmp.width) this.canvas.width = bmp.width;
      if (this.canvas.height !== bmp.height) this.canvas.height = bmp.height;
      this.ctx.drawImage(bmp, 0, 0);
      bmp.close();
      this.frames++;
      if (frame.stamp) {
        const delta = Date.now() / 1000 - frame.stamp;
        if (this.clockOffset === null || delta < this.clockOffset) this.clockOffset = delta;
        this.extraDelayMs = Math.round(1000 * (delta - this.clockOffset));
      }
      const delay = this.extraDelayMs === null ? '' : ` · +${this.extraDelayMs} ms`;
      frameShown('canvas', `${this.canvas.width}×${this.canvas.height} MJPEG${delay}`);
    } catch {
      // a broken frame: skip it
    }
    this.busy = false;
    if (this.latest) this.schedule();
  }
}

function indexOfHeaderEnd(buf) {
  for (let i = 0; i + 3 < buf.length; i++) {
    if (buf[i] === 13 && buf[i + 1] === 10 && buf[i + 2] === 13 && buf[i + 3] === 10) return i;
  }
  return -1;
}

function startCamera() {
  stopCamera();
  const name = $('cameraTopic').value;
  settings.cameraTopic = name;
  saveSettings();
  if (!ros) return showCamPlaceholder('Not connected');
  if (!name) return showCamPlaceholder('No camera topic selected');

  const type = normType(topicTypes[name]);
  const compressed = type.endsWith('CompressedImage');
  // MJPEG for camera images; depth always comes over rosbridge (as 16-bit PNG it's small, and
  // JPEG would destroy the millimetre values)
  if (settings.cameraSource === 'mjpeg' && !compressed && !/depth/.test(name)) return startMjpeg(name);
  cam.frames = 0;
  cam.lastFrame = 0;
  cam.warned = false;
  showCamPlaceholder('Waiting for frames…');
  cameraSub = new ROSLIB.Topic({
    ros, name, messageType: topicTypes[name],
    throttle_rate: Math.round(1000 / settings.fps), queue_length: 1,
    // note: compression 'cbor' would be lighter, but the robot's rosbridge sends nothing with it
  });
  cameraSub.subscribe(compressed ? onCompressedImage : onRawImage);
  log(`Camera: ${name}`);
}

// Video straight from web_video_server, like the factory app: JPEG-compressed on the robot,
// ~14 fps with a few ms latency (raw images over rosbridge took ~1.7 s).
function startMjpeg(name, retried = false) {
  // Browsers block pages from loading anything from local-network devices ("Local Network
  // Access"), so the video comes through the helper, which also serves this page.
  if (!SERVED_BY_HELPER) {
    showCamPlaceholder('Video needs the local helper: open the console with start_mapping_console.bat (http://127.0.0.1:8765/).');
    log('Opened as a file: the browser blocks the robot video. Use start_mapping_console.bat.', 'error');
    return;
  }
  // keep the slashes literal: web_video_server doesn't decode %2F and would subscribe to a
  // non-existent topic (it then sends the stream header and never a frame)
  const topic = encodeURIComponent(name).replace(/%2F/g, '/');
  const query = `topic=${topic}&type=mjpeg&quality=85`;
  const view = mjpegView = new MjpegView(`${location.origin}/video?${query}`, $('camCanvas'));
  cam.frames = 0;
  cam.lastFrame = 0;
  showCamPlaceholder('Waiting for video…');
  // If no frame arrives (robot video server stopped?), have the local
  // helper restart it once; if that doesn't help, fall back to rosbridge for this session.
  setTimeout(async () => {
    if (mjpegView !== view || view.frames > 0) return;
    if (!retried) {
      log('No video frames. Restarting the robot\'s video server…', 'warn');
      stopCamera();
      try {
        const r = await helperFetch('/api/restart-video', { method: 'POST' });
        if (!r.ok) throw new Error(r.error);
        setTimeout(() => startMjpeg(name, true), 3000);
        return;
      } catch {
        log('Could not restart the video server (is the helper running? start_mapping_console.bat).', 'warn');
      }
    }
    log('Video server sends no frames. Using rosbridge video for now (slower).', 'error');
    settings.cameraSource = 'rosbridge';  // not saved: MJPEG stays the default next time
    startCamera();
  }, 5000);
  log(`Camera (MJPEG): ${name}`);
}

function frameShown(el, info) {
  $('camPlaceholder').style.display = 'none';
  $('camImg').style.display = el === 'img' ? 'block' : 'none';
  $('camCanvas').style.display = el === 'canvas' ? 'block' : 'none';
  cam.frames++;
  cam.lastFrame = performance.now();
  cam.info = info;
}

// Colour map for depth, t = 0 (near) .. 1 (far): near = red, then yellow, cyan, far = blue.
function depthColour(t) {
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  return [255 * r, 255 * g, 255 * b];
}

// 16-bit PNG depth (millimetres). Browsers only decode PNGs to 8 bits, which drops the lower
// byte and makes 0-2 m look black, so decode it ourselves (UPNG) and colour it by distance.
function onDepthPng(msg) {
  const png = UPNG.decode(toBytes(msg.data).buffer);
  const { width: w, height: h } = png;
  const src = png.data;  // big-endian 16-bit samples
  // colour scale: up to the 98th percentile of a pixel sample, so near scenes still show detail
  const sample = [];
  for (let i = 0; i < w * h; i += 97) {
    const mm = (src[2 * i] << 8) | src[2 * i + 1];
    if (mm > 0) sample.push(mm);
  }
  sample.sort((a, b) => a - b);
  const maxMm = sample.length ? sample[Math.floor(sample.length * 0.98)] : DEPTH_MAX_MM;
  const cv = $('camCanvas');
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(w, h);
  const out = img.data;
  for (let i = 0; i < w * h; i++) {
    const mm = (src[2 * i] << 8) | src[2 * i + 1];
    const o = 4 * i;
    if (mm > 0) {
      const [r, g, b] = depthColour(Math.min(1, mm / maxMm));
      out[o] = r; out[o + 1] = g; out[o + 2] = b;
    }
    out[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  frameShown('canvas', `${w}×${h} depth · colour scale 0–${(maxMm / 1000).toFixed(1)} m`);
}

function onCompressedImage(msg) {
  if (/mono16|16UC1/.test(msg.format)) return onDepthPng(msg);
  const mime = /png/i.test(msg.format) ? 'image/png' : 'image/jpeg';
  const img = $('camImg');
  img.onload = () => frameShown('img', `${img.naturalWidth}×${img.naturalHeight} · ${msg.format}`);
  img.src = `data:${mime};base64,${toBase64(msg.data)}`;
}

// Converts one pixel of a raw image into RGB. Returns null for unsupported encodings.
function pixelReader(encoding, src, bigEndian) {
  const u16 = (i) => (bigEndian ? (src[i] << 8) | src[i + 1] : src[i] | (src[i + 1] << 8));
  const depthGray = (mm) => (mm <= 0 ? 0 : 255 - Math.min(255, (mm / DEPTH_MAX_MM) * 255));
  const view = new DataView(src.buffer);
  switch (encoding) {
    case 'rgb8': return (r, x) => [src[r + x * 3], src[r + x * 3 + 1], src[r + x * 3 + 2]];
    case 'bgr8': return (r, x) => [src[r + x * 3 + 2], src[r + x * 3 + 1], src[r + x * 3]];
    case 'rgba8': return (r, x) => [src[r + x * 4], src[r + x * 4 + 1], src[r + x * 4 + 2]];
    case 'bgra8': return (r, x) => [src[r + x * 4 + 2], src[r + x * 4 + 1], src[r + x * 4]];
    case 'mono8': return (r, x) => [src[r + x], src[r + x], src[r + x]];
    case 'mono16':
    case '16UC1': return (r, x) => { const g = depthGray(u16(r + x * 2)); return [g, g, g]; };
    case '32FC1': return (r, x) => { const g = depthGray(view.getFloat32(r + x * 4, !bigEndian) * 1000); return [g, g, g]; };
    default: return null;
  }
}

function onRawImage(msg) {
  const { width: w, height: h, step, encoding } = msg;
  const src = toBytes(msg.data);
  const read = pixelReader(encoding, src, !!msg.is_bigendian);
  if (!read) {
    if (!cam.warned) log(`Unsupported image encoding "${encoding}"`, 'error');
    cam.warned = true;
    return;
  }
  const cv = $('camCanvas');
  if (cv.width !== w) cv.width = w;
  if (cv.height !== h) cv.height = h;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(w, h);
  const out = img.data;
  if (encoding === 'bgr8' || encoding === 'rgb8') {
    // fast path for the colour camera
    const [ri, bi] = encoding === 'bgr8' ? [2, 0] : [0, 2];
    for (let y = 0; y < h; y++) {
      let i = y * step;
      let o = y * w * 4;
      for (let x = 0; x < w; x++, i += 3, o += 4) {
        out[o] = src[i + ri]; out[o + 1] = src[i + 1]; out[o + 2] = src[i + bi]; out[o + 3] = 255;
      }
    }
  } else {
    for (let y = 0; y < h; y++) {
      const row = y * step;
      for (let x = 0; x < w; x++) {
        const [r, g, b] = read(row, x);
        const o = (y * w + x) * 4;
        out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  frameShown('canvas', `${w}×${h} · ${encoding}`);
}

setInterval(() => {
  if (!cameraSub && !mjpegView) return;
  const age = (performance.now() - cam.lastFrame) / 1000;
  if (cam.lastFrame && age > 2) {
    $('camStats').textContent = `No frames for ${age.toFixed(0)} s. Is the camera node running?`;
  } else if (cam.lastFrame) {
    $('camStats').textContent = `${cam.info} · ${cam.frames} fps`;
  }
  cam.frames = 0;
}, 1000);

// ---------- lidar ----------

let lastScan = null;

function startScan() {
  if (scanSub) scanSub.unsubscribe();
  scanSub = null;
  lastScan = null;
  const name = $('scanTopic').value;
  settings.scanTopic = name;
  saveSettings();
  drawScan();
  if (!ros || !name) return;
  scanSub = new ROSLIB.Topic({ ros, name, messageType: topicTypes[name], throttle_rate: 200, queue_length: 1 });
  scanSub.subscribe((msg) => { lastScan = msg; drawScan(); });
  log(`Lidar: ${name}`);
}

function drawScan() {
  const cv = $('scanCanvas');
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  const h = cv.clientHeight;
  if (!w || !h) return;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const css = getComputedStyle(document.documentElement);
  const cx = w / 2;
  const cy = h / 2;
  const range = settings.scanRange;
  const scale = (Math.min(w, h) / 2 - 6) / range;

  // 1 m rings
  ctx.strokeStyle = css.getPropertyValue('--scan-grid');
  ctx.lineWidth = 1;
  for (let m = 1; m <= range; m++) {
    ctx.beginPath();
    ctx.arc(cx, cy, m * scale, 0, Math.PI * 2);
    ctx.stroke();
  }

  // robot marker, forward = up
  ctx.fillStyle = css.getPropertyValue('--robot');
  ctx.beginPath();
  ctx.moveTo(cx, cy - 8); ctx.lineTo(cx - 5, cy + 5); ctx.lineTo(cx + 5, cy + 5);
  ctx.closePath();
  ctx.fill();

  if (!lastScan) {
    $('scanStats').textContent = scanSub ? 'Waiting for scans…' : '–';
    return;
  }

  const s = lastScan;
  let valid = 0;
  let nearest = Infinity;
  ctx.fillStyle = css.getPropertyValue('--scan-point');
  s.ranges.forEach((r, i) => {
    // rosbridge sends inf/NaN as null
    if (r == null || !isFinite(r) || r < s.range_min || r > s.range_max) return;
    valid++;
    nearest = Math.min(nearest, r);
    if (r > range) return;
    const a = s.angle_min + i * s.angle_increment;
    const px = cx - Math.sin(a) * r * scale;
    const py = cy - Math.cos(a) * r * scale;
    ctx.fillRect(px - 1, py - 1, 2, 2);
  });
  $('scanStats').textContent =
    `nearest ${isFinite(nearest) ? nearest.toFixed(2) + ' m' : '–'} · ${valid} pts · rings 1 m`;
}

new ResizeObserver(drawScan).observe($('scanCanvas'));

// ---------- driving ----------

const pressed = new Set();
// Left hand drives; arrow keys belong to the arm (see ARM_BINDINGS).
const KEYMAP = {
  w: 'fwd', s: 'back',
  a: 'left', d: 'right',
  q: 'strafeL', e: 'strafeR',
};
let wasMoving = false;

function setupCmdVel() {
  if (cmdVelPub) cmdVelPub.unadvertise();
  cmdVelPub = null;
  const name = $('cmdVelTopic').value.trim();
  settings.cmdVelTopic = name;
  saveSettings();
  if (!ros || !name) return;
  const existing = topicTypes[name];
  if (existing && normType(existing) !== TWIST_TYPE) {
    log(`${name} has type ${existing}, not Twist. Driving is disabled for this topic.`, 'error');
    return;
  }
  if (!existing) log(`${name} does not exist on the robot yet. Commands may go nowhere.`, 'warn');
  cmdVelPub = new ROSLIB.Topic({ ros, name, messageType: 'geometry_msgs/msg/Twist' });
  log(`Velocity topic: ${name}`);
}

function currentTwist() {
  let lx = 0, ly = 0, az = 0;
  if (pressed.has('fwd')) lx += 1;
  if (pressed.has('back')) lx -= 1;
  if (pressed.has('left')) az += 1;
  if (pressed.has('right')) az -= 1;
  if (settings.chassis === 'mecanum') {
    if (pressed.has('strafeL')) ly += 1;
    if (pressed.has('strafeR')) ly -= 1;
  }
  return {
    linear: { x: lx * settings.maxLin, y: ly * settings.maxLin, z: 0 },
    angular: { x: 0, y: 0, z: az * settings.maxAng },
  };
}

function publishTwist(t) {
  if (cmdVelPub) cmdVelPub.publish(t);
}

function sendStop() {
  const zero = { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };
  // sent twice in case one message is lost
  publishTwist(zero);
  publishTwist(zero);
  wasMoving = false;
}

// Releases all held keys/buttons and stops manual driving.
function releaseControls() {
  pressed.clear();
  armKeys.clear();
  armButtons.clear();
  sendStop();
  updatePadHighlight();
}

// Asks automatic driving (the Map view's exploration, map_view.js) to pause; it can be resumed.
function pauseAutonomy(reason) {
  document.dispatchEvent(new CustomEvent('landerpi:pause-auto', { detail: reason }));
}

// STOP button / Space: everything stops, including a running exploration (paused, resumable).
function emergencyStop() {
  releaseControls();
  pauseAutonomy('STOP');
}

// Sends commands at 10 Hz while something is held, plus a stop when released.
setInterval(() => {
  const t = currentTwist();
  const moving = t.linear.x !== 0 || t.linear.y !== 0 || t.angular.z !== 0;
  const armed = $('armed').checked;
  $('twistReadout').textContent = !cmdVelPub ? 'No velocity topic'
    : !armed ? 'Not armed. Nothing is sent.'
      : `vx ${t.linear.x.toFixed(2)}  vy ${t.linear.y.toFixed(2)}  wz ${t.angular.z.toFixed(2)}`;
  if (!armed) return;
  if (moving) publishTwist(t);
  else if (wasMoving) sendStop();
  wasMoving = moving;
}, 100);

function updatePadHighlight() {
  document.querySelectorAll('.pad [data-action]').forEach((b) => {
    b.classList.toggle('active', pressed.has(b.dataset.action));
  });
}

let lastNotArmedHint = 0;

function hintIfNotArmed() {
  if ($('armed').checked || performance.now() - lastNotArmedHint < 3000) return;
  lastNotArmedHint = performance.now();
  log(!ros ? 'Key ignored: not connected.' : 'Key ignored: tick "Motion armed" first.', 'warn');
}

const modifierOf = (e) => (e.ctrlKey ? 'ctrl' : e.shiftKey ? 'shift' : '');

document.addEventListener('keydown', (e) => {
  // Only text fields keep the keys for typing; everything else controls the robot.
  if (e.target.matches('input:not([type]), input[type="text"], input[type="number"], textarea')) return;
  const k = e.key.toLowerCase();
  armMod = modifierOf(e);
  if (k === ' ') { e.preventDefault(); emergencyStop(); return; }
  // Letter keys with Ctrl stay browser shortcuts (Ctrl+R, Ctrl+F, ...).
  if (ARM_KEYS.has(k) && !(k.length === 1 && e.ctrlKey)) {
    e.preventDefault();
    armKeys.add(k);
    hintIfNotArmed();
    return;
  }
  if (k === 'h' && !e.ctrlKey) {
    e.preventDefault();
    if (!e.repeat) armHome();
    return;
  }
  if (k === 'p' && !e.ctrlKey) {
    e.preventDefault();
    if (!e.repeat) (e.shiftKey ? saveSurveyPose : armSurvey)();
    return;
  }
  const action = KEYMAP[k];
  if (action && !e.ctrlKey) {
    e.preventDefault();
    pressed.add(action);
    updatePadHighlight();
    hintIfNotArmed();
    if ($('armed').checked && !e.repeat) pauseAutonomy('manual driving');  // the human takes over
  }
});

document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  armMod = modifierOf(e);
  armKeys.delete(k);
  const action = KEYMAP[k];
  if (action) { pressed.delete(action); updatePadHighlight(); }
});

// Stop manual driving when the window loses focus, so a key release can't be missed (a running
// exploration keeps going: it doesn't depend on held keys).
window.addEventListener('blur', releaseControls);
document.addEventListener('visibilitychange', () => { if (document.hidden) releaseControls(); });
window.addEventListener('beforeunload', () => {
  sendStop();
  stopCamera();  // close the video stream cleanly
});

document.querySelectorAll('.pad [data-action]').forEach((btn) => {
  const action = btn.dataset.action;
  btn.addEventListener('pointerdown', (e) => {
    btn.setPointerCapture(e.pointerId);
    pressed.add(action);
    updatePadHighlight();
    hintIfNotArmed();
    if ($('armed').checked) pauseAutonomy('manual driving');
  });
  for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    btn.addEventListener(ev, () => { pressed.delete(action); updatePadHighlight(); });
  }
});

// ---------- arm & gripper ----------

// Hiwonder bus servos: positions 0-1000, 500 = centre. IDs as reported by our robot.
const SERVO_CMD_TOPIC = '/servo_controller';
const SERVO_STATE_TOPIC = '/controller_manager/servo_states';
const ARM_JOINTS = [
  { id: 1, name: 'Base pan', keys: '← →' },
  { id: 4, name: 'Camera tilt', keys: '↑ ↓' },
  { id: 2, name: 'Shoulder', keys: '⇧ ↑ ↓' },
  { id: 3, name: 'Elbow', keys: 'Ctrl ↑ ↓' },
  { id: 5, name: 'Wrist rotate', keys: '⇧ ← →' },
  { id: 10, name: 'Gripper', keys: 'F − R +', minus: '−', plus: '+' },
];
// Pose the robot takes after boot (read from servo_states). The gripper is left as it is.
const ARM_HOME = { 1: 500, 2: 730, 3: 130, 4: 100, 5: 500 };
// mod: modifier that must be held ('' = none, '*' = any)
const ARM_BINDINGS = [
  { key: 'arrowleft', mod: '', id: 1, dir: 1 }, { key: 'arrowright', mod: '', id: 1, dir: -1 },
  { key: 'arrowup', mod: '', id: 4, dir: 1 }, { key: 'arrowdown', mod: '', id: 4, dir: -1 },
  { key: 'arrowup', mod: 'shift', id: 2, dir: 1 }, { key: 'arrowdown', mod: 'shift', id: 2, dir: -1 },
  { key: 'arrowup', mod: 'ctrl', id: 3, dir: 1 }, { key: 'arrowdown', mod: 'ctrl', id: 3, dir: -1 },
  { key: 'arrowleft', mod: 'shift', id: 5, dir: 1 }, { key: 'arrowright', mod: 'shift', id: 5, dir: -1 },
  { key: 'r', mod: '*', id: 10, dir: 1 }, { key: 'f', mod: '*', id: 10, dir: -1 },
];
const ARM_KEYS = new Set(ARM_BINDINGS.map((b) => b.key));
const ARM_TICK_S = 0.1;

const armKeys = new Set();     // held keyboard keys
const armButtons = new Set();  // held on-screen buttons, as "id:dir"
let armMod = '';
const armMeasured = {};  // id -> position reported by the robot
const armTarget = {};    // id -> position being commanded while a control is held
let armLastCmd = 0;

function startArm() {
  if (servoStateSub) servoStateSub.unsubscribe();
  if (servoPub) servoPub.unadvertise();
  servoStateSub = servoPub = null;
  if (!ros) return;
  if (topicTypes[SERVO_CMD_TOPIC]) {
    servoPub = new ROSLIB.Topic({ ros, name: SERVO_CMD_TOPIC, messageType: 'servo_controller_msgs/msg/ServosPosition' });
  } else {
    log(`${SERVO_CMD_TOPIC} not found. Arm control is disabled.`, 'warn');
  }
  if (topicTypes[SERVO_STATE_TOPIC]) {
    servoStateSub = new ROSLIB.Topic({
      ros, name: SERVO_STATE_TOPIC, messageType: topicTypes[SERVO_STATE_TOPIC], throttle_rate: 200, queue_length: 1,
    });
    servoStateSub.subscribe((msg) => {
      for (const s of msg.servo_state) armMeasured[s.id] = s.position;
      renderJointValues();
    });
  }
}

// Current direction per joint from held keys and buttons, as [[id, +1|-1], ...].
function heldArmJoints() {
  const dirs = {};
  for (const k of armKeys) {
    const b = ARM_BINDINGS.find((x) => x.key === k && (x.mod === '*' || x.mod === armMod));
    if (b) dirs[b.id] = (dirs[b.id] || 0) + b.dir;
  }
  for (const token of armButtons) {
    const [id, dir] = token.split(':').map(Number);
    dirs[id] = (dirs[id] || 0) + dir;
  }
  return Object.entries(dirs).filter(([, d]) => d !== 0).map(([id, d]) => [Number(id), Math.sign(d)]);
}

function sendServos(positions, duration) {
  if (!servoPub || !positions.length) return;
  servoPub.publish({
    duration,
    position_unit: 'pulse',
    position: positions.map(([id, p]) => ({ id, position: Math.round(p) })),
  });
}

// While a control is held, move its joint a small step every tick. Releasing stops it.
setInterval(() => {
  const held = heldArmJoints();
  const now = performance.now();
  if (!held.length) {
    // after a pause, start the next move from the measured position again
    if (now - armLastCmd > 500) for (const id of Object.keys(armTarget)) delete armTarget[id];
    return;
  }
  if (!$('armed').checked || !servoPub) return;
  const step = settings.armSpeed * ARM_TICK_S;
  const positions = [];
  for (const [id, dir] of held) {
    const start = armTarget[id] ?? armMeasured[id];
    if (start == null) continue;
    const sign = settings.armInvert[id] ? -1 : 1;
    armTarget[id] = Math.min(1000, Math.max(0, start + dir * sign * step));
    positions.push([id, armTarget[id]]);
  }
  sendServos(positions, ARM_TICK_S * 1.5);
  armLastCmd = now;
  renderJointValues();
}, ARM_TICK_S * 1000);

function armGoTo(pose, label) {
  if (!$('armed').checked) return hintIfNotArmed();
  if (!servoPub) return log('No arm command topic.', 'warn');
  for (const id of Object.keys(armTarget)) delete armTarget[id];
  sendServos(Object.entries(pose).map(([id, p]) => [Number(id), p]), 1.5);
  log(`Arm moving to ${label} pose`);
}

const armHome = () => armGoTo(ARM_HOME, 'home');
const armSurvey = () => armGoTo(settings.surveyPose, 'survey');

// Saves the arm's current position (servos 1-5, gripper untouched) as the survey pose.
function saveSurveyPose() {
  const pose = {};
  for (const id of [1, 2, 3, 4, 5]) {
    if (armMeasured[id] == null) return log('Arm position unknown: connect first.', 'warn');
    pose[id] = Math.round(armMeasured[id]);
  }
  settings.surveyPose = pose;
  saveSettings();
  log(`Survey pose saved: ${Object.values(pose).join(' / ')}`);
}

function renderJointRows() {
  const body = $('jointRows');
  body.innerHTML = '';
  for (const j of ARM_JOINTS) {
    const tr = document.createElement('div');
    tr.className = 'joint';
    tr.title = `Servo ${j.id}`;
    tr.innerHTML = `
      <div class="jlabel"><span class="jname"></span><span class="jkeys"></span></div>
      <button class="jbtn" data-dir="-1"></button>
      <div class="jpos"><div class="bar"><i id="jbar${j.id}"></i></div><span id="jval${j.id}">–</span></div>
      <button class="jbtn" data-dir="1"></button>
      <label class="rev" title="Reverse the direction of this joint"><input type="checkbox"><span>⇄</span></label>`;
    tr.querySelector('.jname').textContent = j.name;
    tr.querySelector('.jkeys').textContent = j.keys;
    const [minus, plus] = tr.querySelectorAll('.jbtn');
    minus.textContent = j.minus || '−';
    plus.textContent = j.plus || '+';
    for (const btn of [minus, plus]) {
      const token = `${j.id}:${btn.dataset.dir}`;
      btn.addEventListener('pointerdown', (e) => {
        btn.setPointerCapture(e.pointerId);
        armButtons.add(token);
        hintIfNotArmed();
      });
      for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        btn.addEventListener(ev, () => armButtons.delete(token));
      }
    }
    const rev = tr.querySelector('input');
    rev.checked = !!settings.armInvert[j.id];
    rev.addEventListener('change', () => {
      settings.armInvert = { ...settings.armInvert, [j.id]: rev.checked };
      saveSettings();
    });
    body.append(tr);
  }
}

function renderJointValues() {
  for (const j of ARM_JOINTS) {
    const v = armTarget[j.id] ?? armMeasured[j.id];
    $(`jval${j.id}`).textContent = v == null ? '–' : Math.round(v);
    $(`jbar${j.id}`).style.width = v == null ? '0' : `${v / 10}%`;
  }
}

// ---------- wiring ----------

function applyCameraSource() {
  $('fpsLabel').style.display = settings.cameraSource === 'mjpeg' ? 'none' : '';
}

function applyChassis() {
  document.querySelectorAll('.mecanum-only').forEach((b) => { b.disabled = settings.chassis !== 'mecanum'; });
}

function bindSetting(id, key, parse, after) {
  const el = $(id);
  el.value = settings[key];
  el.addEventListener('change', () => {
    settings[key] = parse(el.value);
    saveSettings();
    if (after) after();
  });
}

function showSpeeds() {
  $('maxLinVal').textContent = `${Number(settings.maxLin).toFixed(2)} m/s`;
  $('maxAngVal').textContent = `${Number(settings.maxAng).toFixed(1)} rad/s`;
  $('armSpeedVal').textContent = `${settings.armSpeed} /s`;
}

$('url').value = settings.url;
$('cmdVelTopic').value = settings.cmdVelTopic;
bindSetting('chassis', 'chassis', String, applyChassis);
bindSetting('cameraSource', 'cameraSource', String, () => { applyCameraSource(); startCamera(); });
bindSetting('maxLin', 'maxLin', Number, showSpeeds);
bindSetting('maxAng', 'maxAng', Number, showSpeeds);
bindSetting('fps', 'fps', (v) => Math.min(30, Math.max(1, Number(v) || 10)), startCamera);
bindSetting('scanRange', 'scanRange', (v) => Math.min(15, Math.max(1, Number(v) || 4)), drawScan);
$('maxLin').addEventListener('input', () => { settings.maxLin = Number($('maxLin').value); showSpeeds(); });
$('maxAng').addEventListener('input', () => { settings.maxAng = Number($('maxAng').value); showSpeeds(); });
bindSetting('armSpeed', 'armSpeed', Number, showSpeeds);
$('armSpeed').addEventListener('input', () => { settings.armSpeed = Number($('armSpeed').value); showSpeeds(); });
$('armHomeBtn').addEventListener('click', armHome);
$('armSurveyBtn').addEventListener('click', armSurvey);
$('armSaveSurveyBtn').addEventListener('click', saveSurveyPose);

$('connectBtn').addEventListener('click', () => (ros ? disconnect() : connect()));
$('url').addEventListener('keydown', (e) => { if (e.key === 'Enter') connect(); });
$('refreshBtn').addEventListener('click', refreshTopics);
$('topicFilter').addEventListener('input', renderTopicTable);
$('cameraTopic').addEventListener('change', startCamera);
$('scanTopic').addEventListener('change', startScan);
$('cmdVelTopic').addEventListener('change', setupCmdVel);
$('stopBtn').addEventListener('click', emergencyStop);
function setArmed(on) {
  $('armed').checked = on;
  document.body.classList.toggle('is-armed', on);
}

$('armed').addEventListener('change', () => {
  let on = $('armed').checked;
  if (on && batteryLevel === 'critical' &&
      !confirm('The battery is critical. Moving now drains it further and will soon trigger the robot\'s loud alarm. Arm anyway?')) {
    on = false;
  }
  setArmed(on);
  if (!on) {
    sendStop();
    pauseAutonomy('disarmed');
  }
  // don't keep keyboard focus on the switch
  $('armed').blur();
  log(on ? 'Armed: drive and arm keys are live.' : 'Disarmed.', 'warn');
});

// ---------- robot power, via the local helper (tools/robot_helper.py) ----------

const helperFetch = (path, opts = {}) =>
  fetch(HELPER_URL + path, { ...opts, headers: { 'X-LanderPi-Console': '1' } }).then((r) => r.json());

function setPowerButtons(enabled) {
  $('rebootBtn').disabled = !enabled;
  $('shutdownBtn').disabled = !enabled;
}

async function checkHelper() {
  const el = $('powerStatus');
  el.textContent = 'Checking…';
  try {
    const r = await helperFetch('/api/status');
    el.textContent = r.robot
      ? `Ready · ${r.info.split('\n').join(', ')}${r.wifi ? ` · Wi-Fi ${r.wifi}` : ''}`
      : `Helper running, but the robot is not reachable (${r.info}).`;
    setPowerButtons(r.robot);
  } catch {
    el.textContent = 'Helper not running. Start the console with start_mapping_console.bat.';
    setPowerButtons(false);
  }
}

async function powerAction(action) {
  const question = action === 'reboot'
    ? 'Reboot the robot now? It will be back after 1-2 minutes.'
    : 'Shut down the robot now?\n\nAfterwards wait ~20 s, then switch it off with its power switch. '
      + 'Until you do, the controller board keeps drawing battery power.';
  if (!confirm(question)) return;
  emergencyStop();
  disconnect();
  setPowerButtons(false);
  let r;
  try {
    r = await helperFetch(`/api/${action}`, { method: 'POST' });
    if (!r.ok) throw new Error(r.error || 'failed');
  } catch (e) {
    log(`${action} failed: ${e.message}`, 'error');
    checkHelper();
    return;
  }
  if (action === 'reboot') {
    log('Robot is rebooting. Reconnecting automatically…', 'warn');
    showBanner(r.rejoinWifi
      ? `Robot is rebooting (~1-2 min). The laptop will rejoin "${r.rejoinWifi}" and reconnect by itself.`
      : 'Robot is rebooting (~1-2 min). If the laptop changes Wi-Fi, rejoin the robot hotspot.', 'low');
    reconnectAfterReboot();
  } else {
    log('Robot is shutting down.', 'warn');
    showBanner('Robot is shutting down. Wait ~20 s, then switch it off with the power switch.', 'low');
  }
}

// The robot needs ~30 s to go down and come back; then try every few seconds for 3 minutes.
function reconnectAfterReboot() {
  const giveUp = Date.now() + 240000;  // matches the helper's Wi-Fi rejoin window
  const attempt = () => {
    if ($('status').dataset.state === 'ok') return showBanner(null);
    if (Date.now() > giveUp) return log('The robot did not come back within 3 minutes. Is the laptop on its Wi-Fi?', 'error');
    connect();
    setTimeout(attempt, 8000);
  };
  setTimeout(attempt, 30000);
}

$('rebootBtn').addEventListener('click', () => powerAction('reboot'));
$('shutdownBtn').addEventListener('click', () => powerAction('shutdown'));

// settings & debug drawer
function toggleDrawer(open) {
  document.body.classList.toggle('drawer-open', open);
  if (open) checkHelper();
}
$('drawerBtn').addEventListener('click', () => toggleDrawer(true));
$('batterySound').checked = settings.batterySound;
$('batterySound').addEventListener('change', () => {
  settings.batterySound = $('batterySound').checked;
  saveSettings();
  if (settings.batterySound) chime(1);  // preview
});
$('robotIndicators').checked = settings.robotIndicators;
$('robotIndicators').addEventListener('change', () => {
  settings.robotIndicators = $('robotIndicators').checked;
  saveSettings();
  if (!settings.robotIndicators) clearRobotIndicators();
  else if (batterySamples.length) updateRobotIndicators(batterySamples[batterySamples.length - 1][1], true);
});
$('drawerClose').addEventListener('click', () => toggleDrawer(false));
$('scrim').addEventListener('click', () => toggleDrawer(false));
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') toggleDrawer(false); });

renderJointRows();
applyChassis();
applyCameraSource();
showSpeeds();
showCamPlaceholder('Not connected');
drawScan();

// The JS libraries in vendor/ are downloaded by the helper on its first start (tools/console_vendor.py). Without
// them the console can't connect (roslib) or show 3D (three.js): say so instead of failing silently.
if (SERVED_BY_HELPER) {
  helperFetch('/api/vendor').then((v) => {
    if (!v.missing || !v.missing.length) return;
    const text = `The helper could not download the console's libraries (${v.missing.join(', ')}): no internet? See the helper window.`;
    showBanner(text, 'critical');
    log(text, 'error');
  }).catch(() => {});
}

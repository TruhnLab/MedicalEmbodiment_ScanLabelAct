// The page's small behaviours: reveal on scroll, the teaser's chapters, the demo tabs, videos that play only
// while visible, the replay of the exploration runs, and the 3D room (loaded when it comes near).

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- reveal on scroll ----------
const revealer = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add('in');
    revealer.unobserve(e.target);
  }
}, { rootMargin: '0px 0px -8% 0px' });
$$('.reveal').forEach((el) => revealer.observe(el));

// ---------- segmented controls: the sliding pill follows the pressed button ----------
function placePill(seg) {
  const on = $('button[aria-pressed="true"]', seg), pill = $('.pill', seg);
  if (!on || !pill) return;
  pill.style.width = `${on.offsetWidth}px`;
  pill.style.transform = `translateX(${on.offsetLeft - 4}px)`;
}
function press(seg, btn) {
  $$('button', seg).forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
  placePill(seg);
}
const segs = $$('.seg');
segs.forEach(placePill);
addEventListener('resize', () => segs.forEach(placePill));
document.fonts?.ready.then(() => segs.forEach(placePill));

// ---------- videos play only while on screen ----------
const player = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting && !v.closest('[hidden]')) v.play().catch(() => {});
    else v.pause();
  }
}, { threshold: 0.35 });
$$('video').forEach((v) => player.observe(v));

// ---------- teaser chapters ----------
const teaser = $('#teaser');
const chapters = $$('#chapters button');
if (teaser && chapters.length) {
  const starts = chapters.map((b) => parseFloat(b.dataset.start));
  const tick = () => {
    const t = teaser.currentTime, end = teaser.duration || starts.at(-1) + 10;
    chapters.forEach((b, i) => {
      const a = starts[i], z = starts[i + 1] ?? end;
      const on = t >= a && t < z;
      b.classList.toggle('on', on);
      $('.fill', b).style.width = on ? `${((t - a) / (z - a)) * 100}%` : '0';
    });
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  chapters.forEach((b, i) => b.addEventListener('click', () => { teaser.currentTime = starts[i] + 0.01; teaser.play().catch(() => {}); }));
}

// ---------- demo tabs ----------
const tabs = $('#demoTabs');
if (tabs) {
  for (const btn of $$('button[aria-controls]', tabs)) {
    btn.addEventListener('click', () => {
      press(tabs, btn);
      for (const b of $$('button[aria-controls]', tabs)) {
        const panel = document.getElementById(b.getAttribute('aria-controls'));
        const show = b === btn;
        panel.hidden = !show;
        const v = $('video', panel);
        if (show) { v.preload = 'auto'; v.currentTime = 0; v.play().catch(() => {}); } else v.pause();
      }
    });
  }
}

// ---------- exploration runs: a replay of the map, the path and the camera's coverage ----------
// Per run (assets/data/runs.json): the lidar map (<session>.png), the path with times, and <session>_cov.png, one
// pixel per map cell whose R, G and B are the moments (1-255 over the run, 0 = never) the cell became a surface
// to look at, was seen by the camera, and was given up. The explorer's own coverage figure comes as a series.
const REPLAY_S = 16;  // every run is replayed in this many seconds
const COV = { seen: [34, 197, 94, 150], todo: [245, 158, 11, 175], gave: [135, 146, 166, 190] };

const fmtDate = (s) => new Date(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8))
  .toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const loadImage = (src) => new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });

function runPlayer(view) {
  const canvas = $('#runCanvas', view), ctx = canvas.getContext('2d');
  const play = $('#runPlay', view), slider = $('#runTime', view), clock = $('#runClock', view);
  let run = null, map = null, cov = null, t = 0, playing = false, last = 0, drawnCode = -1, liveStat = null;

  async function load(r) {
    const [img, covImg] = await Promise.all([loadImage(`assets/data/runs/${r.session}.png`),
      r.has_coverage ? loadImage(`assets/data/runs/${r.session}_cov.png`) : null]);
    cov = null;
    if (covImg) {
      const w = covImg.width, h = covImg.height;
      const off = document.createElement('canvas');
      Object.assign(off, { width: w, height: h });
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(covImg, 0, 0);
      const px = octx.getImageData(0, 0, w, h).data;
      const cells = [];
      for (let i = 0; i < w * h; i++) if (px[4 * i]) cells.push(i);
      cov = { off, octx, out: octx.createImageData(w, h), px, cells };
    }
    run = r;
    map = img;
    drawnCode = -1;
  }

  // the coverage at time code c (1-255), only redrawn when the code changes
  function paintCoverage(c) {
    if (!cov || c === drawnCode) return;
    const { px, cells, out } = cov, d = out.data;
    d.fill(0);
    for (const i of cells) {
      const k = 4 * i;
      if (px[k] > c) continue;  // not a surface yet
      const col = px[k + 2] && px[k + 2] <= c ? COV.gave : px[k + 1] && px[k + 1] <= c ? COV.seen : COV.todo;
      d[k] = col[0]; d[k + 1] = col[1]; d[k + 2] = col[2]; d[k + 3] = col[3];
    }
    cov.octx.putImageData(out, 0, 0);
    drawnCode = c;
  }

  function draw() {
    if (!run) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const s = Math.min(canvas.width / run.w, canvas.height / run.h);
    const ox = (canvas.width - run.w * s) / 2, oy = (canvas.height - run.h * s) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(map, ox, oy, run.w * s, run.h * s);
    paintCoverage(1 + Math.round((254 * t) / run.seconds));
    if (cov) ctx.drawImage(cov.off, ox, oy, run.w * s, run.h * s);

    // the path up to t, and the robot where it is at t
    const P = run.path;
    let k = 0;
    while (k + 1 < P.length && P[k + 1][2] <= t) k++;
    let [x, y] = P[k];
    if (k + 1 < P.length && P[k + 1][2] > P[k][2]) {
      const f = Math.min(1, (t - P[k][2]) / (P[k + 1][2] - P[k][2]));
      x += (P[k + 1][0] - x) * f;
      y += (P[k + 1][1] - y) * f;
    }
    const X = (v) => ox + v * s, Y = (v) => oy + v * s;
    ctx.lineJoin = ctx.lineCap = 'round';
    ctx.strokeStyle = '#2f6bff';
    ctx.lineWidth = 2.2 * dpr;
    ctx.beginPath();
    ctx.moveTo(X(P[0][0]), Y(P[0][1]));
    for (let i = 1; i <= k; i++) ctx.lineTo(X(P[i][0]), Y(P[i][1]));
    ctx.lineTo(X(x), Y(y));
    ctx.stroke();
    ctx.lineWidth = 2 * dpr;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(X(P[0][0]), Y(P[0][1]), 4 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.strokeStyle = '#fff';
    ctx.beginPath(); ctx.arc(X(x), Y(y), 6 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();

    slider.value = String(Math.round((1000 * t) / run.seconds));
    clock.innerHTML = `${mmss(t)} / ${mmss(run.seconds)}<em> · ×${Math.round(run.seconds / REPLAY_S)}</em>`;
    if (liveStat && run.coverage_series?.length) {
      let c = 0;
      for (const [ts, v] of run.coverage_series) { if (ts > t) break; c = v; }
      liveStat.textContent = `${Math.round(c * 100)} %`;
    }
  }

  function frame(now) {
    if (!playing) return;
    t = Math.min(run.seconds, t + ((now - last) / 1000) * (run.seconds / REPLAY_S));
    last = now;
    draw();
    if (t >= run.seconds) setPlaying(false);
    else requestAnimationFrame(frame);
  }
  function setPlaying(on) {
    playing = on && !!run;
    play.classList.toggle('on', playing);
    play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
    if (playing) {
      if (t >= run.seconds) t = 0;
      last = performance.now();
      requestAnimationFrame(frame);
    }
  }
  play.addEventListener('click', () => setPlaying(!playing));
  slider.addEventListener('input', () => { if (run) { t = (+slider.value / 1000) * run.seconds; draw(); } });
  new ResizeObserver(() => draw()).observe(canvas);

  return {
    async show(r, stat, autoplay) {
      setPlaying(false);
      await load(r);
      liveStat = stat;
      t = autoplay ? 0 : r.seconds;
      draw();
      if (autoplay) setPlaying(true);
    },
    start() { t = 0; setPlaying(true); },
  };
}

async function runSection() {
  const tabs = $('#runTabs'), view = $('#runView');
  if (!tabs || !view) return;
  const { totals, runs } = await fetch('assets/data/runs.json').then((r) => r.json());
  const player = runPlayer(view);
  let seen = false;

  function show(r, btn) {
    press(tabs, btn);
    $('#runTitle').innerHTML = `${r.room}<small>${fmtDate(r.session)}, ${r.time}</small>`;
    const rows = [
      ['Duration', `${r.minutes.toFixed(1)} min`],
      ['Path driven', `${r.path_m.toFixed(1)} m`],
      ['Area mapped', `${r.area_m2} m²`],
      ['Keyframes', r.keyframes.toLocaleString('en-US')],
      r.coverage ? ['Surfaces observed', `<span id="runLive">${Math.round(r.coverage * 100)} %</span>`] : null,
      r.objects ? ['Labelled objects', r.objects] : null,
    ].filter(Boolean);
    $('#runStats').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    return player.show(r, $('#runLive'), seen && !calm);
  }

  for (const r of runs) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'tab');
    b.textContent = r.room;
    b.addEventListener('click', () => show(r, b));
    tabs.append(b);
  }
  const later = document.createElement('button');
  later.type = 'button';
  later.className = 'later';
  later.disabled = true;
  later.textContent = 'Coming soon';
  later.title = 'Further rooms will be added';
  tabs.append(later);
  await show(runs[0], $('button', tabs));
  // the first replay starts when the viewer comes into sight
  new IntersectionObserver((entries, obs) => {
    if (!entries[0].isIntersecting) return;
    seen = true;
    if (!calm) player.start();
    obs.disconnect();
  }, { threshold: 0.45 }).observe(view);

  $('#runTotals').textContent = `Across all recordings, ${totals.recordings} runs in ${totals.rooms} rooms on ${totals.days} days, `
    + `with ${totals.minutes} minutes and ${totals.path_m} m of driving.`;
}

runSection();

// ---------- the 3D room, loaded when it comes near ----------
const roomHost = $('#roomView');
if (roomHost) {
  const near = new IntersectionObserver(async (entries) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    near.disconnect();
    try {
      const { initRoom } = await import('./room3d.js');
      await initRoom(roomHost);
    } catch (err) {
      console.error(err);
      $('#roomLoading').textContent = 'The 3D view needs WebGL, which this browser does not offer.';
    }
  }, { rootMargin: '600px 0px' });
  near.observe(roomHost);
}

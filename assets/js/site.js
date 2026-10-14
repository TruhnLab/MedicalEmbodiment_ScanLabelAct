// The page's small behaviours: reveal on scroll, the teaser's chapters, the demo tabs, videos that play only
// while visible, the exploration runs, and the 3D room (loaded when it comes near).

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

// ---------- the run wall ----------
const svgNS = 'http://www.w3.org/2000/svg';

// a run's lidar map with the path driven (strokes in map pixels)
function runMap(r) {
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${r.w} ${r.h}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.setAttribute('aria-hidden', 'true');
  const img = document.createElementNS(svgNS, 'image');
  img.setAttribute('href', `assets/data/runs/${r.session}.png`);
  img.setAttribute('width', r.w);
  img.setAttribute('height', r.h);
  const sw = Math.max(r.w, r.h) / 150;
  const line = document.createElementNS(svgNS, 'polyline');
  line.setAttribute('points', r.path);
  line.setAttribute('class', 'trail');
  line.setAttribute('stroke-width', sw.toFixed(1));
  const pts = r.path.split(' ');
  const dot = (p, cls, rad) => {
    const [x, y] = p.split(',');
    const c = document.createElementNS(svgNS, 'circle');
    Object.entries({ cx: x, cy: y, r: rad, class: cls, 'stroke-width': sw * 0.8 }).forEach(([k, v]) => c.setAttribute(k, v));
    return c;
  };
  svg.append(img, line, dot(pts[0], 'start', sw * 1.5), dot(pts.at(-1), 'end', sw * 2.2));
  return svg;
}

// draws the path from start to end, at a pace that grows with its length
function drawTrail(host) {
  const line = $('.trail', host);
  const len = line.getTotalLength();
  host.classList.remove('drawn');
  line.style.transition = 'none';
  line.style.strokeDasharray = `${len}`;
  line.style.strokeDashoffset = calm ? '0' : `${len}`;
  if (calm) { host.classList.add('drawn'); return; }
  line.getBoundingClientRect();  // start the transition from the hidden state
  const secs = 2 + Math.min(4, len / 1500);
  line.style.transition = `stroke-dashoffset ${secs}s cubic-bezier(.4,.1,.4,1)`;
  line.style.strokeDashoffset = '0';
  line.addEventListener('transitionend', () => host.classList.add('drawn'), { once: true });
}

const fmtDate = (s) => new Date(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8))
  .toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });

async function runSection() {
  const tabs = $('#runTabs'), view = $('#runView');
  if (!tabs || !view) return;
  const { totals, runs } = await fetch('assets/data/runs.json').then((r) => r.json());
  let seen = false;

  function show(r, btn) {
    press(tabs, btn);
    $('#runMap').replaceChildren(runMap(r));
    $('#runTitle').innerHTML = `${r.room}<small>${fmtDate(r.session)}, ${r.time}</small>`;
    const rows = [
      ['Duration', `${r.minutes.toFixed(1)} min`],
      ['Path driven', `${r.path_m.toFixed(1)} m`],
      ['Area mapped', `${r.area_m2} m²`],
      ['Keyframes', r.keyframes.toLocaleString('en-US')],
      r.coverage ? ['Surfaces observed', `${Math.round(r.coverage * 100)} %`] : null,
      r.objects ? ['Labelled objects', r.objects] : null,
    ].filter(Boolean);
    $('#runStats').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    if (seen) drawTrail(view);
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
  show(runs[0], $('button', tabs));
  $('#runReplay').addEventListener('click', () => drawTrail(view));
  new IntersectionObserver((entries, obs) => {
    if (!entries[0].isIntersecting) return;
    seen = true;
    drawTrail(view);
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

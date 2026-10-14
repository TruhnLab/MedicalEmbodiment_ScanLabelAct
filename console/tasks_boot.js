// Task console start-up: takes the parts both consoles share from the mapping console's page
// (index.html: top bar with Connect / Armed / STOP / battery / settings, the camera, the drive strip,
// the lidar and arm cards, the settings drawer), puts them into tasks.html's slots, runs the unchanged
// app.js on them, then tasks.js. So driving, the arm, keys (W A S D, Q E, arrows, H, P, Space) and
// names are the same in both consoles, and a new control in the mapping console shows up here too.

const $ = (id) => document.getElementById(id);

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`could not load ${src}`));
    document.body.append(s);
  });
}

try {
  const html = await (await fetch('index.html', { cache: 'no-store' })).text();
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const take = (sel) => {
    const el = doc.querySelector(sel);
    if (!el) throw new Error(`index.html has no "${sel}" any more`);
    return document.importNode(el, true);
  };

  const header = take('header.topbar');
  header.querySelector('.segmented')?.remove();  // Drive / Map / 3D tabs belong to the mapping console
  header.querySelector('.brand').innerHTML = '<span class="logo"></span>LanderPi <span class="muted">Tasks</span>';
  const link = document.createElement('a');
  link.className = 'btn ghost';
  link.href = '/';
  link.target = '_blank';
  link.title = 'The mapping console (SLAM, explore, record, reconstruct, label)';
  link.textContent = 'Mapping console ↗';
  header.querySelector('#drawerBtn').before(link);
  $('topbarSlot').replaceWith(header);

  $('camSlot').replaceWith(take('.cam-view'));
  $('stripSlot').replaceWith(take('section.strip'));
  $('lidarSlot').replaceWith(take('section.lidar'));
  $('armSlot').replaceWith(take('section.armpanel'));
  $('drawerSlot').replaceWith(take('#scrim'), take('#drawer'));

  // M switches the big view here too: room / camera instead of map / drive
  const mHint = [...document.querySelectorAll('.strip .keys span')].find((s) => /map/i.test(s.textContent));
  if (mHint) mHint.innerHTML = '<kbd>M</kbd> room / camera view';

  for (const src of ['vendor/roslib.min.js', 'vendor/pako.min.js', 'vendor/UPNG.js', 'app.js']) await loadScript(src);
  await import('./tasks.js');
} catch (e) {
  document.body.insertAdjacentHTML('afterbegin',
    `<div class="banner" data-level="critical" style="position:static;transform:none;margin:12px">
       The task console could not start: ${e.message}. Is it opened through the helper (start_task_console.bat)? On the first start the helper downloads the console's libraries: see its window.</div>`);
  throw e;
}

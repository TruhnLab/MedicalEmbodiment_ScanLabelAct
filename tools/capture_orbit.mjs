// The labelled room turning (for the teaser) and one still in the objects view (for the "Label" card),
// captured from the page's own 3D viewer through tools/capture.html.
//
//   python -m http.server 8800 --bind 127.0.0.1        (in the page's folder)
//   msedge --headless=new --remote-debugging-port=9333 --enable-unsafe-swiftshader --use-angle=swiftshader ...
//   node capture_orbit.mjs <frames folder>              (from a folder with puppeteer-core installed)
//
// Then: ffmpeg -framerate 30 -i <frames>/%04d.png ... assets/media/orbit.mp4 (tools/make_media.py's encoding);
// the still is written to <frames>/labels.png.
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';

const out = process.argv[2];
fs.mkdirSync(out, { recursive: true });
const browser = await puppeteer.connect({ browserURL: 'http://127.0.0.1:9333', defaultViewport: null });
const page = await browser.newPage();

async function open(w, h) {
  await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:8800/tools/capture.html?w=${w}&h=${h}`, { waitUntil: 'load' });
  await page.waitForFunction(() => window.__ready === true, { timeout: 60000 });
}

// 6 s at 30 fps: a quarter turn; photo colours first, then the objects fade in
const FRAMES = 180, A0 = -2.75, TURN = Math.PI / 2;
await open(1280, 720);
for (let i = 0; i < FRAMES; i++) {
  const k = i / (FRAMES - 1);
  const mix = Math.min(1, Math.max(0, (k - 0.25) / 0.3));
  const fade = mix * mix * (3 - 2 * mix);
  await page.evaluate((a, m) => window.__room.view(a, 0.62, 0.7, m), A0 + TURN * k, fade);
  await page.screenshot({ path: `${out}/${String(i).padStart(4, '0')}.png` });
}

await open(1200, 900);
await page.evaluate(() => window.__room.view(-2.35, 0.8, 0.82, 1));
await page.screenshot({ path: `${out}/labels.png` });
await page.close();
browser.disconnect();
console.log(`${FRAMES} frames and labels.png in ${out}`);

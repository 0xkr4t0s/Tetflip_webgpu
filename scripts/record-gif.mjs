// Records the app as an animated GIF in headless Chromium (WebGPU via SwiftShader when there is no
// GPU). The app's dev-only `advance` hook steps the simulation, so frames are evenly spaced in
// simulated time however slowly they render, and `capture` renders each frame offscreen.
// Usage: node scripts/record-gif.mjs out.gif "scene=dam-break&res=low" [--option=value ...]
//   --seconds=4    simulated time to record          --size=640x360  output size in pixels
//   --every=2      display frames (1/60 s) per GIF    --delay=4       GIF frame delay (1/100 s)
//   --start=0      simulated seconds to skip first    --pause=0.6     extra hold on the first and last frame (s)
//   --zoom=1       camera distance vs. the default    --yaw, --pitch  camera angles (radians)
//   --orbit=0      yaw change over the clip           --ss=2          supersampling factor (1 or 2)
//   --fuzz=4       keep a pixel's previous colour while it changes by at most this much per channel
//   --dither=8     ordered-dither amplitude (0-32): hides banding, at some cost in file size
//   --frames=file  cache of the captured frames: written after recording, and used instead of
//                  recording when it already exists (to try other encoding options quickly)
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'vite';
import { chromium } from 'playwright';
import gifenc from 'gifenc';

const { GIFEncoder, applyPalette, quantize } = gifenc;

const args = process.argv.slice(2);
const [out = 'recording.gif', query = 'res=low'] = args.filter((a) => !a.startsWith('--'));
const opt = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')));
const num = (key, fallback) => (opt[key] === undefined ? fallback : Number(opt[key]));
const seconds = num('seconds', 4), every = num('every', 2), delay = num('delay', 4);
const start = num('start', 0), pause = num('pause', 0.6), ss = num('ss', 2);
const fuzz = num('fuzz', 4), dither = num('dither', 8);
const zoom = num('zoom', 1), orbit = num('orbit', 0), yaw = num('yaw'), pitch = num('pitch');
const [width, height] = (opt.size ?? '640x360').split('x').map(Number);

/** Box-filters an RGBA image down by an integer factor. */
function downsample(src, w, h, f) {
  if (f === 1) return src;
  const ow = Math.floor(w / f), oh = Math.floor(h / f), dst = new Uint8Array(ow * oh * 4), n = f * f;
  for (let y = 0; y < oh; y++)
    for (let x = 0; x < ow; x++)
      for (let c = 0; c < 4; c++) {
        let sum = 0;
        for (let j = 0; j < f; j++) for (let i = 0; i < f; i++) sum += src[((y * f + j) * w + x * f + i) * 4 + c];
        dst[(y * ow + x) * 4 + c] = Math.round(sum / n);
      }
  return dst;
}

async function record() {
  const server = await createServer({ server: { port: 0 }, logLevel: 'error' });
  await server.listen();
  const browser = await chromium.launch({
    headless: true,
    channel: 'chromium',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--disable-vulkan-surface', '--use-webgpu-adapter=swiftshader'],
  });
  const errors = [];
  try {
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: ss });
    page.on('console', (m) => ['error', 'warning'].includes(m.type()) && errors.push(m.text()));
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${server.resolvedUrls.local[0]}?${query}&paused=1`);
    await page.waitForFunction(() => window.__tetflip, null, { timeout: 600_000, polling: 250 });
    const yaw0 = await page.evaluate(({ zoom, yaw, pitch }) => {
      const camera = window.__tetflip.camera;
      camera.distance *= zoom;
      if (Number.isFinite(yaw)) camera.yaw = yaw;
      if (Number.isFinite(pitch)) camera.pitch = pitch;
      return camera.yaw;
    }, { zoom, yaw, pitch });
    await page.evaluate((n) => window.__tetflip.advance(n), Math.round(start * 60));

    const count = Math.round((seconds * 60) / every) + 1;
    const frames = [];
    const t0 = performance.now();
    for (let i = 0; i < count; i++) {
      if (i > 0) await page.evaluate((n) => window.__tetflip.advance(n), every);
      if (orbit) await page.evaluate((y) => (window.__tetflip.camera.yaw = y), yaw0 + (orbit * i) / Math.max(1, count - 1));
      const frame = await page.evaluate(() => window.__tetflip.capture());
      frames.push(downsample(new Uint8Array(Buffer.from(frame.rgba, 'base64')), frame.width, frame.height, Math.round(frame.width / width)));
      if (i % 10 === 0 || i === count - 1) {
        const time = await page.evaluate(() => window.__tetflip.time);
        console.log(`frame ${i + 1}/${count}  t=${time.toFixed(2)} s  ${((performance.now() - t0) / 1000).toFixed(0)} s elapsed`);
      }
    }
    return frames;
  } finally {
    const relevant = errors.filter((e) => !e.includes('404'));
    if (relevant.length) console.error(relevant.join('\n'));
    await browser.close();
    await server.close();
  }
}

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

/** Adds a 4×4 Bayer threshold pattern, so gradients quantize to a stable dither, not bands. */
function ditherPattern(rgba, w, h, amount) {
  if (!amount) return rgba;
  const out = new Uint8Array(rgba.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const offset = ((BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16 - 0.5) * amount;
      const p = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) out[p + c] = Math.min(255, Math.max(0, Math.round(rgba[p + c] + offset)));
      out[p + 3] = 255;
    }
  return out;
}

/**
 * Encodes the frames with one global palette. After the first frame, pixels that keep their
 * colour are written as transparent, so the static parts of the scene cost almost nothing.
 */
function encodeGif(frames, w, h) {
  const all = new Uint8Array(frames.length * w * h * 4);
  frames.forEach((f, i) => all.set(f, i * w * h * 4));
  const palette = quantize(all, 255);
  while (palette.length < 255) palette.push([0, 0, 0]);
  const KEY = 255;
  const gif = GIFEncoder();
  let shown = null;
  frames.forEach((rgba, i) => {
    const index = applyPalette(ditherPattern(rgba, w, h, dither), palette);
    let data = index;
    if (shown) {
      data = new Uint8Array(index.length);
      for (let p = 0; p < index.length; p++) {
        const c = palette[shown[p]];
        const near = Math.max(Math.abs(rgba[p * 4] - c[0]), Math.abs(rgba[p * 4 + 1] - c[1]), Math.abs(rgba[p * 4 + 2] - c[2])) <= fuzz;
        if (index[p] === shown[p] || near) data[p] = KEY;
        else shown[p] = data[p] = index[p];
      }
    } else shown = index.slice();
    const hold = i === 0 || i === frames.length - 1 ? pause * 100 : 0;
    gif.writeFrame(data, w, h, {
      palette: i === 0 ? [...palette, [255, 0, 255]] : undefined,
      delay: Math.round(delay + hold) * 10,
      transparent: i > 0,
      transparentIndex: KEY,
      dispose: 1,
    });
  });
  gif.finish();
  return gif.bytes();
}

let frames;
if (opt.frames && existsSync(opt.frames)) {
  const raw = readFileSync(opt.frames);
  const size = width * height * 4;
  // Copies, not views: gifenc reads `.buffer` and would see the whole file.
  frames = Array.from({ length: raw.length / size }, (_, i) => new Uint8Array(raw.subarray(i * size, (i + 1) * size)));
  console.log(`loaded ${frames.length} frames from ${opt.frames}`);
} else {
  frames = await record();
  if (opt.frames) writeFileSync(opt.frames, Buffer.concat(frames));
}
const bytes = encodeGif(frames, width, height);
writeFileSync(out, bytes);
console.log(`saved ${out} (${width}×${height}, ${frames.length} frames, ${(bytes.length / 1e6).toFixed(2)} MB)`);

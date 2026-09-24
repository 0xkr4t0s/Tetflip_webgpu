// Opens the app in headless Chromium (WebGPU via SwiftShader), runs the simulation until the
// requested simulated time and saves the rendered frame as a PNG. Headless SwiftShader cannot
// present a WebGPU canvas, so the frame is rendered offscreen by the app's dev-only capture hook.
// Usage: node scripts/screenshot.mjs out.png "scene=dam-break&res=low" 0.8
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { createServer } from 'vite';
import { chromium } from 'playwright';

const [out = 'screenshot.png', query = 'res=low', until = '0.5'] = process.argv.slice(2);

function png(width, height, rgba) {
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    const c = Buffer.alloc(4);
    c.writeUInt32BE(crc(td));
    return Buffer.concat([len, td, c]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) rgba.copy(raw, y * (width * 4 + 1) + 1, y * width * 4, (y + 1) * width * 4);
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

const server = await createServer({ server: { port: 0 }, logLevel: 'error' });
await server.listen();
const browser = await chromium.launch({
  headless: true,
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--disable-vulkan-surface', '--use-webgpu-adapter=swiftshader'],
});
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('console', (m) => ['error', 'warning'].includes(m.type()) && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto(`${server.resolvedUrls.local[0]}?${query}`);
  await page.waitForFunction((t) => (window.__tetflip?.time ?? -1) >= t && window.__tetflip.frames > 3, Number(until), { timeout: 1_800_000, polling: 500 });
  const frame = await page.evaluate(() => window.__tetflip.capture());
  const time = await page.evaluate(() => window.__tetflip.time);
  writeFileSync(out, png(frame.width, frame.height, Buffer.from(frame.rgba, 'base64')));
  console.log(`saved ${out} (${frame.width}×${frame.height}) at t=${time.toFixed(3)} s`);
} finally {
  const relevant = errors.filter((e) => !e.includes('404'));
  if (relevant.length) console.error(relevant.join('\n'));
  await browser.close();
  await server.close();
}

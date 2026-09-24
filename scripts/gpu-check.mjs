// Runs browser-side GPU diagnostics in headless Chromium (WebGPU via SwiftShader when no
// hardware adapter is present).
//   node scripts/gpu-check.mjs                      GPU solver vs CPU reference (exit code 1 on failure)
//   node scripts/gpu-check.mjs energy "cells=24"    energy / volume history of the dam break
import { createServer } from 'vite';
import { chromium } from 'playwright';

const [pageName = 'index', query = ''] = process.argv.slice(2);
const server = await createServer({ server: { port: 0 }, logLevel: 'error' });
await server.listen();
const url = `${server.resolvedUrls.local[0]}tests/gpu/${pageName}.html?${query}`;
const browser = await chromium.launch({
  headless: true,
  channel: 'chromium',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader'],
});
let exitCode = 1;
try {
  const page = await browser.newPage();
  page.on('console', (m) => (m.type() === 'error' || m.type() === 'warning') && !m.text().includes('404') && console.error('[page]', m.text()));
  page.on('pageerror', (e) => console.error('[page]', e.message));
  await page.goto(url);
  await page.waitForFunction(() => window.__result !== undefined, null, { timeout: 3_600_000 });
  const result = await page.evaluate(() => window.__result);
  if (result.error) {
    console.error(result.error);
  } else if (!result.checks) {
    console.log(JSON.stringify(result, null, 1));
    exitCode = 0;
  } else {
    console.log(`particles ${result.particles}, nodes ${result.nodes}, tets ${result.tets}; last PCG rᵀM⁻¹r ${result.pcg.initial.toExponential(2)} → ${result.pcg.final.toExponential(2)}`);
    for (const c of result.checks) console.log(`${c.pass ? 'PASS' : 'FAIL'}  ${c.name}: ${c.value.toExponential(3)} (limit ${c.limit})`);
    exitCode = result.pass ? 0 : 1;
  }
} finally {
  await browser.close();
  await server.close();
}
process.exit(exitCode);

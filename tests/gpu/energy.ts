/**
 * Energy diagnostic: runs the dam break on the GPU and reports total mechanical energy per unit
 * particle mass over time for a few parameter variants. Energy must not grow.
 */
import { initGpu } from '../../src/gpu/device';
import { createBccMesh } from '../../src/mesh/bcc';
import { prepareMesh } from '../../src/mesh/prepare';
import { GpuSolver } from '../../src/sim/GpuSolver';
import { defaultParams, type SimParams } from '../../src/sim/params';
import { scenes, seedParticles } from '../../src/sim/scenes';

async function run() {
  const { device } = await initGpu();
  const q = new URLSearchParams(location.search);
  const cells = Number(q.get('cells') ?? 24);
  const seconds = Number(q.get('seconds') ?? 6);
  const variants: Record<string, Partial<SimParams>> = JSON.parse(q.get('variants') ?? '{"default":{}}');
  const scene = scenes.find((s) => s.id === (q.get('scene') ?? 'dam-break'))!;
  const h = Math.max(...scene.size) / cells;
  const mesh = prepareMesh(createBccMesh({ boundsMin: [0, 0, 0], size: scene.size, spacing: h }));
  const seed = seedParticles(scene, h / 2, 1e7);
  const dt = 1 / 120;
  const out: Record<string, string[]> = {};
  for (const [name, overrides] of Object.entries(variants)) {
    const params = { ...defaultParams(), restDensity: 8 / h ** 3, ...overrides };
    const solver = new GpuSolver(device, mesh, seed.count, params);
    solver.setParticles(seed);
    const rows: string[] = [];
    const steps = Math.round(seconds / dt);
    for (let s = 0; s <= steps; s++) {
      if (s % 60 === 0) {
        const pos = new Float32Array(await solver.read('particlePos', seed.count * 16));
        const vel = new Float32Array(await solver.read('particleVel', seed.count * 16));
        let ke = 0, pe = 0, y = 0;
        for (let i = 0; i < seed.count; i++) {
          ke += 0.5 * (vel[i * 4] ** 2 + vel[i * 4 + 1] ** 2 + vel[i * 4 + 2] ** 2);
          pe += 9.81 * pos[i * 4 + 1];
          y += pos[i * 4 + 1];
        }
        rows.push(`t=${(s * dt).toFixed(1)} E=${((ke + pe) / seed.count).toFixed(4)} KE=${(ke / seed.count).toFixed(4)} meanY=${(y / seed.count).toFixed(4)}`);
      }
      const e = device.createCommandEncoder();
      solver.encode(e, dt, 1);
      device.queue.submit([e.finish()]);
    }
    solver.destroy();
    out[name] = rows;
  }
  return { particles: seed.count, tets: mesh.tetCount, out };
}

declare global {
  interface Window {
    __result?: unknown;
  }
}
run()
  .then((r) => (window.__result = r))
  .catch((e) => (window.__result = { error: String(e?.stack ?? e) }));

/**
 * Runs the WebGPU solver next to the CPU reference and compares their fields. Loaded in
 * headless Chromium by scripts/gpu-check.mjs; the verdict is published on window.__result.
 */
import { initGpu } from '../../src/gpu/device';
import { createBccMesh } from '../../src/mesh/bcc';
import { prepareMesh } from '../../src/mesh/prepare';
import { GpuSolver } from '../../src/sim/GpuSolver';
import { defaultParams } from '../../src/sim/params';
import { ReferenceSolver, type ParticleState } from '../../src/sim/reference';
import { seedParticles, type Scene } from '../../src/sim/scenes';

interface Check {
  name: string;
  value: number;
  limit: number;
  pass: boolean;
}

const checks: Check[] = [];
const expectBelow = (name: string, value: number, limit: number) => checks.push({ name, value, limit, pass: value < limit });

/** max |a - b| / max |b| over the selected components. */
function relError(a: ArrayLike<number>, b: ArrayLike<number>, stride = 1, comps = stride, count = b.length / stride): number {
  let err = 0, scale = 1e-12;
  for (let i = 0; i < count; i++)
    for (let c = 0; c < comps; c++) {
      err = Math.max(err, Math.abs(a[i * stride + c] - b[i * stride + c]));
      scale = Math.max(scale, Math.abs(b[i * stride + c]));
    }
  return err / scale;
}

async function run() {
  const { device } = await initGpu();
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', (e as GPUUncapturedErrorEvent).error.message));
  const h = 0.1;
  const scene: Scene = { id: 'dam', name: 'dam', size: [0.8, 0.6, 0.5], shapes: [{ kind: 'box', min: [0, 0, 0], max: [0.3, 0.45, 0.5], velocity: [0.3, 0, 0.1] }] };
  const mesh = prepareMesh(createBccMesh({ boundsMin: [0, 0, 0], size: scene.size, spacing: h }));
  const seed = seedParticles(scene, h / 2, 1e6);
  const params = { ...defaultParams(), restDensity: 8 / h ** 3, pcgIterations: 80 };
  const dt = 1 / 120;

  const particles: ParticleState = { count: seed.count, positions: seed.positions.slice(), velocities: seed.velocities.slice(), tets: new Uint32Array(seed.count).fill(0xffffffff) };
  ReferenceSolver.locateAll(mesh, particles);
  const ref = new ReferenceSolver(mesh, particles, params);

  const gpu = new GpuSolver(device, mesh, seed.count, params);
  gpu.setParticles(seed);
  const gpuTets = new Uint32Array(await gpu.read('particleTet', seed.count * 4));
  let tetMismatch = 0;
  for (let i = 0; i < seed.count; i++) if (gpuTets[i] !== particles.tets[i]) tetMismatch++;
  expectBelow('initial point location mismatches', tetMismatch, 1);

  ref.step(dt);
  const encoder = device.createCommandEncoder();
  gpu.encode(encoder, dt, 1, true);
  device.queue.submit([encoder.finish()]);
  gpu.resolveStats();

  const nodeState = new Float32Array(await gpu.read('nodeState'));
  const phi = Array.from({ length: mesh.nodeCount }, (_, i) => nodeState[i * 4]);
  const density = Array.from({ length: mesh.nodeCount }, (_, i) => nodeState[i * 4 + 1]);
  // φ is only meaningful where particles actually contribute; nodes with tiny total weight
  // amplify the 16.16 fixed-point rounding of the splat (they are air either way).
  const nodeAcc = new Int32Array(await gpu.read('nodeAcc'));
  const weighty = (i: number) => nodeAcc[i * 8 + 7] / 2 ** 20 > 0.05;
  expectBelow('level set φ (nodes with weight > 0.05)', relError(phi.map((v, i) => (weighty(i) ? v : 0)), Array.from(ref.nodePhi, (v, i) => (weighty(i) ? v : 0))), 1e-3);
  expectBelow('level set φ (all nodes)', relError(phi, ref.nodePhi), 2e-3);
  expectBelow('density', relError(density, ref.nodeDensity), 1e-3);
  let signFlips = 0;
  for (let i = 0; i < mesh.nodeCount; i++) if (phi[i] < 0 !== ref.nodePhi[i] < 0) signFlips++;
  expectBelow('liquid classification mismatches', signFlips, 1);
  expectBelow('rhs', relError(new Float32Array(await gpu.read('rhs')), ref.rhs), 1e-3);
  expectBelow('matrix', relError(new Float32Array(await gpu.read('matrix')), ref.matrix), 1e-4);
  expectBelow('pressure', relError(new Float32Array(await gpu.read('pressure')), ref.pressure), 5e-3);
  const refTetVel = new Float64Array(mesh.tetCount * 4);
  for (let t = 0; t < mesh.tetCount; t++) refTetVel.set(ref.tetVel.subarray(t * 3, t * 3 + 3), t * 4);
  expectBelow('tet velocity', relError(new Float32Array(await gpu.read('tetVel')), refTetVel, 4, 3), 5e-3);
  const gv = new Float32Array(await gpu.read('particleVel', seed.count * 16));
  const gp = new Float32Array(await gpu.read('particlePos', seed.count * 16));
  expectBelow('particle velocity', relError(gv, particles.velocities, 4, 3), 5e-3);
  expectBelow('particle position (abs, m)', relError(gp, particles.positions, 4, 3) * 0.8, 1e-4);
  const gt = new Uint32Array(await gpu.read('particleTet', seed.count * 4));
  let moved = 0;
  for (let i = 0; i < seed.count; i++) if (gt[i] !== particles.tets[i]) moved++;
  expectBelow('particle tet mismatches after advection (fraction)', moved / seed.count, 0.01);

  // Longer GPU-only run: stays finite, inside the box and keeps its volume roughly.
  for (let s = 0; s < 120; s++) {
    const e = device.createCommandEncoder();
    gpu.encode(e, dt, 1, s === 119);
    device.queue.submit([e.finish()]);
  }
  gpu.resolveStats();
  const pos = new Float32Array(await gpu.read('particlePos', seed.count * 16));
  let bad = 0, maxX = 0;
  for (let i = 0; i < seed.count; i++) {
    for (let c = 0; c < 3; c++) {
      const v = pos[i * 4 + c];
      if (!Number.isFinite(v) || v < mesh.boundsMin[c] || v > mesh.boundsMax[c]) bad++;
    }
    maxX = Math.max(maxX, pos[i * 4]);
  }
  expectBelow('non-finite or escaped particle coordinates after 1 s', bad, 1);
  expectBelow('dam front has not advanced (negated max x)', -maxX, -0.6);
  await device.queue.onSubmittedWorkDone();
  await new Promise((r) => setTimeout(r, 50));
  expectBelow('PCG residual reduction (final / initial)', gpu.lastStats.final / Math.max(gpu.lastStats.initial, 1e-30), 1e-3);

  return { particles: seed.count, nodes: mesh.nodeCount, tets: mesh.tetCount, pcg: gpu.lastStats, checks, pass: checks.every((c) => c.pass) };
}

declare global {
  interface Window {
    __result?: unknown;
  }
}

run()
  .then((r) => (window.__result = r))
  .catch((e) => (window.__result = { error: String(e?.stack ?? e) }))
  .finally(() => (document.getElementById('out')!.textContent = JSON.stringify(window.__result, null, 2)));

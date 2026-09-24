import { describe, expect, it } from 'vitest';
import { createBccMesh } from '../src/mesh/bcc';
import { prepareMesh } from '../src/mesh/prepare';
import { defaultParams } from '../src/sim/params';
import { ReferenceSolver, type ParticleState } from '../src/sim/reference';
import { seedParticles, type Scene } from '../src/sim/scenes';

const h = 0.1;
const mesh = prepareMesh(createBccMesh({ boundsMin: [0, 0, 0], size: [0.8, 0.6, 0.5], spacing: h }));

const noParticles = (): ParticleState => ({ count: 0, positions: new Float32Array(0), velocities: new Float32Array(0), tets: new Uint32Array(0) });

function particlesFor(scene: Scene, spacing: number): ParticleState {
  const seed = seedParticles(scene, spacing, 1e6);
  const particles = { ...seed, tets: new Uint32Array(seed.count).fill(0xffffffff) };
  ReferenceSolver.locateAll(mesh, particles);
  return particles;
}

/** Sets an exact level set and solves the projection for velocity `u` (+ gravity). */
function projectWith(phi: (x: number, y: number, z: number) => number, u: (x: number, y: number, z: number) => number[], gravity = true) {
  const params = { ...defaultParams(), pcgIterations: 400 };
  if (!gravity) params.gravity = [0, 0, 0];
  const solver = new ReferenceSolver(mesh, noParticles(), params);
  for (let i = 0; i < mesh.nodeCount; i++) solver.nodePhi[i] = phi(mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2]);
  for (let t = 0; t < mesh.tetCount; t++) {
    const c = [0, 0, 0];
    for (let a = 0; a < 4; a++) for (let k = 0; k < 3; k++) c[k] += mesh.positions[mesh.tets[t * 4 + a] * 3 + k] / 4;
    solver.tetVel.set(u(c[0], c[1], c[2]), t * 3);
  }
  solver.applyForces(1 / 60);
  solver.assemble();
  const residual = solver.solvePressure();
  solver.project();
  return { solver, residual };
}

const liquidTets = (solver: ReferenceSolver, all: boolean) => {
  const out: number[] = [];
  for (let t = 0; t < mesh.tetCount; t++) {
    const liquid = [0, 1, 2, 3].map((a) => solver.nodePhi[mesh.tets[t * 4 + a]] < 0);
    if (all ? liquid.every(Boolean) : liquid.some(Boolean)) out.push(t);
  }
  return out;
};

describe('pressure projection', () => {
  it('keeps a flat pool exactly at rest, including tets cut by the free surface (ghost fluid)', () => {
    const { solver, residual } = projectWith((_x, y) => y - 0.27, () => [0, 0, 0]);
    expect(residual).toBeLessThan(1e-9);
    let maxSpeed = 0;
    for (const t of liquidTets(solver, false))
      maxSpeed = Math.max(maxSpeed, Math.hypot(solver.tetVel[t * 3], solver.tetVel[t * 3 + 1], solver.tetVel[t * 3 + 2]));
    expect(maxSpeed).toBeLessThan(1e-6);
  });

  it('produces a velocity field that is weakly divergence free at every liquid node', () => {
    const phi = (x: number, y: number, z: number) => Math.hypot(x - 0.4, y - 0.3, z - 0.25) - 0.22;
    const swirl = (x: number, y: number, z: number) => [Math.sin(7 * y) + x, Math.cos(5 * z) - 2 * y * x, Math.sin(3 * x + z)];
    const { solver } = projectWith(phi, swirl, false);
    let before = 0, after = 0;
    for (let i = 0; i < mesh.nodeCount; i++) {
      if (solver.nodePhi[i] >= 0) continue;
      before = Math.max(before, Math.abs(solver.rhs[i]));
      let div = 0;
      for (let s = mesh.nodeTetOffsets[i]; s < mesh.nodeTetOffsets[i + 1]; s++) {
        const e = mesh.nodeTetEntries[s], t = e >> 2, a = e & 3;
        for (let c = 0; c < 3; c++) div += mesh.tetVolume[t] * mesh.tetPlanes[t * 16 + a * 4 + c] * solver.tetVel[t * 3 + c];
      }
      after = Math.max(after, Math.abs(div));
    }
    expect(before).toBeGreaterThan(0);
    expect(after).toBeLessThan(before * 1e-6);
  });

  it('produces a symmetric matrix', () => {
    const { solver } = projectWith((x, y) => y - 0.3 - 0.1 * Math.sin(8 * x), () => [0, 0, 0]);
    const entry = (i: number, j: number) => {
      for (let e = mesh.rowOffsets[i]; e < mesh.rowOffsets[i + 1]; e++) if (mesh.rowColumns[e] === j) return solver.matrix[e];
      return 0;
    };
    for (let i = 0; i < mesh.nodeCount; i++)
      for (let e = mesh.rowOffsets[i]; e < mesh.rowOffsets[i + 1]; e++) {
        const j = mesh.rowColumns[e];
        expect(solver.matrix[e]).toBeCloseTo(entry(j, i), 9);
      }
  });
});

describe('full step', () => {
  it('keeps a particle-seeded pool nearly at rest', () => {
    const particles = particlesFor({ id: 'pool', name: 'pool', size: [0.8, 0.6, 0.5], shapes: [{ kind: 'box', min: [0, 0, 0], max: [0.8, 0.3, 0.5] }] }, h / 2);
    const solver = new ReferenceSolver(mesh, particles, { ...defaultParams(), restDensity: 8 / h ** 3 });
    for (let s = 0; s < 40; s++) solver.step(1 / 60);
    let maxSpeed = 0, sumSq = 0, meanY = 0;
    for (let i = 0; i < particles.count; i++) {
      const speed = Math.hypot(particles.velocities[i * 4], particles.velocities[i * 4 + 1], particles.velocities[i * 4 + 2]);
      maxSpeed = Math.max(maxSpeed, speed);
      sumSq += speed * speed;
      meanY += particles.positions[i * 4 + 1] / particles.count;
    }
    // A column of this height falling freely would reach ~2.4 m/s.
    expect(Math.sqrt(sumSq / particles.count)).toBeLessThan(0.1);
    expect(maxSpeed).toBeLessThan(0.5);
    expect(meanY).toBeGreaterThan(0.143); // initial mean height is 0.15
  });

  it('runs a dam break without blowing up or losing particles', () => {
    const particles = particlesFor({ id: 'dam', name: 'dam', size: [0.8, 0.6, 0.5], shapes: [{ kind: 'box', min: [0, 0, 0], max: [0.3, 0.45, 0.5] }] }, h / 2);
    const solver = new ReferenceSolver(mesh, particles, { ...defaultParams(), restDensity: 8 / h ** 3 });
    for (let s = 0; s < 60; s++) solver.step(1 / 120);
    let maxX = 0;
    for (let i = 0; i < particles.count; i++) {
      for (let c = 0; c < 3; c++) {
        expect(Number.isFinite(particles.positions[i * 4 + c])).toBe(true);
        expect(particles.positions[i * 4 + c]).toBeGreaterThanOrEqual(mesh.boundsMin[c]);
        expect(particles.positions[i * 4 + c]).toBeLessThanOrEqual(mesh.boundsMax[c]);
      }
      maxX = Math.max(maxX, particles.positions[i * 4]);
    }
    // The front should have travelled well past the initial column after 0.5 s.
    expect(maxX).toBeGreaterThan(0.6);
  });
});

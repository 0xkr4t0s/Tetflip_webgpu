import { describe, expect, it } from 'vitest';
import { createBccMesh } from '../src/mesh/bcc';
import { barycentric, locate, prepareMesh } from '../src/mesh/prepare';
import type { Vec3 } from '../src/math/vec3';

const mesh = prepareMesh(createBccMesh({ boundsMin: [0, 0, 0], size: [1, 0.75, 0.5], spacing: 0.125 }));

// Deterministic pseudo-random numbers so failures are reproducible.
function rng(seed: number) {
  return () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
}

describe('BCC mesh', () => {
  it('has the expected node and tetrahedron counts', () => {
    const [nx, ny, nz] = [8, 6, 4];
    const interior = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
    const boundary = 2 * (ny * nz + nx * nz + nx * ny);
    expect(mesh.nodeCount).toBe((nx + 1) * (ny + 1) * (nz + 1) + nx * ny * nz + boundary);
    expect(mesh.tetCount).toBe((interior + boundary) * 4);
  });

  it('consists of positively oriented tetrahedra that fill the box exactly', () => {
    let total = 0;
    for (let t = 0; t < mesh.tetCount; t++) {
      expect(mesh.tetVolume[t]).toBeGreaterThan(0);
      total += mesh.tetVolume[t];
    }
    expect(total).toBeCloseTo(1 * 0.75 * 0.5, 6);
  });

  it('has symmetric face adjacency, with boundary faces only on the box surface', () => {
    const { tetNeighbors, tets, positions } = mesh;
    for (let t = 0; t < mesh.tetCount; t++) {
      for (let k = 0; k < 4; k++) {
        const n = tetNeighbors[t * 4 + k];
        if (n >= 0) {
          expect(Array.from(tetNeighbors.subarray(n * 4, n * 4 + 4))).toContain(t);
          continue;
        }
        // A boundary face must lie in one of the box planes.
        const face = [0, 1, 2, 3].filter((a) => a !== k).map((a) => tets[t * 4 + a]);
        const onPlane = [0, 1, 2].some((axis) =>
          [mesh.boundsMin[axis], mesh.boundsMax[axis]].some((v) => face.every((f) => Math.abs(positions[f * 3 + axis] - v) < 1e-9)),
        );
        expect(onPlane).toBe(true);
      }
    }
  });

  it('computes barycentric planes that interpolate linear functions exactly', () => {
    const f = (p: Vec3) => 2 * p[0] - 3 * p[1] + 0.5 * p[2] + 1;
    const rand = rng(7);
    for (let s = 0; s < 200; s++) {
      const t = Math.floor(rand() * mesh.tetCount);
      const w = [rand(), rand(), rand(), rand()];
      const sum = w.reduce((a, b) => a + b);
      const p: Vec3 = [0, 0, 0];
      for (let a = 0; a < 4; a++) {
        const n = mesh.tets[t * 4 + a];
        for (let c = 0; c < 3; c++) p[c] += (w[a] / sum) * mesh.positions[n * 3 + c];
      }
      const b = barycentric(mesh.tetPlanes, t, p);
      let interp = 0;
      for (let a = 0; a < 4; a++) {
        expect(b[a]).toBeCloseTo(w[a] / sum, 4);
        const n = mesh.tets[t * 4 + a];
        interp += b[a] * f([mesh.positions[n * 3], mesh.positions[n * 3 + 1], mesh.positions[n * 3 + 2]]);
      }
      expect(interp).toBeCloseTo(f(p), 4);
    }
  });

  it('has no positive off-diagonal stiffness couplings (no obtuse dihedral angles)', () => {
    for (let t = 0; t < mesh.tetCount; t++)
      for (let a = 0; a < 4; a++)
        for (let b = a + 1; b < 4; b++) {
          let k = 0;
          for (let c = 0; c < 3; c++) k += mesh.tetPlanes[t * 16 + a * 4 + c] * mesh.tetPlanes[t * 16 + b * 4 + c];
          expect(k).toBeLessThan(1e-9);
        }
  });

  it('builds CSR rows that start with the diagonal and whose slot maps resolve correctly', () => {
    for (let i = 0; i < mesh.nodeCount; i++) {
      const start = mesh.rowOffsets[i];
      expect(mesh.rowColumns[start]).toBe(i);
      for (let s = mesh.nodeTetOffsets[i]; s < mesh.nodeTetOffsets[i + 1]; s++) {
        const entry = mesh.nodeTetEntries[s];
        const t = entry >> 2;
        expect(mesh.tets[entry]).toBe(i);
        for (let a = 0; a < 4; a++) {
          const slot = (mesh.nodeTetSlots[s] >>> (8 * a)) & 0xff;
          expect(mesh.rowColumns[start + slot]).toBe(mesh.tets[t * 4 + a]);
        }
      }
    }
  });

  it('locates random points from the seed grid and by walking from a far-away tet', () => {
    const rand = rng(11);
    for (let s = 0; s < 500; s++) {
      const p: Vec3 = [rand() * 1, rand() * 0.75, rand() * 0.5];
      for (const start of [-1, 0, mesh.tetCount - 1]) {
        const t = locate(mesh, p, start);
        const b = barycentric(mesh.tetPlanes, t, p);
        expect(Math.min(...b)).toBeGreaterThanOrEqual(-1e-5);
      }
    }
  });
});

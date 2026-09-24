import type { Vec3 } from '../math/vec3';

export type Shape =
  | { kind: 'box'; min: Vec3; max: Vec3; velocity?: Vec3 }
  | { kind: 'sphere'; center: Vec3; radius: number; velocity?: Vec3 };

export interface Scene {
  id: string;
  name: string;
  /** Domain extent in metres; the domain spans [0, size]. */
  size: Vec3;
  shapes: Shape[];
}

export const scenes: Scene[] = [
  {
    id: 'dam-break',
    name: 'Dam break',
    size: [1.6, 0.8, 0.6],
    shapes: [{ kind: 'box', min: [0, 0, 0], max: [0.5, 0.6, 0.6] }],
  },
  {
    id: 'double-dam',
    name: 'Double dam break',
    size: [1.6, 0.8, 0.8],
    shapes: [
      { kind: 'box', min: [0, 0, 0], max: [0.45, 0.55, 0.4] },
      { kind: 'box', min: [1.15, 0, 0.4], max: [1.6, 0.55, 0.8] },
    ],
  },
  {
    id: 'drop',
    name: 'Drop into pool',
    size: [1.0, 1.0, 1.0],
    shapes: [
      { kind: 'box', min: [0, 0, 0], max: [1, 0.22, 1] },
      { kind: 'sphere', center: [0.5, 0.65, 0.5], radius: 0.16, velocity: [0, -1.5, 0] },
    ],
  },
  {
    id: 'column',
    name: 'Column collapse',
    size: [1.2, 0.8, 1.2],
    shapes: [{ kind: 'box', min: [0.42, 0, 0.42], max: [0.78, 0.7, 0.78] }],
  },
  {
    id: 'wave',
    name: 'Sloshing wave',
    size: [1.6, 0.6, 0.4],
    shapes: [{ kind: 'box', min: [0, 0, 0], max: [1.6, 0.18, 0.4], velocity: [1.2, 0, 0] }],
  },
];

export interface ParticleSeed {
  count: number;
  /** xyz + padding per particle. */
  positions: Float32Array;
  /** xyz + padding per particle. */
  velocities: Float32Array;
}

function inside(shape: Shape, p: Vec3): boolean {
  if (shape.kind === 'box') return p.every((v, a) => v >= shape.min[a] && v <= shape.max[a]);
  const d = [0, 1, 2].map((a) => p[a] - shape.center[a]);
  return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= shape.radius * shape.radius;
}

/**
 * Fills the scene's shapes with particles on a jittered lattice of the given spacing.
 * Particles are kept a small margin away from the domain walls.
 */
export function seedParticles(scene: Scene, spacing: number, maxParticles: number, seed = 1): ParticleSeed {
  let s = seed >>> 0;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
  const margin = spacing * 0.25;
  const dims = scene.size.map((l) => Math.floor(l / spacing));
  const positions: number[] = [];
  const velocities: number[] = [];
  for (let k = 0; k < dims[2]; k++)
    for (let j = 0; j < dims[1]; j++)
      for (let i = 0; i < dims[0]; i++) {
        const p: Vec3 = [i, j, k].map((c, a) => {
          const v = (c + 0.5 + (rand() - 0.5) * 0.6) * spacing;
          return Math.min(scene.size[a] - margin, Math.max(margin, v));
        }) as Vec3;
        const shape = scene.shapes.find((sh) => inside(sh, p));
        if (!shape) continue;
        if (positions.length / 4 >= maxParticles) break;
        const v = shape.velocity ?? [0, 0, 0];
        positions.push(p[0], p[1], p[2], 0);
        velocities.push(v[0], v[1], v[2], 0);
      }
  return { count: positions.length / 4, positions: Float32Array.from(positions), velocities: Float32Array.from(velocities) };
}

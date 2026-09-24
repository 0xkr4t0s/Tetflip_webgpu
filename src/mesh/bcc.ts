import type { Vec3 } from '../math/vec3';

/** Raw tetrahedral mesh: node positions plus 4 node indices per tetrahedron. */
export interface TetMesh {
  readonly nodeCount: number;
  readonly tetCount: number;
  /** xyz per node. */
  readonly positions: Float64Array;
  /** 4 node indices per tetrahedron, positively oriented. */
  readonly tets: Uint32Array;
  readonly boundsMin: Vec3;
  readonly boundsMax: Vec3;
  /** Characteristic edge length (the lattice spacing for BCC meshes). */
  readonly spacing: number;
}

export interface BccOptions {
  boundsMin: Vec3;
  /** Requested extent; each axis is rounded to a whole number of lattice cells. */
  size: Vec3;
  /** Lattice spacing h. */
  spacing: number;
}

/**
 * Body-centred cubic tetrahedralization of an axis-aligned box.
 *
 * Nodes are the cube corners (primary lattice) plus the cube centres (dual lattice).
 * Every face shared by two cubes produces four tetrahedra, each spanned by one edge of that
 * face and the two cube centres: these are the congruent, well-shaped BCC tetrahedra used in
 * the TETFLIP paper. Faces on the domain boundary get an extra node at the face centre and
 * close the mesh with four tetrahedra, each exactly half of a BCC tetrahedron mirrored across
 * the wall. The mesh fills the box exactly and has no obtuse dihedral angles, so every
 * off-diagonal pressure-matrix coupling is non-positive.
 */
export function createBccMesh(opts: BccOptions): TetMesh {
  const h = opts.spacing;
  const nx = Math.max(1, Math.round(opts.size[0] / h));
  const ny = Math.max(1, Math.round(opts.size[1] / h));
  const nz = Math.max(1, Math.round(opts.size[2] / h));
  const [ox, oy, oz] = opts.boundsMin;

  const primaryCount = (nx + 1) * (ny + 1) * (nz + 1);
  const centerCount = nx * ny * nz;
  const boundaryFaces = 2 * (ny * nz + nx * nz + nx * ny);
  const nodeCount = primaryCount + centerCount + boundaryFaces;
  const positions = new Float64Array(nodeCount * 3);

  const P = (i: number, j: number, k: number) => i + (nx + 1) * (j + (ny + 1) * k);
  const C = (i: number, j: number, k: number) => primaryCount + i + nx * (j + ny * k);

  for (let k = 0; k <= nz; k++)
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i <= nx; i++) {
        const n = P(i, j, k) * 3;
        positions[n] = ox + i * h;
        positions[n + 1] = oy + j * h;
        positions[n + 2] = oz + k * h;
      }
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const n = C(i, j, k) * 3;
        positions[n] = ox + (i + 0.5) * h;
        positions[n + 1] = oy + (j + 0.5) * h;
        positions[n + 2] = oz + (k + 0.5) * h;
      }

  const interiorFaces = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
  const tetCount = (interiorFaces + boundaryFaces) * 4;
  const tets = new Uint32Array(tetCount * 4);
  let t = 0;

  const emit = (a: number, b: number, c: number, d: number) => {
    // Orient so that the signed volume is positive.
    if (signedVolume(positions, a, b, c, d) < 0) [c, d] = [d, c];
    tets[t * 4] = a;
    tets[t * 4 + 1] = b;
    tets[t * 4 + 2] = c;
    tets[t * 4 + 3] = d;
    t++;
  };
  // A face is described by its four corners in cyclic order.
  const interior = (q: number[], c0: number, c1: number) => {
    for (let e = 0; e < 4; e++) emit(q[e], q[(e + 1) % 4], c0, c1);
  };
  let faceNode = primaryCount + centerCount;
  const boundary = (q: number[], c: number) => {
    const f = faceNode++;
    for (let a = 0; a < 3; a++)
      positions[f * 3 + a] = (positions[q[0] * 3 + a] + positions[q[1] * 3 + a] + positions[q[2] * 3 + a] + positions[q[3] * 3 + a]) / 4;
    for (let e = 0; e < 4; e++) emit(q[e], q[(e + 1) % 4], c, f);
  };

  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i <= nx; i++) {
        const q = [P(i, j, k), P(i, j + 1, k), P(i, j + 1, k + 1), P(i, j, k + 1)];
        if (i === 0) boundary(q, C(0, j, k));
        else if (i === nx) boundary(q, C(nx - 1, j, k));
        else interior(q, C(i - 1, j, k), C(i, j, k));
      }
  for (let k = 0; k < nz; k++)
    for (let j = 0; j <= ny; j++)
      for (let i = 0; i < nx; i++) {
        const q = [P(i, j, k), P(i + 1, j, k), P(i + 1, j, k + 1), P(i, j, k + 1)];
        if (j === 0) boundary(q, C(i, 0, k));
        else if (j === ny) boundary(q, C(i, ny - 1, k));
        else interior(q, C(i, j - 1, k), C(i, j, k));
      }
  for (let k = 0; k <= nz; k++)
    for (let j = 0; j < ny; j++)
      for (let i = 0; i < nx; i++) {
        const q = [P(i, j, k), P(i + 1, j, k), P(i + 1, j + 1, k), P(i, j + 1, k)];
        if (k === 0) boundary(q, C(i, j, 0));
        else if (k === nz) boundary(q, C(i, j, nz - 1));
        else interior(q, C(i, j, k - 1), C(i, j, k));
      }

  return {
    nodeCount,
    tetCount,
    positions,
    tets,
    boundsMin: [ox, oy, oz],
    boundsMax: [ox + nx * h, oy + ny * h, oz + nz * h],
    spacing: h,
  };
}

export function signedVolume(pos: Float64Array, a: number, b: number, c: number, d: number): number {
  const ax = pos[a * 3], ay = pos[a * 3 + 1], az = pos[a * 3 + 2];
  const ux = pos[b * 3] - ax, uy = pos[b * 3 + 1] - ay, uz = pos[b * 3 + 2] - az;
  const vx = pos[c * 3] - ax, vy = pos[c * 3 + 1] - ay, vz = pos[c * 3 + 2] - az;
  const wx = pos[d * 3] - ax, wy = pos[d * 3 + 1] - ay, wz = pos[d * 3 + 2] - az;
  return (ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)) / 6;
}

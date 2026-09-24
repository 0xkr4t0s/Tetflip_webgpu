import type { TetMesh } from './bcc';
import type { Vec3 } from '../math/vec3';

/**
 * A tetrahedral mesh with every static quantity the solver needs precomputed.
 * All of it is uploaded to the GPU once; only per-step fields change afterwards.
 */
export interface PreparedMesh extends TetMesh {
  /** Signed volume per tetrahedron (positive). */
  readonly tetVolume: Float32Array;
  /**
   * Barycentric-coordinate planes: for tet t and local vertex a, entries [16t + 4a .. +3]
   * hold (∇σ_a, c_a) such that σ_a(p) = ∇σ_a · p + c_a. The gradients also define the
   * per-tetrahedron pressure gradient operator [∇] of the paper.
   */
  readonly tetPlanes: Float32Array;
  /** Neighbour across the face opposite local vertex k, or -1 on the domain boundary. */
  readonly tetNeighbors: Int32Array;
  /** Sum of the volumes of all tetrahedra incident to a node. */
  readonly nodeVolume: Float32Array;
  /** Node → incident tetrahedra (CSR). Entries encode tet * 4 + localIndex. */
  readonly nodeTetOffsets: Uint32Array;
  readonly nodeTetEntries: Uint32Array;
  /** Node → node adjacency (CSR, the pressure matrix sparsity). Row i starts with i itself. */
  readonly rowOffsets: Uint32Array;
  readonly rowColumns: Uint32Array;
  /**
   * For each node→tet entry, the column slots (relative to the row start) of the tet's four
   * nodes, packed as four bytes. Lets the matrix assembly scatter without searching.
   */
  readonly nodeTetSlots: Uint32Array;
  readonly locator: Locator;
}

/** Uniform grid of seed tetrahedra for jump-and-walk point location. */
export interface Locator {
  readonly origin: Vec3;
  readonly cellSize: number;
  readonly dims: [number, number, number];
  /** A tetrahedron containing each cell's centre. */
  readonly seeds: Uint32Array;
}

export function prepareMesh(mesh: TetMesh): PreparedMesh {
  const geometry = computeGeometry(mesh);
  const tetNeighbors = computeTetNeighbors(mesh);
  const incidence = computeIncidence(mesh);
  const rows = computeRows(mesh, incidence.nodeTetOffsets, incidence.nodeTetEntries);
  const base = { ...mesh, ...geometry, tetNeighbors, ...incidence, ...rows };
  const nodeVolume = new Float32Array(mesh.nodeCount);
  for (let t = 0; t < mesh.tetCount; t++)
    for (let a = 0; a < 4; a++) nodeVolume[mesh.tets[t * 4 + a]] += geometry.tetVolume[t];
  return { ...base, nodeVolume, locator: buildLocator(base, mesh.spacing * 0.5) };
}

function computeGeometry(mesh: TetMesh) {
  const { tetCount, tets, positions: x } = mesh;
  const tetVolume = new Float32Array(tetCount);
  const tetPlanes = new Float32Array(tetCount * 16);
  for (let t = 0; t < tetCount; t++) {
    const n0 = tets[t * 4] * 3, n1 = tets[t * 4 + 1] * 3, n2 = tets[t * 4 + 2] * 3, n3 = tets[t * 4 + 3] * 3;
    const e1: Vec3 = [x[n1] - x[n0], x[n1 + 1] - x[n0 + 1], x[n1 + 2] - x[n0 + 2]];
    const e2: Vec3 = [x[n2] - x[n0], x[n2 + 1] - x[n0 + 1], x[n2 + 2] - x[n0 + 2]];
    const e3: Vec3 = [x[n3] - x[n0], x[n3 + 1] - x[n0 + 1], x[n3 + 2] - x[n0 + 2]];
    const c23 = crossArr(e2, e3), c31 = crossArr(e3, e1), c12 = crossArr(e1, e2);
    const det = e1[0] * c23[0] + e1[1] * c23[1] + e1[2] * c23[2];
    tetVolume[t] = det / 6;
    const g: Vec3[] = [[0, 0, 0], [c23[0] / det, c23[1] / det, c23[2] / det], [c31[0] / det, c31[1] / det, c31[2] / det], [c12[0] / det, c12[1] / det, c12[2] / det]];
    g[0] = [-(g[1][0] + g[2][0] + g[3][0]), -(g[1][1] + g[2][1] + g[3][1]), -(g[1][2] + g[2][2] + g[3][2])];
    const nodes = [n0, n1, n2, n3];
    for (let a = 0; a < 4; a++) {
      const n = nodes[a];
      const o = t * 16 + a * 4;
      tetPlanes[o] = g[a][0];
      tetPlanes[o + 1] = g[a][1];
      tetPlanes[o + 2] = g[a][2];
      tetPlanes[o + 3] = 1 - (g[a][0] * x[n] + g[a][1] * x[n + 1] + g[a][2] * x[n + 2]);
    }
  }
  return { tetVolume, tetPlanes };
}

const crossArr = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Matches faces by bucketing them on their smallest node index. */
function computeTetNeighbors(mesh: TetMesh): Int32Array {
  const { tetCount, nodeCount, tets } = mesh;
  const faceCount = tetCount * 4;
  const faceKey = (f: number): [number, number, number] => {
    const t = f >> 2, k = f & 3;
    const v: number[] = [];
    for (let a = 0; a < 4; a++) if (a !== k) v.push(tets[t * 4 + a]);
    v.sort((p, q) => p - q);
    return [v[0], v[1], v[2]];
  };
  const keys = new Uint32Array(faceCount * 3);
  const bucketStart = new Uint32Array(nodeCount + 1);
  for (let f = 0; f < faceCount; f++) {
    const k = faceKey(f);
    keys.set(k, f * 3);
    bucketStart[k[0] + 1]++;
  }
  for (let i = 0; i < nodeCount; i++) bucketStart[i + 1] += bucketStart[i];
  const fill = bucketStart.slice(0, nodeCount);
  const bucket = new Uint32Array(faceCount);
  for (let f = 0; f < faceCount; f++) bucket[fill[keys[f * 3]]++] = f;

  const neighbors = new Int32Array(faceCount).fill(-1);
  for (let i = 0; i < nodeCount; i++) {
    for (let s = bucketStart[i]; s < bucketStart[i + 1]; s++) {
      const f = bucket[s];
      if (neighbors[f] !== -1) continue;
      for (let r = s + 1; r < bucketStart[i + 1]; r++) {
        const g = bucket[r];
        if (keys[f * 3 + 1] === keys[g * 3 + 1] && keys[f * 3 + 2] === keys[g * 3 + 2]) {
          neighbors[f] = g >> 2;
          neighbors[g] = f >> 2;
          break;
        }
      }
    }
  }
  return neighbors;
}

function computeIncidence(mesh: TetMesh) {
  const { tetCount, nodeCount, tets } = mesh;
  const nodeTetOffsets = new Uint32Array(nodeCount + 1);
  for (let e = 0; e < tetCount * 4; e++) nodeTetOffsets[tets[e] + 1]++;
  for (let i = 0; i < nodeCount; i++) nodeTetOffsets[i + 1] += nodeTetOffsets[i];
  const fill = nodeTetOffsets.slice(0, nodeCount);
  const nodeTetEntries = new Uint32Array(tetCount * 4);
  for (let e = 0; e < tetCount * 4; e++) nodeTetEntries[fill[tets[e]]++] = e;
  return { nodeTetOffsets, nodeTetEntries };
}

function computeRows(mesh: TetMesh, nodeTetOffsets: Uint32Array, nodeTetEntries: Uint32Array) {
  const { nodeCount, tets } = mesh;
  const rowOffsets = new Uint32Array(nodeCount + 1);
  const columns: number[] = [];
  const nodeTetSlots = new Uint32Array(nodeTetEntries.length);
  const row: number[] = [];
  for (let i = 0; i < nodeCount; i++) {
    row.length = 0;
    row.push(i);
    for (let s = nodeTetOffsets[i]; s < nodeTetOffsets[i + 1]; s++) {
      const t = nodeTetEntries[s] >> 2;
      for (let a = 0; a < 4; a++) {
        const j = tets[t * 4 + a];
        if (!row.includes(j)) row.push(j);
      }
    }
    const rest = row.slice(1).sort((p, q) => p - q);
    row.length = 1;
    row.push(...rest);
    if (row.length > 255) throw new Error(`Node ${i} has ${row.length - 1} neighbours; at most 254 are supported`);
    for (let s = nodeTetOffsets[i]; s < nodeTetOffsets[i + 1]; s++) {
      const t = nodeTetEntries[s] >> 2;
      let packed = 0;
      for (let a = 0; a < 4; a++) packed |= row.indexOf(tets[t * 4 + a]) << (8 * a);
      nodeTetSlots[s] = packed >>> 0;
    }
    rowOffsets[i + 1] = rowOffsets[i] + row.length;
    for (const j of row) columns.push(j);
  }
  return { rowOffsets, rowColumns: Uint32Array.from(columns), nodeTetSlots };
}

type LocatorInput = TetMesh & { tetPlanes: Float32Array };

function buildLocator(mesh: LocatorInput, cellSize: number): Locator {
  const origin = mesh.boundsMin;
  const dims: [number, number, number] = [0, 1, 2].map((a) =>
    Math.max(1, Math.ceil((mesh.boundsMax[a] - mesh.boundsMin[a]) / cellSize - 1e-9)),
  ) as [number, number, number];
  const seeds = new Uint32Array(dims[0] * dims[1] * dims[2]).fill(0xffffffff);
  const { positions: x, tets } = mesh;
  const lo: Vec3 = [0, 0, 0], hi: Vec3 = [0, 0, 0];
  for (let t = 0; t < mesh.tetCount; t++) {
    for (let a = 0; a < 3; a++) {
      lo[a] = Infinity;
      hi[a] = -Infinity;
    }
    for (let v = 0; v < 4; v++) {
      const n = tets[t * 4 + v] * 3;
      for (let a = 0; a < 3; a++) {
        lo[a] = Math.min(lo[a], x[n + a]);
        hi[a] = Math.max(hi[a], x[n + a]);
      }
    }
    // Cells whose centres fall inside the tet's bounding box.
    const c0 = [0, 1, 2].map((a) => Math.max(0, Math.ceil((lo[a] - origin[a]) / cellSize - 0.5)));
    const c1 = [0, 1, 2].map((a) => Math.min(dims[a] - 1, Math.floor((hi[a] - origin[a]) / cellSize - 0.5)));
    for (let k = c0[2]; k <= c1[2]; k++)
      for (let j = c0[1]; j <= c1[1]; j++)
        for (let i = c0[0]; i <= c1[0]; i++) {
          const cell = i + dims[0] * (j + dims[1] * k);
          if (seeds[cell] !== 0xffffffff) continue;
          const p: Vec3 = [origin[0] + (i + 0.5) * cellSize, origin[1] + (j + 0.5) * cellSize, origin[2] + (k + 0.5) * cellSize];
          if (minBarycentric(mesh.tetPlanes, t, p) >= -1e-6) seeds[cell] = t;
        }
  }
  // Cells whose centre lies outside the mesh (only possible for non-box domains) walk from tet 0.
  for (let c = 0; c < seeds.length; c++) if (seeds[c] === 0xffffffff) seeds[c] = 0;
  return { origin: [...origin], cellSize, dims, seeds };
}

export function barycentric(planes: Float32Array, t: number, p: Vec3, out: number[] = [0, 0, 0, 0]): number[] {
  for (let a = 0; a < 4; a++) {
    const o = t * 16 + a * 4;
    out[a] = planes[o] * p[0] + planes[o + 1] * p[1] + planes[o + 2] * p[2] + planes[o + 3];
  }
  return out;
}

function minBarycentric(planes: Float32Array, t: number, p: Vec3): number {
  const b = barycentric(planes, t, p);
  return Math.min(b[0], b[1], b[2], b[3]);
}

/**
 * Jump-and-walk point location. Starts at `start` (or at the locator seed when start < 0)
 * and repeatedly steps across the face with the most negative barycentric coordinate.
 * Returns the containing tet, or the last visited tet if the point lies outside the mesh.
 * Mirrors `locate()` in the WGSL solver.
 */
export function locate(mesh: PreparedMesh, p: Vec3, start = -1, maxSteps = 64): number {
  let t = start >= 0 ? start : seedFor(mesh.locator, p);
  const b = [0, 0, 0, 0];
  for (let step = 0; step < maxSteps; step++) {
    barycentric(mesh.tetPlanes, t, p, b);
    let k = 0;
    for (let a = 1; a < 4; a++) if (b[a] < b[k]) k = a;
    if (b[k] >= -1e-5) return t;
    const next = mesh.tetNeighbors[t * 4 + k];
    if (next < 0) return t;
    t = next;
  }
  return t;
}

export function seedFor(loc: Locator, p: Vec3): number {
  const c = [0, 1, 2].map((a) => Math.min(loc.dims[a] - 1, Math.max(0, Math.floor((p[a] - loc.origin[a]) / loc.cellSize))));
  return loc.seeds[c[0] + loc.dims[0] * (c[1] + loc.dims[1] * c[2])];
}

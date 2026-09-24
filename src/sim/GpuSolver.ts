import type { PreparedMesh } from '../mesh/prepare';
import type { Vec3 } from '../math/vec3';
import { createBuffer, Kernel, readBuffer } from '../gpu/kernel';
import { DENSITY_DEADZONE, particleSpacing, SEPARATION_MAX_STEP, SEPARATION_RADIUS, SEPARATION_SURFACE_BAND, type SimParams } from './params';
import type { ParticleSeed } from './scenes';

import common from './shaders/common.wgsl?raw';
import geom from './shaders/geom.wgsl?raw';
import walk from './shaders/walk.wgsl?raw';
import p2g from './shaders/p2g.wgsl?raw';
import nodeGather from './shaders/node_gather.wgsl?raw';
import tetGather from './shaders/tet_gather.wgsl?raw';
import extrapolate from './shaders/extrapolate.wgsl?raw';
import forces from './shaders/forces.wgsl?raw';
import assemble from './shaders/assemble.wgsl?raw';
import prepareSolve from './shaders/prepare_solve.wgsl?raw';
import matvec from './shaders/matvec.wgsl?raw';
import pcgInit from './shaders/pcg_init.wgsl?raw';
import reduce from './shaders/reduce.wgsl?raw';
import pcgUpdate from './shaders/pcg_update.wgsl?raw';
import pcgDirection from './shaders/pcg_direction.wgsl?raw';
import project from './shaders/project.wgsl?raw';
import tetToNodes from './shaders/tet_to_nodes.wgsl?raw';
import g2p from './shaders/g2p.wgsl?raw';
import advect from './shaders/advect.wgsl?raw';
import locate from './shaders/locate.wgsl?raw';
import hashCount from './shaders/hash_count.wgsl?raw';
import hashScatter from './shaders/hash_scatter.wgsl?raw';
import scan from './shaders/scan.wgsl?raw';
import separate from './shaders/separate.wgsl?raw';
import separateApply from './shaders/separate_apply.wgsl?raw';

const WG = 128;
const MAX_ROW = 32;
const UNIFORM_WORDS = 48;
const SCAN_BLOCK = 512;

export interface ForceBrush {
  position: Vec3;
  velocity: Vec3;
  radius: number;
  /** Blend rate towards `velocity` per second; 0 disables the brush. */
  strength: number;
}

export interface PressureStats {
  /** Preconditioned residual rᵀM⁻¹r before and after the CG iterations of the last solve. */
  initial: number;
  final: number;
}

const S = GPUBufferUsage.STORAGE;
const COPY = GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/**
 * TETFLIP on the GPU. Owns every simulation buffer and encodes substeps as a single compute
 * pass. Each kernel mirrors one phase of `ReferenceSolver` (see reference.ts for the maths).
 */
export class GpuSolver {
  readonly device: GPUDevice;
  readonly mesh: PreparedMesh;
  readonly maxParticles: number;
  params: SimParams;
  particleCount = 0;
  /** Substep length of the most recent encode (used to scale pressure visualisation). */
  lastDt = 1 / 120;
  force: ForceBrush = { position: [0, 0, 0], velocity: [0, 0, 0], radius: 0.1, strength: 0 };

  /** Buffers other systems (the renderer, tests) may read. */
  readonly buffers: Record<string, GPUBuffer> = {};
  private readonly uniformData = new ArrayBuffer(UNIFORM_WORDS * 4);
  private readonly kernels: Record<string, Kernel> = {};
  private readonly groups: Record<string, GPUBindGroup> = {};
  private readonly numPartials: number;
  /** Neighbour grid for the position correction (cell size = separation radius). */
  private readonly hashDims: [number, number, number];
  private readonly hashCellSize: number;
  /** Element counts of each level of the multi-level prefix scan over the hash cells. */
  private readonly scanLevels: number[] = [];
  private statsStaging: GPUBuffer;
  private statsPending = false;
  lastStats: PressureStats = { initial: 0, final: 0 };

  constructor(device: GPUDevice, mesh: PreparedMesh, maxParticles: number, params: SimParams) {
    this.device = device;
    this.mesh = mesh;
    this.maxParticles = maxParticles;
    this.params = params;
    this.numPartials = Math.ceil(mesh.nodeCount / WG);
    // Particles are seeded at half the lattice spacing; the grid is fixed per mesh.
    this.hashCellSize = SEPARATION_RADIUS * (params.restDensity > 0 ? particleSpacing(params) : mesh.spacing / 2);
    this.hashDims = [0, 1, 2].map((a) => Math.max(1, Math.ceil((mesh.boundsMax[a] - mesh.boundsMin[a]) / this.hashCellSize))) as [number, number, number];
    for (let count = this.hashDims[0] * this.hashDims[1] * this.hashDims[2] + 1; ; count = Math.ceil(count / SCAN_BLOCK)) {
      this.scanLevels.push(count);
      if (count <= SCAN_BLOCK) break;
    }
    for (let i = 0; i < mesh.nodeCount; i++) {
      const len = mesh.rowOffsets[i + 1] - mesh.rowOffsets[i];
      if (len > MAX_ROW) throw new Error(`Pressure row ${i} has ${len} entries; the GPU solver supports at most ${MAX_ROW}`);
    }
    this.createBuffers();
    this.createKernels();
    this.statsStaging = device.createBuffer({ label: 'stats staging', size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  }

  private createBuffers(): void {
    const { device, mesh } = this;
    const n = mesh.nodeCount, m = mesh.tetCount;
    const b = this.buffers;
    const make = (label: string, usage: GPUBufferUsageFlags, data: ArrayBufferView | number) => (b[label] = createBuffer(device, label, usage, data));

    const nodePos = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      nodePos.set(mesh.positions.subarray(i * 3, i * 3 + 3), i * 4);
      nodePos[i * 4 + 3] = mesh.nodeVolume[i];
    }
    const tetGeom = new Float32Array(m * 20);
    for (let t = 0; t < m; t++) {
      tetGeom.set(mesh.tetPlanes.subarray(t * 16, t * 16 + 16), t * 20);
      tetGeom[t * 20 + 16] = mesh.tetVolume[t];
      for (let a = 0; a < 4; a++)
        for (let c = 0; c < 3; c++) tetGeom[t * 20 + 17 + c] += mesh.positions[mesh.tets[t * 4 + a] * 3 + c] / 4;
    }
    const nodeOffsets = new Uint32Array((n + 1) * 2);
    for (let i = 0; i <= n; i++) {
      nodeOffsets[i * 2] = mesh.nodeTetOffsets[i];
      nodeOffsets[i * 2 + 1] = mesh.rowOffsets[i];
    }
    const nodeTetInfo = new Uint32Array(mesh.nodeTetEntries.length * 2);
    for (let s = 0; s < mesh.nodeTetEntries.length; s++) {
      nodeTetInfo[s * 2] = mesh.nodeTetEntries[s];
      nodeTetInfo[s * 2 + 1] = mesh.nodeTetSlots[s];
    }

    make('sim', GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, UNIFORM_WORDS * 4);
    make('nodePos', S | COPY, nodePos);
    make('tetNodes', S | COPY, mesh.tets);
    make('tetGeom', S | COPY, tetGeom);
    make('tetNeighbors', S, mesh.tetNeighbors);
    make('nodeOffsets', S, nodeOffsets);
    make('nodeTetInfo', S, nodeTetInfo);
    make('rowColumns', S, mesh.rowColumns);
    make('locSeeds', S, mesh.locator.seeds);

    const P = this.maxParticles;
    make('particlePos', S | COPY | GPUBufferUsage.VERTEX, P * 16);
    make('particleVel', S | COPY | GPUBufferUsage.VERTEX, P * 16);
    make('particleTet', S | COPY, P * 4);
    make('tetAcc', S | COPY, m * 16);
    make('nodeAcc', S | COPY, n * 32);
    make('tetVel', S | COPY, m * 16);
    make('tetVelTmp', S | COPY, m * 16);
    make('tetVelOld', S | COPY, m * 16);
    make('nodeState', S | COPY, n * 16);
    make('nodeVel', S | COPY, n * 32);
    make('matrix', S | COPY, mesh.rowColumns.length * 4);
    for (const name of ['rhs', 'pressure', 'invDiag', 'r', 'p', 'q']) make(name, S | COPY, n * 4);
    make('partials', S | COPY, this.numPartials * 4);
    make('scalars', S | COPY, 32);

    make('particleRank', S | COPY, P * 4);
    make('sortedIndex', S | COPY, P * 4);
    make('posTmp', S | COPY, P * 16);
    // Level 0 of the scan is the cell-count array itself (one extra entry receives the total).
    this.scanLevels.forEach((count, level) => {
      make(level === 0 ? 'cellCounts' : `scanSums${level}`, S | COPY, count * 4);
      make(`scanParams${level}`, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, new Uint32Array([count, 0, 0, 0]));
    });
    make(`scanSums${this.scanLevels.length}`, S | COPY, 16);
  }

  private createKernels(): void {
    const { device, buffers: b } = this;
    const k = this.kernels;
    const g = this.groups;
    const kernel = (name: string, src: string, parts: string[] = [], constants?: Record<string, number>) =>
      (k[name] = new Kernel(device, name, [common, ...parts, src].join('\n'), constants));

    kernel('p2g', p2g, [geom]);
    kernel('nodeGather', nodeGather);
    kernel('tetGather', tetGather, [geom]);
    kernel('extrapolate', extrapolate);
    kernel('forces', forces);
    kernel('assemble', assemble, [geom]);
    kernel('prepareSolve', prepareSolve);
    kernel('matvec', matvec);
    kernel('pcgInit', pcgInit);
    for (const mode of [0, 1, 2]) kernel(`reduce${mode}`, reduce, [], { MODE: mode });
    kernel('pcgUpdate', pcgUpdate);
    kernel('pcgDirection', pcgDirection);
    kernel('project', project, [geom]);
    kernel('tetToNodes', tetToNodes, [geom]);
    kernel('g2p', g2p, [geom]);
    kernel('advect', advect, [geom, walk]);
    kernel('locate', locate, [geom, walk]);
    kernel('hashCount', hashCount);
    kernel('hashScatter', hashScatter);
    kernel('separate', separate, [geom]);
    kernel('separateApply', separateApply, [geom, walk]);
    k.scanBlocks = new Kernel(device, 'scanBlocks', scan, { SCAN_ADD: 0 });
    k.scanAdd = new Kernel(device, 'scanAdd', scan, { SCAN_ADD: 1 });

    for (const name of ['p2g', 'nodeGather', 'tetGather', 'forces', 'assemble', 'prepareSolve', 'reduce0', 'reduce1', 'reduce2', 'pcgDirection', 'project', 'tetToNodes', 'g2p', 'advect', 'locate', 'hashCount', 'separateApply'])
      g[name] = k[name].bind(b);
    g.hashScatter = k.hashScatter.bind({ ...b, cellStart: b.cellCounts });
    g.separate = k.separate.bind({ ...b, cellStart: b.cellCounts });
    this.scanLevels.forEach((_, level) => {
      const res = { params: b[`scanParams${level}`], data: level === 0 ? b.cellCounts : b[`scanSums${level}`], sums: b[`scanSums${level + 1}`] };
      g[`scanBlocks${level}`] = k.scanBlocks.bind(res);
      g[`scanAdd${level}`] = k.scanAdd.bind(res);
    });
    g.extrapolateA = k.extrapolate.bind({ ...b, src: b.tetVel, dst: b.tetVelTmp });
    g.extrapolateB = k.extrapolate.bind({ ...b, src: b.tetVelTmp, dst: b.tetVel });
    g.matvecInit = k.matvec.bind({ ...b, vin: b.pressure, vout: b.q });
    g.matvecLoop = k.matvec.bind({ ...b, vin: b.p, vout: b.q });
    g.pcgInit = k.pcgInit.bind({ ...b, ax: b.q });
    g.pcgUpdate = k.pcgUpdate.bind({ ...b, x: b.pressure });
  }

  /** Uploads new particles and locates their tetrahedra. Resets the pressure warm start. */
  setParticles(seed: ParticleSeed): void {
    const count = Math.min(seed.count, this.maxParticles);
    this.particleCount = count;
    const q = this.device.queue;
    q.writeBuffer(this.buffers.particlePos, 0, seed.positions.subarray(0, count * 4));
    q.writeBuffer(this.buffers.particleVel, 0, seed.velocities.subarray(0, count * 4));
    this.writeUniforms(1 / 60);
    const encoder = this.device.createCommandEncoder({ label: 'reset' });
    encoder.clearBuffer(this.buffers.pressure);
    const pass = encoder.beginComputePass({ label: 'locate' });
    this.dispatch(pass, 'locate', 'locate', count);
    pass.end();
    q.submit([encoder.finish()]);
  }

  private writeUniforms(dt: number): void {
    const { mesh, params } = this;
    const f = new Float32Array(this.uniformData);
    const u = new Uint32Array(this.uniformData);
    f.set(mesh.boundsMin, 0);
    f[3] = mesh.spacing;
    f.set(mesh.boundsMax, 4);
    f[7] = dt;
    f.set(params.gravity, 8);
    f[11] = params.flipRatio;
    f.set(mesh.locator.origin, 12);
    f[15] = mesh.locator.cellSize;
    u.set(mesh.locator.dims, 16);
    u[19] = this.particleCount;
    u[20] = mesh.nodeCount;
    u[21] = mesh.tetCount;
    f[22] = params.surfaceRadius * mesh.spacing;
    f[23] = params.ghostClamp;
    f[24] = params.volumeCorrection;
    f[25] = params.restDensity;
    f[26] = DENSITY_DEADZONE;
    u[27] = this.numPartials;
    f.set(this.force.position, 28);
    f[31] = this.force.radius;
    f.set(this.force.velocity, 32);
    f[35] = this.force.strength;
    u.set(this.hashDims, 36);
    f[39] = this.hashCellSize;
    const spacing = params.restDensity > 0 ? particleSpacing(params) : mesh.spacing / 2;
    f[40] = Math.min(1, params.separation * dt) * spacing;
    f[41] = spacing;
    f[42] = SEPARATION_RADIUS * spacing;
    f[43] = SEPARATION_MAX_STEP * spacing;
    f[44] = SEPARATION_SURFACE_BAND * spacing;
    this.device.queue.writeBuffer(this.buffers.sim, 0, this.uniformData);
  }

  private dispatch(pass: GPUComputePassEncoder, kernel: string, group: string, count: number): void {
    pass.setPipeline(this.kernels[kernel].pipeline);
    pass.setBindGroup(0, this.groups[group]);
    pass.dispatchWorkgroups(Math.ceil(count / WG));
  }

  private dispatchSingle(pass: GPUComputePassEncoder, kernel: string): void {
    pass.setPipeline(this.kernels[kernel].pipeline);
    pass.setBindGroup(0, this.groups[kernel]);
    pass.dispatchWorkgroups(1);
  }

  /**
   * Encodes `substeps` simulation steps of length dt each. Uniforms are written immediately,
   * so all substeps encoded before the next submit share the same dt and parameters.
   */
  encode(encoder: GPUCommandEncoder, dt: number, substeps = 1, collectStats = false): void {
    if (this.particleCount === 0) return;
    this.lastDt = dt;
    this.writeUniforms(dt);
    const P = this.particleCount, n = this.mesh.nodeCount, m = this.mesh.tetCount;
    const b = this.buffers;
    for (let s = 0; s < substeps; s++) {
      encoder.clearBuffer(b.tetAcc);
      encoder.clearBuffer(b.nodeAcc);
      const pass = encoder.beginComputePass({ label: 'tetflip step' });
      this.dispatch(pass, 'p2g', 'p2g', P);
      this.dispatch(pass, 'nodeGather', 'nodeGather', n);
      this.dispatch(pass, 'tetGather', 'tetGather', m);
      const passes = 2 * Math.ceil(this.params.extrapolationPasses / 2);
      for (let e = 0; e < passes; e++) this.dispatch(pass, 'extrapolate', e % 2 === 0 ? 'extrapolateA' : 'extrapolateB', m);
      this.dispatch(pass, 'forces', 'forces', m);
      this.dispatch(pass, 'assemble', 'assemble', n);
      this.dispatch(pass, 'prepareSolve', 'prepareSolve', n);
      this.dispatch(pass, 'matvec', 'matvecInit', n);
      this.dispatch(pass, 'pcgInit', 'pcgInit', n);
      this.dispatchSingle(pass, 'reduce0');
      for (let it = 0; it < this.params.pcgIterations; it++) {
        this.dispatch(pass, 'matvec', 'matvecLoop', n);
        this.dispatchSingle(pass, 'reduce1');
        this.dispatch(pass, 'pcgUpdate', 'pcgUpdate', n);
        this.dispatchSingle(pass, 'reduce2');
        this.dispatch(pass, 'pcgDirection', 'pcgDirection', n);
      }
      this.dispatch(pass, 'project', 'project', m);
      for (let e = 0; e < passes; e++) this.dispatch(pass, 'extrapolate', e % 2 === 0 ? 'extrapolateA' : 'extrapolateB', m);
      this.dispatch(pass, 'tetToNodes', 'tetToNodes', n);
      this.dispatch(pass, 'g2p', 'g2p', P);
      this.dispatch(pass, 'advect', 'advect', P);
      pass.end();
      if (this.params.separation > 0) this.encodeSeparation(encoder);
    }
    if (collectStats && !this.statsPending) {
      encoder.copyBufferToBuffer(b.scalars, 0, this.statsStaging, 0, 32);
      this.statsPending = true;
    }
  }

  /** Counting-sorts particles into the neighbour grid, then applies the position correction. */
  private encodeSeparation(encoder: GPUCommandEncoder): void {
    const P = this.particleCount;
    encoder.clearBuffer(this.buffers.cellCounts);
    const pass = encoder.beginComputePass({ label: 'position correction' });
    this.dispatch(pass, 'hashCount', 'hashCount', P);
    const levels = this.scanLevels;
    for (let l = 0; l < levels.length; l++) {
      pass.setPipeline(this.kernels.scanBlocks.pipeline);
      pass.setBindGroup(0, this.groups[`scanBlocks${l}`]);
      pass.dispatchWorkgroups(Math.ceil(levels[l] / SCAN_BLOCK));
    }
    for (let l = levels.length - 2; l >= 0; l--) {
      pass.setPipeline(this.kernels.scanAdd.pipeline);
      pass.setBindGroup(0, this.groups[`scanAdd${l}`]);
      pass.dispatchWorkgroups(Math.ceil(levels[l] / SCAN_BLOCK));
    }
    this.dispatch(pass, 'hashScatter', 'hashScatter', P);
    this.dispatch(pass, 'separate', 'separate', P);
    this.dispatch(pass, 'separateApply', 'separateApply', P);
    pass.end();
  }

  /** Call after submitting a command buffer encoded with collectStats. */
  resolveStats(): void {
    if (!this.statsPending || this.statsStaging.mapState !== 'unmapped') return;
    this.statsStaging.mapAsync(GPUMapMode.READ).then(
      () => {
        const s = new Float32Array(this.statsStaging.getMappedRange().slice(0));
        this.statsStaging.unmap();
        this.lastStats = { initial: s[4], final: s[0] };
        this.statsPending = false;
      },
      () => (this.statsPending = false),
    );
  }

  read(name: string, bytes?: number): Promise<ArrayBuffer> {
    return readBuffer(this.device, this.buffers[name], bytes);
  }

  destroy(): void {
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    this.statsStaging.destroy();
  }
}

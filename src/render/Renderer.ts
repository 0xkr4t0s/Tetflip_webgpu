import type { PreparedMesh } from '../mesh/prepare';
import type { GpuSolver } from '../sim/GpuSolver';
import type { Vec3 } from '../math/vec3';
import { normalize } from '../math/vec3';
import type { OrbitCamera } from './Camera';

import common from './shaders/common.wgsl?raw';
import background from './shaders/background.wgsl?raw';
import spheres from './shaders/spheres.wgsl?raw';
import blur from './shaders/blur.wgsl?raw';
import composite from './shaders/composite.wgsl?raw';
import blit from './shaders/blit.wgsl?raw';
import lines from './shaders/lines.wgsl?raw';

export type RenderMode = 'fluid' | 'particles';

export interface RenderOptions {
  mode: RenderMode;
  showMesh: boolean;
  showBox: boolean;
  /** Mesh slice position along z, as a fraction of the domain depth. */
  slice: number;
  /** Particle render radius in units of the particle spacing. */
  particleScale: number;
  fluidColor: Vec3;
  absorption: number;
}

export const defaultRenderOptions = (): RenderOptions => ({
  mode: 'fluid',
  showMesh: false,
  showBox: true,
  slice: 0.5,
  particleScale: 0.9,
  fluidColor: [0.25, 0.62, 0.95],
  absorption: 6,
});

const CAMERA_FLOATS = 92;
const DEPTH_FORMAT: GPUTextureFormat = 'depth32float';
const FLUID_DEPTH_FORMAT: GPUTextureFormat = 'rg32float';
const THICKNESS_FORMAT: GPUTextureFormat = 'r16float';
const SCENE_FORMAT: GPUTextureFormat = 'rgba16float';

type Entry = 'uniform' | 'storage' | 'texture' | 'unfilterable' | 'sampler';

interface Targets {
  width: number;
  height: number;
  scene: GPUTexture;
  fluidDepth: GPUTexture;
  fluidDepthTmp: GPUTexture;
  depth: GPUTexture;
  thickness: GPUTexture;
  thicknessTmp: GPUTexture;
  groups: Record<string, GPUBindGroup>;
}

export class Renderer {
  readonly options = defaultRenderOptions();
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly format: GPUTextureFormat;
  private readonly cameraBuffer: GPUBuffer;
  private readonly cameraData = new Float32Array(CAMERA_FLOATS);
  private readonly sampler: GPUSampler;
  private readonly layouts: Record<string, GPUBindGroupLayout> = {};
  private readonly pipelines: Record<string, GPURenderPipeline> = {};
  private targets: Targets | null = null;
  private solver: GpuSolver | null = null;
  private mesh: PreparedMesh | null = null;
  private solverGroups: Record<string, GPUBindGroup> = {};
  private sliceIndices: GPUBuffer | null = null;
  private sliceCount = 0;
  private sliceBuiltFor = -1;
  private time = 0;

  constructor(device: GPUDevice, canvas: HTMLCanvasElement) {
    this.device = device;
    this.canvas = canvas;
    this.context = canvas.getContext('webgpu') as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.cameraBuffer = device.createBuffer({ label: 'camera', size: CAMERA_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.createPipelines();
  }

  private layout(name: string, entries: Entry[]): GPUBindGroupLayout {
    const vis = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    return (this.layouts[name] = this.device.createBindGroupLayout({
      label: name,
      entries: entries.map((e, binding): GPUBindGroupLayoutEntry => {
        if (e === 'uniform') return { binding, visibility: vis, buffer: { type: 'uniform' } };
        if (e === 'storage') return { binding, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } };
        if (e === 'sampler') return { binding, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } };
        return { binding, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: e === 'texture' ? 'float' : 'unfilterable-float' } };
      }),
    }));
  }

  private createPipelines(): void {
    const d = this.device;
    const L = {
      camera: this.layout('camera', ['uniform']),
      spheres: this.layout('spheres', ['uniform', 'storage', 'storage']),
      blur: this.layout('blur', ['uniform', 'unfilterable']),
      gauss: this.layout('gauss', ['uniform', 'unfilterable']),
      composite: this.layout('composite', ['uniform', 'unfilterable', 'texture', 'texture', 'sampler']),
      blit: this.layout('blit', ['uniform', 'texture']),
      lines: this.layout('lines', ['uniform', 'storage', 'storage', 'storage']),
    };
    const mod = (label: string, src: string) => d.createShaderModule({ label, code: `${common}\n${src}` });
    const m = { background: mod('background', background), spheres: mod('spheres', spheres), blur: mod('blur', blur), composite: mod('composite', composite), blit: mod('blit', blit), lines: mod('lines', lines) };
    const alpha: GPUBlendState = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    const overlayDepth: GPUDepthStencilState = { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' };
    const pipe = (
      name: string,
      layout: GPUBindGroupLayout,
      module: GPUShaderModule,
      vs: string,
      fs: string,
      format: GPUTextureFormat,
      extra: { blend?: GPUBlendState; depth?: GPUDepthStencilState; topology?: GPUPrimitiveTopology; constants?: Record<string, number> } = {},
    ) =>
      (this.pipelines[name] = d.createRenderPipeline({
        label: name,
        layout: d.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: vs, constants: extra.constants },
        fragment: { module, entryPoint: fs, targets: [{ format, blend: extra.blend }], constants: extra.constants },
        primitive: { topology: extra.topology ?? 'triangle-list' },
        depthStencil: extra.depth,
      }));

    pipe('background', L.camera, m.background, 'vs', 'fs', SCENE_FORMAT);
    pipe('fluidDepth', L.spheres, m.spheres, 'vs', 'fsDepth', FLUID_DEPTH_FORMAT, { depth: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' } });
    pipe('thickness', L.spheres, m.spheres, 'vs', 'fsThickness', THICKNESS_FORMAT, {
      blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } },
    });
    pipe('blurH', L.blur, m.blur, 'vs', 'fsDepth', FLUID_DEPTH_FORMAT, { constants: { DIRECTION: 0 } });
    pipe('blurV', L.blur, m.blur, 'vs', 'fsDepth', FLUID_DEPTH_FORMAT, { constants: { DIRECTION: 1 } });
    pipe('gaussH', L.gauss, m.blur, 'vs', 'fsGaussian', THICKNESS_FORMAT, { constants: { DIRECTION: 0 } });
    pipe('gaussV', L.gauss, m.blur, 'vs', 'fsGaussian', THICKNESS_FORMAT, { constants: { DIRECTION: 1 } });
    pipe('composite', L.composite, m.composite, 'vs', 'fs', this.format, { depth: overlayDepth });
    pipe('blit', L.blit, m.blit, 'vs', 'fs', this.format, { depth: overlayDepth });
    pipe('particles', L.spheres, m.spheres, 'vs', 'fsShaded', this.format, { depth: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' } });
    pipe('mesh', L.lines, m.lines, 'vsMesh', 'fs', this.format, { blend: alpha, depth: overlayDepth, topology: 'line-list' });
    pipe('box', L.lines, m.lines, 'vsBox', 'fs', this.format, { blend: alpha, depth: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' }, topology: 'line-list' });
  }

  /** Points the renderer at a (new) solver's buffers. */
  setSolver(solver: GpuSolver): void {
    this.solver = solver;
    this.mesh = solver.mesh;
    const b = solver.buffers;
    const group = (layout: string, buffers: GPUBuffer[]) =>
      this.device.createBindGroup({
        layout: this.layouts[layout],
        entries: [{ binding: 0, resource: { buffer: this.cameraBuffer } }, ...buffers.map((buffer, i) => ({ binding: i + 1, resource: { buffer } }))],
      });
    this.solverGroups = {
      spheres: group('spheres', [b.particlePos, b.particleVel]),
      lines: group('lines', [b.nodePos, b.nodeState, b.pressure]),
    };
    this.sliceBuiltFor = -1;
  }

  private ensureTargets(): Targets {
    const width = Math.max(1, this.canvas.width), height = Math.max(1, this.canvas.height);
    if (this.targets && this.targets.width === width && this.targets.height === height) return this.targets;
    if (this.targets) for (const t of [this.targets.scene, this.targets.fluidDepth, this.targets.fluidDepthTmp, this.targets.depth, this.targets.thickness, this.targets.thicknessTmp]) t.destroy();
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const tex = (label: string, format: GPUTextureFormat, w = width, h = height) => this.device.createTexture({ label, size: [w, h], format, usage });
    const hw = Math.max(1, width >> 1), hh = Math.max(1, height >> 1);
    const t = {
      width,
      height,
      scene: tex('scene', SCENE_FORMAT),
      fluidDepth: tex('fluid depth', FLUID_DEPTH_FORMAT),
      fluidDepthTmp: tex('fluid depth tmp', FLUID_DEPTH_FORMAT),
      depth: tex('depth', DEPTH_FORMAT),
      thickness: tex('thickness', THICKNESS_FORMAT, hw, hh),
      thicknessTmp: tex('thickness tmp', THICKNESS_FORMAT, hw, hh),
      groups: {} as Record<string, GPUBindGroup>,
    };
    const cam = { binding: 0, resource: { buffer: this.cameraBuffer } };
    const g = (layout: string, ...resources: GPUBindingResource[]) =>
      this.device.createBindGroup({ layout: this.layouts[layout], entries: [cam, ...resources.map((resource, i) => ({ binding: i + 1, resource }))] });
    t.groups = {
      camera: g('camera'),
      blurH: g('blur', t.fluidDepth.createView()),
      blurV: g('blur', t.fluidDepthTmp.createView()),
      gaussH: g('gauss', t.thickness.createView()),
      gaussV: g('gauss', t.thicknessTmp.createView()),
      composite: g('composite', t.fluidDepth.createView(), t.thickness.createView(), t.scene.createView(), this.sampler),
      blit: g('blit', t.scene.createView()),
    };
    this.targets = t;
    return t;
  }

  private writeCamera(camera: OrbitCamera, t: Targets): void {
    const mesh = this.mesh!;
    const o = this.options;
    const c = this.cameraData;
    c.set(camera.view, 0);
    c.set(camera.proj, 16);
    c.set(camera.invProj, 32);
    c.set(camera.invView, 48);
    c.set(camera.eye, 64);
    c[67] = this.time;
    c[68] = t.width;
    c[69] = t.height;
    c[70] = camera.near;
    c[71] = camera.far;
    c.set(mesh.boundsMin, 72);
    c[75] = mesh.boundsMin[1] - 0.002;
    c.set(mesh.boundsMax, 76);
    c.set(normalize([-0.45, 0.8, 0.35]), 80);
    c.set(o.fluidColor, 84);
    c[87] = o.absorption;
    const height = mesh.boundsMax[1] - mesh.boundsMin[1];
    const dt = this.solver?.lastDt ?? 1 / 120;
    c[88] = o.particleScale * mesh.spacing * 0.5;
    c[89] = 1 / Math.sqrt(2 * 9.81 * height);
    c[90] = 1 / (9.81 * height * dt);
    c[91] = o.mode === 'fluid' ? 0 : 1;
    this.device.queue.writeBuffer(this.cameraBuffer, 0, c);
  }

  /** Line-list indices of all tet edges whose tet centroid lies in a slab around the slice. */
  private buildSlice(): void {
    const mesh = this.mesh!;
    const key = Math.round(this.options.slice * 1000);
    if (key === this.sliceBuiltFor) return;
    this.sliceBuiltFor = key;
    const z0 = mesh.boundsMin[2] + this.options.slice * (mesh.boundsMax[2] - mesh.boundsMin[2]);
    const half = mesh.spacing * 0.5;
    const seen = new Set<number>();
    const idx: number[] = [];
    const n = mesh.nodeCount;
    for (let t = 0; t < mesh.tetCount; t++) {
      let cz = 0;
      for (let a = 0; a < 4; a++) cz += mesh.positions[mesh.tets[t * 4 + a] * 3 + 2] * 0.25;
      if (Math.abs(cz - z0) > half) continue;
      for (let a = 0; a < 4; a++)
        for (let b = a + 1; b < 4; b++) {
          const i = mesh.tets[t * 4 + a], j = mesh.tets[t * 4 + b];
          const k = Math.min(i, j) * n + Math.max(i, j);
          if (seen.has(k)) continue;
          seen.add(k);
          idx.push(i, j);
        }
    }
    this.sliceIndices?.destroy();
    const data = Uint32Array.from(idx.length ? idx : [0, 0]);
    this.sliceIndices = this.device.createBuffer({ label: 'slice indices', size: data.byteLength, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.sliceIndices, 0, data);
    this.sliceCount = idx.length;
  }

  get colorFormat(): GPUTextureFormat {
    return this.format;
  }

  /** Encodes a frame into the canvas, or into `target` (same format and size) when given. */
  encode(encoder: GPUCommandEncoder, camera: OrbitCamera, dt: number, target?: GPUTextureView): void {
    if (!this.solver || !this.mesh) return;
    this.time += dt;
    const t = this.ensureTargets();
    camera.update(t.width / t.height);
    this.writeCamera(camera, t);
    const particles = this.solver.particleCount;
    const o = this.options;
    const P = this.pipelines;

    const colorPass = (view: GPUTextureView, clear = true): GPURenderPassColorAttachment => ({
      view,
      loadOp: clear ? 'clear' : 'load',
      storeOp: 'store',
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
    });
    const fullscreen = (target: GPUTextureView, pipeline: GPURenderPipeline, group: GPUBindGroup) => {
      const pass = encoder.beginRenderPass({ colorAttachments: [colorPass(target)] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
    };

    fullscreen(t.scene.createView(), P.background, t.groups.camera);

    if (o.mode === 'fluid' && particles > 0) {
      const depthPass = encoder.beginRenderPass({
        colorAttachments: [colorPass(t.fluidDepth.createView())],
        depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
      });
      depthPass.setPipeline(P.fluidDepth);
      depthPass.setBindGroup(0, this.solverGroups.spheres);
      depthPass.draw(6, particles);
      depthPass.end();

      const thickPass = encoder.beginRenderPass({ colorAttachments: [colorPass(t.thickness.createView())] });
      thickPass.setPipeline(P.thickness);
      thickPass.setBindGroup(0, this.solverGroups.spheres);
      thickPass.draw(6, particles);
      thickPass.end();

      for (let i = 0; i < 3; i++) {
        fullscreen(t.fluidDepthTmp.createView(), P.blurH, t.groups.blurH);
        fullscreen(t.fluidDepth.createView(), P.blurV, t.groups.blurV);
      }
      fullscreen(t.thicknessTmp.createView(), P.gaussH, t.groups.gaussH);
      fullscreen(t.thickness.createView(), P.gaussV, t.groups.gaussV);
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [colorPass(target ?? this.context.getCurrentTexture().createView())],
      depthStencilAttachment: { view: t.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    if (o.mode === 'fluid' && particles > 0) {
      pass.setPipeline(P.composite);
      pass.setBindGroup(0, t.groups.composite);
    } else {
      pass.setPipeline(P.blit);
      pass.setBindGroup(0, t.groups.blit);
    }
    pass.draw(3);
    if (o.mode === 'particles' && particles > 0) {
      pass.setPipeline(P.particles);
      pass.setBindGroup(0, this.solverGroups.spheres);
      pass.draw(6, particles);
    }
    if (o.showMesh) {
      this.buildSlice();
      if (this.sliceCount > 0 && this.sliceIndices) {
        pass.setPipeline(P.mesh);
        pass.setBindGroup(0, this.solverGroups.lines);
        pass.setIndexBuffer(this.sliceIndices, 'uint32');
        pass.drawIndexed(this.sliceCount);
      }
    }
    if (o.showBox) {
      pass.setPipeline(P.box);
      pass.setBindGroup(0, this.solverGroups.lines);
      pass.draw(24);
    }
    pass.end();
  }
}

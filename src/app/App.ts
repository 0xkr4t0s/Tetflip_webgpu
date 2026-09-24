import GUI from 'lil-gui';
import { initGpu } from '../gpu/device';
import { buildMeshAsync } from '../mesh/buildMesh';
import type { PreparedMesh } from '../mesh/prepare';
import type { Vec3 } from '../math/vec3';
import { attachCameraControls, OrbitCamera } from '../render/Camera';
import { Renderer } from '../render/Renderer';
import { GpuSolver } from '../sim/GpuSolver';
import { defaultParams, type SimParams } from '../sim/params';
import { scenes, seedParticles, type Scene } from '../sim/scenes';

/** Lattice cells along the longest domain axis. */
const RESOLUTIONS: Record<string, number> = { 'Low (fast)': 40, Medium: 64, High: 88 };
/** Particles are seeded 2 × 2 × 2 per lattice cell. */
const PARTICLES_PER_AXIS = 2;
const FRAME_DT = 1 / 60;

interface Settings {
  scene: string;
  resolution: string;
  paused: boolean;
  timeScale: number;
  substeps: number;
  brush: boolean;
  brushRadius: number;
  brushStrength: number;
}

export class App {
  private readonly canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private renderer!: Renderer;
  private readonly camera = new OrbitCamera();
  private solver: GpuSolver | null = null;
  private mesh: PreparedMesh | null = null;
  private scene: Scene = scenes[0];
  private readonly params: SimParams = defaultParams();
  private readonly settings: Settings = {
    scene: scenes[0].id,
    resolution: 'Medium',
    paused: false,
    timeScale: 1,
    substeps: 2,
    brush: false,
    brushRadius: 0.15,
    brushStrength: 40,
  };
  private gui: GUI | null = null;
  private loading = false;
  private simTime = 0;
  private frameCount = 0;
  private fps = 60;
  private lastFrame = performance.now();
  private lastStats = 0;
  private brushActive = false;
  private brushPoint: Vec3 | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  /** Applies ?scene=<id>&res=low|medium|high&view=particles&mesh=1&paused=1 from the URL. */
  private applyUrlOptions(): void {
    const q = new URLSearchParams(location.search);
    const scene = q.get('scene');
    if (scene && scenes.some((sc) => sc.id === scene)) this.settings.scene = scene;
    const res = q.get('res')?.toLowerCase();
    const match = Object.keys(RESOLUTIONS).find((k) => k.toLowerCase().startsWith(res ?? '\0'));
    if (match) this.settings.resolution = match;
    if (q.get('view') === 'particles') this.renderer.options.mode = 'particles';
    if (q.get('mesh') === '1') this.renderer.options.showMesh = true;
    if (q.get('paused') === '1') this.settings.paused = true;
  }

  async start(): Promise<void> {
    const gpu = await initGpu();
    this.device = gpu.device;
    this.device.addEventListener('uncapturederror', (e) => console.error('WebGPU error:', (e as GPUUncapturedErrorEvent).error.message));
    this.device.lost.then((info) => {
      if (info.reason !== 'destroyed') this.showError(new Error(`The GPU device was lost: ${info.message}`));
    });
    this.renderer = new Renderer(this.device, this.canvas);
    this.applyUrlOptions();
    this.observeSize();
    this.attachInput();
    this.buildGui();
    await this.load();
    requestAnimationFrame(this.frame);
    if (import.meta.env.DEV) Object.assign(window, { __tetflip: this });
  }

  /** Simulated seconds since the last reset (exposed for automated checks). */
  get time(): number {
    return this.simTime;
  }

  /**
   * Renders one frame offscreen and returns it as base64 RGBA (for automated screenshots in
   * environments where the canvas cannot be presented, e.g. headless SwiftShader).
   */
  async capture(): Promise<{ width: number; height: number; rgba: string }> {
    const width = this.canvas.width, height = this.canvas.height;
    const texture = this.device.createTexture({ size: [width, height], format: this.renderer.colorFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const buffer = this.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = this.device.createCommandEncoder();
    this.renderer.encode(encoder, this.camera, 0, texture.createView());
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
    this.device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(width * height * 4);
    const bgra = this.renderer.colorFormat.startsWith('bgra');
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const i = y * bytesPerRow + x * 4, o = (y * width + x) * 4;
        out[o] = src[i + (bgra ? 2 : 0)];
        out[o + 1] = src[i + 1];
        out[o + 2] = src[i + (bgra ? 0 : 2)];
        out[o + 3] = 255;
      }
    buffer.unmap();
    buffer.destroy();
    texture.destroy();
    let binary = '';
    for (let i = 0; i < out.length; i += 0x8000) binary += String.fromCharCode(...out.subarray(i, i + 0x8000));
    return { width, height, rgba: btoa(binary) };
  }

  /** Frames rendered so far (exposed for automated checks). */
  get frames(): number {
    return this.frameCount;
  }

  showError(err: unknown): void {
    console.error(err);
    document.getElementById('overlay')!.hidden = true;
    document.getElementById('error-text')!.textContent = err instanceof Error ? err.message : String(err);
    document.getElementById('error')!.hidden = false;
  }

  private observeSize(): void {
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const max = this.device.limits.maxTextureDimension2D;
      this.canvas.width = Math.min(max, Math.max(1, Math.round(this.canvas.clientWidth * dpr)));
      this.canvas.height = Math.min(max, Math.max(1, Math.round(this.canvas.clientHeight * dpr)));
    };
    new ResizeObserver(resize).observe(this.canvas);
    resize();
  }

  /** (Re)builds the mesh and solver for the current scene and resolution. */
  private async load(): Promise<void> {
    if (this.loading) return;
    this.loading = true;
    const overlay = document.getElementById('overlay')!;
    overlay.hidden = false;
    try {
      this.scene = scenes.find((s) => s.id === this.settings.scene) ?? scenes[0];
      const longest = Math.max(...this.scene.size);
      const spacing = longest / RESOLUTIONS[this.settings.resolution];
      document.getElementById('overlay-text')!.textContent = 'Building tetrahedral mesh…';
      const mesh = await buildMeshAsync({ size: this.scene.size, spacing });
      document.getElementById('overlay-text')!.textContent = 'Uploading to the GPU…';
      await new Promise((r) => setTimeout(r, 0));
      const seed = this.seed(mesh);
      this.params.restDensity = PARTICLES_PER_AXIS ** 3 / mesh.spacing ** 3;
      this.solver?.destroy();
      this.solver = new GpuSolver(this.device, mesh, seed.count, this.params);
      this.solver.setParticles(seed);
      this.mesh = mesh;
      this.renderer.setSolver(this.solver);
      this.camera.frame(mesh.boundsMin, mesh.boundsMax);
      this.simTime = 0;
    } finally {
      this.loading = false;
      overlay.hidden = true;
    }
  }

  private seed(mesh: PreparedMesh) {
    return seedParticles(this.scene, mesh.spacing / PARTICLES_PER_AXIS, 4_000_000);
  }

  private reset(): void {
    if (!this.solver || !this.mesh) return;
    this.solver.setParticles(this.seed(this.mesh));
    this.simTime = 0;
  }

  private frame = (now: number): void => {
    requestAnimationFrame(this.frame);
    const elapsed = Math.min(0.25, (now - this.lastFrame) / 1000);
    this.lastFrame = now;
    this.fps += (1 / Math.max(elapsed, 1e-3) - this.fps) * 0.05;
    if (!this.solver || this.loading) return;

    const encoder = this.device.createCommandEncoder({ label: 'frame' });
    const collect = this.frameCount++ % 20 === 0;
    if (!this.settings.paused) {
      const substeps = this.settings.substeps;
      const dt = (FRAME_DT * this.settings.timeScale) / substeps;
      this.solver.encode(encoder, dt, substeps, collect);
      this.simTime += dt * substeps;
    }
    this.renderer.encode(encoder, this.camera, elapsed);
    this.device.queue.submit([encoder.finish()]);
    this.solver.resolveStats();
    if (now - this.lastStats > 250) {
      this.lastStats = now;
      this.updateStats();
    }
  };

  private updateStats(): void {
    const s = this.solver!, m = this.mesh!;
    const pcg = s.lastStats.initial > 0 ? Math.sqrt(s.lastStats.final / s.lastStats.initial) : 0;
    const rows: [string, string][] = [
      ['fps', this.fps.toFixed(0)],
      ['particles', s.particleCount.toLocaleString()],
      ['tetrahedra', m.tetCount.toLocaleString()],
      ['nodes', m.nodeCount.toLocaleString()],
      ['sim time', `${this.simTime.toFixed(2)} s`],
      ['CG residual', pcg > 0 ? pcg.toExponential(1) : '–'],
    ];
    document.getElementById('stats')!.innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  }

  private attachInput(): void {
    attachCameraControls(this.canvas, this.camera, {
      wantsBrush: (e) => e.button === 0 && (e.shiftKey || this.settings.brush),
      brushStart: (e) => {
        this.brushActive = true;
        this.brushPoint = this.brushTarget(e);
      },
      brushMove: (e) => {
        const p = this.brushTarget(e);
        if (!this.solver || !p) return;
        const prev = this.brushPoint ?? p;
        const v = p.map((c, i) => (c - prev[i]) / FRAME_DT) as Vec3;
        const speed = Math.hypot(...v);
        const cap = 3;
        const scale = speed > cap ? cap / speed : 1;
        this.solver.force = {
          position: p,
          velocity: v.map((c) => c * scale) as Vec3,
          radius: this.settings.brushRadius,
          strength: this.settings.brushStrength,
        };
        this.brushPoint = p;
      },
      brushEnd: () => {
        this.brushActive = false;
        this.brushPoint = null;
        if (this.solver) this.solver.force = { ...this.solver.force, strength: 0 };
      },
    });
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      if (e.code === 'Space') {
        this.settings.paused = !this.settings.paused;
        e.preventDefault();
      } else if (e.code === 'KeyR') this.reset();
      else if (e.code === 'KeyV') this.renderer.options.mode = this.renderer.options.mode === 'fluid' ? 'particles' : 'fluid';
      else if (e.code === 'KeyM') this.renderer.options.showMesh = !this.renderer.options.showMesh;
      else return;
      this.gui?.controllersRecursive().forEach((c) => c.updateDisplay());
    });
  }

  /** Where a screen ray passes through the tank: the midpoint of its segment inside the box. */
  private brushTarget(e: PointerEvent): Vec3 | null {
    if (!this.mesh) return null;
    const rect = this.canvas.getBoundingClientRect();
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((e.clientY - rect.top) / rect.height) * 2;
    const { origin, dir } = this.camera.ray(ndcX, ndcY);
    let t0 = 0, t1 = Infinity;
    for (let a = 0; a < 3; a++) {
      const inv = 1 / dir[a];
      let ta = (this.mesh.boundsMin[a] - origin[a]) * inv;
      let tb = (this.mesh.boundsMax[a] - origin[a]) * inv;
      if (ta > tb) [ta, tb] = [tb, ta];
      t0 = Math.max(t0, ta);
      t1 = Math.min(t1, tb);
    }
    if (t0 > t1) return this.brushActive ? this.brushPoint : null;
    const t = (t0 + t1) / 2;
    return origin.map((o, a) => o + dir[a] * t) as Vec3;
  }

  private buildGui(): void {
    const gui = (this.gui = new GUI({ title: 'TETFLIP' }));
    const s = this.settings;
    const p = this.params;
    const r = this.renderer.options;

    const sim = gui.addFolder('Simulation');
    sim.add(s, 'scene', Object.fromEntries(scenes.map((sc) => [sc.name, sc.id]))).name('scene').onChange(() => this.load());
    sim.add(s, 'resolution', Object.keys(RESOLUTIONS)).name('mesh resolution').onChange(() => this.load());
    sim.add(s, 'paused').name('paused').listen();
    sim.add({ reset: () => this.reset() }, 'reset').name('reset scene (R)');
    sim.add(s, 'timeScale', 0.1, 2, 0.05).name('time scale');
    sim.add(s, 'substeps', 1, 6, 1).name('substeps / frame');

    const phys = gui.addFolder('Physics');
    phys.add(p.gravity, '1', -20, 0, 0.1).name('gravity (m/s²)');
    phys.add(p, 'flipRatio', 0.8, 1, 0.005).name('FLIP ratio');
    phys.add(p, 'pcgIterations', 5, 200, 1).name('CG iterations');
    phys.add(p, 'volumeCorrection', 0, 3, 0.05).name('volume correction');
    phys.add(p, 'surfaceRadius', 0.3, 0.8, 0.01).name('surface radius (h)');
    phys.close();

    const view = gui.addFolder('Rendering');
    view.add(r, 'mode', { 'liquid surface': 'fluid', particles: 'particles' }).name('view (V)').listen();
    view.add(r, 'showMesh').name('mesh slice (M)').listen();
    view.add(r, 'slice', 0, 1, 0.01).name('slice position');
    view.add(r, 'showBox').name('tank outline');
    view.add(r, 'particleScale', 0.4, 1.6, 0.05).name('particle size');
    view.add(r, 'absorption', 0, 15, 0.1).name('absorption');
    view.addColor(r, 'fluidColor').name('liquid colour');

    const brush = gui.addFolder('Interaction');
    brush.add(s, 'brush').name('drag pushes liquid');
    brush.add(s, 'brushRadius', 0.04, 0.4, 0.01).name('brush radius (m)');
    brush.add(s, 'brushStrength', 5, 100, 1).name('brush strength');
    brush.close();

    if (window.innerWidth < 720) gui.close();
  }
}

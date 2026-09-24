import { invert, lookAt, multiply, perspective, transformPoint, type Mat4 } from '../math/mat4';
import { add, normalize, scale, sub, type Vec3 } from '../math/vec3';

export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 3;
  yaw = -0.6;
  pitch = 0.35;
  readonly fovy = (40 * Math.PI) / 180;
  readonly near = 0.02;
  readonly far = 60;

  eye: Vec3 = [0, 0, 0];
  view: Mat4 = new Float32Array(16);
  proj: Mat4 = new Float32Array(16);
  invView: Mat4 = new Float32Array(16);
  invProj: Mat4 = new Float32Array(16);
  viewProj: Mat4 = new Float32Array(16);

  /** Frames an axis-aligned box. */
  frame(min: Vec3, max: Vec3): void {
    this.target = scale(add(min, max), 0.5);
    this.target[1] = min[1] + (max[1] - min[1]) * 0.35;
    const extent = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    this.distance = extent * 1.35;
    this.yaw = -0.55;
    this.pitch = 0.42;
  }

  update(aspect: number): void {
    const cp = Math.cos(this.pitch);
    const dir: Vec3 = [Math.sin(this.yaw) * cp, Math.sin(this.pitch), Math.cos(this.yaw) * cp];
    this.eye = add(this.target, scale(dir, this.distance));
    this.view = lookAt(this.eye, this.target, [0, 1, 0]);
    this.proj = perspective(this.fovy, aspect, this.near, this.far);
    this.invView = invert(this.view);
    this.invProj = invert(this.proj);
    this.viewProj = multiply(this.proj, this.view);
  }

  orbit(dx: number, dy: number): void {
    this.yaw -= dx * 0.006;
    this.pitch = Math.min(1.45, Math.max(-0.2, this.pitch + dy * 0.006));
  }

  zoom(factor: number): void {
    this.distance = Math.min(20, Math.max(0.3, this.distance * factor));
  }

  pan(dx: number, dy: number): void {
    const s = this.distance * 0.0015;
    const right: Vec3 = [this.view[0], this.view[4], this.view[8]];
    const up: Vec3 = [this.view[1], this.view[5], this.view[9]];
    this.target = add(this.target, add(scale(right, -dx * s), scale(up, dy * s)));
  }

  /** World-space ray through normalised device coordinates. */
  ray(ndcX: number, ndcY: number): { origin: Vec3; dir: Vec3 } {
    const pView = transformPoint(this.invProj, [ndcX, ndcY, 0.5]);
    const pWorld = transformPoint(this.invView, pView);
    return { origin: this.eye, dir: normalize(sub(pWorld, this.eye)) };
  }
}

export interface PointerHandlers {
  /** Return true to route this drag to the force brush instead of the camera. */
  wantsBrush(e: PointerEvent): boolean;
  brushStart(e: PointerEvent): void;
  brushMove(e: PointerEvent): void;
  brushEnd(): void;
}

/** Mouse/touch camera controls: drag to orbit, right/middle drag to pan, wheel or pinch to zoom. */
export function attachCameraControls(canvas: HTMLCanvasElement, camera: OrbitCamera, handlers: PointerHandlers): void {
  const pointers = new Map<number, { x: number; y: number }>();
  let mode: 'orbit' | 'pan' | 'brush' | null = null;
  let pinch = 0;

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      if (mode === 'brush') handlers.brushEnd();
      const [a, b] = [...pointers.values()];
      pinch = Math.hypot(a.x - b.x, a.y - b.y);
      mode = 'pan';
      return;
    }
    if (handlers.wantsBrush(e)) {
      mode = 'brush';
      handlers.brushStart(e);
    } else mode = e.button === 0 && !e.ctrlKey ? 'orbit' : 'pan';
  });
  canvas.addEventListener('pointermove', (e) => {
    const prev = pointers.get(e.pointerId);
    if (!prev) return;
    const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (pinch > 0) camera.zoom(pinch / d);
      pinch = d;
      camera.pan(dx / 2, dy / 2);
    } else if (mode === 'orbit') camera.orbit(dx, dy);
    else if (mode === 'pan') camera.pan(dx, dy);
    else if (mode === 'brush') handlers.brushMove(e);
  });
  const end = (e: PointerEvent) => {
    pointers.delete(e.pointerId);
    if (mode === 'brush') handlers.brushEnd();
    mode = pointers.size === 1 ? 'orbit' : null;
    pinch = 0;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      camera.zoom(Math.exp(e.deltaY * 0.001));
    },
    { passive: false },
  );
}

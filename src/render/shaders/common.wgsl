// Shared by all render shaders. Mirrors the camera uniform written in Renderer.ts.
struct Camera {
  view: mat4x4f,
  proj: mat4x4f,
  invProj: mat4x4f,
  invView: mat4x4f,
  eye: vec3f, time: f32,
  resolution: vec2f, near: f32, far: f32,
  boundsMin: vec3f, floorY: f32,
  boundsMax: vec3f, _pad1: f32,
  lightDir: vec3f, _pad2: f32,
  fluidColor: vec3f, absorption: f32,
  // particleRadius, speedScale (1 / reference speed), pressureScale, colorMode
  params: vec4f,
}

@group(0) @binding(0) var<uniform> cam: Camera;

// ACES filmic tone curve (Narkowicz fit) followed by sRGB-like gamma. Everything upstream of
// the final output stays in linear HDR.
fn toDisplay(c: vec3f) -> vec3f {
  let x = c * 0.9;
  let t = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
  return pow(clamp(t, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
}

fn skyColor(dir: vec3f) -> vec3f {
  let t = clamp(dir.y, 0.0, 1.0);
  var c = mix(vec3f(0.42, 0.47, 0.55), vec3f(0.10, 0.17, 0.32), pow(t, 0.6));
  c += vec3f(0.30, 0.28, 0.26) * exp(-abs(dir.y) * 10.0);
  let sun = max(dot(dir, cam.lightDir), 0.0);
  c += vec3f(1.0, 0.93, 0.82) * (pow(sun, 1200.0) * 60.0 + pow(sun, 16.0) * 0.35);
  return c;
}

// Soft darkening of the floor under and around the tank.
fn floorShade(p: vec3f) -> f32 {
  let c = 0.5 * (cam.boundsMin.xz + cam.boundsMax.xz);
  let h = 0.5 * (cam.boundsMax.xz - cam.boundsMin.xz);
  let q = abs(p.xz - c) - h;
  let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
  return 0.55 + 0.45 * smoothstep(-0.05, 0.35, d);
}

// Environment seen along a world-space ray: a checkered floor that fades into the sky.
fn environment(origin: vec3f, dir: vec3f) -> vec3f {
  let sky = skyColor(dir);
  if (dir.y >= -1e-4) { return sky; }
  let t = (cam.floorY - origin.y) / dir.y;
  if (t <= 0.0) { return sky; }
  let p = origin + dir * t;
  let cell = floor(p.xz / 0.1);
  let checker = select(0.20, 0.26, (i32(cell.x) + i32(cell.y)) % 2 == 0);
  // Thin grout lines between tiles.
  let g = abs(fract(p.xz / 0.1) - 0.5);
  let grout = smoothstep(0.47, 0.5, max(g.x, g.y));
  var c = vec3f(mix(checker, 0.14, grout * 0.6)) * vec3f(0.96, 0.95, 1.0) * floorShade(p);
  let fog = 1.0 - exp(-t * 0.12);
  return mix(c, sky, fog);
}

fn worldRay(uv: vec2f) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let v = cam.invProj * vec4f(ndc, 0.5, 1.0);
  let dirView = normalize(v.xyz / v.w);
  return normalize((cam.invView * vec4f(dirView, 0.0)).xyz);
}

fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let v = cam.invProj * vec4f(ndc, 0.5, 1.0);
  let dir = v.xyz / v.w;
  return dir * (depth / -dir.z);
}

// Blue → cyan → white ramp for speed visualisation.
fn speedRamp(s: f32) -> vec3f {
  let t = clamp(s, 0.0, 1.0);
  return mix(mix(vec3f(0.05, 0.2, 0.75), vec3f(0.1, 0.75, 0.95), smoothstep(0.0, 0.5, t)), vec3f(1.0, 1.0, 1.0), smoothstep(0.5, 1.0, t));
}

struct FullscreenOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

fn fullscreen(vi: u32) -> FullscreenOut {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: FullscreenOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

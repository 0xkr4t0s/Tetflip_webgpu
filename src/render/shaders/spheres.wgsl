// Particles drawn as ray-cast sphere impostors (instanced camera-facing quads).
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read> particleVel: array<vec4f>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) corner: vec2f,
  @location(1) center: vec3f,
  @location(2) speed: f32,
}

const CORNERS = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1), vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let corner = CORNERS[vi];
  let radius = cam.params.x;
  let center = (cam.view * vec4f(particlePos[ii].xyz, 1.0)).xyz;
  var o: VOut;
  o.pos = cam.proj * vec4f(center + vec3f(corner * radius, 0.0), 1.0);
  o.corner = corner;
  o.center = center;
  o.speed = length(particleVel[ii].xyz);
  return o;
}

struct Surface { normal: vec3f, pos: vec3f, depth: f32 }

fn sphereSurface(in: VOut) -> Surface {
  let r2 = dot(in.corner, in.corner);
  if (r2 > 1.0) { discard; }
  let n = vec3f(in.corner, sqrt(1.0 - r2));
  let p = in.center + n * cam.params.x;
  let clip = cam.proj * vec4f(p, 1.0);
  return Surface(n, p, clip.z / clip.w);
}

struct DepthOut {
  @location(0) data: vec4f, // linear depth, speed
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fsDepth(in: VOut) -> DepthOut {
  let s = sphereSurface(in);
  return DepthOut(vec4f(-s.pos.z, in.speed, 0.0, 1.0), s.depth);
}

@fragment
fn fsThickness(in: VOut) -> @location(0) vec4f {
  let r2 = dot(in.corner, in.corner);
  if (r2 > 1.0) { discard; }
  return vec4f(2.0 * sqrt(1.0 - r2) * cam.params.x, 0.0, 0.0, 1.0);
}

struct ShadedOut {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@fragment
fn fsShaded(in: VOut) -> ShadedOut {
  let s = sphereSurface(in);
  let lightView = normalize((cam.view * vec4f(cam.lightDir, 0.0)).xyz);
  let base = speedRamp(in.speed * cam.params.y);
  let diffuse = max(dot(s.normal, lightView), 0.0);
  let h = normalize(lightView + vec3f(0.0, 0.0, 1.0));
  let spec = pow(max(dot(s.normal, h), 0.0), 60.0) * 0.35;
  let c = base * (0.3 + 0.8 * diffuse) + vec3f(spec);
  return ShadedOut(vec4f(toDisplay(c), 1.0), s.depth);
}

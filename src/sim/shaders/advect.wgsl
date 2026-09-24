// Midpoint (RK2) advection through the projected velocity field, wall clamping and
// re-location of each particle's tetrahedron.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> particleVel: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> particleTet: array<u32>;
@group(0) @binding(4) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(5) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(6) var<storage, read> tetNeighbors: array<vec4i>;
@group(0) @binding(7) var<storage, read> tetVel: array<vec4f>;
@group(0) @binding(8) var<storage, read> nodeVel: array<vec4f>;

fn velocityAt(t: u32, p: vec3f) -> vec3f {
  let st = subTet(clampBary(baryAt(t, p)));
  var w = st.w;
  var v = st.wc * tetVel[t].xyz;
  var nodes = tetNodes[t];
  for (var a = 0u; a < 4u; a++) { v += w[a] * nodeVel[2u * nodes[a]].xyz; }
  return v;
}

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let margin = sim.spacing * 0.02;
  let lo = sim.boundsMin + vec3f(margin);
  let hi = sim.boundsMax - vec3f(margin);
  let p = particlePos[i];
  let t = particleTet[i];
  let v1 = velocityAt(t, p.xyz);
  let mid = clamp(p.xyz + 0.5 * sim.dt * v1, lo, hi);
  let tm = locate(mid, t);
  let x = p.xyz + sim.dt * velocityAt(tm, mid);
  let xc = clamp(x, lo, hi);
  // Particles pushed into a wall lose the velocity component that points into it.
  var v = particleVel[i].xyz;
  v = select(v, max(v, vec3f(0.0)), x < lo);
  v = select(v, min(v, vec3f(0.0)), x > hi);
  particlePos[i] = vec4f(xc, p.w);
  particleVel[i] = vec4f(v, particleVel[i].w);
  particleTet[i] = locate(xc, tm);
}

// Mesh → particle: FLIP/PIC blend using the subdivided-tet interpolation.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> particleVel: array<vec4f>;
@group(0) @binding(3) var<storage, read> particleTet: array<u32>;
@group(0) @binding(4) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(5) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(6) var<storage, read> tetVel: array<vec4f>;
@group(0) @binding(7) var<storage, read> tetVelOld: array<vec4f>;
@group(0) @binding(8) var<storage, read> nodeVel: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let t = particleTet[i];
  let st = subTet(clampBary(baryAt(t, particlePos[i].xyz)));
  var w = st.w;
  let u = tetVel[t].xyz;
  var pic = st.wc * u;
  var delta = st.wc * (u - tetVelOld[t].xyz);
  var nodes = tetNodes[t];
  for (var a = 0u; a < 4u; a++) {
    if (w[a] == 0.0) { continue; }
    pic += w[a] * nodeVel[2u * nodes[a]].xyz;
    delta += w[a] * nodeVel[2u * nodes[a] + 1u].xyz;
  }
  let v = particleVel[i];
  particleVel[i] = vec4f(mix(pic, v.xyz + delta, sim.flipRatio), v.w);
}

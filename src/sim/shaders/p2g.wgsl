// Particle → mesh splat: the adjoint of the subdivided-tet interpolation, plus the
// bary-weighted particle offsets used for the Zhu–Bridson level set.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read> particleVel: array<vec4f>;
@group(0) @binding(3) var<storage, read> particleTet: array<u32>;
@group(0) @binding(4) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(5) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(6) var<storage, read> nodePos: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> tetAcc: array<atomic<i32>>;
@group(0) @binding(8) var<storage, read_write> nodeAcc: array<atomic<i32>>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let p = particlePos[i].xyz;
  let v = clamp(particleVel[i].xyz, vec3f(-64.0), vec3f(64.0));
  let t = particleTet[i];
  var b = clampBary(baryAt(t, p));
  let st = subTet(b);
  var w = st.w;

  if (st.wc > 0.0) {
    atomicAdd(&tetAcc[t * 4u], toFixed(st.wc * v.x));
    atomicAdd(&tetAcc[t * 4u + 1u], toFixed(st.wc * v.y));
    atomicAdd(&tetAcc[t * 4u + 2u], toFixed(st.wc * v.z));
    atomicAdd(&tetAcc[t * 4u + 3u], toFixed(st.wc));
  }
  var nodes = tetNodes[t];
  for (var a = 0u; a < 4u; a++) {
    let n = nodes[a];
    let o = n * 8u;
    if (w[a] > 0.0) {
      atomicAdd(&nodeAcc[o], toFixed(w[a] * v.x));
      atomicAdd(&nodeAcc[o + 1u], toFixed(w[a] * v.y));
      atomicAdd(&nodeAcc[o + 2u], toFixed(w[a] * v.z));
      atomicAdd(&nodeAcc[o + 3u], toFixed(w[a]));
    }
    if (b[a] > 0.0) {
      let rel = b[a] * (p - nodePos[n].xyz) / sim.spacing;
      atomicAdd(&nodeAcc[o + 4u], toFine(rel.x));
      atomicAdd(&nodeAcc[o + 5u], toFine(rel.y));
      atomicAdd(&nodeAcc[o + 6u], toFine(rel.z));
      atomicAdd(&nodeAcc[o + 7u], toFine(b[a]));
    }
  }
}

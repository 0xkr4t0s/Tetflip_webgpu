// u_t ← u_t - [∇]_t p̂, with ghost pressures for the tet's air nodes. Marks tets touching the
// liquid as valid (w = 1) so the following extrapolation overwrites pure-air tets.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(2) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(3) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(4) var<storage, read> pressure: array<f32>;
@group(0) @binding(5) var<storage, read_write> tetVel: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= sim.tetCount) { return; }
  let nodes = tetNodes[t];
  var phi = vec4f(nodeState[nodes.x].x, nodeState[nodes.y].x, nodeState[nodes.z].x, nodeState[nodes.w].x);
  if (all(phi >= vec4f(0.0))) {
    tetVel[t].w = 0.0;
    return;
  }
  let pl = vec4f(pressure[nodes.x], pressure[nodes.y], pressure[nodes.z], pressure[nodes.w]);
  var P = loadPlanes(t);
  let V = tetVolume(t);
  var grad = vec3f(0.0);
  for (var a = 0u; a < 4u; a++) {
    var pa = pl[a];
    if (phi[a] >= 0.0) { pa = dot(ghostWeights(P, V, phi, a), pl); }
    grad += P[a].xyz * pa;
  }
  let u = tetVel[t];
  tetVel[t] = vec4f(u.xyz - grad, 1.0);
}

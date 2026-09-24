// Per tet: its own splat plus its volume share of each node's splat, normalised.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> tetAcc: array<i32>;
@group(0) @binding(2) var<storage, read> nodeAcc: array<i32>;
@group(0) @binding(3) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(4) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(5) var<storage, read> nodePos: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> tetVel: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= sim.tetCount) { return; }
  var acc = vec3f(fromFixed(tetAcc[t * 4u]), fromFixed(tetAcc[t * 4u + 1u]), fromFixed(tetAcc[t * 4u + 2u]));
  var wsum = fromFixed(tetAcc[t * 4u + 3u]);
  let V = tetVolume(t);
  var nodes = tetNodes[t];
  for (var a = 0u; a < 4u; a++) {
    let n = nodes[a];
    let share = V / nodePos[n].w;
    let o = n * 8u;
    acc += share * vec3f(fromFixed(nodeAcc[o]), fromFixed(nodeAcc[o + 1u]), fromFixed(nodeAcc[o + 2u]));
    wsum += share * fromFixed(nodeAcc[o + 3u]);
  }
  tetVel[t] = select(vec4f(0.0), vec4f(acc / wsum, 1.0), wsum > 1e-4);
}

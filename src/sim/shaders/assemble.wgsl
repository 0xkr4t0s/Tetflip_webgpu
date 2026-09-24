// Per node: one row of A = [∇]ᵀV[∇] with ghost-fluid terms, and b = [∇]ᵀV u (+ the volume
// correction source). Air nodes get identity rows. Mirrors ReferenceSolver.assemble.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> nodeOffsets: array<vec2u>; // (first incident tet entry, row start)
@group(0) @binding(2) var<storage, read> nodeTetInfo: array<vec2u>; // (tet * 4 + local, packed column slots)
@group(0) @binding(3) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(4) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(5) var<storage, read> tetVel: array<vec4f>;
@group(0) @binding(6) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(8) var<storage, read_write> rhs: array<f32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.nodeCount) { return; }
  let row = nodeOffsets[i].y;
  let rowLen = nodeOffsets[i + 1u].y - row;
  let state = nodeState[i];
  if (state.x >= 0.0) {
    matrix[row] = 1.0;
    for (var e = 1u; e < rowLen; e++) { matrix[row + e] = 0.0; }
    rhs[i] = 0.0;
    return;
  }
  var vals: array<f32, MAX_ROW>;
  var b = 0.0;
  for (var s = nodeOffsets[i].x; s < nodeOffsets[i + 1u].x; s++) {
    let info = nodeTetInfo[s];
    let t = info.x >> 2u;
    let li = info.x & 3u;
    let slots = info.y;
    var P = loadPlanes(t);
    let V = tetVolume(t);
    let nodes = tetNodes[t];
    var phi = vec4f(nodeState[nodes.x].x, nodeState[nodes.y].x, nodeState[nodes.z].x, nodeState[nodes.w].x);
    let gi = P[li].xyz;
    b += V * dot(gi, tetVel[t].xyz);
    for (var j = 0u; j < 4u; j++) {
      let kij = V * dot(gi, P[j].xyz);
      if (phi[j] < 0.0) {
        vals[(slots >> (8u * j)) & 0xffu] += kij;
        continue;
      }
      if (kij >= -1e-5 * V * dot(P[j].xyz, P[j].xyz)) { continue; }
      var w = ghostWeights(P, V, phi, j);
      for (var l = 0u; l < 4u; l++) {
        if (w[l] != 0.0) { vals[(slots >> (8u * l)) & 0xffu] += kij * w[l]; }
      }
    }
  }
  let excess = max(state.y - 1.0 - sim.densityDeadzone, 0.0);
  b += sim.volumeCorrection * excess * state.z * 0.25 / sim.dt;
  for (var e = 0u; e < rowLen; e++) { matrix[row + e] = vals[e]; }
  rhs[i] = b;
}

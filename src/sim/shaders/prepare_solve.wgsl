// Jacobi preconditioner and warm start: air nodes are pinned to zero pressure.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> nodeOffsets: array<vec2u>;
@group(0) @binding(2) var<storage, read> matrix: array<f32>;
@group(0) @binding(3) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> invDiag: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure: array<f32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.nodeCount) { return; }
  invDiag[i] = 1.0 / matrix[nodeOffsets[i].y];
  if (nodeState[i].x >= 0.0) { pressure[i] = 0.0; }
}

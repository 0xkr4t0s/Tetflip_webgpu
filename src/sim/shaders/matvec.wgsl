// out = A · vin, plus per-workgroup partial sums of vin · out.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> nodeOffsets: array<vec2u>;
@group(0) @binding(2) var<storage, read> rowColumns: array<u32>;
@group(0) @binding(3) var<storage, read> matrix: array<f32>;
@group(0) @binding(4) var<storage, read> vin: array<f32>;
@group(0) @binding(5) var<storage, read_write> vout: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  var d = 0.0;
  if (i < sim.nodeCount) {
    var s = 0.0;
    for (var e = nodeOffsets[i].y; e < nodeOffsets[i + 1u].y; e++) { s += matrix[e] * vin[rowColumns[e]]; }
    vout[i] = s;
    d = vin[i] * s;
  }
  scratch[lid] = d;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partials[wid.x] = scratch[0]; }
}

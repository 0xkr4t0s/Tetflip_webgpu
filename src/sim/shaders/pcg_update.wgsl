// x += α p, r -= α q, partial sums of r · M⁻¹ r.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> scalars: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<f32>;
@group(0) @binding(3) var<storage, read> q: array<f32>;
@group(0) @binding(4) var<storage, read> invDiag: array<f32>;
@group(0) @binding(5) var<storage, read_write> x: array<f32>;
@group(0) @binding(6) var<storage, read_write> r: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<f32>;

var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  var d = 0.0;
  if (i < sim.nodeCount) {
    let alpha = scalars[1];
    x[i] += alpha * p[i];
    let ri = r[i] - alpha * q[i];
    r[i] = ri;
    d = ri * ri * invDiag[i];
  }
  scratch[lid] = d;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partials[wid.x] = scratch[0]; }
}

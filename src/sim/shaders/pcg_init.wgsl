// r = b - A x, p = M⁻¹ r, partial sums of r · M⁻¹ r.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> rhs: array<f32>;
@group(0) @binding(2) var<storage, read> ax: array<f32>;
@group(0) @binding(3) var<storage, read> invDiag: array<f32>;
@group(0) @binding(4) var<storage, read_write> r: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let i = gid.x;
  var d = 0.0;
  if (i < sim.nodeCount) {
    let ri = rhs[i] - ax[i];
    let zi = ri * invDiag[i];
    r[i] = ri;
    p[i] = zi;
    d = ri * zi;
  }
  scratch[lid] = d;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
    workgroupBarrier();
  }
  if (lid == 0u) { partials[wid.x] = scratch[0]; }
}

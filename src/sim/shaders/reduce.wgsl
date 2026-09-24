// Sums the workgroup partials and updates the CG scalars. scalars = [rz, alpha, beta, pq, rz0].
override MODE: u32 = 0u; // 0: initial rz, 1: alpha = rz / pq, 2: beta = rz_new / rz

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> scalars: array<f32>;

const RG: u32 = 256u;
var<workgroup> scratch: array<f32, RG>;

@compute @workgroup_size(RG)
fn main(@builtin(local_invocation_index) lid: u32) {
  var s = 0.0;
  for (var k = lid; k < sim.numPartials; k += RG) { s += partials[k]; }
  scratch[lid] = s;
  workgroupBarrier();
  for (var stride = RG / 2u; stride > 0u; stride >>= 1u) {
    if (lid < stride) { scratch[lid] += scratch[lid + stride]; }
    workgroupBarrier();
  }
  if (lid != 0u) { return; }
  let total = scratch[0];
  if (MODE == 0u) {
    scalars[0] = total;
    scalars[4] = total;
  } else if (MODE == 1u) {
    scalars[3] = total;
    scalars[1] = select(0.0, scalars[0] / total, total > 0.0);
  } else {
    scalars[2] = select(0.0, total / scalars[0], scalars[0] > 0.0);
    scalars[0] = total;
  }
}

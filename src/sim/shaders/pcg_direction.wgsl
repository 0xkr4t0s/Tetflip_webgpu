// p = M⁻¹ r + β p.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> scalars: array<f32>;
@group(0) @binding(2) var<storage, read> r: array<f32>;
@group(0) @binding(3) var<storage, read> invDiag: array<f32>;
@group(0) @binding(4) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.nodeCount) { return; }
  p[i] = r[i] * invDiag[i] + scalars[2] * p[i];
}

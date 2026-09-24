// Extends velocities into tets without particle data by averaging valid face neighbours.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(3) var<storage, read> tetNeighbors: array<vec4i>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= sim.tetCount) { return; }
  let u = src[t];
  if (u.w > 0.0) {
    dst[t] = u;
    return;
  }
  var nb = tetNeighbors[t];
  var acc = vec3f(0.0);
  var count = 0.0;
  for (var k = 0u; k < 4u; k++) {
    if (nb[k] < 0) { continue; }
    let v = src[u32(nb[k])];
    if (v.w > 0.0) {
      acc += v.xyz;
      count += 1.0;
    }
  }
  dst[t] = select(u, vec4f(acc / count, 1.0), count > 0.0);
}

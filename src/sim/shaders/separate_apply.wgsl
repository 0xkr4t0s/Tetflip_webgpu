// Commits the corrected positions and re-locates each particle's tetrahedron.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> posTmp: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> particlePos: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> particleTet: array<u32>;
@group(0) @binding(4) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(5) var<storage, read> tetNeighbors: array<vec4i>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let x = posTmp[i].xyz;
  particlePos[i] = vec4f(x, particlePos[i].w);
  particleTet[i] = locate(x, particleTet[i]);
}

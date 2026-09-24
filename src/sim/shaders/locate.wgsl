// Initial point location from the seed grid (used after (re)seeding particles).
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> particleTet: array<u32>;
@group(0) @binding(3) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(4) var<storage, read> tetNeighbors: array<vec4i>;
@group(0) @binding(5) var<storage, read> locSeeds: array<u32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let p = particlePos[i].xyz;
  let c = clamp(vec3i(floor((p - sim.locOrigin) / sim.locCellSize)), vec3i(0), vec3i(sim.locDims) - vec3i(1));
  let cell = u32(c.x) + sim.locDims.x * (u32(c.y) + sim.locDims.y * u32(c.z));
  particleTet[i] = locate(p, locSeeds[cell]);
}

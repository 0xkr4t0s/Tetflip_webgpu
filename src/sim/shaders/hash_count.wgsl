// Counting sort, pass 1: per-cell particle counts and each particle's rank within its cell.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particleRank: array<u32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  particleRank[i] = atomicAdd(&cellCounts[hashIndex(hashCell(particlePos[i].xyz))], 1u);
}

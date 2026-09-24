// Counting sort, pass 3: particle indices grouped by cell (cellStart is the exclusive scan).
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read> particleRank: array<u32>;
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<storage, read_write> sortedIndex: array<u32>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  sortedIndex[cellStart[hashIndex(hashCell(particlePos[i].xyz))] + particleRank[i]] = i;
}

// Particle position correction (Ando et al. 2012, TETFLIP §3): push apart particles closer than
// the rest spacing, with walls as mirrors; tangential only near the surface. Mirrors
// separateParticles() in reference.ts. Writes candidate positions to posTmp.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> particlePos: array<vec4f>;
@group(0) @binding(2) var<storage, read> particleTet: array<u32>;
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIndex: array<u32>;
@group(0) @binding(5) var<storage, read> tetNodes: array<vec4u>;
@group(0) @binding(6) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(7) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(8) var<storage, read_write> posTmp: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.particleCount) { return; }
  let xi = particlePos[i].xyz;
  let R = sim.separationRadius;
  var push = vec3f(0.0);
  let c = hashCell(xi);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cc = c + vec3i(dx, dy, dz);
        if (any(cc < vec3i(0)) || any(cc >= vec3i(sim.hashDims))) { continue; }
        let cell = hashIndex(cc);
        for (var k = cellStart[cell]; k < cellStart[cell + 1u]; k++) {
          let j = sortedIndex[k];
          if (j == i) { continue; }
          let d = xi - particlePos[j].xyz;
          let dist = length(d);
          if (dist >= R || dist < 1e-6 * R) { continue; }
          push += d / dist * (1.0 - dist / R);
        }
      }
    }
  }
  let dl = xi - sim.boundsMin;
  let dh = sim.boundsMax - xi;
  push += select(vec3f(0.0), vec3f(1.0) - 2.0 * dl / R, dl < vec3f(R * 0.5));
  push -= select(vec3f(0.0), vec3f(1.0) - 2.0 * dh / R, dh < vec3f(R * 0.5));
  var delta = push * sim.separationStep;

  // Tangential only near the surface (linear level set of the particle's tet).
  let t = particleTet[i];
  let nodes = tetNodes[t];
  let phiN = vec4f(nodeState[nodes.x].x, nodeState[nodes.y].x, nodeState[nodes.z].x, nodeState[nodes.w].x);
  var P = loadPlanes(t);
  let b = max(baryFrom(P, xi), vec4f(0.0));
  let phi = dot(b, phiN);
  let grad = P[0].xyz * phiN.x + P[1].xyz * phiN.y + P[2].xyz * phiN.z + P[3].xyz * phiN.w;
  let g2 = dot(grad, grad);
  if (phi > -sim.separationBand && g2 > 1e-12) { delta -= dot(delta, grad) / g2 * grad; }

  let len = length(delta);
  if (len > sim.separationMaxStep) { delta *= sim.separationMaxStep / len; }
  let margin = sim.spacing * 0.02;
  posTmp[i] = vec4f(clamp(xi + delta, sim.boundsMin + vec3f(margin), sim.boundsMax - vec3f(margin)), 0.0);
}

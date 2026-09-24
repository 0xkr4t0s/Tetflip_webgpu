// Per node: Zhu–Bridson level set φ = |x - x̄| - r and relative particle density.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> nodeAcc: array<i32>;
@group(0) @binding(2) var<storage, read> nodePos: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> nodeState: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.nodeCount) { return; }
  let o = i * 8u;
  let x = nodePos[i];
  let wsum = fromFine(nodeAcc[o + 7u]);
  // Σ_p σ_n(x_p) ≈ density · ∫σ_n = density · V_n / 4.
  let rest = sim.restDensity * x.w * 0.25;
  let density = select(1.0, wsum / rest, rest > 0.0);
  var phi = sim.spacing;
  if (wsum >= 1e-4) {
    var off = vec3f(fromFine(nodeAcc[o + 4u]), fromFine(nodeAcc[o + 5u]), fromFine(nodeAcc[o + 6u])) * sim.spacing / wsum;
    // Wall nodes only see particles on one side; mirroring them cancels the normal offset.
    let eps = 1e-4 * sim.spacing;
    let onWall = (abs(x.xyz - sim.boundsMin) < vec3f(eps)) | (abs(x.xyz - sim.boundsMax) < vec3f(eps));
    off = select(off, vec3f(0.0), onWall);
    phi = length(off) - sim.surfaceRadius;
  }
  nodeState[i] = vec4f(phi, density, x.w, 0.0);
}

// Shared declarations for every simulation kernel. Mirrors `SimUniforms` in GpuSolver.ts.
struct Sim {
  boundsMin: vec3f, spacing: f32,
  boundsMax: vec3f, dt: f32,
  gravity: vec3f, flipRatio: f32,
  locOrigin: vec3f, locCellSize: f32,
  locDims: vec3u, particleCount: u32,
  nodeCount: u32, tetCount: u32, surfaceRadius: f32, ghostClamp: f32,
  volumeCorrection: f32, restDensity: f32, densityDeadzone: f32, numPartials: u32,
  forcePos: vec3f, forceRadius: f32,
  forceVel: vec3f, forceStrength: f32,
}

const WG: u32 = 128u;
const MAX_ROW: u32 = 32u;
// WebGPU has no float atomics, so particle-to-mesh splats accumulate in fixed point:
// velocities in 16.16 (range for fast flows), level-set sums in 12.20 (they stay O(10) but
// low-weight air nodes need the precision for accurate ghost pressures).
const FIXED_SCALE: f32 = 65536.0;
const FINE_SCALE: f32 = 1048576.0;

fn toFixed(v: f32) -> i32 { return i32(round(v * FIXED_SCALE)); }
fn fromFixed(v: i32) -> f32 { return f32(v) / FIXED_SCALE; }
fn toFine(v: f32) -> i32 { return i32(round(v * FINE_SCALE)); }
fn fromFine(v: i32) -> f32 { return f32(v) / FINE_SCALE; }

// σ_a(p) = ∇σ_a · p + c_a, with the planes stored as (∇σ_a, c_a).
fn baryFrom(P: array<vec4f, 4>, p: vec3f) -> vec4f {
  return vec4f(
    dot(P[0].xyz, p) + P[0].w,
    dot(P[1].xyz, p) + P[1].w,
    dot(P[2].xyz, p) + P[2].w,
    dot(P[3].xyz, p) + P[3].w,
  );
}

fn clampBary(b: vec4f) -> vec4f {
  let c = max(b, vec4f(0.0));
  return c / max(c.x + c.y + c.z + c.w, 1e-12);
}

// Weights on the tetrahedron subdivided around its centre: the point lies in the sub-tet
// formed by the centre and the face opposite the smallest barycentric coordinate m.
// Centre weight is 4m, node weights are σ_a - m (zero for the opposite vertex).
struct SubTet { wc: f32, w: vec4f }

fn subTet(b: vec4f) -> SubTet {
  let m = min(min(b.x, b.y), min(b.z, b.w));
  return SubTet(4.0 * m, b - vec4f(m));
}

// Symmetric ghost-fluid weights for air node g (paper Eqs. 6–12): p_G = Σ_l w_l p_l over the
// liquid nodes l, with θ_l = K_lG / Σ K_lG. Couplings within rounding of zero count as zero
// (BCC tets have right dihedral angles); a positive coupling means a badly shaped tet and
// falls back to p_G = 0. Must match ReferenceSolver.ghostWeights.
fn ghostWeights(Pin: array<vec4f, 4>, V: f32, phiIn: vec4f, g: u32) -> vec4f {
  var P = Pin;
  var phi = phiIn;
  var w = vec4f(0.0);
  let gradG = P[g].xyz;
  let tol = 1e-5 * V * dot(gradG, gradG);
  var S = 0.0;
  var liquidCount = 0.0;
  for (var l = 0u; l < 4u; l++) {
    if (l == g || phi[l] >= 0.0) { continue; }
    let k = V * dot(P[l].xyz, gradG);
    if (k > tol) { return vec4f(0.0); }
    if (k < -tol) { w[l] = k; }
    S += w[l];
    liquidCount += 1.0;
  }
  if (liquidCount == 0.0) { return vec4f(0.0); }
  var phiL = 0.0;
  for (var l = 0u; l < 4u; l++) {
    if (l == g || phi[l] >= 0.0) { continue; }
    w[l] = select(1.0 / liquidCount, w[l] / S, S < -tol);
    phiL += w[l] * phi[l];
  }
  let ratio = max(phi[g] / min(phiL, -1e-6 * sim.spacing), -sim.ghostClamp);
  return w * ratio;
}

/**
 * CPU reference implementation of one TETFLIP step.
 *
 * Every phase here has a WGSL counterpart in `src/sim/shaders/` and follows it one to one, so
 * the GPU solver can be checked against this code (see `scripts/gpu-check.mjs`) and the
 * numerics can be unit tested without a GPU. It favours clarity over speed.
 *
 * Discretization (Ando, Thuerey & Wojtan 2013, Section 3):
 *  - velocity: one 3-vector per tetrahedron (at its barycentre),
 *  - pressure: one scalar per node, linear inside each tetrahedron,
 *  - projection: minimise Σ_t V_t |u_t - [∇]_t q|², i.e. solve [∇]ᵀV[∇] q = [∇]ᵀV u
 *    (q = Δt p / ρ) and set u ← u - [∇] q,
 *  - free surface: symmetric ghost-fluid coefficients (Eq. 12),
 *  - particle transfers: barycentric interpolation on the tetrahedron subdivided around its
 *    centre (Section 3, "Velocity Interpolation"); the particle-to-mesh transfer is its adjoint.
 */
import type { PreparedMesh } from '../mesh/prepare';
import { barycentric, locate } from '../mesh/prepare';
import type { Vec3 } from '../math/vec3';
import { DENSITY_DEADZONE, type SimParams } from './params';

export interface ParticleState {
  count: number;
  /** xyzw per particle (w unused). */
  positions: Float32Array;
  /** xyzw per particle (w unused). */
  velocities: Float32Array;
  /** Containing tetrahedron of each particle. */
  tets: Uint32Array;
}

/** Barycentric coordinates clamped to the tetrahedron, as used by both transfers. */
function clampedBary(mesh: PreparedMesh, t: number, p: Vec3): number[] {
  const b = barycentric(mesh.tetPlanes, t, p).map((v) => Math.max(v, 0));
  const s = b[0] + b[1] + b[2] + b[3];
  return b.map((v) => v / s);
}

/**
 * Weights of the subdivided tetrahedron containing the point: the tet centre (weight wc) and
 * the three nodes of the face opposite local vertex k (weights w[j] for j ≠ k, w[k] = 0).
 */
export function subTetWeights(b: number[]): { k: number; wc: number; w: number[] } {
  let k = 0;
  for (let a = 1; a < 4; a++) if (b[a] < b[k]) k = a;
  const bk = b[k];
  return { k, wc: 4 * bk, w: b.map((v, a) => (a === k ? 0 : v - bk)) };
}

export class ReferenceSolver {
  readonly params: SimParams;
  readonly mesh: PreparedMesh;
  readonly particles: ParticleState;

  // Per-tet fields (vec3 as xyz triplets).
  readonly tetVel: Float64Array;
  readonly tetVelOld: Float64Array;
  readonly tetValid: Uint8Array;
  // Per-node fields.
  readonly nodePhi: Float64Array;
  /** Particle density at each node relative to the rest density (1 = as seeded). */
  readonly nodeDensity: Float64Array;
  readonly pressure: Float64Array;
  readonly nodeVel: Float64Array;
  readonly nodeDelta: Float64Array;
  // Pressure system (CSR on the static node adjacency).
  readonly matrix: Float64Array;
  readonly rhs: Float64Array;

  constructor(mesh: PreparedMesh, particles: ParticleState, params: SimParams) {
    this.mesh = mesh;
    this.particles = particles;
    this.params = params;
    this.tetVel = new Float64Array(mesh.tetCount * 3);
    this.tetVelOld = new Float64Array(mesh.tetCount * 3);
    this.tetValid = new Uint8Array(mesh.tetCount);
    this.nodePhi = new Float64Array(mesh.nodeCount);
    this.nodeDensity = new Float64Array(mesh.nodeCount);
    this.pressure = new Float64Array(mesh.nodeCount);
    this.nodeVel = new Float64Array(mesh.nodeCount * 3);
    this.nodeDelta = new Float64Array(mesh.nodeCount * 3);
    this.matrix = new Float64Array(mesh.rowColumns.length);
    this.rhs = new Float64Array(mesh.nodeCount);
  }

  static locateAll(mesh: PreparedMesh, particles: ParticleState): void {
    for (let i = 0; i < particles.count; i++) {
      const start = particles.tets[i] < mesh.tetCount ? particles.tets[i] : -1;
      particles.tets[i] = locate(mesh, particlePos(particles, i), start);
    }
  }

  step(dt: number): void {
    this.particleToMesh();
    this.extrapolate();
    this.applyForces(dt);
    this.assemble(dt);
    this.solvePressure();
    this.project();
    this.tetToNodes();
    this.meshToParticles();
    this.advect(dt);
  }

  /** Adjoint of the subdivided-tet interpolation, plus the Zhu–Bridson level set. */
  particleToMesh(): void {
    const { mesh, particles } = this;
    const n = mesh.nodeCount, m = mesh.tetCount, h = mesh.spacing;
    const tetAcc = new Float64Array(m * 4);
    const nodeAcc = new Float64Array(n * 4);
    const phiAcc = new Float64Array(n * 4);
    for (let i = 0; i < particles.count; i++) {
      const p = particlePos(particles, i);
      const v = [particles.velocities[i * 4], particles.velocities[i * 4 + 1], particles.velocities[i * 4 + 2]];
      const t = particles.tets[i];
      const b = clampedBary(mesh, t, p);
      const { wc, w } = subTetWeights(b);
      for (let c = 0; c < 3; c++) tetAcc[t * 4 + c] += wc * v[c];
      tetAcc[t * 4 + 3] += wc;
      for (let a = 0; a < 4; a++) {
        const node = mesh.tets[t * 4 + a];
        for (let c = 0; c < 3; c++) {
          nodeAcc[node * 4 + c] += w[a] * v[c];
          phiAcc[node * 4 + c] += b[a] * (p[c] - mesh.positions[node * 3 + c]);
        }
        nodeAcc[node * 4 + 3] += w[a];
        phiAcc[node * 4 + 3] += b[a];
      }
    }
    // Level set: distance to the weighted mean of nearby particles minus a radius. Nodes on a
    // wall see particles on one side only; mirroring them cancels the wall-normal offset.
    const radius = this.params.surfaceRadius * h;
    for (let node = 0; node < n; node++) {
      const wsum = phiAcc[node * 4 + 3];
      // Σ_p σ_n(x_p) ≈ density · ∫σ_n = density · V_n / 4.
      const rest = this.params.restDensity * mesh.nodeVolume[node] * 0.25;
      this.nodeDensity[node] = rest > 0 ? wsum / rest : 1;
      if (wsum < 1e-4) {
        this.nodePhi[node] = h;
        continue;
      }
      let d2 = 0;
      for (let c = 0; c < 3; c++) {
        const x = mesh.positions[node * 3 + c];
        const onWall = Math.abs(x - mesh.boundsMin[c]) < 1e-4 * h || Math.abs(x - mesh.boundsMax[c]) < 1e-4 * h;
        const off = onWall ? 0 : phiAcc[node * 4 + c] / wsum;
        d2 += off * off;
      }
      this.nodePhi[node] = Math.sqrt(d2) - radius;
    }
    // Gather: tet weight = its own splat + its share of each node's splat.
    for (let t = 0; t < m; t++) {
      let wsum = tetAcc[t * 4 + 3];
      const acc = [tetAcc[t * 4], tetAcc[t * 4 + 1], tetAcc[t * 4 + 2]];
      for (let a = 0; a < 4; a++) {
        const node = mesh.tets[t * 4 + a];
        const share = mesh.tetVolume[t] / mesh.nodeVolume[node];
        for (let c = 0; c < 3; c++) acc[c] += share * nodeAcc[node * 4 + c];
        wsum += share * nodeAcc[node * 4 + 3];
      }
      const valid = wsum > 1e-4;
      this.tetValid[t] = valid ? 1 : 0;
      for (let c = 0; c < 3; c++) this.tetVel[t * 3 + c] = valid ? acc[c] / wsum : 0;
    }
  }

  /** Extends velocities into empty tets by averaging valid face neighbours. */
  extrapolate(): void {
    const { mesh } = this;
    for (let pass = 0; pass < this.params.extrapolationPasses; pass++) {
      const valid = this.tetValid.slice();
      const vel = this.tetVel.slice();
      for (let t = 0; t < mesh.tetCount; t++) {
        if (valid[t]) continue;
        const acc = [0, 0, 0];
        let count = 0;
        for (let k = 0; k < 4; k++) {
          const nb = mesh.tetNeighbors[t * 4 + k];
          if (nb < 0 || !valid[nb]) continue;
          for (let c = 0; c < 3; c++) acc[c] += vel[nb * 3 + c];
          count++;
        }
        if (count === 0) continue;
        for (let c = 0; c < 3; c++) this.tetVel[t * 3 + c] = acc[c] / count;
        this.tetValid[t] = 1;
      }
    }
  }

  applyForces(dt: number): void {
    this.tetVelOld.set(this.tetVel);
    const g = this.params.gravity;
    for (let t = 0; t < this.mesh.tetCount; t++) for (let c = 0; c < 3; c++) this.tetVel[t * 3 + c] += g[c] * dt;
  }

  /** Local stiffness entry V_t ∇σ_a · ∇σ_b. */
  private K(t: number, a: number, b: number): number {
    const P = this.mesh.tetPlanes, o = t * 16;
    return this.mesh.tetVolume[t] * (P[o + a * 4] * P[o + b * 4] + P[o + a * 4 + 1] * P[o + b * 4 + 1] + P[o + a * 4 + 2] * P[o + b * 4 + 2]);
  }

  /**
   * Ghost-fluid weights for air node g of tet t: p_G = Σ_l w_l p_l over the tet's liquid nodes
   * (paper Eqs. 6–12). With θ_l = K_lG / Σ K_lG the embedded matrix stays symmetric. BCC
   * tetrahedra have right dihedral angles, so some K_lG are exactly zero; if every coupling is
   * zero the ghost value cannot affect the matrix and uniform θ is used for the velocity
   * update. A positive coupling marks a badly shaped tet, which falls back to p_G = 0.
   */
  ghostWeights(t: number, g: number): number[] {
    const { mesh, nodePhi } = this;
    const w = [0, 0, 0, 0];
    const tol = 1e-5 * this.K(t, g, g);
    let S = 0, liquidCount = 0;
    for (let l = 0; l < 4; l++) {
      if (l === g || nodePhi[mesh.tets[t * 4 + l]] >= 0) continue;
      const k = this.K(t, l, g);
      if (k > tol) return [0, 0, 0, 0];
      w[l] = k < -tol ? k : 0; // couplings within rounding of zero count as zero
      S += w[l];
      liquidCount++;
    }
    if (liquidCount === 0) return w;
    let phiL = 0;
    for (let l = 0; l < 4; l++) {
      if (l === g || nodePhi[mesh.tets[t * 4 + l]] >= 0) continue;
      w[l] = S < -tol ? w[l] / S : 1 / liquidCount; // θ_l
      phiL += w[l] * nodePhi[mesh.tets[t * 4 + l]];
    }
    const phiG = nodePhi[mesh.tets[t * 4 + g]];
    const ratio = Math.max(phiG / Math.min(phiL, -1e-6 * mesh.spacing), -this.params.ghostClamp);
    for (let l = 0; l < 4; l++) w[l] *= ratio;
    return w;
  }

  /**
   * Assembles A = [∇]ᵀV[∇] (with ghost-fluid terms) and b = [∇]ᵀV u over liquid nodes.
   * Because G ᵀV u = -∫σ_i ∇·u, adding s·V_n/4 to b makes the projected field reach divergence
   * s at node i; the volume correction uses that to relax over-packed regions.
   */
  assemble(dt = 1): void {
    const { mesh, nodePhi, matrix, rhs } = this;
    matrix.fill(0);
    rhs.fill(0);
    for (let i = 0; i < mesh.nodeCount; i++) {
      const row = mesh.rowOffsets[i];
      if (nodePhi[i] >= 0) {
        matrix[row] = 1; // air: identity row, q = 0
        continue;
      }
      const excess = Math.max(this.nodeDensity[i] - 1 - DENSITY_DEADZONE, 0);
      rhs[i] += (this.params.volumeCorrection * excess * mesh.nodeVolume[i] * 0.25) / dt;
      for (let s = mesh.nodeTetOffsets[i]; s < mesh.nodeTetOffsets[i + 1]; s++) {
        const entry = mesh.nodeTetEntries[s];
        const t = entry >> 2, li = entry & 3;
        const slots = mesh.nodeTetSlots[s];
        const V = mesh.tetVolume[t], o = t * 16 + li * 4;
        const P = mesh.tetPlanes;
        rhs[i] += V * (P[o] * this.tetVel[t * 3] + P[o + 1] * this.tetVel[t * 3 + 1] + P[o + 2] * this.tetVel[t * 3 + 2]);
        for (let j = 0; j < 4; j++) {
          const nj = mesh.tets[t * 4 + j];
          if (nodePhi[nj] < 0) {
            matrix[row + ((slots >>> (8 * j)) & 0xff)] += this.K(t, li, j);
            continue;
          }
          const kiG = this.K(t, li, j);
          if (kiG >= -1e-5 * this.K(t, j, j)) continue;
          const w = this.ghostWeights(t, j);
          for (let l = 0; l < 4; l++) if (w[l] !== 0) matrix[row + ((slots >>> (8 * l)) & 0xff)] += kiG * w[l];
        }
      }
    }
  }

  multiply(x: Float64Array, out: Float64Array): void {
    const { mesh, matrix } = this;
    for (let i = 0; i < mesh.nodeCount; i++) {
      let s = 0;
      for (let e = mesh.rowOffsets[i]; e < mesh.rowOffsets[i + 1]; e++) s += matrix[e] * x[mesh.rowColumns[e]];
      out[i] = s;
    }
  }

  /** Jacobi-preconditioned CG with a fixed iteration count, warm started from the last solve. */
  solvePressure(): number {
    const n = this.mesh.nodeCount;
    const x = this.pressure, b = this.rhs, A = this.matrix, rows = this.mesh.rowOffsets;
    for (let i = 0; i < n; i++) if (this.nodePhi[i] >= 0) x[i] = 0;
    const r = new Float64Array(n), z = new Float64Array(n), p = new Float64Array(n), q = new Float64Array(n);
    this.multiply(x, q);
    let rz = 0;
    for (let i = 0; i < n; i++) {
      r[i] = b[i] - q[i];
      z[i] = r[i] / A[rows[i]];
      p[i] = z[i];
      rz += r[i] * z[i];
    }
    for (let it = 0; it < this.params.pcgIterations; it++) {
      this.multiply(p, q);
      let pq = 0;
      for (let i = 0; i < n; i++) pq += p[i] * q[i];
      const alpha = pq > 0 ? rz / pq : 0;
      let rzNew = 0;
      for (let i = 0; i < n; i++) {
        x[i] += alpha * p[i];
        r[i] -= alpha * q[i];
        z[i] = r[i] / A[rows[i]];
        rzNew += r[i] * z[i];
      }
      const beta = rz > 0 ? rzNew / rz : 0;
      rz = rzNew;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    let rr = 0;
    for (let i = 0; i < n; i++) rr += r[i] * r[i];
    return Math.sqrt(rr);
  }

  /** u_t ← u_t - [∇]_t p̂ where p̂ holds liquid pressures and ghost values for air nodes. */
  project(): void {
    const { mesh, nodePhi, pressure } = this;
    const P = mesh.tetPlanes;
    for (let t = 0; t < mesh.tetCount; t++) {
      let anyLiquid = false;
      for (let a = 0; a < 4; a++) if (nodePhi[mesh.tets[t * 4 + a]] < 0) anyLiquid = true;
      if (!anyLiquid) continue;
      const ph = [0, 0, 0, 0];
      for (let a = 0; a < 4; a++) {
        const node = mesh.tets[t * 4 + a];
        if (nodePhi[node] < 0) {
          ph[a] = pressure[node];
          continue;
        }
        const w = this.ghostWeights(t, a);
        for (let l = 0; l < 4; l++) ph[a] += w[l] * pressure[mesh.tets[t * 4 + l]];
      }
      for (let a = 0; a < 4; a++)
        for (let c = 0; c < 3; c++) this.tetVel[t * 3 + c] -= P[t * 16 + a * 4 + c] * ph[a];
    }
  }

  /** Volume-weighted node averages of the new velocity and of its change during the step. */
  tetToNodes(): void {
    const { mesh } = this;
    for (let i = 0; i < mesh.nodeCount; i++) {
      const v = [0, 0, 0], d = [0, 0, 0];
      for (let s = mesh.nodeTetOffsets[i]; s < mesh.nodeTetOffsets[i + 1]; s++) {
        const t = mesh.nodeTetEntries[s] >> 2;
        const V = mesh.tetVolume[t];
        for (let c = 0; c < 3; c++) {
          v[c] += V * this.tetVel[t * 3 + c];
          d[c] += V * (this.tetVel[t * 3 + c] - this.tetVelOld[t * 3 + c]);
        }
      }
      for (let c = 0; c < 3; c++) {
        this.nodeVel[i * 3 + c] = v[c] / mesh.nodeVolume[i];
        this.nodeDelta[i * 3 + c] = d[c] / mesh.nodeVolume[i];
      }
    }
  }

  /** Interpolates the tet-centre field `tetField` with node averages `nodeField` at p in tet t. */
  interpolate(t: number, p: Vec3, tetField: Float64Array, nodeField: Float64Array, tetOld?: Float64Array): Vec3 {
    const { wc, w } = subTetWeights(clampedBary(this.mesh, t, p));
    const out: Vec3 = [0, 0, 0];
    for (let c = 0; c < 3; c++) {
      const tv = tetField[t * 3 + c] - (tetOld ? tetOld[t * 3 + c] : 0);
      out[c] = wc * tv;
      for (let a = 0; a < 4; a++) out[c] += w[a] * nodeField[this.mesh.tets[t * 4 + a] * 3 + c];
    }
    return out;
  }

  meshToParticles(): void {
    const { particles, params } = this;
    for (let i = 0; i < particles.count; i++) {
      const p = particlePos(particles, i);
      const t = particles.tets[i];
      const pic = this.interpolate(t, p, this.tetVel, this.nodeVel);
      const delta = this.interpolate(t, p, this.tetVel, this.nodeDelta, this.tetVelOld);
      for (let c = 0; c < 3; c++) {
        const flip = particles.velocities[i * 4 + c] + delta[c];
        particles.velocities[i * 4 + c] = params.flipRatio * flip + (1 - params.flipRatio) * pic[c];
      }
    }
  }

  /** Midpoint (RK2) advection through the projected mesh velocity, then wall clamping. */
  advect(dt: number): void {
    const { mesh, particles } = this;
    const lo = mesh.boundsMin, hi = mesh.boundsMax;
    const margin = mesh.spacing * 0.02;
    for (let i = 0; i < particles.count; i++) {
      const p = particlePos(particles, i);
      const t = particles.tets[i];
      const v1 = this.interpolate(t, p, this.tetVel, this.nodeVel);
      const mid: Vec3 = [0, 1, 2].map((c) => clamp(p[c] + 0.5 * dt * v1[c], lo[c] + margin, hi[c] - margin)) as Vec3;
      const tm = locate(mesh, mid, t);
      const v2 = this.interpolate(tm, mid, this.tetVel, this.nodeVel);
      for (let c = 0; c < 3; c++) {
        let x = p[c] + dt * v2[c];
        if (x < lo[c] + margin) {
          x = lo[c] + margin;
          particles.velocities[i * 4 + c] = Math.max(particles.velocities[i * 4 + c], 0);
        } else if (x > hi[c] - margin) {
          x = hi[c] - margin;
          particles.velocities[i * 4 + c] = Math.min(particles.velocities[i * 4 + c], 0);
        }
        particles.positions[i * 4 + c] = x;
      }
      particles.tets[i] = locate(mesh, particlePos(particles, i), tm);
    }
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function particlePos(particles: ParticleState, i: number): Vec3 {
  return [particles.positions[i * 4], particles.positions[i * 4 + 1], particles.positions[i * 4 + 2]];
}

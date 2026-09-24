import type { Vec3 } from '../math/vec3';

export interface SimParams {
  gravity: Vec3;
  /** 1 = pure FLIP, 0 = pure PIC. */
  flipRatio: number;
  /** Fixed number of Jacobi-preconditioned CG iterations per substep (warm started). */
  pcgIterations: number;
  /** Particle radius used for the Zhu–Bridson level set, in units of the lattice spacing. */
  surfaceRadius: number;
  /** Upper bound on |φ_G / φ̃_L| in the ghost-fluid coefficients (paper Eq. 12). */
  ghostClamp: number;
  /** Number of passes that extend tet velocities into tets that received no particle data. */
  extrapolationPasses: number;
  /**
   * Strength of the volume correction: where particles are packed denser than at seeding,
   * the projection targets a small positive divergence that spreads them out again
   * (a lightweight stand-in for the paper's particle position correction). 0 disables it.
   */
  volumeCorrection: number;
  /** Particles per unit volume at rest, set from the seeding density. */
  restDensity: number;
}

/** Relative over-density tolerated before the volume correction kicks in (filters sampling noise). */
export const DENSITY_DEADZONE = 0.2;

export const defaultParams = (): SimParams => ({
  gravity: [0, -9.81, 0],
  flipRatio: 0.97,
  pcgIterations: 60,
  surfaceRadius: 0.5,
  ghostClamp: 10,
  extrapolationPasses: 2,
  volumeCorrection: 1,
  restDensity: 0,
});

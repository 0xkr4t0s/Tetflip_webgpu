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
  /**
   * Rate (1/s) of the particle position correction of Ando et al. 2012, used by the paper to
   * stop FLIP particles from clustering: particles closer than the rest spacing are pushed
   * apart, tangentially only near the surface. 0 disables it.
   */
  separation: number;
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
  separation: 10,
});

/** Rest spacing between particles implied by the rest density. */
export const particleSpacing = (params: SimParams): number => Math.cbrt(1 / params.restDensity);

/** Neighbours closer than this (in units of the particle spacing) are pushed apart. */
export const SEPARATION_RADIUS = 1.0;
/** Largest position correction per step, in units of the particle spacing. */
export const SEPARATION_MAX_STEP = 0.25;
/** Within this distance of the surface (in particle spacings) corrections are tangential only. */
export const SEPARATION_SURFACE_BAND = 2.0;

# Algorithm notes

These notes cover the numerics implemented in `src/sim/reference.ts` (CPU, f64) and
`src/sim/shaders/*.wgsl` (GPU, f32). The two implementations mirror each other phase by phase,
and `npm run test:gpu` checks that they agree.

## Discretization (Ando et al. 2013, §3)

| Quantity | Location | Representation |
| --- | --- | --- |
| velocity `u_t` | tetrahedron barycentre | 3-vector, piecewise constant |
| pressure `p_i` | mesh node | scalar, piecewise linear |
| level set `φ_i` | mesh node | scalar |

A linear pressure field has a constant gradient on each tetrahedron:

```
[∇]_t p = Σ_a ∇σ_a p_a ,
```

where `σ_a` are the barycentric coordinates of tet `t`. Each tet stores its four barycentric
planes `(∇σ_a, c_a)` with `σ_a(x) = ∇σ_a·x + c_a`. The same planes are used for point location,
interpolation and the gradient operator.

### Projection

The projection finds the smallest change in kinetic energy that makes `u` divergence free:

```
q = argmin Σ_t V_t |u_t − [∇]_t q|²    (q = Δt p / ρ)
⇒  A q = b,   A = [∇]ᵀ V [∇],   b = [∇]ᵀ V u,   u ← u − [∇] q
```

`A` is the P1 finite-element stiffness matrix: `A_ij = Σ_t V_t ∇σ_i·∇σ_j`. Its sparsity is the
node adjacency, which is fixed for a static mesh, so the CSR pattern and, for each
(node, incident tet) pair, the column slots of the tet's four nodes are precomputed. Assembly on
the GPU then needs no searching and no atomics: each thread owns one row.

`b_i = Σ_t V_t ∇σ_i·u_t = −∫σ_i ∇·u + ∮σ_i u·n`. Driving it to zero gives both
incompressibility and, at the domain walls, zero normal flux. The box walls therefore need no
special handling in the solve.

### Free surface: ghost fluid (Eqs. 6–12)

Nodes with `φ ≥ 0` are air. For an air node `G` of a tet with liquid nodes `L`, the ghost
pressure is a linear combination of the liquid pressures:

```
p_G = Σ_l w_l p_l,     w_l = (φ_G / φ̃_L) θ_l,     θ_l = K_lG / Σ_k K_kG,     φ̃_L = Σ θ_l φ_l
```

`K` is the tet's local stiffness matrix. This choice of `θ` makes the embedded matrix
symmetric, since the liquid-row contribution `K_iG w_l ∝ K_iG K_lG` is symmetric in `i, l`, so
CG still applies. `|φ_G / φ̃_L|` is clamped by `ghostClamp`.

Two special cases:

- **Right dihedral angles.** BCC tetrahedra have 90° dihedral angles, so some `K_lG` are exactly
  zero, or zero up to rounding. Couplings within `1e-5 K_GG` of zero are treated as zero. If all
  of a ghost's couplings vanish, the ghost cannot affect the matrix, and uniform `θ` is used only
  for the velocity update.
- **Obtuse tets.** A positive coupling marks a poorly shaped tet, which falls back to first-order
  `p_G = 0` (the paper blends this more gradually, Eq. 13). The mesh used here has none (see
  below).

With exact linear `φ`, a pool at rest is reproduced **exactly**: the test
`keeps a flat pool exactly at rest` checks this to 1e-6.

### Velocity interpolation (§3, "Velocity Interpolation")

Tet-centre velocities are averaged to the nodes, weighted by volume. To interpolate at `x`, the
tet is subdivided virtually around its centre `c`. `x` lies in the sub-tet formed by `c` and
the face opposite the vertex with the smallest barycentric coordinate `m`, and there:

```
u(x) = 4m · u_t + Σ_a (σ_a − m) · ũ_a
```

This is C⁰ and reproduces the tet-centre sample exactly. **Particle → mesh** uses the transpose
of this operator, normalised by the transposed weights, so the two transfers are adjoint.

### FLIP update

`u_old` is the mesh velocity before forces and projection. Particles receive

```
v ← α (v + I(u − u_old)) + (1 − α) I(u),     α = flipRatio
```

and are advected with RK2 through `I(u)`, followed by jump-and-walk point location.

## Level set

Zhu–Bridson: `φ_i = |x_i − x̄_i| − r`, where `x̄_i` is the average of nearby particle positions
weighted by the linear hat function of node `i`, and `r = surfaceRadius · h`. On a wall node the
wall-normal component of `x_i − x̄_i` is zeroed, which is equivalent to mirroring the particles
across the wall. Without this, wall nodes would turn into free surface.

## Volume correction

FLIP particles slowly bunch up. Where the relative particle density at a liquid node exceeds
`1 + 0.2`, a small positive divergence `s = κ (ρ/ρ₀ − 1.2) / Δt` is requested by adding
`s V_i / 4` to `b_i`. This lightweight substitute for the paper's particle position correction
keeps the liquid's volume within a few percent over long runs.

## Mesh

The domain is tetrahedralized with a body-centred cubic (BCC) lattice:

- **Nodes**: cube corners, cube centres, and the centre of every boundary face.
- **Interior**: every face shared by two cubes yields four tets, each built from one edge of
  that face and the two cube centres. These are congruent tetrahedra whose dihedral angles are
  60° and 90°.
- **Boundary**: every boundary face yields four tets, each built from one face edge, the cube
  centre and the face-centre node. Each is exactly half a BCC tet.

No dihedral angle is obtuse, so every off-diagonal of `A` is ≤ 0. This makes `A` an M-matrix,
and the ghost-fluid coefficients are always valid.

Precomputed per mesh: tet planes, volumes and centroids, face neighbours, node→tet incidence
(CSR), node adjacency (CSR, the matrix pattern), column-slot maps, and a seed grid with
spacing `h/2` for jump-and-walk point location.

## GPU notes

- WebGPU has no float atomics, so the particle splat uses fixed-point `atomicAdd`: 16.16 for
  velocities and 12.20 for the level-set sums. Low-weight air nodes need the extra precision
  for accurate ghost pressures.
- The CG solve runs a fixed number of iterations (warm started from the previous step), with
  dot products reduced on the GPU, so a step never waits on the CPU. Residuals are read back
  asynchronously for the stats panel only.
- Each kernel declares its own bindings; `src/gpu/kernel.ts` parses them to build the bind group
  layout, so bind groups are created from a name → buffer map.

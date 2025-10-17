# TETFLIP Algorithm Documentation

## Introduction

TETFLIP is a hybrid Eulerian-Lagrangian method for fluid simulation that combines:
- **Tetrahedral meshes** for adaptive spatial discretization
- **FLIP (Fluid-Implicit-Particle)** method for advection
- **Pressure projection** for incompressibility

## Algorithm Overview

### Main Simulation Loop

Each simulation timestep follows these stages:

```
1. Particle to Grid (P2G) Transfer
2. Apply Body Forces
3. Solve Pressure Poisson Equation
4. Apply Pressure Gradient
5. Grid to Particle (G2P) Transfer
6. Advect Particles
7. Handle Collisions
8. (Optional) Adapt Mesh
```

## Detailed Algorithm Steps

### 1. Particle to Grid Transfer (P2G)

**Purpose**: Transfer particle velocities to mesh nodes

**Method**: Weighted average using barycentric coordinates

```
For each particle p:
    1. Find containing tetrahedron T
    2. Compute barycentric coordinates (w0, w1, w2, w3)
    3. For each node i in T:
        v_node[i] += w[i] * v_particle
        weight[i] += w[i]
        
For each node i:
    v_node[i] /= weight[i]  // Normalize
```

**Key Points**:
- Uses barycentric interpolation for smooth transfer
- Maintains momentum conservation
- Handles particles near mesh boundaries

### 2. Apply Body Forces

**Purpose**: Add external forces (gravity, wind, etc.)

**Method**: Explicit force integration

```
For each node i:
    v_node[i] += gravity * dt
```

**Key Points**:
- Can add any external force field
- Simple explicit Euler integration
- Can be extended for other forces

### 3. Pressure Solve

**Purpose**: Enforce incompressibility constraint (∇·v = 0)

**Method**: Solve Poisson equation ∇²p = ρ/dt · ∇·v

#### 3.1 Compute Divergence

```
For each tetrahedron T:
    div = approximate_divergence(T)
    For each node i in T:
        divergence[i] += div
```

#### 3.2 Solve for Pressure

Using Jacobi iteration (simplified):

```
For iteration = 1 to max_iterations:
    For each node i:
        p_new[i] = (sum(p_neighbors) - rhs[i]) / num_neighbors
    
    if converged:
        break
```

**Key Points**:
- Can use more advanced solvers (Conjugate Gradient, Multigrid)
- Boundary conditions are important
- Convergence tolerance affects accuracy

### 4. Apply Pressure Gradient

**Purpose**: Update velocities to be divergence-free

**Method**: v_new = v_old - dt * ∇p

```
For each node i:
    grad_p = compute_pressure_gradient(i)
    v_node[i] -= dt * grad_p
```

**Key Points**:
- Projects velocity onto divergence-free space
- Satisfies incompressibility
- Preserves tangential velocity components

### 5. Grid to Particle Transfer (G2P)

**Purpose**: Transfer updated velocities back to particles

**Method**: FLIP/PIC blend

#### Pure PIC (Particle-in-Cell)
```
v_particle = interpolate(v_grid)
```

#### Pure FLIP (Fluid-Implicit-Particle)
```
v_particle = v_particle + interpolate(v_grid_new - v_grid_old)
```

#### Blended FLIP/PIC
```
v_pic = interpolate(v_grid_new)
v_flip = v_particle + interpolate(v_grid_new - v_grid_old)
v_particle = (1 - α) * v_pic + α * v_flip
```

where α is the FLIP ratio (typically 0.95-0.99)

**Key Points**:
- FLIP reduces numerical dissipation
- PIC adds stability
- Blend ratio is tunable parameter

### 6. Advect Particles

**Purpose**: Move particles according to their velocities

**Method**: Explicit Euler integration

```
For each particle p:
    position[p] += velocity[p] * dt
```

**Key Points**:
- Can use higher-order integration (RK2, RK4)
- Timestep size affects stability
- CFL condition should be respected

### 7. Handle Collisions

**Purpose**: Enforce domain boundaries and solid obstacles

**Method**: Collision detection and response

```
For each particle p:
    if outside_domain(p):
        project_to_boundary(p)
        apply_friction_and_restitution(p)
```

**Key Points**:
- Coefficient of restitution controls bounciness
- Friction affects sliding behavior
- Can handle complex geometry

### 8. Mesh Adaptation (Advanced)

**Purpose**: Refine mesh in regions of interest, coarsen elsewhere

**Criteria**:
- High velocity gradients
- Surface proximity
- Curvature
- User-defined features

**Operations**:
- Edge split (refinement)
- Edge collapse (coarsening)
- Face swap (quality improvement)
- Node relocation (smoothing)

## Mathematical Foundations

### Navier-Stokes Equations

The incompressible Navier-Stokes equations:

```
∂v/∂t + (v·∇)v = -1/ρ ∇p + ν∇²v + g
∇·v = 0
```

Where:
- v: velocity field
- p: pressure
- ρ: density
- ν: kinematic viscosity
- g: body forces

### Operator Splitting

TETFLIP uses operator splitting to solve different terms separately:

1. **Advection**: ∂v/∂t + (v·∇)v = 0
2. **Body Forces**: ∂v/∂t = g
3. **Viscosity**: ∂v/∂t = ν∇²v
4. **Pressure**: ∂v/∂t = -1/ρ ∇p, subject to ∇·v = 0

### Barycentric Coordinates

For a point P inside tetrahedron with vertices V0, V1, V2, V3:

```
P = w0*V0 + w1*V1 + w2*V2 + w3*V3
```

where w0 + w1 + w2 + w3 = 1 and all wi ≥ 0

Computed using volume ratios:
```
w[i] = Volume(P, other three vertices) / Volume(tetrahedron)
```

### Pressure Poisson Equation

Taking divergence of pressure update:

```
∇·v_new = 0
∇·(v_old - dt/ρ ∇p) = 0
∇·v_old = dt/ρ ∇²p
∇²p = ρ/dt ∇·v_old
```

This is a Poisson equation for pressure p.

## Performance Considerations

### Timestep Selection

CFL condition:
```
dt ≤ CFL * h / |v_max|
```

where:
- h: minimum mesh spacing
- v_max: maximum velocity
- CFL: Courant number (typically 0.5-1.0)

### Spatial Resolution

- Finer mesh → better detail, slower simulation
- Coarser mesh → faster simulation, less detail
- Adaptive meshing balances both

### Solver Convergence

- More iterations → better accuracy, slower
- Better preconditioner → faster convergence
- Multigrid methods → O(n) complexity

## Implementation Tips

### Numerical Stability

1. **Clamp velocities** to prevent explosions
2. **Use implicit methods** for stiff terms (viscosity)
3. **Limit timestep** based on CFL condition
4. **Check for NaN/Inf** values

### Memory Management

1. **Pre-allocate buffers** to avoid reallocations
2. **Use typed arrays** for better performance
3. **Minimize copies** between CPU and GPU
4. **Batch operations** for efficiency

### GPU Acceleration

1. **Particle operations** are embarrassingly parallel
2. **Sparse linear solvers** benefit from GPU
3. **Mesh operations** need careful synchronization
4. **Use compute shaders** for physics

## References

1. Ando, R., Thuerey, N., & Wojtan, C. (2013). "A highly adaptive liquid simulator on tetrahedral meshes." *SIGGRAPH 2013*

2. Bridson, R. (2015). "Fluid Simulation for Computer Graphics" (2nd ed.)

3. Zhu, Y., & Bridson, R. (2005). "Animating sand as a fluid." *SIGGRAPH 2005*

4. Stam, J. (1999). "Stable fluids." *SIGGRAPH 1999*

5. Losasso, F., Talton, J., Kwatra, N., & Fedkiw, R. (2008). "Two-way coupled SPH and particle level set fluid simulation." *IEEE TVCG*

## Common Issues and Solutions

### Issue: Particles leak through boundaries
**Solution**: Use smaller timestep, improve collision detection, or add particle resampling

### Issue: Simulation becomes unstable
**Solution**: Reduce timestep, clamp velocities, check for degenerate mesh elements

### Issue: Pressure solver doesn't converge
**Solution**: Increase iterations, improve initial guess, use better preconditioner

### Issue: Fluid appears too viscous
**Solution**: Increase FLIP ratio, reduce artificial damping, check for numerical dissipation

### Issue: Performance is slow
**Solution**: Use GPU compute shaders, optimize mesh resolution, implement spatial acceleration structures

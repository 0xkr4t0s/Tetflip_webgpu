# TETFLIP WebGPU - Liquid Simulator

A real-time liquid simulation implementation using the TETFLIP technique from the paper ["A Highly Adaptive Liquid Simulator on Tetrahedral Meshes"](https://dl.acm.org/doi/10.1145/2461912.2461982) by Ryoichi Ando, Nils Thuerey, and Chris Wojtan (SIGGRAPH 2013), implemented in WebGPU.

## Overview

TETFLIP combines the strengths of:
- **Tetrahedral Meshes**: Adaptive unstructured mesh for spatial discretization
- **FLIP Method**: Fluid-Implicit-Particle method for advection with reduced numerical dissipation
- **Pressure Projection**: Enforces incompressibility constraint
- **Adaptive Refinement**: Dynamic mesh adaptation based on flow features (planned feature)

This implementation provides a fully interactive 3D liquid simulation that runs entirely in the browser using WebGPU for GPU-accelerated physics computation and rendering.

## Features

### Current Features
- ✅ Tetrahedral mesh generation and management
- ✅ FLIP particle advection system
- ✅ Pressure projection solver for incompressibility
- ✅ Interactive 3D visualization with WebGPU
- ✅ Real-time parameter adjustment
- ✅ Dam break scenario demonstration
- ✅ Particle and mesh wireframe rendering
- ✅ Camera controls (orbit and zoom)

### Planned Features
- 🔲 GPU-accelerated compute shaders for physics
- 🔲 Adaptive mesh refinement based on flow features
- 🔲 Surface reconstruction and rendering
- 🔲 Multiple simulation scenarios
- 🔲 Viscosity effects
- 🔲 Two-way rigid body coupling
- 🔲 Performance optimizations

## Prerequisites

To run this simulator, you need:
- A modern web browser with WebGPU support:
  - Chrome/Edge 113+ (with WebGPU enabled)
  - Firefox Nightly (with WebGPU enabled)
  - Safari Technology Preview (with WebGPU enabled)
- A GPU that supports WebGPU

## Installation & Usage

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/0xkr4t0s/Tetflip_webgpu.git
cd Tetflip_webgpu
```

2. Start a local web server (WebGPU requires a secure context):
```bash
# Using Python 3
python -m http.server 8000

# Or using Node.js with http-server
npx http-server -p 8000
```

3. Open your browser and navigate to:
```
http://localhost:8000
```

4. Click "Start Simulation" to begin!

## Architecture

### Project Structure
```
Tetflip_webgpu/
├── index.html          # Main HTML page
├── styles.css          # Styling
├── src/
│   ├── main.js         # Application entry point
│   ├── simulator.js    # TETFLIP simulation core
│   ├── mesh.js         # Tetrahedral mesh management
│   ├── particles.js    # Particle system
│   ├── pressure_solver.js  # Incompressibility solver
│   └── renderer.js     # WebGPU rendering
└── README.md
```

### Core Components

#### 1. TetFlipSimulator (`simulator.js`)
The main simulation loop that orchestrates:
- Particle-to-mesh velocity transfer (P2G)
- Body force application (gravity)
- Pressure solve for incompressibility
- Mesh-to-particle velocity transfer (G2P)
- Particle advection
- Collision handling

#### 2. TetrahedralMesh (`mesh.js`)
Manages the tetrahedral mesh:
- Regular grid-based mesh generation
- Barycentric coordinate computation
- Particle containment queries
- Mesh topology and connectivity

#### 3. ParticleSystem (`particles.js`)
Handles fluid particles:
- Position and velocity storage
- Particle initialization
- Data management

#### 4. PressureSolver (`pressure_solver.js`)
Enforces incompressibility:
- Divergence computation
- Pressure Poisson equation solve (Jacobi iteration)
- Pressure gradient application

#### 5. Renderer (`renderer.js`)
WebGPU-based visualization:
- Particle rendering (point sprites)
- Mesh wireframe rendering
- Camera controls
- MVP matrix computation

## TETFLIP Algorithm

The simulation follows these steps each frame:

1. **Particle to Grid (P2G)**: Transfer particle velocities to mesh nodes using barycentric interpolation
2. **Body Forces**: Apply gravity and other external forces
3. **Pressure Solve**: Solve Poisson equation ∇²p = ρ/Δt · ∇·v to find pressure field
4. **Pressure Projection**: Update velocities to be divergence-free: v_new = v_old - Δt·∇p
5. **Grid to Particle (G2P)**: Transfer updated velocities back to particles (FLIP method)
6. **Advection**: Move particles according to their velocities
7. **Collision**: Handle boundary collisions with domain walls

## Controls

### Simulation Controls
- **Start/Pause**: Control simulation playback
- **Reset**: Reset to initial dam break scenario

### Parameters
- **Time Step**: Simulation time step (affects stability and speed)
- **Gravity**: Gravitational acceleration
- **Viscosity**: Fluid viscosity (WIP)
- **Particle Count**: Number of particles (requires reset)

### Rendering Options
- **Show Particles**: Toggle particle visualization
- **Show Mesh**: Toggle tetrahedral mesh wireframe
- **Show Velocity Field**: Toggle velocity vector visualization (WIP)

### Camera Controls
- **Left Mouse Drag**: Rotate camera
- **Mouse Wheel**: Zoom in/out

## Technical Details

### FLIP vs PIC
The implementation uses the FLIP (Fluid-Implicit-Particle) method, which reduces numerical dissipation compared to PIC (Particle-in-Cell):

- **PIC**: v_particle = interpolate(v_grid)
- **FLIP**: v_particle = v_particle + interpolate(v_grid_new - v_grid_old)

The current implementation uses a FLIP ratio of 0.95 (95% FLIP, 5% PIC) for stability.

### Tetrahedral Mesh
The mesh uses a regular grid subdivided into tetrahedra. Each cube is split into 5 tetrahedra using a consistent pattern to avoid gaps. The current implementation uses an 8×8×8 grid resolution.

### Pressure Solver
The pressure solver uses Jacobi iterations to solve the Poisson equation. While simple, this method may require many iterations for convergence. Future versions will implement conjugate gradient or multigrid methods for better performance.

## Performance

Current performance metrics (approximate):
- **Grid Resolution**: 8×8×8 (2,880 tetrahedra)
- **Particles**: 1,000 (adjustable)
- **Frame Rate**: 60 FPS (browser dependent)
- **Computation**: CPU-based (GPU compute shaders planned)

## Limitations & Future Work

Current limitations:
- Physics computation is CPU-based (not GPU-accelerated yet)
- Fixed mesh resolution (no adaptive refinement)
- Simplified pressure solver
- No surface reconstruction
- Limited to single-phase fluids

Future improvements:
- Implement GPU compute shaders for physics
- Add adaptive mesh refinement
- Surface reconstruction and marching cubes
- SPH-style surface tension
- Viscosity implementation
- Multiple fluid scenarios
- Performance profiling and optimization

## References

1. Ando, R., Thuerey, N., & Wojtan, C. (2013). A highly adaptive liquid simulator on tetrahedral meshes. *ACM Transactions on Graphics (TOG)*, 32(4), 1-10.

2. Bridson, R. (2015). *Fluid Simulation for Computer Graphics* (2nd ed.). CRC Press.

3. Zhu, Y., & Bridson, R. (2005). Animating sand as a fluid. *ACM Transactions on Graphics (TOG)*, 24(3), 965-972.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Based on the TETFLIP technique by Ryoichi Ando, Nils Thuerey, and Chris Wojtan
- Inspired by the computer graphics and fluid simulation research community
- Built with WebGPU for modern web-based GPU computing
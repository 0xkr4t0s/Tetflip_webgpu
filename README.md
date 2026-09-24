# TETFLIP · WebGPU

Real-time liquid simulation on **tetrahedral meshes**, running entirely on the GPU with WebGPU.
It implements the discretization from [*A Highly Adaptive Liquid Simulator on Tetrahedral
Meshes*](https://doi.org/10.1145/2461912.2461982) (Ando, Thuerey & Wojtan, SIGGRAPH 2013) and
renders the result as a smooth, refractive liquid surface.

- **TETFLIP pressure projection** (paper §3): velocities at tetrahedron centres, pressures at
  mesh nodes, and the projection `[∇]ᵀV[∇] p = [∇]ᵀV u` solved with Jacobi-preconditioned CG.
- **Second-order free surface**: the paper's symmetric ghost-fluid coefficients (Eq. 12).
- **FLIP particles** with the paper's subdivided-tetrahedron velocity interpolation and its
  adjoint for particle-to-mesh transfer.
- **Everything on the GPU**: 17 small WGSL compute kernels, with no CPU round trips during a step.
- **Screen-space fluid rendering**: sphere-impostor depth and thickness, bilateral smoothing, and
  Fresnel, refraction and Beer–Lambert absorption.
- **Verified**: the GPU solver is checked field by field against an f64 CPU reference, and the
  reference is unit tested (exact hydrostatics, divergence-free projection, matrix symmetry).

## Running it

You need Node 20+ and a browser with WebGPU (Chrome/Edge 113+, Safari 26+, or Firefox 141+ on
Windows).

```bash
npm install
npm run dev          # http://localhost:5173
```

### Controls

| Input | Action |
| --- | --- |
| drag | orbit |
| right-drag / two-finger drag | pan |
| wheel / pinch | zoom |
| <kbd>Shift</kbd> + drag (or enable *drag pushes liquid*) | push the liquid |
| <kbd>Space</kbd> | pause |
| <kbd>R</kbd> | reset the scene |
| <kbd>V</kbd> | toggle liquid surface / particles |
| <kbd>M</kbd> | toggle the tetrahedral mesh slice (coloured by pressure) |

The panel on the right switches scenes (dam break, double dam break, drop into a pool, column
collapse, sloshing wave), mesh resolution, physics parameters and rendering options.

URL parameters let you share a setup: `?scene=drop&res=high&view=particles&mesh=1&paused=1`.

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Vite dev server |
| `npm run build` | type check and production build into `dist/` |
| `npm test` | unit tests (mesh invariants and the CPU reference solver) |
| `npm run typecheck` | TypeScript only |
| `npm run test:gpu` | runs the WebGPU solver next to the CPU reference in headless Chromium and compares every intermediate field |
| `npm run screenshot -- out.png "scene=drop&res=low" 1.0` | renders the app headlessly at simulated time 1.0 s |

`test:gpu` and `screenshot` use Playwright's Chromium. When no GPU is available they fall back to
SwiftShader (slow, but good enough to validate correctness). Install the browser once with
`npx playwright install chromium`.

## How it works

Each substep runs this pipeline on the GPU (`src/sim/shaders/`):

1. **p2g**: every particle splats its velocity into its tetrahedron's centre and three of its
   nodes (the adjoint of the interpolation in step 9). It also splats barycentric-weighted
   offsets for the level set. WebGPU has no float atomics, so the splats use fixed-point
   `atomicAdd`.
2. **node_gather**: Zhu–Bridson level set `φ = |x − x̄| − r` and particle density at nodes.
3. **tet_gather / extrapolate**: normalised tet-centre velocities, then extended into empty tets.
4. **forces**: gravity and the interactive brush. The pre-force velocity is kept for FLIP.
5. **assemble**: one row per node of `A = [∇]ᵀV[∇]` with ghost-fluid terms for air neighbours,
   into a CSR pattern precomputed from the mesh, plus `b = [∇]ᵀV u`.
6. **PCG**: Jacobi-preconditioned conjugate gradients, warm started, with dot products reduced
   on the GPU and a fixed iteration count, so nothing is read back.
7. **project**: `u_t ← u_t − [∇]_t p̂`, where `p̂` includes the ghost pressures.
8. **tet_to_nodes**: volume-weighted node averages of the new velocity and of its change.
9. **g2p / advect**: FLIP/PIC blend using the subdivided-tet interpolation, then RK2 advection
   and jump-and-walk point location through tet face adjacency.

The mesh is a **body-centred cubic** tetrahedralization of the box: cube corners plus cube
centres, with four congruent tetrahedra per interior face. Boundary faces get an extra node at
their centre, so every boundary tet is half a BCC tet. As a result the mesh has no obtuse
dihedral angles, the pressure matrix is an M-matrix, and the ghost-fluid boundary condition stays
second-order everywhere. The mesh, adjacency, CSR sparsity and point-location seed grid are
built once on the CPU in a Web Worker.

The paper also re-meshes adaptively as the liquid moves. This project uses a static mesh, but
the solver works with any tetrahedral mesh (nothing in the GPU code assumes BCC structure), so
adaptive meshing can be added later.

See [`docs/ALGORITHM.md`](docs/ALGORITHM.md) for the maths and [`docs/tetflip_paper.pdf`](docs/tetflip_paper.pdf)
for the paper.

## Project layout

```
src/
  app/App.ts            UI, scene loading, frame loop, force brush
  gpu/                  device setup, Kernel helper (layouts from WGSL bindings)
  mesh/                 BCC mesh generation, precomputed topology, point location, worker
  sim/
    GpuSolver.ts        buffers and per-substep dispatch sequence
    reference.ts        CPU reference of the same algorithm (f64)
    shaders/*.wgsl      compute kernels
    scenes.ts params.ts
  render/
    Renderer.ts         screen-space fluid pipeline and debug views
    Camera.ts           orbit camera and pointer controls
    shaders/*.wgsl
tests/                  Vitest unit tests; tests/gpu/ holds the browser-side GPU check
scripts/                headless GPU check and screenshot tools
```

## License

MIT. See [LICENSE](LICENSE).

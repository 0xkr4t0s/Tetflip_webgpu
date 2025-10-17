# Development Guide

## Getting Started

### Prerequisites

1. **Node.js** (optional, for development tools)
2. **Modern browser** with WebGPU support:
   - Chrome 113+ (enable via `chrome://flags/#enable-webgpu`)
   - Edge 113+
   - Firefox Nightly
   - Safari Technology Preview

3. **Text editor/IDE** (VS Code recommended)

### Project Setup

```bash
git clone https://github.com/0xkr4t0s/Tetflip_webgpu.git
cd Tetflip_webgpu

# Start local server
python -m http.server 8000
# OR
npx http-server -p 8000

# Open browser to http://localhost:8000
```

## Project Structure

```
Tetflip_webgpu/
├── index.html              # Main application page
├── styles.css              # Styling
├── src/                    # Source code
│   ├── main.js            # Application entry point
│   ├── simulator.js       # TETFLIP simulation core
│   ├── mesh.js            # Tetrahedral mesh
│   ├── particles.js       # Particle system
│   ├── pressure_solver.js # Pressure projection
│   └── renderer.js        # WebGPU rendering
├── docs/                   # Documentation
│   └── ALGORITHM.md       # Algorithm documentation
├── README.md              # Project readme
└── LICENSE                # MIT License
```

## Code Architecture

### Module Dependencies

```
main.js
  ├── simulator.js
  │   ├── mesh.js
  │   ├── particles.js
  │   └── pressure_solver.js
  └── renderer.js
```

### Key Classes

#### 1. App (main.js)
- Entry point
- UI event handling
- Animation loop
- Statistics tracking

#### 2. TetFlipSimulator (simulator.js)
- Main simulation logic
- TETFLIP algorithm implementation
- Parameter management

#### 3. TetrahedralMesh (mesh.js)
- Mesh generation
- Topology queries
- Barycentric coordinates

#### 4. ParticleSystem (particles.js)
- Particle storage
- Position/velocity management

#### 5. PressureSolver (pressure_solver.js)
- Divergence computation
- Pressure Poisson solve
- Pressure gradient application

#### 6. Renderer (renderer.js)
- WebGPU rendering
- Camera controls
- Shader management

## Development Workflow

### Adding New Features

1. **Create a branch**
```bash
git checkout -b feature/your-feature
```

2. **Make changes**
3. **Test locally**
4. **Commit with clear messages**
```bash
git commit -m "Add feature: description"
```

5. **Push and create PR**
```bash
git push origin feature/your-feature
```

### Testing Changes

#### Manual Testing
1. Open browser console (F12)
2. Check for errors
3. Verify visual output
4. Test parameter changes
5. Check performance (FPS)

#### Code Validation
```bash
# Check JavaScript syntax
node --check src/*.js

# Format code (if using prettier)
npx prettier --write src/*.js
```

## Common Development Tasks

### 1. Adding New Simulation Parameters

**In simulator.js:**
```javascript
constructor(device) {
    // Add parameter
    this.myNewParameter = defaultValue;
}

setMyNewParameter(value) {
    this.myNewParameter = value;
}
```

**In main.js:**
```javascript
setupEventListeners() {
    document.getElementById('mySlider').addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        this.simulator.setMyNewParameter(value);
    });
}
```

**In index.html:**
```html
<input type="range" id="mySlider" min="0" max="10" step="0.1" value="5">
```

### 2. Modifying Rendering

**In renderer.js:**
```javascript
render(particles, mesh, velocityField) {
    // Add your custom rendering code
    this.renderMyCustomVisualization();
}
```

### 3. Changing Simulation Algorithm

**In simulator.js:**
```javascript
step() {
    // Modify simulation steps
    this.myCustomStep();
}
```

### 4. Adding WebGPU Compute Shaders

```javascript
// Create shader module
const shaderCode = `
    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        // Your compute shader code
    }
`;

const shaderModule = device.createShaderModule({ code: shaderCode });

// Create compute pipeline
const pipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
        module: shaderModule,
        entryPoint: 'main'
    }
});

// Dispatch compute work
const commandEncoder = device.createCommandEncoder();
const passEncoder = commandEncoder.beginComputePass();
passEncoder.setPipeline(pipeline);
passEncoder.setBindGroup(0, bindGroup);
passEncoder.dispatchWorkgroups(Math.ceil(count / 256));
passEncoder.end();
device.queue.submit([commandEncoder.finish()]);
```

## Performance Optimization

### Profiling

Use browser DevTools:
1. Open Performance tab
2. Record simulation
3. Analyze bottlenecks

### Optimization Strategies

#### JavaScript Optimization
- Use typed arrays (Float32Array, Uint32Array)
- Minimize object allocations
- Avoid unnecessary array copies
- Use efficient data structures

#### WebGPU Optimization
- Batch render calls
- Minimize buffer uploads
- Use appropriate buffer usage flags
- Implement GPU compute for physics

#### Algorithm Optimization
- Spatial acceleration structures (octree, BVH)
- Better linear solvers (CG, multigrid)
- Adaptive timestep
- Level of detail (LOD)

## Debugging Tips

### Common Issues

#### WebGPU Not Available
```javascript
if (!navigator.gpu) {
    console.error('WebGPU not supported');
    // Fallback or error message
}
```

#### Shader Compilation Errors
```javascript
const shaderModule = device.createShaderModule({ code: shaderCode });
const info = await shaderModule.getCompilationInfo();
console.log(info.messages);
```

#### Buffer Size Mismatches
```javascript
// Always check buffer sizes
console.log('Expected size:', expectedSize);
console.log('Actual size:', buffer.size);
```

#### NaN/Infinity Values
```javascript
// Add validation
if (!isFinite(value)) {
    console.warn('Invalid value detected:', value);
    value = 0; // or some default
}
```

### Debugging Tools

1. **Console Logging**
```javascript
console.log('Particle count:', this.getParticleCount());
console.log('Simulation time:', this.simulationTime);
```

2. **Visual Debugging**
- Render mesh wireframe
- Show velocity vectors
- Color-code by property (pressure, velocity magnitude)

3. **Performance Monitoring**
```javascript
const start = performance.now();
this.expensiveOperation();
const end = performance.now();
console.log('Operation took:', end - start, 'ms');
```

## Code Style Guidelines

### Naming Conventions
- **Classes**: PascalCase (e.g., `TetFlipSimulator`)
- **Functions/Methods**: camelCase (e.g., `computePressure`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_PARTICLES`)
- **Private properties**: prefix with underscore (e.g., `_internalState`)

### Comments
```javascript
/**
 * Brief description of function
 * 
 * @param {type} paramName - Description
 * @returns {type} Description
 */
function myFunction(paramName) {
    // Implementation
}
```

### Code Organization
- One class per file
- Group related methods together
- Keep functions focused and small
- Use meaningful variable names

## Contributing

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Make changes with clear commits
4. Test thoroughly
5. Update documentation
6. Submit pull request

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] No console errors
- [ ] Performance is acceptable
- [ ] Documentation updated
- [ ] Comments explain complex logic

## Resources

### WebGPU
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WebGPU Samples](https://webgpu.github.io/webgpu-samples/)
- [MDN WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API)

### Fluid Simulation
- [Bridson's Book](http://www.cs.ubc.ca/~rbridson/fluidsimulation/)
- [SIGGRAPH Course Notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/)

### JavaScript
- [MDN JavaScript](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
- [JavaScript.info](https://javascript.info/)

## Support

For questions or issues:
- Open an issue on GitHub
- Check existing documentation
- Review algorithm documentation

## License

This project is licensed under the MIT License - see LICENSE file for details.

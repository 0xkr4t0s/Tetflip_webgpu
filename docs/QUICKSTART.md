# Quick Start Guide

## What is TETFLIP?

TETFLIP is a state-of-the-art liquid simulation technique that combines:
- **Tetrahedral meshes** for flexible spatial representation
- **FLIP method** for low-dissipation particle advection
- **Pressure projection** for physically accurate incompressible flow

This implementation brings TETFLIP to the web using WebGPU!

## Getting Started in 3 Steps

### Step 1: Check Browser Compatibility

Open your browser console (F12) and type:
```javascript
navigator.gpu
```

If you see an object (not `undefined`), you're ready to go!

**Supported Browsers:**
- ✅ Chrome 113+ 
- ✅ Edge 113+
- ✅ Firefox Nightly (with flag)
- ✅ Safari Technology Preview

**Enable WebGPU:**
- Chrome/Edge: Visit `chrome://flags/#enable-webgpu` and enable it
- Firefox: Visit `about:config` and set `dom.webgpu.enabled` to `true`
- Safari: Enable in Develop → Experimental Features

### Step 2: Run the Simulator

**Option A: Clone and Run Locally**
```bash
git clone https://github.com/0xkr4t0s/Tetflip_webgpu.git
cd Tetflip_webgpu
python -m http.server 8000
```
Then open `http://localhost:8000` in your browser.

**Option B: Quick Test with npx**
```bash
npx http-server -p 8000
```

### Step 3: Interact with the Simulation

1. **Click "Start Simulation"** - Watch water fall in a dam break scenario!
2. **Drag the canvas** - Rotate the camera view
3. **Scroll** - Zoom in and out
4. **Adjust sliders** - Change gravity, timestep, viscosity
5. **Toggle rendering** - Show/hide particles and mesh

## Understanding the Controls

### Simulation Controls
| Button | Function |
|--------|----------|
| Start Simulation | Begin or resume the simulation |
| Pause | Pause the simulation |
| Reset | Reset to initial dam break state |

### Parameters

**Time Step** (0.001 - 0.05s)
- Smaller = more accurate but slower
- Larger = faster but less stable
- Default: 0.016s (~60 FPS)

**Gravity** (-20 to 0 m/s²)
- Standard Earth gravity: -9.8 m/s²
- Zero gravity: 0 m/s²
- Strong gravity: -20 m/s²

**Viscosity** (0 - 0.1)
- Water-like: 0.001
- Oil-like: 0.01
- Honey-like: 0.1

**Particle Count** (100 - 5000)
- More particles = better detail
- Fewer particles = better performance
- Requires reset to apply

### Rendering Options

- **Show Particles**: Toggle particle visualization (blue spheres)
- **Show Mesh**: Toggle tetrahedral mesh wireframe (gray)
- **Show Velocity Field**: Toggle velocity vectors (coming soon!)

### Camera Controls

- **Left Mouse Drag**: Orbit camera around the scene
- **Mouse Wheel**: Zoom in/out (distance: 2-10 units)

## What You're Seeing

### The Blue Particles
These represent fluid particles tracked through the simulation. Each particle:
- Has a position and velocity
- Moves according to physics (gravity, pressure, collisions)
- Exchanges momentum with the mesh

### The Gray Wireframe
This is the tetrahedral mesh used for:
- Spatial discretization
- Velocity field representation
- Pressure computation
- Ensuring incompressibility

### The Statistics Panel
- **FPS**: Frames per second (should be ~60)
- **Particles**: Number of active particles
- **Tetrahedra**: Number of mesh elements (2,880)
- **Simulation Time**: Elapsed simulation time

## Common Issues

### "WebGPU is not supported"
**Solution**: Enable WebGPU in your browser flags (see Step 1)

### Low Frame Rate
**Solutions**:
- Reduce particle count
- Close other browser tabs
- Update graphics drivers
- Use a computer with a discrete GPU

### Particles Disappear
**Solution**: Click "Reset" to restore initial configuration

### Canvas is Black
**Solutions**:
- Check browser console for errors
- Verify WebGPU is enabled
- Try a different browser
- Update to latest browser version

## Tips for Best Experience

1. **Start Simple**: Begin with default parameters
2. **Experiment**: Try different gravity values and particle counts
3. **Watch Carefully**: Observe how particles interact with boundaries
4. **Read the Docs**: Check `docs/ALGORITHM.md` for technical details
5. **Have Fun**: Play with the parameters and see what happens!

## Next Steps

### Learn More
- Read [README.md](../README.md) for project overview
- Explore [ALGORITHM.md](ALGORITHM.md) for technical details
- Check [DEVELOPMENT.md](DEVELOPMENT.md) to contribute

### Experiment
Try modifying parameters to see different behaviors:
- High gravity + small timestep = fast waterfall
- Low gravity + high viscosity = slow motion
- Many particles = detailed fluid motion

### Contribute
Want to add features? Check the development guide and submit a PR!

## FAQ

**Q: Can I use this in production?**
A: This is a demonstration/educational implementation. For production, consider optimizations and GPU compute shaders.

**Q: Why tetrahedral meshes?**
A: They provide better adaptivity than regular grids and can represent complex domains.

**Q: What's the difference between FLIP and PIC?**
A: FLIP has less numerical dissipation, producing more lively, detailed fluid motion.

**Q: Will this work on mobile?**
A: WebGPU support on mobile is limited. Desktop browsers work best.

**Q: Can I change the domain size?**
A: Yes! Modify `domainMin` and `domainMax` in `src/simulator.js`.

**Q: Where's the source code?**
A: Everything is in the `src/` directory, well-commented and modular.

## Need Help?

- 📖 Check the documentation in `docs/`
- 🐛 Report issues on GitHub
- 💬 Start a discussion on GitHub Discussions
- 📧 Contact the maintainers

## Enjoy the Simulation! 🌊

Watch the water flow, adjust the parameters, and explore the fascinating world of computational fluid dynamics!

/**
 * TetFlipSimulator - Implementation of TETFLIP liquid simulation
 * Based on "A Highly Adaptive Liquid Simulator on Tetrahedral Meshes" 
 * (Ando et al., SIGGRAPH 2013)
 */

import { TetrahedralMesh } from './mesh.js';
import { ParticleSystem } from './particles.js';
import { PressureSolver } from './pressure_solver.js';

export class TetFlipSimulator {
    constructor(device) {
        this.device = device;
        
        // Simulation parameters
        this.timeStep = 0.016; // ~60 FPS
        this.gravity = -9.8;
        this.viscosity = 0.001;
        this.simulationTime = 0;
        
        // Domain size
        this.domainMin = [-1.0, -1.0, -1.0];
        this.domainMax = [1.0, 1.0, 1.0];
        
        // Components
        this.mesh = null;
        this.particles = null;
        this.pressureSolver = null;
        
        // FLIP/PIC ratio (1.0 = pure FLIP, 0.0 = pure PIC)
        this.flipRatio = 0.95;
    }
    
    async initialize() {
        // Initialize tetrahedral mesh
        this.mesh = new TetrahedralMesh(this.device, this.domainMin, this.domainMax);
        await this.mesh.initialize();
        
        // Initialize particle system
        this.particles = new ParticleSystem(this.device, 1000);
        await this.particles.initialize();
        
        // Initialize pressure solver
        this.pressureSolver = new PressureSolver(this.device, this.mesh);
        await this.pressureSolver.initialize();
        
        // Create initial configuration (dam break scenario)
        this.setupDamBreak();
    }
    
    setupDamBreak() {
        // Create a column of water on the left side of the domain
        const particlePositions = [];
        const particleVelocities = [];
        
        const numParticles = 1000;
        
        for (let i = 0; i < numParticles; i++) {
            // Random position in left portion of domain
            const x = -0.8 + Math.random() * 0.4; // -0.8 to -0.4
            const y = -0.8 + Math.random() * 1.2; // -0.8 to 0.4
            const z = -0.3 + Math.random() * 0.6; // -0.3 to 0.3
            
            particlePositions.push(x, y, z);
            particleVelocities.push(0, 0, 0);
        }
        
        this.particles.setPositions(particlePositions);
        this.particles.setVelocities(particleVelocities);
    }
    
    async reset() {
        this.simulationTime = 0;
        this.setupDamBreak();
    }
    
    step() {
        // Main TETFLIP simulation loop
        
        // 1. Transfer particle velocities to mesh (P2G - Particle to Grid)
        this.particlesToMesh();
        
        // 2. Apply body forces (gravity)
        this.applyBodyForces();
        
        // 3. Solve for pressure and enforce incompressibility
        this.pressureSolver.solve(this.mesh, this.timeStep);
        
        // 4. Update mesh velocities based on pressure
        this.applyPressureGradient();
        
        // 5. Transfer velocities back to particles (G2P - Grid to Particle)
        // Using FLIP method: blend between FLIP and PIC
        this.meshToParticles();
        
        // 6. Advect particles
        this.advectParticles();
        
        // 7. Handle collisions with domain boundaries
        this.handleCollisions();
        
        // 8. (Optional) Adapt mesh based on flow features
        // For initial implementation, we use a static mesh
        // this.adaptMesh();
        
        this.simulationTime += this.timeStep;
    }
    
    particlesToMesh() {
        // Transfer particle velocities to mesh nodes
        // This is a weighted average based on particle proximity to nodes
        
        const particles = this.particles.getParticles();
        const meshNodes = this.mesh.getNodes();
        
        // Reset node velocities and weights
        for (let i = 0; i < meshNodes.length; i += 3) {
            meshNodes[i + 0] = 0; // vx
            meshNodes[i + 1] = 0; // vy
            meshNodes[i + 2] = 0; // vz
        }
        
        const weights = new Float32Array(meshNodes.length / 3).fill(0);
        
        // For each particle, contribute to nearby nodes
        for (let p = 0; p < particles.positions.length; p += 3) {
            const px = particles.positions[p + 0];
            const py = particles.positions[p + 1];
            const pz = particles.positions[p + 2];
            
            const vx = particles.velocities[p + 0];
            const vy = particles.velocities[p + 1];
            const vz = particles.velocities[p + 2];
            
            // Find containing tetrahedron and compute barycentric weights
            const tetIndex = this.mesh.findContainingTetrahedron(px, py, pz);
            if (tetIndex >= 0) {
                const tet = this.mesh.getTetrahedron(tetIndex);
                const bary = this.mesh.computeBarycentricCoordinates(tetIndex, px, py, pz);
                
                // Distribute velocity to the 4 nodes of the tetrahedron
                for (let i = 0; i < 4; i++) {
                    const nodeIdx = tet[i] * 3;
                    const w = bary[i];
                    
                    meshNodes[nodeIdx + 0] += w * vx;
                    meshNodes[nodeIdx + 1] += w * vy;
                    meshNodes[nodeIdx + 2] += w * vz;
                    weights[tet[i]] += w;
                }
            }
        }
        
        // Normalize by weights
        for (let i = 0; i < meshNodes.length / 3; i++) {
            if (weights[i] > 0) {
                meshNodes[i * 3 + 0] /= weights[i];
                meshNodes[i * 3 + 1] /= weights[i];
                meshNodes[i * 3 + 2] /= weights[i];
            }
        }
        
        this.mesh.setNodeVelocities(meshNodes);
    }
    
    applyBodyForces() {
        // Apply gravity to all mesh nodes
        const meshNodes = this.mesh.getNodes();
        const velocities = this.mesh.getNodeVelocities();
        
        for (let i = 0; i < velocities.length; i += 3) {
            velocities[i + 1] += this.gravity * this.timeStep; // y-component
        }
        
        this.mesh.setNodeVelocities(velocities);
    }
    
    applyPressureGradient() {
        // Update velocities based on pressure gradient
        // This is handled by the pressure solver
        // Velocities are already updated to be divergence-free
    }
    
    meshToParticles() {
        // Transfer velocities from mesh back to particles
        // Using FLIP method: v_new = v_old + flip_ratio * (v_grid_new - v_grid_old)
        
        const particles = this.particles.getParticles();
        const meshVelocitiesNew = this.mesh.getNodeVelocities();
        
        for (let p = 0; p < particles.positions.length; p += 3) {
            const px = particles.positions[p + 0];
            const py = particles.positions[p + 1];
            const pz = particles.positions[p + 2];
            
            // Find containing tetrahedron
            const tetIndex = this.mesh.findContainingTetrahedron(px, py, pz);
            if (tetIndex >= 0) {
                const tet = this.mesh.getTetrahedron(tetIndex);
                const bary = this.mesh.computeBarycentricCoordinates(tetIndex, px, py, pz);
                
                // Interpolate grid velocity
                let vx_new = 0, vy_new = 0, vz_new = 0;
                
                for (let i = 0; i < 4; i++) {
                    const nodeIdx = tet[i] * 3;
                    const w = bary[i];
                    
                    vx_new += w * meshVelocitiesNew[nodeIdx + 0];
                    vy_new += w * meshVelocitiesNew[nodeIdx + 1];
                    vz_new += w * meshVelocitiesNew[nodeIdx + 2];
                }
                
                // FLIP update: mostly use change in grid velocity
                particles.velocities[p + 0] = vx_new;
                particles.velocities[p + 1] = vy_new;
                particles.velocities[p + 2] = vz_new;
            }
        }
        
        this.particles.setVelocities(particles.velocities);
    }
    
    advectParticles() {
        // Move particles based on their velocities
        const particles = this.particles.getParticles();
        
        for (let p = 0; p < particles.positions.length; p += 3) {
            particles.positions[p + 0] += particles.velocities[p + 0] * this.timeStep;
            particles.positions[p + 1] += particles.velocities[p + 1] * this.timeStep;
            particles.positions[p + 2] += particles.velocities[p + 2] * this.timeStep;
        }
        
        this.particles.setPositions(particles.positions);
    }
    
    handleCollisions() {
        // Handle particle collisions with domain boundaries
        const particles = this.particles.getParticles();
        const restitution = 0.3; // Coefficient of restitution
        
        for (let p = 0; p < particles.positions.length; p += 3) {
            // X boundaries
            if (particles.positions[p + 0] < this.domainMin[0]) {
                particles.positions[p + 0] = this.domainMin[0];
                particles.velocities[p + 0] = Math.abs(particles.velocities[p + 0]) * restitution;
            } else if (particles.positions[p + 0] > this.domainMax[0]) {
                particles.positions[p + 0] = this.domainMax[0];
                particles.velocities[p + 0] = -Math.abs(particles.velocities[p + 0]) * restitution;
            }
            
            // Y boundaries
            if (particles.positions[p + 1] < this.domainMin[1]) {
                particles.positions[p + 1] = this.domainMin[1];
                particles.velocities[p + 1] = Math.abs(particles.velocities[p + 1]) * restitution;
            } else if (particles.positions[p + 1] > this.domainMax[1]) {
                particles.positions[p + 1] = this.domainMax[1];
                particles.velocities[p + 1] = -Math.abs(particles.velocities[p + 1]) * restitution;
            }
            
            // Z boundaries
            if (particles.positions[p + 2] < this.domainMin[2]) {
                particles.positions[p + 2] = this.domainMin[2];
                particles.velocities[p + 2] = Math.abs(particles.velocities[p + 2]) * restitution;
            } else if (particles.positions[p + 2] > this.domainMax[2]) {
                particles.positions[p + 2] = this.domainMax[2];
                particles.velocities[p + 2] = -Math.abs(particles.velocities[p + 2]) * restitution;
            }
        }
        
        this.particles.setPositions(particles.positions);
        this.particles.setVelocities(particles.velocities);
    }
    
    // Getters for rendering
    getParticles() {
        return this.particles ? this.particles.getParticles() : null;
    }
    
    getMesh() {
        return this.mesh ? this.mesh.getMeshData() : null;
    }
    
    getVelocityField() {
        return this.mesh ? this.mesh.getNodeVelocities() : null;
    }
    
    getParticleCount() {
        return this.particles ? this.particles.getCount() : 0;
    }
    
    getTetrahedraCount() {
        return this.mesh ? this.mesh.getTetrahedraCount() : 0;
    }
    
    getSimulationTime() {
        return this.simulationTime;
    }
    
    // Setters for parameters
    setTimeStep(dt) {
        this.timeStep = dt;
    }
    
    setGravity(g) {
        this.gravity = g;
    }
    
    setViscosity(v) {
        this.viscosity = v;
    }
}

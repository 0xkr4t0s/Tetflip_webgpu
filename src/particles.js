/**
 * ParticleSystem - Manages fluid particles for TETFLIP simulation
 */

export class ParticleSystem {
    constructor(device, maxParticles) {
        this.device = device;
        this.maxParticles = maxParticles;
        this.count = 0;
        
        this.positions = null;  // Particle positions [x, y, z, ...]
        this.velocities = null; // Particle velocities [vx, vy, vz, ...]
    }
    
    async initialize() {
        // Initialize particle buffers
        this.positions = new Float32Array(this.maxParticles * 3);
        this.velocities = new Float32Array(this.maxParticles * 3);
        this.count = 0;
    }
    
    setPositions(positions) {
        if (positions.length > this.maxParticles * 3) {
            console.warn('Too many particles, truncating');
            positions = positions.slice(0, this.maxParticles * 3);
        }
        
        this.positions.set(positions);
        this.count = positions.length / 3;
    }
    
    setVelocities(velocities) {
        if (velocities.length > this.maxParticles * 3) {
            console.warn('Too many velocities, truncating');
            velocities = velocities.slice(0, this.maxParticles * 3);
        }
        
        this.velocities.set(velocities);
    }
    
    getParticles() {
        return {
            positions: this.positions,
            velocities: this.velocities,
            count: this.count
        };
    }
    
    getCount() {
        return this.count;
    }
}

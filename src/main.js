import { TetFlipSimulator } from './simulator.js';
import { Renderer } from './renderer.js';

class App {
    constructor() {
        this.canvas = document.getElementById('renderCanvas');
        this.simulator = null;
        this.renderer = null;
        this.isRunning = false;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fps = 0;
        
        this.init();
    }
    
    async init() {
        // Check WebGPU support
        if (!navigator.gpu) {
            document.getElementById('webgpu-error').style.display = 'block';
            console.error('WebGPU not supported');
            return;
        }
        
        try {
            // Initialize WebGPU
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error('No GPU adapter found');
            }
            
            const device = await adapter.requestDevice();
            
            // Initialize simulator and renderer
            this.simulator = new TetFlipSimulator(device);
            this.renderer = new Renderer(this.canvas, device);
            
            await this.simulator.initialize();
            await this.renderer.initialize();
            
            // Setup UI event listeners
            this.setupEventListeners();
            
            // Update initial statistics
            this.updateStats();
            
            console.log('TETFLIP Simulator initialized successfully');
        } catch (error) {
            console.error('Failed to initialize:', error);
            document.getElementById('webgpu-error').textContent = 
                `Failed to initialize WebGPU: ${error.message}`;
            document.getElementById('webgpu-error').style.display = 'block';
        }
    }
    
    setupEventListeners() {
        // Simulation controls
        document.getElementById('startBtn').addEventListener('click', () => this.start());
        document.getElementById('pauseBtn').addEventListener('click', () => this.pause());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        
        // Parameter sliders
        document.getElementById('timeStepSlider').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('timeStepValue').textContent = value.toFixed(3);
            if (this.simulator) {
                this.simulator.setTimeStep(value);
            }
        });
        
        document.getElementById('gravitySlider').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('gravityValue').textContent = value.toFixed(1);
            if (this.simulator) {
                this.simulator.setGravity(value);
            }
        });
        
        document.getElementById('viscositySlider').addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            document.getElementById('viscosityValue').textContent = value.toFixed(3);
            if (this.simulator) {
                this.simulator.setViscosity(value);
            }
        });
        
        document.getElementById('particleCountSlider').addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            document.getElementById('particleCountValue').textContent = value;
        });
        
        // Rendering options
        document.getElementById('showParticles').addEventListener('change', (e) => {
            if (this.renderer) {
                this.renderer.setShowParticles(e.target.checked);
            }
        });
        
        document.getElementById('showMesh').addEventListener('change', (e) => {
            if (this.renderer) {
                this.renderer.setShowMesh(e.target.checked);
            }
        });
        
        document.getElementById('showVelocity').addEventListener('change', (e) => {
            if (this.renderer) {
                this.renderer.setShowVelocity(e.target.checked);
            }
        });
        
        // Handle canvas resize
        window.addEventListener('resize', () => this.handleResize());
        this.handleResize();
    }
    
    handleResize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        
        if (this.renderer) {
            this.renderer.handleResize();
        }
    }
    
    start() {
        if (!this.isRunning) {
            this.isRunning = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('pauseBtn').disabled = false;
            this.lastFrameTime = performance.now();
            this.animate();
        }
    }
    
    pause() {
        this.isRunning = false;
        document.getElementById('startBtn').disabled = false;
        document.getElementById('pauseBtn').disabled = true;
    }
    
    async reset() {
        const wasRunning = this.isRunning;
        this.pause();
        
        if (this.simulator) {
            await this.simulator.reset();
        }
        
        this.updateStats();
        
        if (wasRunning) {
            this.start();
        }
    }
    
    animate() {
        if (!this.isRunning) return;
        
        const currentTime = performance.now();
        const deltaTime = (currentTime - this.lastFrameTime) / 1000;
        this.lastFrameTime = currentTime;
        
        // Update FPS
        this.frameCount++;
        if (this.frameCount % 60 === 0) {
            this.fps = Math.round(1 / deltaTime);
        }
        
        // Step simulation
        if (this.simulator) {
            this.simulator.step();
        }
        
        // Render
        if (this.renderer && this.simulator) {
            this.renderer.render(
                this.simulator.getParticles(),
                this.simulator.getMesh(),
                this.simulator.getVelocityField()
            );
        }
        
        // Update statistics
        this.updateStats();
        
        // Continue animation loop
        requestAnimationFrame(() => this.animate());
    }
    
    updateStats() {
        document.getElementById('fps').textContent = this.fps;
        
        if (this.simulator) {
            document.getElementById('particleCount').textContent = 
                this.simulator.getParticleCount();
            document.getElementById('tetCount').textContent = 
                this.simulator.getTetrahedraCount();
            document.getElementById('simTime').textContent = 
                this.simulator.getSimulationTime().toFixed(3);
        }
    }
}

// Initialize app when DOM is loaded
window.addEventListener('DOMContentLoaded', () => {
    new App();
});

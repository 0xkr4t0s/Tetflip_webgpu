/**
 * Renderer - WebGPU renderer for TETFLIP simulation
 */

export class Renderer {
    constructor(canvas, device) {
        this.canvas = canvas;
        this.device = device;
        this.context = null;
        
        // Rendering options
        this.showParticles = true;
        this.showMesh = true;
        this.showVelocity = false;
        
        // Camera parameters
        this.cameraPosition = [0, 0, 5];
        this.cameraRotation = [0, 0];
        
        // Rendering resources
        this.pipeline = null;
        this.particlePipeline = null;
        this.meshPipeline = null;
        
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
    }
    
    async initialize() {
        // Get WebGPU context
        this.context = this.canvas.getContext('webgpu');
        
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: presentationFormat,
            alphaMode: 'opaque'
        });
        
        // Create render pipelines
        await this.createPipelines(presentationFormat);
        
        // Setup mouse controls for camera
        this.setupMouseControls();
    }
    
    async createPipelines(format) {
        // Shader code for particle rendering
        const particleShaderCode = `
            struct Uniforms {
                mvpMatrix: mat4x4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            }
            
            @vertex
            fn vertexMain(@location(0) position: vec3<f32>) -> VertexOutput {
                var output: VertexOutput;
                output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
                output.color = vec4<f32>(0.3, 0.6, 1.0, 1.0);
                return output;
            }
            
            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
                return input.color;
            }
        `;
        
        // Shader code for mesh rendering
        const meshShaderCode = `
            struct Uniforms {
                mvpMatrix: mat4x4<f32>,
            }
            
            @group(0) @binding(0) var<uniform> uniforms: Uniforms;
            
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) color: vec4<f32>,
            }
            
            @vertex
            fn vertexMain(@location(0) position: vec3<f32>) -> VertexOutput {
                var output: VertexOutput;
                output.position = uniforms.mvpMatrix * vec4<f32>(position, 1.0);
                output.color = vec4<f32>(0.8, 0.8, 0.8, 0.3);
                return output;
            }
            
            @fragment
            fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
                return input.color;
            }
        `;
        
        // Create shader modules
        const particleShaderModule = this.device.createShaderModule({
            code: particleShaderCode
        });
        
        const meshShaderModule = this.device.createShaderModule({
            code: meshShaderCode
        });
        
        // Create particle pipeline
        this.particlePipeline = await this.device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: particleShaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 12,
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x3'
                    }]
                }]
            },
            fragment: {
                module: particleShaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'point-list'
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });
        
        // Create mesh pipeline
        this.meshPipeline = await this.device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: meshShaderModule,
                entryPoint: 'vertexMain',
                buffers: [{
                    arrayStride: 12,
                    attributes: [{
                        shaderLocation: 0,
                        offset: 0,
                        format: 'float32x3'
                    }]
                }]
            },
            fragment: {
                module: meshShaderModule,
                entryPoint: 'fragmentMain',
                targets: [{
                    format: format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        },
                        alpha: {
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha',
                            operation: 'add'
                        }
                    }
                }]
            },
            primitive: {
                topology: 'line-list',
                stripIndexFormat: undefined
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus'
            }
        });
        
        // Create depth texture
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
    
    setupMouseControls() {
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaX = e.clientX - this.lastMouseX;
                const deltaY = e.clientY - this.lastMouseY;
                
                this.cameraRotation[0] += deltaY * 0.01;
                this.cameraRotation[1] += deltaX * 0.01;
                
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
        
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.cameraPosition[2] += e.deltaY * 0.01;
            this.cameraPosition[2] = Math.max(2, Math.min(10, this.cameraPosition[2]));
        });
    }
    
    render(particles, mesh, velocityField) {
        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
        
        const renderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store'
            }
        };
        
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        
        // Compute MVP matrix
        const mvpMatrix = this.computeMVPMatrix();
        const mvpBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(mvpBuffer, 0, mvpMatrix);
        
        // Render particles
        if (this.showParticles && particles && particles.count > 0) {
            const particleBuffer = this.device.createBuffer({
                size: particles.positions.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            });
            this.device.queue.writeBuffer(particleBuffer, 0, particles.positions);
            
            const bindGroup = this.device.createBindGroup({
                layout: this.particlePipeline.getBindGroupLayout(0),
                entries: [{
                    binding: 0,
                    resource: { buffer: mvpBuffer }
                }]
            });
            
            passEncoder.setPipeline(this.particlePipeline);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.setVertexBuffer(0, particleBuffer);
            passEncoder.draw(particles.count, 1, 0, 0);
        }
        
        // Render mesh wireframe
        if (this.showMesh && mesh && mesh.tetCount > 0) {
            const edges = this.extractMeshEdges(mesh);
            
            if (edges.length > 0) {
                const edgeBuffer = this.device.createBuffer({
                    size: edges.byteLength,
                    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
                });
                this.device.queue.writeBuffer(edgeBuffer, 0, edges);
                
                const bindGroup = this.device.createBindGroup({
                    layout: this.meshPipeline.getBindGroupLayout(0),
                    entries: [{
                        binding: 0,
                        resource: { buffer: mvpBuffer }
                    }]
                });
                
                passEncoder.setPipeline(this.meshPipeline);
                passEncoder.setBindGroup(0, bindGroup);
                passEncoder.setVertexBuffer(0, edgeBuffer);
                passEncoder.draw(edges.length / 3, 1, 0, 0);
            }
        }
        
        passEncoder.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }
    
    extractMeshEdges(mesh) {
        // Extract unique edges from tetrahedra for wireframe rendering
        const edgeSet = new Set();
        const edges = [];
        
        for (let t = 0; t < mesh.tetCount; t++) {
            const i = t * 4;
            const n0 = mesh.tetrahedra[i];
            const n1 = mesh.tetrahedra[i + 1];
            const n2 = mesh.tetrahedra[i + 2];
            const n3 = mesh.tetrahedra[i + 3];
            
            // Each tetrahedron has 6 edges
            const tetEdges = [
                [n0, n1], [n0, n2], [n0, n3],
                [n1, n2], [n1, n3], [n2, n3]
            ];
            
            for (const [a, b] of tetEdges) {
                const key = a < b ? `${a}-${b}` : `${b}-${a}`;
                if (!edgeSet.has(key)) {
                    edgeSet.add(key);
                    
                    // Add both vertices of the edge
                    edges.push(
                        mesh.nodes[a * 3], mesh.nodes[a * 3 + 1], mesh.nodes[a * 3 + 2],
                        mesh.nodes[b * 3], mesh.nodes[b * 3 + 1], mesh.nodes[b * 3 + 2]
                    );
                }
            }
        }
        
        return new Float32Array(edges);
    }
    
    computeMVPMatrix() {
        // Create model-view-projection matrix
        const fov = Math.PI / 4;
        const aspect = this.canvas.width / this.canvas.height;
        const near = 0.1;
        const far = 100.0;
        
        // Projection matrix
        const f = 1.0 / Math.tan(fov / 2);
        const rangeInv = 1.0 / (near - far);
        
        const projection = new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * rangeInv, -1,
            0, 0, near * far * rangeInv * 2, 0
        ]);
        
        // View matrix (camera transform)
        const rx = this.cameraRotation[0];
        const ry = this.cameraRotation[1];
        
        const cosRx = Math.cos(rx);
        const sinRx = Math.sin(rx);
        const cosRy = Math.cos(ry);
        const sinRy = Math.sin(ry);
        
        const view = new Float32Array([
            cosRy, sinRx * sinRy, -cosRx * sinRy, 0,
            0, cosRx, sinRx, 0,
            sinRy, -sinRx * cosRy, cosRx * cosRy, 0,
            0, 0, -this.cameraPosition[2], 1
        ]);
        
        // Multiply projection * view
        return this.multiplyMatrices(projection, view);
    }
    
    multiplyMatrices(a, b) {
        const result = new Float32Array(16);
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                result[i * 4 + j] = 0;
                for (let k = 0; k < 4; k++) {
                    result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                }
            }
        }
        
        return result;
    }
    
    handleResize() {
        if (this.depthTexture) {
            this.depthTexture.destroy();
        }
        
        this.depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
    }
    
    setShowParticles(show) {
        this.showParticles = show;
    }
    
    setShowMesh(show) {
        this.showMesh = show;
    }
    
    setShowVelocity(show) {
        this.showVelocity = show;
    }
}

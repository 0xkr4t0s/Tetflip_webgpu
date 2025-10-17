/**
 * TetrahedralMesh - Manages the tetrahedral mesh for TETFLIP simulation
 */

export class TetrahedralMesh {
    constructor(device, domainMin, domainMax) {
        this.device = device;
        this.domainMin = domainMin;
        this.domainMax = domainMax;
        
        this.nodes = null;           // Node positions [x, y, z, ...]
        this.nodeVelocities = null;  // Node velocities [vx, vy, vz, ...]
        this.tetrahedra = null;      // Tetrahedra indices [n0, n1, n2, n3, ...]
        this.nodeCount = 0;
        this.tetCount = 0;
    }
    
    async initialize() {
        // Create initial tetrahedral mesh
        // For simplicity, we'll create a regular grid-based tetrahedral mesh
        this.createRegularMesh(8, 8, 8);
    }
    
    createRegularMesh(nx, ny, nz) {
        // Create a regular grid and tetrahedralize each cube into 5 tetrahedra
        
        const nodes = [];
        const tetrahedra = [];
        
        const dx = (this.domainMax[0] - this.domainMin[0]) / nx;
        const dy = (this.domainMax[1] - this.domainMin[1]) / ny;
        const dz = (this.domainMax[2] - this.domainMin[2]) / nz;
        
        // Create grid nodes
        for (let k = 0; k <= nz; k++) {
            for (let j = 0; j <= ny; j++) {
                for (let i = 0; i <= nx; i++) {
                    const x = this.domainMin[0] + i * dx;
                    const y = this.domainMin[1] + j * dy;
                    const z = this.domainMin[2] + k * dz;
                    nodes.push(x, y, z);
                }
            }
        }
        
        // Create tetrahedra
        // Each cube is split into 5 tetrahedra
        const getNodeIndex = (i, j, k) => {
            return i + j * (nx + 1) + k * (nx + 1) * (ny + 1);
        };
        
        for (let k = 0; k < nz; k++) {
            for (let j = 0; j < ny; j++) {
                for (let i = 0; i < nx; i++) {
                    // Cube vertices
                    const n000 = getNodeIndex(i, j, k);
                    const n100 = getNodeIndex(i + 1, j, k);
                    const n010 = getNodeIndex(i, j + 1, k);
                    const n110 = getNodeIndex(i + 1, j + 1, k);
                    const n001 = getNodeIndex(i, j, k + 1);
                    const n101 = getNodeIndex(i + 1, j, k + 1);
                    const n011 = getNodeIndex(i, j + 1, k + 1);
                    const n111 = getNodeIndex(i + 1, j + 1, k + 1);
                    
                    // Split cube into 5 tetrahedra
                    // Using a consistent pattern to avoid gaps
                    tetrahedra.push(n000, n100, n110, n101);
                    tetrahedra.push(n000, n110, n010, n011);
                    tetrahedra.push(n000, n101, n001, n011);
                    tetrahedra.push(n110, n101, n111, n011);
                    tetrahedra.push(n000, n110, n101, n011);
                }
            }
        }
        
        this.nodes = new Float32Array(nodes);
        this.nodeVelocities = new Float32Array(nodes.length).fill(0);
        this.tetrahedra = new Uint32Array(tetrahedra);
        this.nodeCount = this.nodes.length / 3;
        this.tetCount = this.tetrahedra.length / 4;
    }
    
    getNodes() {
        return this.nodes;
    }
    
    getNodeVelocities() {
        return this.nodeVelocities;
    }
    
    setNodeVelocities(velocities) {
        this.nodeVelocities = new Float32Array(velocities);
    }
    
    getTetrahedra() {
        return this.tetrahedra;
    }
    
    getTetrahedron(index) {
        // Return the 4 node indices of a tetrahedron
        const i = index * 4;
        return [
            this.tetrahedra[i],
            this.tetrahedra[i + 1],
            this.tetrahedra[i + 2],
            this.tetrahedra[i + 3]
        ];
    }
    
    findContainingTetrahedron(x, y, z) {
        // Find which tetrahedron contains the point (x, y, z)
        // Simple linear search for now (could be optimized with spatial data structure)
        
        for (let t = 0; t < this.tetCount; t++) {
            const tet = this.getTetrahedron(t);
            const bary = this.computeBarycentricCoordinates(t, x, y, z);
            
            // Check if all barycentric coordinates are non-negative
            if (bary[0] >= 0 && bary[1] >= 0 && bary[2] >= 0 && bary[3] >= 0) {
                return t;
            }
        }
        
        return -1; // Not found
    }
    
    computeBarycentricCoordinates(tetIndex, x, y, z) {
        // Compute barycentric coordinates for point (x, y, z) in tetrahedron
        const tet = this.getTetrahedron(tetIndex);
        
        // Get node positions
        const p0 = [this.nodes[tet[0] * 3], this.nodes[tet[0] * 3 + 1], this.nodes[tet[0] * 3 + 2]];
        const p1 = [this.nodes[tet[1] * 3], this.nodes[tet[1] * 3 + 1], this.nodes[tet[1] * 3 + 2]];
        const p2 = [this.nodes[tet[2] * 3], this.nodes[tet[2] * 3 + 1], this.nodes[tet[2] * 3 + 2]];
        const p3 = [this.nodes[tet[3] * 3], this.nodes[tet[3] * 3 + 1], this.nodes[tet[3] * 3 + 2]];
        
        // Compute volume of tetrahedron using determinant
        const volume = this.tetrahedronVolume(p0, p1, p2, p3);
        
        if (Math.abs(volume) < 1e-10) {
            // Degenerate tetrahedron
            return [0.25, 0.25, 0.25, 0.25];
        }
        
        const p = [x, y, z];
        
        // Compute sub-volumes
        const v0 = this.tetrahedronVolume(p, p1, p2, p3);
        const v1 = this.tetrahedronVolume(p0, p, p2, p3);
        const v2 = this.tetrahedronVolume(p0, p1, p, p3);
        const v3 = this.tetrahedronVolume(p0, p1, p2, p);
        
        // Barycentric coordinates
        return [
            v0 / volume,
            v1 / volume,
            v2 / volume,
            v3 / volume
        ];
    }
    
    tetrahedronVolume(p0, p1, p2, p3) {
        // Compute signed volume of tetrahedron using scalar triple product
        const v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        const v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        const v3 = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
        
        // v1 . (v2 x v3)
        const cross = [
            v2[1] * v3[2] - v2[2] * v3[1],
            v2[2] * v3[0] - v2[0] * v3[2],
            v2[0] * v3[1] - v2[1] * v3[0]
        ];
        
        return (v1[0] * cross[0] + v1[1] * cross[1] + v1[2] * cross[2]) / 6.0;
    }
    
    getMeshData() {
        // Return mesh data for rendering
        return {
            nodes: this.nodes,
            tetrahedra: this.tetrahedra,
            nodeCount: this.nodeCount,
            tetCount: this.tetCount
        };
    }
    
    getTetrahedraCount() {
        return this.tetCount;
    }
}

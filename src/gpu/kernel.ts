type BindingType = GPUBufferBindingType;

interface BindingInfo {
  name: string;
  binding: number;
  type: BindingType;
}

const BINDING_RE = /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<([^>]*)>)?\s+(\w+)\s*:/g;

/** Reads the `@group(0) @binding(n) var<...> name` declarations of a WGSL module. */
export function parseBindings(code: string): BindingInfo[] {
  const out: BindingInfo[] = [];
  for (const m of code.matchAll(BINDING_RE)) {
    const space = (m[2] ?? '').replace(/\s/g, '');
    const type: BindingType = space === 'uniform' ? 'uniform' : space === 'storage,read_write' ? 'storage' : 'read-only-storage';
    out.push({ binding: Number(m[1]), type, name: m[3] });
  }
  return out.sort((a, b) => a.binding - b.binding);
}

/**
 * A compute pipeline whose bind group layout is derived from the shader's own binding
 * declarations, so bind groups can be created from a name → buffer map.
 */
export class Kernel {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  readonly bindings: BindingInfo[];
  private readonly device: GPUDevice;
  private readonly layout: GPUBindGroupLayout;

  constructor(device: GPUDevice, label: string, code: string, constants?: Record<string, number>) {
    this.device = device;
    this.label = label;
    this.bindings = parseBindings(code);
    this.layout = device.createBindGroupLayout({
      label,
      entries: this.bindings.map((b) => ({ binding: b.binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: b.type } })),
    });
    const module = device.createShaderModule({ label, code });
    this.pipeline = device.createComputePipeline({
      label,
      layout: device.createPipelineLayout({ label, bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: 'main', constants },
    });
  }

  bind(resources: Record<string, GPUBuffer>, label = this.label): GPUBindGroup {
    return this.device.createBindGroup({
      label,
      layout: this.layout,
      entries: this.bindings.map((b) => {
        const buffer = resources[b.name];
        if (!buffer) throw new Error(`${this.label}: no buffer supplied for binding '${b.name}'`);
        return { binding: b.binding, resource: { buffer } };
      }),
    });
  }
}

export function createBuffer(
  device: GPUDevice,
  label: string,
  usage: GPUBufferUsageFlags,
  dataOrSize: ArrayBufferView | number,
): GPUBuffer {
  const size = typeof dataOrSize === 'number' ? dataOrSize : dataOrSize.byteLength;
  // Bindings must be non-empty and sizes 4-byte aligned.
  const buffer = device.createBuffer({ label, size: Math.max(16, Math.ceil(size / 4) * 4), usage, mappedAtCreation: typeof dataOrSize !== 'number' });
  if (typeof dataOrSize !== 'number') {
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(dataOrSize.buffer, dataOrSize.byteOffset, dataOrSize.byteLength));
    buffer.unmap();
  }
  return buffer;
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, size = source.size): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.destroy();
  return copy;
}

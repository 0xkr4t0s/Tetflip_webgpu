import type { Vec3 } from '../math/vec3';
import type { PreparedMesh } from './prepare';

export interface MeshRequest {
  size: Vec3;
  spacing: number;
}

/** Builds and prepares a BCC mesh off the main thread. */
export function buildMeshAsync(request: MeshRequest): Promise<PreparedMesh> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./meshWorker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (e: MessageEvent<PreparedMesh>) => {
      worker.terminate();
      resolve(e.data);
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message));
    };
    worker.postMessage(request);
  });
}

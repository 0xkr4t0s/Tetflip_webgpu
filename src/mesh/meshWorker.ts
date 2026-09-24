/// <reference lib="webworker" />
import { createBccMesh } from './bcc';
import { prepareMesh } from './prepare';
import type { MeshRequest } from './buildMesh';

self.onmessage = (e: MessageEvent<MeshRequest>) => {
  const mesh = prepareMesh(createBccMesh({ boundsMin: [0, 0, 0], size: e.data.size, spacing: e.data.spacing }));
  const transfer = Object.values(mesh)
    .filter((v): v is ArrayBufferView => ArrayBuffer.isView(v))
    .map((v) => v.buffer as ArrayBuffer);
  transfer.push(mesh.locator.seeds.buffer as ArrayBuffer);
  (self as unknown as Worker).postMessage(mesh, transfer);
};

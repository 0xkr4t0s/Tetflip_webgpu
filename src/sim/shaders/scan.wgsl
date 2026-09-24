// In-place exclusive prefix sum over u32, one level of a multi-level scan. SCAN_ADD = 0 scans
// 512-element blocks and writes each block's total to `sums`; SCAN_ADD = 1 adds the scanned
// block totals back into the elements of each block.
override SCAN_ADD: u32 = 0u;

struct ScanParams { count: u32 }

@group(0) @binding(0) var<uniform> params: ScanParams;
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

const THREADS: u32 = 256u;
var<workgroup> partial: array<u32, THREADS>;

@compute @workgroup_size(THREADS)
fn main(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3u) {
  let i0 = wid.x * 2u * THREADS + 2u * lid;
  let i1 = i0 + 1u;
  if (SCAN_ADD == 1u) {
    let offset = sums[wid.x];
    if (i0 < params.count) { data[i0] += offset; }
    if (i1 < params.count) { data[i1] += offset; }
    return;
  }
  let a = select(0u, data[i0], i0 < params.count);
  let b = select(0u, data[i1], i1 < params.count);
  partial[lid] = a + b;
  workgroupBarrier();
  for (var offset = 1u; offset < THREADS; offset <<= 1u) {
    let v = select(0u, partial[max(lid, offset) - offset], lid >= offset);
    workgroupBarrier();
    partial[lid] += v;
    workgroupBarrier();
  }
  let exclusive = partial[lid] - (a + b);
  if (i0 < params.count) { data[i0] = exclusive; }
  if (i1 < params.count) { data[i1] = exclusive + a; }
  if (lid == THREADS - 1u) { sums[wid.x] = partial[lid]; }
}

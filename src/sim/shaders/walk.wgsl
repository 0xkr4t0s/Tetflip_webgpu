// Jump-and-walk point location (mirrors `locate` in mesh/prepare.ts). Requires `tetGeom`
// and `tetNeighbors`. Steps across the face with the most negative barycentric coordinate.
fn locate(p: vec3f, start: u32) -> u32 {
  var t = start;
  for (var step = 0u; step < 64u; step++) {
    var b = baryAt(t, p);
    var k = 0u;
    for (var a = 1u; a < 4u; a++) {
      if (b[a] < b[k]) { k = a; }
    }
    if (b[k] >= -1e-5) { return t; }
    var nb = tetNeighbors[t];
    let next = nb[k];
    if (next < 0) { return t; }
    t = u32(next);
  }
  return t;
}

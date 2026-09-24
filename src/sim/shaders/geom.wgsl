// Tetrahedron geometry helpers. Requires a `tetGeom` binding: five vec4 per tet, the four
// barycentric planes followed by (volume, centroid.xyz).
fn loadPlanes(t: u32) -> array<vec4f, 4> {
  let o = t * 5u;
  return array<vec4f, 4>(tetGeom[o], tetGeom[o + 1u], tetGeom[o + 2u], tetGeom[o + 3u]);
}

fn tetVolume(t: u32) -> f32 { return tetGeom[t * 5u + 4u].x; }

fn baryAt(t: u32, p: vec3f) -> vec4f { return baryFrom(loadPlanes(t), p); }

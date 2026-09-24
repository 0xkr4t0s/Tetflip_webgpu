@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FullscreenOut { return fullscreen(vi); }

@fragment
fn fs(in: FullscreenOut) -> @location(0) vec4f {
  return vec4f(environment(cam.eye, worldRay(in.uv)), 1.0);
}

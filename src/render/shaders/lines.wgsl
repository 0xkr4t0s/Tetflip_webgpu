// Tank outline and the tetrahedral mesh slice (coloured by the solver's nodal pressure).
@group(0) @binding(1) var<storage, read> nodePos: array<vec4f>;
@group(0) @binding(2) var<storage, read> nodeState: array<vec4f>;
@group(0) @binding(3) var<storage, read> pressure: array<f32>;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec4f,
}

@vertex
fn vsMesh(@builtin(vertex_index) node: u32) -> VOut {
  var o: VOut;
  o.pos = cam.proj * cam.view * vec4f(nodePos[node].xyz, 1.0);
  let liquid = nodeState[node].x < 0.0;
  let p = clamp(pressure[node] * cam.params.z, 0.0, 1.0);
  let hot = mix(vec3f(0.15, 0.45, 1.0), vec3f(1.0, 0.45, 0.2), p);
  o.color = select(vec4f(0.75, 0.78, 0.82, 0.18), vec4f(hot, 0.9), liquid);
  return o;
}

const BOX = array<vec3f, 24>(
  vec3f(0, 0, 0), vec3f(1, 0, 0), vec3f(1, 0, 0), vec3f(1, 0, 1), vec3f(1, 0, 1), vec3f(0, 0, 1), vec3f(0, 0, 1), vec3f(0, 0, 0),
  vec3f(0, 1, 0), vec3f(1, 1, 0), vec3f(1, 1, 0), vec3f(1, 1, 1), vec3f(1, 1, 1), vec3f(0, 1, 1), vec3f(0, 1, 1), vec3f(0, 1, 0),
  vec3f(0, 0, 0), vec3f(0, 1, 0), vec3f(1, 0, 0), vec3f(1, 1, 0), vec3f(1, 0, 1), vec3f(1, 1, 1), vec3f(0, 0, 1), vec3f(0, 1, 1),
);

@vertex
fn vsBox(@builtin(vertex_index) vi: u32) -> VOut {
  var o: VOut;
  let p = mix(cam.boundsMin, cam.boundsMax, BOX[vi]);
  o.pos = cam.proj * cam.view * vec4f(p, 1.0);
  o.color = vec4f(0.9, 0.93, 1.0, 0.55);
  return o;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f { return in.color; }

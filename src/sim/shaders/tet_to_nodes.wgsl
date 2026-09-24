// Per node: volume-weighted averages of the projected velocity and of its change this step.
// nodeVel[2i] = velocity, nodeVel[2i + 1] = FLIP delta.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> nodeOffsets: array<vec2u>;
@group(0) @binding(2) var<storage, read> nodeTetInfo: array<vec2u>;
@group(0) @binding(3) var<storage, read> tetGeom: array<vec4f>;
@group(0) @binding(4) var<storage, read> tetVel: array<vec4f>;
@group(0) @binding(5) var<storage, read> tetVelOld: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> nodeVel: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= sim.nodeCount) { return; }
  var v = vec3f(0.0);
  var d = vec3f(0.0);
  var vol = 0.0;
  for (var s = nodeOffsets[i].x; s < nodeOffsets[i + 1u].x; s++) {
    let t = nodeTetInfo[s].x >> 2u;
    let V = tetVolume(t);
    let u = tetVel[t].xyz;
    v += V * u;
    d += V * (u - tetVelOld[t].xyz);
    vol += V;
  }
  nodeVel[2u * i] = vec4f(v / vol, 0.0);
  nodeVel[2u * i + 1u] = vec4f(d / vol, 0.0);
}

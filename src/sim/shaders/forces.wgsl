// Saves the pre-force velocity (for the FLIP update) and applies gravity and the
// interactive force brush.
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> tetVel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> tetVelOld: array<vec4f>;
@group(0) @binding(3) var<storage, read> tetGeom: array<vec4f>;

@compute @workgroup_size(WG)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= sim.tetCount) { return; }
  var u = tetVel[t];
  tetVelOld[t] = u;
  var v = u.xyz + sim.gravity * sim.dt;
  if (sim.forceStrength > 0.0) {
    let d = distance(tetGeom[t * 5u + 4u].yzw, sim.forcePos);
    if (d < sim.forceRadius) {
      let k = clamp(sim.forceStrength * (1.0 - d / sim.forceRadius) * sim.dt, 0.0, 1.0);
      v = mix(v, sim.forceVel, k);
    }
  }
  tetVel[t] = vec4f(v, u.w);
}

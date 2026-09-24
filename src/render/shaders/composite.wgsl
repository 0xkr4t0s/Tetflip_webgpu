// Shades the smoothed fluid depth as a refractive, absorbing liquid over the background.
@group(0) @binding(1) var depthTex: texture_2d<f32>;
@group(0) @binding(2) var thicknessTex: texture_2d<f32>;
@group(0) @binding(3) var sceneTex: texture_2d<f32>;
@group(0) @binding(4) var linearSampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FullscreenOut { return fullscreen(vi); }

fn loadViewPos(px: vec2i) -> vec3f {
  let dims = vec2i(textureDimensions(depthTex));
  let q = clamp(px, vec2i(0), dims - vec2i(1));
  let d = textureLoad(depthTex, q, 0).x;
  return viewPosFromDepth((vec2f(q) + 0.5) / vec2f(dims), d);
}

@fragment
fn fs(in: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let sample = textureLoad(depthTex, px, 0);
  let background = textureSampleLevel(sceneTex, linearSampler, in.uv, 0.0);
  if (sample.x <= 0.0) { return vec4f(toDisplay(background.rgb), 1.0); }

  // Normal from the depth buffer, using the one-sided difference with the smaller jump.
  let p = viewPosFromDepth(in.uv, sample.x);
  let px0 = loadViewPos(px - vec2i(1, 0));
  let px1 = loadViewPos(px + vec2i(1, 0));
  let py0 = loadViewPos(px - vec2i(0, 1));
  let py1 = loadViewPos(px + vec2i(0, 1));
  var ddx = px1 - p;
  if (abs(p.z - px0.z) < abs(px1.z - p.z) || textureLoad(depthTex, px + vec2i(1, 0), 0).x <= 0.0) { ddx = p - px0; }
  var ddy = py1 - p;
  if (abs(p.z - py0.z) < abs(py1.z - p.z) || textureLoad(depthTex, px + vec2i(0, 1), 0).x <= 0.0) { ddy = p - py0; }
  var n = normalize(cross(ddy, ddx));
  if (n.z < 0.0) { n = -n; }

  let thickness = textureSampleLevel(thicknessTex, linearSampler, in.uv, 0.0).x;
  let v = normalize(-p);
  let nWorld = normalize((cam.invView * vec4f(n, 0.0)).xyz);
  let vWorld = normalize((cam.invView * vec4f(v, 0.0)).xyz);
  let pWorld = (cam.invView * vec4f(p, 1.0)).xyz;

  // Refraction: offset the background lookup along the normal, attenuated by Beer–Lambert.
  let offset = n.xy * vec2f(1.0, -1.0) * min(thickness, 0.4) * 0.08;
  let refracted = textureSampleLevel(sceneTex, linearSampler, in.uv + offset, 0.0).rgb;
  let absorb = (vec3f(1.0) - cam.fluidColor) * cam.absorption;
  // Beer–Lambert transmission plus a little in-scattered light so thick liquid is not black.
  let transmitted = refracted * exp(-absorb * thickness) + cam.fluidColor * 0.08 * (1.0 - exp(-thickness * 3.0));

  let r = reflect(-vWorld, nWorld);
  let reflected = environment(pWorld, r);
  let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
  let h = normalize(cam.lightDir + vWorld);
  let spec = pow(max(dot(nWorld, h), 0.0), 350.0) * 3.0;

  var color = mix(transmitted, reflected, fresnel) + vec3f(spec);
  // Fast-moving liquid picks up a hint of white water.
  let foam = smoothstep(0.55, 1.1, sample.y * cam.params.y);
  color = mix(color, vec3f(0.92, 0.95, 1.0), foam * 0.45);
  return vec4f(toDisplay(color), 1.0);
}

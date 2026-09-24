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

fn tangent(px: vec2i, p: vec3f, step: vec2i) -> vec3f {
  let dims = vec2i(textureDimensions(depthTex));
  let validM = textureLoad(depthTex, clamp(px - step, vec2i(0), dims - vec2i(1)), 0).x > 0.0;
  let validP = textureLoad(depthTex, clamp(px + step, vec2i(0), dims - vec2i(1)), 0).x > 0.0;
  let back = p - loadViewPos(px - step);
  let fwd = loadViewPos(px + step) - p;
  if (!validM) { return fwd; }
  if (!validP) { return back; }
  let disagreement = abs(back.z - fwd.z) / (max(abs(back.z), abs(fwd.z)) + 1e-3 * cam.params.x);
  let gentle = select(fwd, back, abs(back.z) < abs(fwd.z));
  return mix(0.5 * (back + fwd), gentle, smoothstep(0.4, 0.9, disagreement));
}

// View-space normal at a pixel, or zero for background pixels.
fn normalAt(px: vec2i) -> vec3f {
  let dims = vec2i(textureDimensions(depthTex));
  let q = clamp(px, vec2i(0), dims - vec2i(1));
  let d = textureLoad(depthTex, q, 0).x;
  if (d <= 0.0) { return vec3f(0.0); }
  let p = viewPosFromDepth((vec2f(q) + 0.5) / vec2f(dims), d);
  var n = normalize(cross(tangent(q, p, vec2i(0, 1)), tangent(q, p, vec2i(1, 0))));
  if (n.z < 0.0) { n = -n; }
  return n;
}

@fragment
fn fs(in: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let sample = textureLoad(depthTex, px, 0);
  let background = textureSampleLevel(sceneTex, linearSampler, in.uv, 0.0);
  if (sample.x <= 0.0) { return vec4f(toDisplay(background.rgb), 1.0); }

  // Normal from the depth buffer: central differences on smooth regions, the gentler one-sided
  // difference across silhouettes, blended smoothly so neighbouring pixels agree.
  // A small cross-shaped average removes pixel-scale normal noise that would otherwise sparkle.
  let p = viewPosFromDepth(in.uv, sample.x);
  var nsum = 2.0 * normalAt(px);
  nsum += normalAt(px + vec2i(1, 0)) + normalAt(px - vec2i(1, 0));
  nsum += normalAt(px + vec2i(0, 1)) + normalAt(px - vec2i(0, 1));
  let n = normalize(nsum);

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
  let spec = pow(max(dot(nWorld, h), 0.0), 160.0) * 1.2;

  var color = mix(transmitted, reflected, fresnel) + vec3f(spec);
  // Fast-moving liquid picks up a hint of white water.
  let foam = smoothstep(0.7, 1.3, sample.y * cam.params.y);
  color = mix(color, vec3f(0.9, 0.94, 1.0), foam * 0.3);
  return vec4f(toDisplay(color), 1.0);
}

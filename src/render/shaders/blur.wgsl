// Separable bilateral filter of the fluid depth: smooths the sphere bumps into a surface while
// preserving silhouettes. DIRECTION 0 = horizontal, 1 = vertical.
override DIRECTION: u32 = 0u;

@group(0) @binding(1) var src: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FullscreenOut { return fullscreen(vi); }

@fragment
fn fsDepth(in: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let center = textureLoad(src, px, 0);
  let d = center.x;
  if (d <= 0.0) { return center; }
  let radiusWorld = cam.params.x;
  // Projected particle radius in pixels sets the kernel width.
  let radiusPx = radiusWorld * cam.proj[1][1] * 0.5 * cam.resolution.y / d;
  let r = i32(clamp(radiusPx * 4.0, 1.0, 32.0));
  let sigma = f32(r) * 0.5 + 0.5;
  let rangeSigma = radiusWorld * 1.5;
  let step = select(vec2i(0, 1), vec2i(1, 0), DIRECTION == 0u);
  let dims = vec2i(textureDimensions(src));

  // Local depth slope along the filter direction, so taps on a surface seen at a grazing angle
  // are compared against the plane through the centre rather than rejected as edges.
  // The baseline spans about a particle so particle-scale bumps don't dominate the estimate.
  let baseline = max(2, r / 3);
  let dm = textureLoad(src, clamp(px - step * baseline, vec2i(0), dims - vec2i(1)), 0).x;
  let dp = textureLoad(src, clamp(px + step * baseline, vec2i(0), dims - vec2i(1)), 0).x;
  var slope = 0.0;
  if (dm > 0.0 && dp > 0.0) {
    // On a plane the one-sided slopes agree; across a silhouette they do not, and then the
    // gentler side is the surface this pixel belongs to.
    let sl = (d - dm) / f32(baseline);
    let sr = (dp - d) / f32(baseline);
    let disagreement = abs(sl - sr) / (max(abs(sl), abs(sr)) + 0.1 * radiusWorld);
    let gentle = select(sr, sl, abs(sl) < abs(sr));
    slope = mix(0.5 * (sl + sr), gentle, smoothstep(0.3, 0.8, disagreement));
  }

  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -r; i <= r; i++) {
    let q = clamp(px + step * i, vec2i(0), dims - vec2i(1));
    let di = textureLoad(src, q, 0).x;
    if (di <= 0.0) { continue; }
    let fi = f32(i);
    let expected = d + slope * fi;
    let dd = (di - expected) / rangeSigma;
    let w = exp(-fi * fi / (2.0 * sigma * sigma) - dd * dd);
    // Average the offsets from the plane so the slope itself is preserved.
    sum += (di - slope * fi) * w;
    wsum += w;
  }
  return vec4f(sum / wsum, center.y, 0.0, 1.0);
}

@fragment
fn fsGaussian(in: FullscreenOut) -> @location(0) vec4f {
  let px = vec2i(in.pos.xy);
  let step = select(vec2i(0, 1), vec2i(1, 0), DIRECTION == 0u);
  let dims = vec2i(textureDimensions(src));
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -8; i <= 8; i++) {
    let q = clamp(px + step * i, vec2i(0), dims - vec2i(1));
    let fi = f32(i);
    let w = exp(-fi * fi / 18.0);
    sum += textureLoad(src, q, 0).x * w;
    wsum += w;
  }
  return vec4f(sum / wsum, 0.0, 0.0, 1.0);
}

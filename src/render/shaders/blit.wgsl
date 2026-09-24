@group(0) @binding(1) var sceneTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FullscreenOut { return fullscreen(vi); }

@fragment
fn fs(in: FullscreenOut) -> @location(0) vec4f { return vec4f(toDisplay(textureLoad(sceneTex, vec2i(in.pos.xy), 0).rgb), 1.0); }

#version 420

// Source 0: quad corner (±2, ±2) — per-vertex
layout(location = 0) in vec2 vertex;
// Source 1: sorted splat index — per-instance (divisor = 1)
layout(location = 8) in float uv0;

// Ogre 1.12 parses uniforms with a source scan that treats an unknown
// sampler type as a uniform block and then skips every uniform up to the
// next brace. Keep the sampler declarations last so the matrices survive
// that scan. Humble's GLSL parser does not register usamplerBuffer.
uniform mat4          world_view_matrix;
uniform mat4          projectionMatrix;
uniform vec3          camera_position_in_splat_frame;
uniform int           sh_degree;
uniform float         u_min_peak_opacity;
uniform float         u_alpha_threshold;

// ROI clip box (axis-aligned, in the scene_node_'s local frame which is the
// Reference Frame's coordinates on the host side).
uniform int           u_clip_enabled;
uniform vec3          u_clip_min;
uniform vec3          u_clip_max;

// Splat-tight forward-z warp bounds used by the WBOIT weight function.
// Far = -1 signals "unset" → fragment shaders fall back to gl_FragCoord.z.
uniform float         u_splat_z_near;
uniform float         u_splat_z_far;

// Sampler types stay last. Ogre 1.12's GLSL scanner does not know
// usamplerBuffer and stops at the first one. layout(binding) pins the
// units, because that scanner also never records these two uniforms.
// Compact base TBO: 2 uvec4 texels per splat (32 B).
layout(binding = 0) uniform usamplerBuffer u_splats;
// Optional SH TBO, sampled only when sh_degree > 0.
layout(binding = 1) uniform samplerBuffer u_sh;

out vec4  vColor;
out vec2  vPosition;
out float v_splat_z_warped;     // splat-center view-z warped into [0,1]

// ── SH constants (Y_l^m coefficients) ──────────────────────────────────────
const float SH_C1    =  0.4886025119029199;
const float SH_C2_0  =  1.0925484305920792;
const float SH_C2_1  = -1.0925484305920792;
const float SH_C2_2  =  0.31539156525252005;
const float SH_C2_3  = -1.0925484305920792;
const float SH_C2_4  =  0.5462742152960396;
const float SH_C3_0  = -0.5900435899266435;
const float SH_C3_1  =  2.890611442640554;
const float SH_C3_2  = -0.4570457994644658;
const float SH_C3_3  =  0.3731763325901154;
const float SH_C3_4  = -0.4570457994644658;
const float SH_C3_5  =  1.445305721320277;
const float SH_C3_6  = -0.5900435899266435;

// Number of non-DC SH coefficients for a given degree.
int shCoeffsNonDC(int d) { return (d + 1) * (d + 1) - 1; }

vec3 fetchSH(int base, int k) { return texelFetch(u_sh, base + k).rgb; }

// Evaluate SH orders 1..sh_degree at direction d, returning RGB increment
// to add to the DC base colour.
vec3 evalSHOrders1Plus(int splat_id, vec3 d)
{
    int base = splat_id * shCoeffsNonDC(sh_degree);
    vec3 rgb = vec3(0.0);

    // Order 1 (3 coeffs)
    rgb += -SH_C1 * d.y * fetchSH(base, 0)
         +  SH_C1 * d.z * fetchSH(base, 1)
         + -SH_C1 * d.x * fetchSH(base, 2);

    if (sh_degree >= 2) {
        float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
        float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;
        rgb += SH_C2_0 * xy               * fetchSH(base, 3)
             + SH_C2_1 * yz               * fetchSH(base, 4)
             + SH_C2_2 * (2.0*zz-xx-yy)   * fetchSH(base, 5)
             + SH_C2_3 * xz               * fetchSH(base, 6)
             + SH_C2_4 * (xx - yy)        * fetchSH(base, 7);

        if (sh_degree >= 3) {
            rgb += SH_C3_0 * d.y * (3.0*xx - yy)              * fetchSH(base, 8)
                 + SH_C3_1 * d.z * xy                          * fetchSH(base, 9)
                 + SH_C3_2 * d.y * (4.0*zz - xx - yy)          * fetchSH(base, 10)
                 + SH_C3_3 * d.z * (2.0*zz - 3.0*xx - 3.0*yy)  * fetchSH(base, 11)
                 + SH_C3_4 * d.x * (4.0*zz - xx - yy)          * fetchSH(base, 12)
                 + SH_C3_5 * d.z * (xx - yy)                   * fetchSH(base, 13)
                 + SH_C3_6 * d.x * (xx - 3.0*yy)               * fetchSH(base, 14);
        }
    }
    return rgb;
}

void main()
{
    int splat_id = int(uv0);
    int base     = splat_id * 2;

    uvec4 t0 = texelFetch(u_splats, base + 0);

    vec3 center = uintBitsToFloat(t0.xyz);

    // ROI clip: reject splats whose centre is outside [Clip Min, Clip Max].
    // z = 2 places the quad past the far plane → rasteriser discards.
    if (u_clip_enabled != 0) {
        if (any(lessThan(center, u_clip_min)) ||
            any(greaterThan(center, u_clip_max))) {
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }
    }

    // ── Project center ────────────────────────────────────────────────────────
    vec4 camspace = world_view_matrix * vec4(center, 1.0);
    vec4 pos2d    = projectionMatrix * camspace;

    float bounds = 1.2 * pos2d.w;
    if (pos2d.z < -pos2d.w || pos2d.z > pos2d.w
        || pos2d.x < -bounds || pos2d.x > bounds
        || pos2d.y < -bounds || pos2d.y > bounds) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 t1 = texelFetch(u_splats, base + 1);
    vec4 rgba = unpackUnorm4x8(t1.z);
    if (rgba.a < max(u_min_peak_opacity, u_alpha_threshold)) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec2 c0001 = unpackHalf2x16(t0.w);
    vec2 c0211 = unpackHalf2x16(t1.x);
    vec2 c1222 = unpackHalf2x16(t1.y);
    float c00 = c0001.x, c01 = c0001.y;
    float c02 = c0211.x, c11 = c0211.y;
    float c12 = c1222.x, c22 = c1222.y;

    // ── EWA: 3D covariance → 2D screen covariance ─────────────────────────────
    mat3 J = mat3(
        -projectionMatrix[0][0] / camspace.z, 0.0,
        (projectionMatrix[0][0] * camspace.x) / (camspace.z * camspace.z),
        0.0, -projectionMatrix[1][1] / camspace.z,
        (projectionMatrix[1][1] * camspace.y) / (camspace.z * camspace.z),
        0.0, 0.0, 0.0
    );

    mat3 cov3d = mat3(
        c00, c01, c02,
        c01, c11, c12,
        c02, c12, c22
    );

    mat3 R   = transpose(mat3(world_view_matrix));
    mat3 T   = R * J;
    mat3 cov = transpose(T) * cov3d * T;

    vec3 vCenter = vec3(pos2d) / pos2d.w;

    float d1 = cov[0][0], od = cov[0][1], d2 = cov[1][1];
    float mid     = 0.5 * (d1 + d2);
    float radius  = length(vec2((d1 - d2) * 0.5, od));
    float lambda1 = mid + radius;
    float lambda2 = mid - radius;
    vec2 diag = od == 0.0 ? (d1 >= d2 ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
                         : normalize(vec2(od, lambda1 - d1));
    vec2 v1   = sqrt(max(2. * lambda1, 0.0)) * diag;
    vec2 v2   = sqrt(max(2. * lambda2, 0.0)) * vec2(diag.y, -diag.x);

    // ── Colour: uint8 DC base, plus optional SH1..3 contribution ─────────────
    vec3 rgb = rgba.rgb;
    if (sh_degree > 0) {
        vec3 direction = normalize(center - camera_position_in_splat_frame);
        rgb = clamp(rgb + evalSHOrders1Plus(splat_id, direction), 0.0, 1.0);
    }
    vColor = vec4(rgb, rgba.a);
    // The fragment shader rejects exp(-r²) * opacity below its cutoff.
    // Bound the quad by that radius so faint splats shade fewer pixels.
    float radius_limit = 2.0;
    if (u_alpha_threshold > 0.0) {
        radius_limit = min(radius_limit, sqrt(log(rgba.a / u_alpha_threshold)) + 0.01);
    }
    vPosition = vertex * (0.5 * radius_limit);

    // Splat-tight warped depth in [0,1]; sentinel -1 if bounds unset.  Ogre
    // view space looks down -Z, so forward distance is -camspace.z.  All four
    // quad vertices share this value, so it interpolates to a constant per
    // splat — exactly the per-splat depth WBOIT's weight function wants.
    float z_span = u_splat_z_far - u_splat_z_near;
    v_splat_z_warped = (z_span > 1e-3)
        ? clamp((-camspace.z - u_splat_z_near) / z_span, 0.0, 1.0)
        : -1.0;

    gl_Position = vec4(
        vCenter.xy + vPosition.x * v1 + vPosition.y * v2,
        vCenter.z, 1.0);
}

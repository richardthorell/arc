#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform sky_constants
{
    vec4 sun_direction_exposure;
    vec4 sky_tint_rayleigh;
    vec4 sky_params;
} constants;

void main()
{
    vec2 uv = in_uv * 2.0 - vec2(1.0);
    float view_y = clamp(uv.y, -1.0, 1.0);
    vec3 sun_dir = normalize(constants.sun_direction_exposure.xyz);
    float exposure = max(constants.sun_direction_exposure.w, 0.0);
    vec3 tint = constants.sky_tint_rayleigh.rgb;
    float rayleigh = max(constants.sky_tint_rayleigh.w, 0.0);
    float mie = max(constants.sky_params.x, 0.0);
    float ozone = max(constants.sky_params.y, 0.0);
    float sun_size = max(constants.sky_params.z, 0.001);
    float sun_intensity = max(constants.sky_params.w, 0.0);

    float horizon = 1.0 - abs(view_y);
    vec3 zenith_color = tint * (0.38 + 0.42 * rayleigh);
    vec3 horizon_color = mix(vec3(0.72, 0.80, 0.88), tint, 0.35) * (0.85 + 0.15 * mie);
    vec3 sky = mix(horizon_color, zenith_color, smoothstep(-0.15, 0.95, view_y));

    float sun_height = clamp(-sun_dir.y * 0.5 + 0.5, 0.0, 1.0);
    vec2 sun_screen = normalize(vec2(sun_dir.x, -sun_dir.y) + vec2(0.0001)) * (0.35 + 0.45 * sun_height);
    float sun_distance = length(uv - sun_screen);
    float sun_disk = smoothstep(sun_size * 1.8, sun_size, sun_distance);
    float sun_glow = exp(-sun_distance * (5.5 / max(mie, 0.05))) * 0.28 * sun_intensity;
    vec3 sun_color = mix(vec3(1.0, 0.66, 0.38), vec3(1.0, 0.96, 0.82), sun_height);

    sky += sun_color * (sun_disk * sun_intensity + sun_glow);
    sky = mix(sky, sky * vec3(0.92, 0.96, 1.04), clamp(ozone, 0.0, 1.0) * 0.18);
    sky = vec3(1.0) - exp(-sky * max(exposure, 0.001));
    out_color = vec4(sky, 1.0);
}

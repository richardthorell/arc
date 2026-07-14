#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D sky_hdri;

layout(push_constant) uniform sky_constants
{
    vec4 camera_forward_fov;
    vec4 camera_up_aspect;
    vec4 sun_direction_intensity;
    vec4 moon_direction_phase;
    vec4 sky_color_source;
    vec4 atmosphere;
    vec4 cumulus;
    vec4 cirrus;
} constants;

const float PI = 3.14159265359;

float hash13(vec3 value)
{
    value = fract(value * 0.1031);
    value += dot(value, value.yzx + 33.33);
    return fract((value.x + value.y) * value.z);
}

float value_noise(vec2 value)
{
    vec2 cell = floor(value);
    vec2 local = fract(value);
    local = local * local * (3.0 - 2.0 * local);
    float a = hash13(vec3(cell, 11.0));
    float b = hash13(vec3(cell + vec2(1.0, 0.0), 11.0));
    float c = hash13(vec3(cell + vec2(0.0, 1.0), 11.0));
    float d = hash13(vec3(cell + vec2(1.0), 11.0));
    return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}

float fbm(vec2 value, int octaves)
{
    float result = 0.0;
    float amplitude = 0.55;
    for (int octave = 0; octave < 4; ++octave)
    {
        if (octave >= octaves)
            break;
        result += value_noise(value) * amplitude;
        value = value * 2.03 + vec2(17.1, 9.2);
        amplitude *= 0.48;
    }
    return result;
}

float cloud_layer(vec3 ray, vec4 settings, float frequency, int octaves)
{
    if (settings.x <= 0.0001 || ray.y <= -0.08)
        return 0.0;
    float horizon_fade = smoothstep(-0.06, 0.18, ray.y);
    vec2 projected = ray.xz / max(ray.y + 0.32, 0.08);
    float shape = fbm(projected * frequency + settings.zw, octaves);
    float threshold = 1.0 - settings.x * 0.78;
    return smoothstep(threshold - 0.13, threshold + 0.08, shape) * settings.y * horizon_fade;
}

vec3 physical_sky(vec3 ray, vec3 toward_sun)
{
    float rayleigh_strength = max(constants.atmosphere.x, 0.0);
    float mie_strength = max(constants.atmosphere.y, 0.0);
    float ozone = max(constants.atmosphere.z, 0.0);
    float height = clamp(ray.y * 0.5 + 0.5, 0.0, 1.0);
    float horizon = exp(-max(ray.y, 0.0) * 5.0);
    float sun_height = clamp(toward_sun.y * 0.5 + 0.5, 0.0, 1.0);
    float twilight = smoothstep(-0.12, 0.12, toward_sun.y);

    vec3 zenith = constants.sky_color_source.rgb * (0.16 + 0.48 * rayleigh_strength);
    vec3 horizon_day = mix(vec3(0.70, 0.79, 0.91), constants.sky_color_source.rgb, 0.32);
    vec3 sunset = vec3(1.0, 0.22 + 0.35 * sun_height, 0.055);
    vec3 horizon_color = mix(sunset, horizon_day, smoothstep(-0.03, 0.25, toward_sun.y));
    vec3 sky = mix(zenith, horizon_color, horizon);

    float mu = clamp(dot(ray, toward_sun), -1.0, 1.0);
    float rayleigh_phase = 0.0596831 * (1.0 + mu * mu);
    float g = clamp(0.72 + mie_strength * 0.12, 0.0, 0.92);
    float mie_phase = (1.0 - g * g) /
        max(4.0 * PI * pow(1.0 + g * g - 2.0 * g * mu, 1.5), 0.001);
    vec3 scatter = vec3(0.36, 0.58, 1.0) * rayleigh_phase * rayleigh_strength +
        vec3(1.0, 0.62, 0.31) * mie_phase * mie_strength;
    sky += scatter * (0.18 + 0.82 * twilight);
    sky *= mix(vec3(0.91, 0.96, 1.06), vec3(1.0), 1.0 - clamp(ozone, 0.0, 1.0) * 0.18);
    sky *= mix(0.025, 1.0, smoothstep(-0.16, 0.05, toward_sun.y));
    return max(sky, vec3(0.0));
}

vec2 equirectangular_uv(vec3 ray, float rotation)
{
    float longitude = atan(ray.z, ray.x) + rotation;
    float latitude = asin(clamp(ray.y, -1.0, 1.0));
    return vec2(fract(longitude / (2.0 * PI) + 0.5), 0.5 - latitude / PI);
}

void main()
{
    vec2 ndc = in_uv * 2.0 - 1.0;
    vec3 forward = normalize(constants.camera_forward_fov.xyz);
    vec3 up = normalize(constants.camera_up_aspect.xyz);
    vec3 right = normalize(cross(forward, up));
    vec3 ray = normalize(
        forward +
        right * ndc.x * constants.camera_forward_fov.w * constants.camera_up_aspect.w +
        up * ndc.y * constants.camera_forward_fov.w);
    vec3 toward_sun = normalize(-constants.sun_direction_intensity.xyz);
    vec3 toward_moon = normalize(-constants.moon_direction_phase.xyz);

    uint sky_settings = floatBitsToUint(constants.sky_color_source.w);
    int source = int(sky_settings & 3u);
    float star_density = float((sky_settings >> 2u) & 1023u) / 1023.0;
    float star_intensity = float((sky_settings >> 12u) & 4095u) / 4095.0 * 16.0;
    uint sun_settings = floatBitsToUint(constants.sun_direction_intensity.w);
    float sun_intensity = float(sun_settings & 65535u) / 65535.0 * 32.0;
    float sun_radius = radians(float(sun_settings >> 16u) / 65535.0 * 5.0);
    uint moon_settings = floatBitsToUint(constants.moon_direction_phase.w);
    float moon_phase = float(moon_settings & 1023u) / 1023.0;
    float moon_intensity = float((moon_settings >> 10u) & 1023u) / 1023.0 * 8.0;
    float moon_radius = radians(float((moon_settings >> 20u) & 1023u) / 1023.0 * 5.0);
    bool moon_enabled = (moon_settings & (1u << 30u)) != 0u;
    vec3 sky;
    if (source == 1)
        sky = texture(sky_hdri, equirectangular_uv(ray, constants.atmosphere.x)).rgb;
    else if (source == 2)
        sky = constants.sky_color_source.rgb;
    else
        sky = physical_sky(ray, toward_sun);

    float night = smoothstep(0.04, -0.14, toward_sun.y);
    if (night > 0.0 && star_density > 0.0)
    {
        vec3 star_cell = floor(ray * 720.0);
        float seed = hash13(star_cell);
        float star = smoothstep(1.0 - star_density * 0.018, 1.0, seed);
        float brightness = pow(hash13(star_cell + 19.0), 5.0) * 3.2 + 0.18;
        sky += vec3(0.72, 0.82, 1.0) * star * brightness * star_intensity * night;
    }

    float moon_distance = acos(clamp(dot(ray, toward_moon), -1.0, 1.0));
    float moon_disk = 1.0 - smoothstep(moon_radius * 0.82, moon_radius * 1.12, moon_distance);
    if (moon_enabled && moon_disk > 0.0 && night > 0.0)
    {
        vec3 tangent = normalize(cross(abs(toward_moon.y) > 0.95 ? vec3(1, 0, 0) : vec3(0, 1, 0), toward_moon));
        float phase_coordinate = dot(ray - toward_moon, tangent) / max(moon_radius, 0.0001);
        float terminator = moon_phase * 2.0 - 1.0;
        float lit = smoothstep(terminator - 0.08, terminator + 0.08, phase_coordinate);
        sky += vec3(0.78, 0.84, 0.92) * moon_disk * (0.08 + 0.92 * lit) * moon_intensity * night;
    }

    float cumulus = cloud_layer(ray, constants.cumulus, 1.35, 4);
    float cirrus = cloud_layer(ray, constants.cirrus, 3.8, 2) * 0.62;
    float cloud = clamp(cumulus + cirrus * (1.0 - cumulus), 0.0, 1.0);
    float sun_alignment = pow(max(dot(ray, toward_sun), 0.0), 12.0);
    vec3 cloud_color = mix(vec3(0.46, 0.50, 0.56), vec3(1.0, 0.96, 0.88),
        clamp(0.35 + toward_sun.y * 0.5 + sun_alignment, 0.0, 1.0));
    sky = mix(sky, cloud_color, cloud);

    float sun_distance = acos(clamp(dot(ray, toward_sun), -1.0, 1.0));
    float sun_disk = 1.0 - smoothstep(sun_radius * 0.82, sun_radius * 1.2, sun_distance);
    float sun_glow = exp(-sun_distance * 34.0) * 0.22;
    vec3 sun_color = mix(vec3(1.0, 0.42, 0.12), vec3(1.0, 0.96, 0.82),
        smoothstep(-0.04, 0.25, toward_sun.y));
    sky += sun_color * (sun_disk * sun_intensity + sun_glow * min(sun_intensity, 2.0)) * (1.0 - cloud);
    sky *= max(constants.atmosphere.w, 0.0);

    // The viewport target is currently SDR; this is the sole sky output transform.
    sky = clamp((sky * (2.51 * sky + 0.03)) / (sky * (2.43 * sky + 0.59) + 0.14), 0.0, 1.0);
    out_color = vec4(sky, 1.0);
}

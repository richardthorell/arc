#include <arc/editor/material_preview.h>
#include <arc/editor/material_preview_realizer.h>

#include <nlohmann/json.hpp>

#if defined(ARC_RENDER_HAS_STB)
#include <stb_image.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::editor
{
namespace
{

using color3 = std::array<float, 3>;

constexpr float preview_sphere_radius = 0.82f;
constexpr float minimum_roughness = 0.045f;
constexpr float dielectric_reflectance = 0.04f;
constexpr float preview_exposure = 1.15f;
constexpr std::uint32_t preview_texture_max_size = 512u;

color3 add(color3 a, color3 b) noexcept
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
color3 mul(color3 a, color3 b) noexcept
{
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2]};
}
color3 mul(color3 value, float scale) noexcept
{
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}
float dot(color3 a, color3 b) noexcept
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
color3 cross(color3 a, color3 b) noexcept
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
color3 normalize(color3 value) noexcept
{
    const float length = std::sqrt(std::max(dot(value, value), 1.0e-12f));
    return mul(value, 1.0f / length);
}
color3 mix(color3 a, color3 b, float amount) noexcept
{
    return add(mul(a, 1.0f - amount), mul(b, amount));
}

float linear_from_srgb(float value) noexcept
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float srgb_from_linear(float value) noexcept
{
    value = std::max(value, 0.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

float aces_tonemap(float value) noexcept
{
    value *= preview_exposure;
    return std::clamp((value * (2.51f * value + 0.03f)) / (value * (2.43f * value + 0.59f) + 0.14f), 0.0f, 1.0f);
}

struct sampled_texture
{
    render::texture_data data;
    bool valid{};
};

using preview_texture_cache = std::map<std::filesystem::path, sampled_texture>;

struct graph_texture_binding
{
    std::string path;
    std::string pin;
};

struct graph_texture_bindings
{
    graph_texture_binding base_color;
    graph_texture_binding metallic;
    graph_texture_binding roughness;
    graph_texture_binding normal;
    graph_texture_binding ambient_occlusion;
    graph_texture_binding emissive;
    graph_texture_binding opacity;
};

struct loaded_graph_texture_binding
{
    sampled_texture texture;
    std::string pin;
};

struct loaded_graph_texture_bindings
{
    loaded_graph_texture_binding base_color;
    loaded_graph_texture_binding metallic;
    loaded_graph_texture_binding roughness;
    loaded_graph_texture_binding normal;
    loaded_graph_texture_binding ambient_occlusion;
    loaded_graph_texture_binding emissive;
    loaded_graph_texture_binding opacity;
};

sampled_texture load_preview_texture_uncached(const std::filesystem::path& path)
{
    if (path.empty()) return {};

#if defined(ARC_RENDER_HAS_STB)
    const auto extension = path.extension().string();
    if (extension != ".dds" && extension != ".DDS" && extension != ".hdr" && extension != ".HDR")
    {
        std::ifstream stream(path, std::ios::binary);
        if (stream)
        {
            stream.seekg(0, std::ios::end);
            const auto encoded_size = stream.tellg();
            stream.seekg(0, std::ios::beg);
            if (encoded_size > 0 && encoded_size <= static_cast<std::streamoff>(std::numeric_limits<int>::max()))
            {
                std::vector<stbi_uc> encoded(static_cast<std::size_t>(encoded_size));
                stream.read(reinterpret_cast<char*>(encoded.data()), encoded_size);
                if (stream)
                {
                    int width{};
                    int height{};
                    int channels{};
                    stbi_uc* decoded = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width,
                                                             &height, &channels, STBI_rgb_alpha);
                    if (decoded && width > 0 && height > 0)
                    {
                        const auto source_width = static_cast<std::uint32_t>(width);
                        const auto source_height = static_cast<std::uint32_t>(height);
                        const auto source_max = std::max(source_width, source_height);
                        const float scale = source_max > preview_texture_max_size
                                                ? static_cast<float>(preview_texture_max_size) /
                                                      static_cast<float>(source_max)
                                                : 1.0f;
                        const auto target_width =
                            std::max(1u, static_cast<std::uint32_t>(std::lround(source_width * scale)));
                        const auto target_height =
                            std::max(1u, static_cast<std::uint32_t>(std::lround(source_height * scale)));

                        render::texture_data texture;
                        texture.name = path.filename().string();
                        texture.width = target_width;
                        texture.height = target_height;
                        texture.format = render::texture_format::rgba8_srgb;
                        texture.mip_levels = 1;
                        texture.pixels.resize(static_cast<std::size_t>(target_width) * target_height * 4u);

                        for (std::uint32_t y = 0; y < target_height; ++y)
                        {
                            const auto source_y = std::min(
                                source_height - 1u,
                                static_cast<std::uint32_t>((static_cast<std::uint64_t>(y) * source_height) /
                                                           target_height));
                            for (std::uint32_t x = 0; x < target_width; ++x)
                            {
                                const auto source_x = std::min(
                                    source_width - 1u,
                                    static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * source_width) /
                                                               target_width));
                                const auto source_offset =
                                    (static_cast<std::size_t>(source_y) * source_width + source_x) * 4u;
                                const auto target_offset =
                                    (static_cast<std::size_t>(y) * target_width + x) * 4u;
                                std::memcpy(texture.pixels.data() + target_offset, decoded + source_offset, 4u);
                            }
                        }
                        stbi_image_free(decoded);
                        return {std::move(texture), true};
                    }
                    if (decoded) stbi_image_free(decoded);
                }
            }
        }
    }
#endif

    const auto result = render::load_texture_asset(path);
    if (!result.succeeded() || !result.texture.has_pixels()) return {};
    const bool supported = result.texture.format == render::texture_format::rgba8_unorm ||
                           result.texture.format == render::texture_format::rgba8_srgb ||
                           result.texture.format == render::texture_format::rgba32f;
    return {result.texture, supported};
}

sampled_texture load_preview_texture(const std::filesystem::path& root, const std::string& path,
                                     preview_texture_cache& cache)
{
    if (path.empty()) return {};
    const auto resolved = resolve_material_texture_path(root, path).lexically_normal();
    if (const auto found = cache.find(resolved); found != cache.end()) return found->second;
    auto loaded = load_preview_texture_uncached(resolved);
    cache.emplace(resolved, loaded);
    return loaded;
}

std::array<float, 4> sample(const sampled_texture& texture, float u, float v, bool color_data)
{
    if (!texture.valid) return {1.0f, 1.0f, 1.0f, 1.0f};
    const auto& data = texture.data;
    u -= std::floor(u);
    v = std::clamp(v, 0.0f, 1.0f);
    const auto x = std::min(data.width - 1u, static_cast<std::uint32_t>(u * data.width));
    const auto y = std::min(data.height - 1u, static_cast<std::uint32_t>((1.0f - v) * data.height));
    const auto pixel = static_cast<std::size_t>(y * data.width + x);
    std::array<float, 4> result{};
    if (data.format == render::texture_format::rgba32f)
        std::memcpy(result.data(), data.pixels.data() + pixel * sizeof(float) * 4u, sizeof(result));
    else
    {
        const auto offset = pixel * 4u;
        for (std::size_t channel = 0; channel < 4u; ++channel)
            result[channel] = static_cast<float>(std::to_integer<std::uint8_t>(data.pixels[offset + channel])) / 255.0f;
    }
    if (color_data && data.format == render::texture_format::rgba8_srgb)
    {
        result[0] = linear_from_srgb(result[0]);
        result[1] = linear_from_srgb(result[1]);
        result[2] = linear_from_srgb(result[2]);
    }
    return result;
}

float sample_scalar(const std::array<float, 4>& value, std::string_view pin) noexcept
{
    if (pin == "g") return value[1];
    if (pin == "b") return value[2];
    if (pin == "a") return value[3];
    return value[0];
}

bool texture_sample_node(const nlohmann::json& node)
{
    if (!node.is_object()) return false;
    const auto type = node.value("type", "");
    return type == "textureSample" || type == "textureSample2D";
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

graph_texture_bindings load_graph_texture_bindings(const std::filesystem::path& material_path)
{
    graph_texture_bindings result;
    const auto source = read_text_file(material_path);
    if (source.empty()) return result;

    const auto document = nlohmann::json::parse(source, nullptr, false);
    if (document.is_discarded() || !document.contains("graph") || !document["graph"].is_object()) return result;
    const auto& graph = document["graph"];
    if (!graph.contains("nodes") || !graph["nodes"].is_array() || !graph.contains("connections") ||
        !graph["connections"].is_array())
        return result;

    std::map<std::string, const nlohmann::json*> nodes;
    std::string output_node;
    for (const auto& node : graph["nodes"])
    {
        if (!node.is_object()) continue;
        const auto id = node.value("id", "");
        if (id.empty()) continue;
        nodes.emplace(id, &node);
        if (node.value("type", "") == "output") output_node = id;
    }
    if (output_node.empty()) return result;

    std::map<std::pair<std::string, std::string>, std::pair<std::string, std::string>> inputs;
    for (const auto& connection : graph["connections"])
    {
        if (!connection.is_object()) continue;
        const auto from = connection.value("from", nlohmann::json::object());
        const auto to = connection.value("to", nlohmann::json::object());
        const auto source_node = from.value("nodeId", "");
        const auto source_pin = from.value("pin", "");
        const auto target_node = to.value("nodeId", "");
        const auto target_pin = to.value("pin", "");
        if (!source_node.empty() && !target_node.empty() && !target_pin.empty())
            inputs[{target_node, target_pin}] = {source_node, source_pin};
    }

    const auto resolve_texture = [&](std::string_view output_pin, bool allow_normal_map = false)
    {
        graph_texture_binding binding;
        const auto output_source = inputs.find({output_node, std::string(output_pin)});
        if (output_source == inputs.end()) return binding;

        auto source_node_id = output_source->second.first;
        auto source_pin = output_source->second.second;
        auto source_node = nodes.find(source_node_id);
        if (source_node == nodes.end()) return binding;

        if (allow_normal_map && (*source_node->second).value("type", "") == "normalMap")
        {
            const auto normal_input = inputs.find({source_node_id, "texture"});
            if (normal_input == inputs.end()) return binding;
            source_node_id = normal_input->second.first;
            source_pin = normal_input->second.second;
            source_node = nodes.find(source_node_id);
            if (source_node == nodes.end()) return binding;
        }

        if (!texture_sample_node(*source_node->second)) return binding;
        const auto values = source_node->second->value("values", nlohmann::json::object());
        binding.path = values.value("texture", "");
        binding.pin = source_pin;
        return binding;
    };

    result.base_color = resolve_texture("baseColor");
    result.metallic = resolve_texture("metallic");
    result.roughness = resolve_texture("roughness");
    result.normal = resolve_texture("normal", true);
    result.ambient_occlusion = resolve_texture("ao");
    result.emissive = resolve_texture("emissive");
    result.opacity = resolve_texture("opacity");
    return result;
}

loaded_graph_texture_binding load_graph_binding(const std::filesystem::path& root, const graph_texture_binding& binding,
                                                preview_texture_cache& cache)
{
    return {.texture = load_preview_texture(root, binding.path, cache), .pin = binding.pin};
}

loaded_graph_texture_bindings load_graph_bindings(const std::filesystem::path& root,
                                                  const graph_texture_bindings& bindings,
                                                  preview_texture_cache& cache)
{
    return {
        .base_color = load_graph_binding(root, bindings.base_color, cache),
        .metallic = load_graph_binding(root, bindings.metallic, cache),
        .roughness = load_graph_binding(root, bindings.roughness, cache),
        .normal = load_graph_binding(root, bindings.normal, cache),
        .ambient_occlusion = load_graph_binding(root, bindings.ambient_occlusion, cache),
        .emissive = load_graph_binding(root, bindings.emissive, cache),
        .opacity = load_graph_binding(root, bindings.opacity, cache),
    };
}

float distribution_ggx(float n_dot_h, float roughness) noexcept
{
    const float alpha = roughness * roughness;
    const float alpha2 = alpha * alpha;
    const float denominator = n_dot_h * n_dot_h * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / std::max(std::numbers::pi_v<float> * denominator * denominator, 1.0e-5f);
}

float geometry_schlick(float n_dot_direction, float roughness) noexcept
{
    const float k = ((roughness + 1.0f) * (roughness + 1.0f)) * 0.125f;
    return n_dot_direction / std::max(n_dot_direction * (1.0f - k) + k, 1.0e-5f);
}

color3 fresnel_schlick(float cos_theta, color3 f0) noexcept
{
    const float factor = std::pow(std::clamp(1.0f - cos_theta, 0.0f, 1.0f), 5.0f);
    return add(f0, mul({1.0f - f0[0], 1.0f - f0[1], 1.0f - f0[2]}, factor));
}

color3 evaluate_light(color3 normal, color3 view, color3 light, color3 radiance, color3 base_color, float metallic,
                      float roughness)
{
    const float n_dot_l = std::max(dot(normal, light), 0.0f);
    const float n_dot_v = std::max(dot(normal, view), 0.0f);
    if (n_dot_l <= 0.0f || n_dot_v <= 0.0f) return {};
    const auto half_vector = normalize(add(view, light));
    const float n_dot_h = std::max(dot(normal, half_vector), 0.0f);
    const float v_dot_h = std::max(dot(view, half_vector), 0.0f);
    const auto f0 = mix({dielectric_reflectance, dielectric_reflectance, dielectric_reflectance}, base_color, metallic);
    const auto fresnel = fresnel_schlick(v_dot_h, f0);
    const float distribution = distribution_ggx(n_dot_h, roughness);
    const float geometry = geometry_schlick(n_dot_l, roughness) * geometry_schlick(n_dot_v, roughness);
    const auto specular = mul(fresnel, distribution * geometry / std::max(4.0f * n_dot_l * n_dot_v, 1.0e-5f));
    const auto diffuse_weight = mul({1.0f - fresnel[0], 1.0f - fresnel[1], 1.0f - fresnel[2]}, 1.0f - metallic);
    const auto diffuse = mul(mul(diffuse_weight, base_color), 1.0f / std::numbers::pi_v<float>);
    return mul(mul(add(diffuse, specular), radiance), n_dot_l);
}

color3 preview_background(float x, float y) noexcept
{
    const float radial = std::clamp(1.0f - std::sqrt(x * x + y * y) * 0.55f, 0.0f, 1.0f);
    const float vertical = std::clamp((y + 1.0f) * 0.5f, 0.0f, 1.0f);
    return mix({0.018f, 0.024f, 0.029f}, {0.085f, 0.105f, 0.118f}, radial * (0.45f + vertical * 0.35f));
}

} // namespace

material_preview_result render_material_preview(const material_asset& asset, const std::filesystem::path& asset_root,
                                                std::uint32_t size)
{
    size = std::clamp(size, 32u, 256u);

    material_asset preview_asset = asset;
    graph_texture_bindings graph_bindings;
    if (asset.graph_reserved && !asset.path.empty())
    {
        const auto realized = load_material_preview_descriptor(asset.path);
        if (realized.succeeded)
        {
            const auto alpha_mode = preview_asset.material.alpha_mode;
            const auto shading_model = preview_asset.material.shading_model;
            const auto domain = preview_asset.material.domain;
            const bool double_sided = preview_asset.material.double_sided;
            preview_asset.material = realized.material;
            preview_asset.material.name = asset.name;
            preview_asset.material.alpha_mode = alpha_mode;
            preview_asset.material.shading_model = shading_model;
            preview_asset.material.domain = domain;
            preview_asset.material.double_sided = double_sided;
        }
        graph_bindings = load_graph_texture_bindings(asset.path);
    }

    preview_texture_cache texture_cache;
    const auto base_map = load_preview_texture(asset_root, preview_asset.textures.base_color, texture_cache);
    const auto metallic_roughness_map =
        load_preview_texture(asset_root, preview_asset.textures.metallic_roughness, texture_cache);
    const auto normal_map = load_preview_texture(asset_root, preview_asset.textures.normal, texture_cache);
    const auto ao_map = load_preview_texture(asset_root, preview_asset.textures.ao, texture_cache);
    const auto emissive_map = load_preview_texture(asset_root, preview_asset.textures.emissive, texture_cache);
    const auto graph_maps = load_graph_bindings(asset_root, graph_bindings, texture_cache);

    render::texture_data output;
    output.name = preview_asset.name + " Preview";
    output.width = size;
    output.height = size;
    output.format = render::texture_format::rgba8_srgb;
    output.pixels.resize(static_cast<std::size_t>(size) * size * 4u);

    const auto& material = preview_asset.material;
    const color3 base_factor{material.base_color[0], material.base_color[1], material.base_color[2]};
    const color3 emissive_factor{material.emissive_factor[0], material.emissive_factor[1], material.emissive_factor[2]};
    const color3 view{0.0f, 0.0f, 1.0f};
    const auto key_light = normalize(color3{-0.48f, 0.62f, 0.76f});
    const auto fill_light = normalize(color3{0.72f, 0.18f, 0.54f});
    const auto rim_light = normalize(color3{0.32f, 0.70f, -0.64f});

    for (std::uint32_t pixel_y = 0; pixel_y < size; ++pixel_y)
    {
        for (std::uint32_t pixel_x = 0; pixel_x < size; ++pixel_x)
        {
            const float x = (2.0f * (static_cast<float>(pixel_x) + 0.5f) / size - 1.0f) / preview_sphere_radius;
            const float y = (1.0f - 2.0f * (static_cast<float>(pixel_y) + 0.5f) / size) / preview_sphere_radius;
            const float radius_squared = x * x + y * y;
            const auto background = preview_background(x, y);
            auto color = background;
            if (radius_squared <= 1.0f)
            {
                auto normal = normalize(color3{x, y, std::sqrt(std::max(0.0f, 1.0f - radius_squared))});
                const float u = 0.5f + std::atan2(normal[0], normal[2]) / (2.0f * std::numbers::pi_v<float>);
                const float v = 0.5f + std::asin(std::clamp(normal[1], -1.0f, 1.0f)) / std::numbers::pi_v<float>;
                const auto base_sample = sample(base_map, u, v, true);
                const auto mr_sample = sample(metallic_roughness_map, u, v, false);
                const auto ao_sample = sample(ao_map, u, v, false);
                const auto emissive_sample = sample(emissive_map, u, v, true);
                const auto graph_base_sample = sample(graph_maps.base_color.texture, u, v, true);
                const auto graph_metallic_sample = sample(graph_maps.metallic.texture, u, v, false);
                const auto graph_roughness_sample = sample(graph_maps.roughness.texture, u, v, false);
                const auto graph_ao_sample = sample(graph_maps.ambient_occlusion.texture, u, v, false);
                const auto graph_emissive_sample = sample(graph_maps.emissive.texture, u, v, true);
                const auto graph_opacity_sample = sample(graph_maps.opacity.texture, u, v, false);
                const auto active_normal_map = graph_maps.normal.texture.valid ? graph_maps.normal.texture : normal_map;
                const auto normal_sample = sample(active_normal_map, u, v, false);
                if (active_normal_map.valid)
                {
                    auto tangent = normalize(color3{normal[2], 0.0f, -normal[0]});
                    const auto bitangent = normalize(cross(normal, tangent));
                    const color3 tangent_normal = normalize({(normal_sample[0] * 2.0f - 1.0f) * material.normal_scale,
                                                             (normal_sample[1] * 2.0f - 1.0f) * material.normal_scale,
                                                             std::max(normal_sample[2] * 2.0f - 1.0f, 0.01f)});
                    normal = normalize(add(add(mul(tangent, tangent_normal[0]), mul(bitangent, tangent_normal[1])),
                                           mul(normal, tangent_normal[2])));
                }
                const color3 base_color = graph_maps.base_color.texture.valid
                                              ? color3{graph_base_sample[0], graph_base_sample[1], graph_base_sample[2]}
                                              : mul(base_factor, {base_sample[0], base_sample[1], base_sample[2]});
                const float metallic =
                    graph_maps.metallic.texture.valid
                        ? std::clamp(sample_scalar(graph_metallic_sample, graph_maps.metallic.pin), 0.0f, 1.0f)
                        : std::clamp(material.metallic * (metallic_roughness_map.valid ? mr_sample[2] : 1.0f), 0.0f,
                                     1.0f);
                const float roughness =
                    graph_maps.roughness.texture.valid
                        ? std::clamp(sample_scalar(graph_roughness_sample, graph_maps.roughness.pin), minimum_roughness,
                                     1.0f)
                        : std::clamp(material.roughness * (metallic_roughness_map.valid ? mr_sample[1] : 1.0f),
                                     minimum_roughness, 1.0f);
                const float ao =
                    graph_maps.ambient_occlusion.texture.valid
                        ? std::clamp(sample_scalar(graph_ao_sample, graph_maps.ambient_occlusion.pin), 0.0f, 1.0f)
                        : std::clamp(1.0f + ((ao_map.valid ? ao_sample[0] : 1.0f) - 1.0f) * material.occlusion_strength,
                                     0.0f, 1.0f);
                const float base_energy = 1.0f - material.clear_coat_factor * dielectric_reflectance;
                color = mul(mul(base_color, 0.055f * base_energy), ao);
                color = add(color, mul(evaluate_light(normal, view, key_light, {4.3f, 3.9f, 3.45f}, base_color,
                                                      metallic, roughness),
                                       base_energy));
                color = add(color, mul(evaluate_light(normal, view, fill_light, {0.75f, 0.95f, 1.25f}, base_color,
                                                      metallic, roughness),
                                       base_energy));
                color = add(color, mul(evaluate_light(normal, view, rim_light, {0.42f, 0.58f, 0.82f}, base_color,
                                                      metallic, roughness),
                                       base_energy));
                if (material.clear_coat_factor > 0.0f)
                {
                    const auto coat =
                        evaluate_light(normal, view, key_light, {4.3f, 3.9f, 3.45f},
                                       {dielectric_reflectance, dielectric_reflectance, dielectric_reflectance}, 1.0f,
                                       std::clamp(material.clear_coat_roughness, minimum_roughness, 1.0f));
                    color = add(color, mul(coat, material.clear_coat_factor));
                }
                if (material.shading_model == render::material_shading_model::skin && material.subsurface_factor > 0.0f)
                {
                    const float wrap = std::clamp((dot(normal, key_light) + 0.35f) / 1.35f, 0.0f, 1.0f);
                    const color3 subsurface{material.subsurface_color[0], material.subsurface_color[1],
                                            material.subsurface_color[2]};
                    color = add(color, mul(mul(base_color, subsurface), wrap * material.subsurface_factor * 0.28f));
                }
                if (material.transmission_factor > 0.0f)
                {
                    const auto attenuation =
                        render::beer_lambert_attenuation(material.attenuation_color, material.attenuation_distance,
                                                         std::max(material.thickness_factor, 0.0f));
                    const color3 transmitted = mul(background, {attenuation[0], attenuation[1], attenuation[2]});
                    color = mix(color, transmitted, std::clamp(material.transmission_factor, 0.0f, 1.0f));
                }
                const float emissive_scale = material.emissive_luminance_nits > 0.0f
                                                 ? material.emissive_luminance_nits / 100.0f
                                                 : material.emissive_strength;
                const color3 emissive =
                    graph_maps.emissive.texture.valid
                        ? color3{graph_emissive_sample[0], graph_emissive_sample[1], graph_emissive_sample[2]}
                        : mul(emissive_factor, {emissive_sample[0], emissive_sample[1], emissive_sample[2]});
                color = add(color, mul(emissive, emissive_scale));
                const float opacity =
                    graph_maps.opacity.texture.valid
                        ? std::clamp(sample_scalar(graph_opacity_sample, graph_maps.opacity.pin), 0.0f, 1.0f)
                        : std::clamp(material.base_color[3] * base_sample[3], 0.0f, 1.0f);
                if (material.alpha_mode == render::material_alpha_mode::masked && opacity < material.alpha_cutoff)
                    color = background;
                else if (material.alpha_mode == render::material_alpha_mode::blend)
                    color = mix(background, color, opacity);
            }

            const auto offset = static_cast<std::size_t>(pixel_y * size + pixel_x) * 4u;
            for (std::size_t channel = 0; channel < 3u; ++channel)
                output.pixels[offset + channel] = static_cast<std::byte>(
                    std::clamp(std::lround(srgb_from_linear(aces_tonemap(color[channel])) * 255.0f), 0l, 255l));
            output.pixels[offset + 3u] = std::byte{255};
        }
    }
    return {std::move(output), "rendered PBR material sphere preview"};
}

} // namespace arc::editor

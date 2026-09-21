#include <arc/editor/texture_preview_realizer.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{
constexpr std::string_view preview_texture_token = "__arc_texture_preview_target__";
constexpr std::string_view checker_texture_token = "__arc_texture_preview_checker__";

nlohmann::json node(std::string id, std::string type, nlohmann::json values = nlohmann::json::object())
{
    return {{"id", std::move(id)}, {"type", std::move(type)}, {"values", std::move(values)}};
}

nlohmann::json connection(std::string id, std::string from_node, std::string from_pin, std::string to_node,
                          std::string to_pin)
{
    return {{"id", std::move(id)},
            {"from", {{"nodeId", std::move(from_node)}, {"pin", std::move(from_pin)}}},
            {"to", {{"nodeId", std::move(to_node)}, {"pin", std::move(to_pin)}}}};
}
} // namespace

render::texture_data select_texture_preview_mip(render::texture_data texture, std::uint32_t mip)
{
    if (texture.dimension != render::texture_dimension::texture_2d || texture.array_layers != 1u ||
        texture.mips.empty() || mip >= texture.mips.size())
        return {};

    const auto selected = texture.mips[mip];
    const bool encoded = texture.has_encoded_mips();
    auto& source = encoded ? texture.encoded : texture.pixels;
    if (selected.offset > source.size() || selected.size > source.size() - selected.offset) return {};

    std::vector<std::byte> bytes(source.begin() + static_cast<std::ptrdiff_t>(selected.offset),
                                 source.begin() + static_cast<std::ptrdiff_t>(selected.offset + selected.size));
    texture.width = selected.width;
    texture.height = selected.height;
    texture.depth = 1;
    texture.mip_levels = 1;
    texture.mips = {{.width = selected.width, .height = selected.height, .offset = 0u, .size = selected.size}};
    if (encoded)
    {
        texture.encoded = std::move(bytes);
        texture.pixels.clear();
    }
    else
    {
        texture.pixels = std::move(bytes);
        texture.encoded.clear();
    }
    return texture;
}

render::texture_data make_texture_preview_checker()
{
    constexpr std::uint32_t extent = 64u;
    constexpr std::uint32_t cell = 8u;
    render::texture_data texture;
    texture.name = "Editor Texture Preview Checker";
    texture.width = extent;
    texture.height = extent;
    texture.depth = 1;
    texture.dimension = render::texture_dimension::texture_2d;
    texture.format = render::texture_format::rgba8_unorm;
    texture.color_space = render::texture_color_space::linear;
    texture.mip_levels = 1;
    texture.pixels.resize(static_cast<std::size_t>(extent) * extent * 4u);
    for (std::uint32_t y = 0; y < extent; ++y)
        for (std::uint32_t x = 0; x < extent; ++x)
        {
            const bool light = ((x / cell) + (y / cell)) % 2u == 0u;
            const std::byte value{static_cast<std::uint8_t>(light ? 96u : 56u)};
            const auto offset = (static_cast<std::size_t>(y) * extent + x) * 4u;
            texture.pixels[offset] = value;
            texture.pixels[offset + 1u] = value;
            texture.pixels[offset + 2u] = value;
            texture.pixels[offset + 3u] = std::byte{255u};
        }
    texture.mips = {{.width = extent, .height = extent, .offset = 0u, .size = texture.pixels.size()}};
    return texture;
}

material_preview_descriptor_result realize_texture_preview_material(std::uint32_t width, std::uint32_t height,
                                                                    const texture_preview_shader_options& options)
{
    using json = nlohmann::json;
    const float exposure_scale = std::exp2(std::clamp(options.exposure, -16.0f, 16.0f));
    const bool any_rgb = options.red || options.green || options.blue;

    json nodes = json::array();
    json connections = json::array();
    nodes.push_back(node("out", "output"));
    nodes.push_back(node("preview", "textureSample", {{"texture", preview_texture_token}, {"dimension", "2d"}}));
    nodes.push_back(
        node("mask", "vector3",
             {{"value", {options.red ? 1.0f : 0.0f, options.green ? 1.0f : 0.0f, options.blue ? 1.0f : 0.0f}}}));
    nodes.push_back(node("masked", "multiply"));
    nodes.push_back(node("exposure", "vector3", {{"value", {exposure_scale, exposure_scale, exposure_scale}}}));
    nodes.push_back(node("exposed", "multiply"));

    connections.push_back(connection("preview-mask", "preview", "rgb", "masked", "a"));
    connections.push_back(connection("mask-value", "mask", "value", "masked", "b"));
    connections.push_back(connection("masked-exposure", "masked", "result", "exposed", "a"));
    connections.push_back(connection("exposure-value", "exposure", "value", "exposed", "b"));

    if (options.nearest)
    {
        nodes.push_back(node("uv", "texCoord"));
        nodes.push_back(
            node("dimensions", "vector2",
                 {{"value", {static_cast<float>(std::max(1u, width)), static_cast<float>(std::max(1u, height))}}}));
        nodes.push_back(node("uv-scaled", "multiply"));
        nodes.push_back(node("uv-floor", "floor"));
        nodes.push_back(node("half", "vector2", {{"value", {0.5f, 0.5f}}}));
        nodes.push_back(node("uv-centered", "add"));
        nodes.push_back(node("uv-nearest", "divide"));
        connections.push_back(connection("uv-scale-a", "uv", "uv", "uv-scaled", "a"));
        connections.push_back(connection("uv-scale-b", "dimensions", "value", "uv-scaled", "b"));
        connections.push_back(connection("uv-floor-in", "uv-scaled", "result", "uv-floor", "value"));
        connections.push_back(connection("uv-center-a", "uv-floor", "result", "uv-centered", "a"));
        connections.push_back(connection("uv-center-b", "half", "value", "uv-centered", "b"));
        connections.push_back(connection("uv-nearest-a", "uv-centered", "result", "uv-nearest", "a"));
        connections.push_back(connection("uv-nearest-b", "dimensions", "value", "uv-nearest", "b"));
        connections.push_back(connection("preview-uv", "uv-nearest", "result", "preview", "uv"));
    }

    if (!any_rgb && options.alpha)
    {
        nodes.push_back(node("alpha-white", "vector3", {{"value", {1.0f, 1.0f, 1.0f}}}));
        nodes.push_back(node("alpha-color", "multiply"));
        connections.push_back(connection("alpha-value", "preview", "a", "alpha-color", "a"));
        connections.push_back(connection("alpha-white-value", "alpha-white", "value", "alpha-color", "b"));
        connections.push_back(connection("alpha-output", "alpha-color", "result", "out", "emissive"));
    }
    else if (any_rgb && options.alpha)
    {
        nodes.push_back(node("checker", "textureSample", {{"texture", checker_texture_token}, {"dimension", "2d"}}));
        nodes.push_back(node("alpha-composite", "lerp"));
        connections.push_back(connection("checker-color", "checker", "rgb", "alpha-composite", "a"));
        connections.push_back(connection("preview-color", "exposed", "result", "alpha-composite", "b"));
        connections.push_back(connection("preview-alpha", "preview", "a", "alpha-composite", "t"));
        connections.push_back(connection("composite-output", "alpha-composite", "result", "out", "emissive"));
    }
    else
    {
        connections.push_back(connection("preview-output", "exposed", "result", "out", "emissive"));
    }

    json authored{{"version", 4},
                  {"name", "Texture Preview"},
                  {"domain", "surface"},
                  {"blendMode", "opaque"},
                  {"shadingModel", "unlit"},
                  {"doubleSided", true},
                  {"graph", {{"version", 1}, {"nodes", std::move(nodes)}, {"connections", std::move(connections)}}}};
    return realize_material_preview_descriptor(authored.dump(), "Texture Preview");
}

} // namespace arc::editor

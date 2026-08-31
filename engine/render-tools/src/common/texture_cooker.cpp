#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <string>

namespace arc::render::tools
{
namespace
{

using json = nlohmann::json;

std::string lowercase(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return result;
}

texture_format format_for_color_space(texture_format format, texture_color_space color_space) noexcept
{
    const bool srgb = color_space == texture_color_space::srgb;
    switch (format)
    {
        case texture_format::rgba8_unorm:
        case texture_format::rgba8_srgb:
            return srgb ? texture_format::rgba8_srgb : texture_format::rgba8_unorm;
        case texture_format::bc1_rgba_unorm:
        case texture_format::bc1_rgba_srgb:
            return srgb ? texture_format::bc1_rgba_srgb : texture_format::bc1_rgba_unorm;
        case texture_format::bc2_rgba_unorm:
        case texture_format::bc2_rgba_srgb:
            return srgb ? texture_format::bc2_rgba_srgb : texture_format::bc2_rgba_unorm;
        case texture_format::bc3_rgba_unorm:
        case texture_format::bc3_rgba_srgb:
            return srgb ? texture_format::bc3_rgba_srgb : texture_format::bc3_rgba_unorm;
        case texture_format::bc7_rgba_unorm:
        case texture_format::bc7_rgba_srgb:
            return srgb ? texture_format::bc7_rgba_srgb : texture_format::bc7_rgba_unorm;
        default:
            return format;
    }
}

assets::asset_cook_result cook_failure(const assets::asset_cook_context& context, std::string message)
{
    return {.error = {.code = assets::asset_error_code::import_failed,
                      .guid = context.asset.guid,
                      .path = context.source.source_path,
                      .message = std::move(message)}};
}

} // namespace

std::string_view texture_import_preset_name(texture_import_preset preset) noexcept
{
    switch (preset)
    {
        case texture_import_preset::custom:
            return "custom";
        case texture_import_preset::color:
            return "color";
        case texture_import_preset::normal_map:
            return "normal_map";
        case texture_import_preset::data:
            return "data";
        case texture_import_preset::hdr:
            return "hdr";
        case texture_import_preset::ui:
            return "ui";
        case texture_import_preset::environment:
            return "environment";
    }
    return "custom";
}

std::string_view texture_semantic_name(texture_semantic semantic) noexcept
{
    switch (semantic)
    {
        case texture_semantic::generic_color:
            return "generic_color";
        case texture_semantic::base_color:
            return "base_color";
        case texture_semantic::emissive:
            return "emissive";
        case texture_semantic::normal:
            return "normal";
        case texture_semantic::metallic_roughness:
            return "metallic_roughness";
        case texture_semantic::occlusion:
            return "occlusion";
        case texture_semantic::clear_coat:
            return "clear_coat";
        case texture_semantic::anisotropy:
            return "anisotropy";
        case texture_semantic::thickness:
            return "thickness";
        case texture_semantic::transmission:
            return "transmission";
        case texture_semantic::lightmap:
            return "lightmap";
        case texture_semantic::environment:
            return "environment";
    }
    return "generic_color";
}

std::string_view texture_color_space_name(texture_color_space color_space) noexcept
{
    return color_space == texture_color_space::linear ? "linear" : "srgb";
}

std::string_view texture_streaming_mode_name(texture_streaming_mode mode) noexcept
{
    switch (mode)
    {
        case texture_streaming_mode::resident:
            return "resident";
        case texture_streaming_mode::streamed_mips:
            return "streamed_mips";
        case texture_streaming_mode::virtual_tiles:
            return "virtual_tiles";
    }
    return "resident";
}

std::optional<texture_import_preset> parse_texture_import_preset(std::string_view value) noexcept
{
    const auto text = lowercase(value);
    if (text == "custom") return texture_import_preset::custom;
    if (text == "color" || text == "default") return texture_import_preset::color;
    if (text == "normal_map" || text == "normalmap" || text == "normal") return texture_import_preset::normal_map;
    if (text == "data" || text == "mask") return texture_import_preset::data;
    if (text == "hdr") return texture_import_preset::hdr;
    if (text == "ui") return texture_import_preset::ui;
    if (text == "environment") return texture_import_preset::environment;
    return std::nullopt;
}

std::optional<texture_semantic> parse_texture_semantic(std::string_view value) noexcept
{
    const auto text = lowercase(value);
    if (text == "generic_color" || text == "genericcolor") return texture_semantic::generic_color;
    if (text == "base_color" || text == "basecolor") return texture_semantic::base_color;
    if (text == "emissive") return texture_semantic::emissive;
    if (text == "normal") return texture_semantic::normal;
    if (text == "metallic_roughness" || text == "metallicroughness") return texture_semantic::metallic_roughness;
    if (text == "occlusion") return texture_semantic::occlusion;
    if (text == "clear_coat" || text == "clearcoat") return texture_semantic::clear_coat;
    if (text == "anisotropy") return texture_semantic::anisotropy;
    if (text == "thickness") return texture_semantic::thickness;
    if (text == "transmission") return texture_semantic::transmission;
    if (text == "lightmap") return texture_semantic::lightmap;
    if (text == "environment") return texture_semantic::environment;
    return std::nullopt;
}

std::optional<texture_color_space> parse_texture_color_space(std::string_view value) noexcept
{
    const auto text = lowercase(value);
    if (text == "linear") return texture_color_space::linear;
    if (text == "srgb") return texture_color_space::srgb;
    return std::nullopt;
}

std::optional<texture_streaming_mode> parse_texture_streaming_mode(std::string_view value) noexcept
{
    const auto text = lowercase(value);
    if (text == "resident") return texture_streaming_mode::resident;
    if (text == "streamed_mips" || text == "streamedmips") return texture_streaming_mode::streamed_mips;
    if (text == "virtual_tiles" || text == "virtualtiles") return texture_streaming_mode::virtual_tiles;
    return std::nullopt;
}

texture_import_settings texture_import_settings_for_preset(texture_import_preset preset) noexcept
{
    texture_import_settings settings;
    settings.preset = preset;
    switch (preset)
    {
        case texture_import_preset::color:
            settings.semantic = texture_semantic::base_color;
            settings.color_space = texture_color_space::srgb;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            break;
        case texture_import_preset::normal_map:
            settings.semantic = texture_semantic::normal;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            break;
        case texture_import_preset::data:
            settings.semantic = texture_semantic::metallic_roughness;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            break;
        case texture_import_preset::hdr:
            settings.semantic = texture_semantic::generic_color;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            break;
        case texture_import_preset::ui:
            settings.semantic = texture_semantic::generic_color;
            settings.color_space = texture_color_space::srgb;
            settings.streaming_mode = texture_streaming_mode::resident;
            break;
        case texture_import_preset::environment:
            settings.semantic = texture_semantic::environment;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            break;
        case texture_import_preset::custom:
            break;
    }
    return settings;
}

texture_import_settings_result parse_texture_import_settings(std::string_view canonical_json,
                                                             std::uint32_t settings_version)
{
    texture_import_settings settings;
    if (canonical_json.empty() || canonical_json == "{}") return texture_import_settings_result::success(settings);
    const auto document = json::parse(canonical_json, nullptr, false);
    if (!document.is_object())
        return texture_import_settings_result::failure("texture import settings must be a JSON object");

    if (const auto field = document.find("streamingMode"); field != document.end())
    {
        if (!field->is_string())
            return texture_import_settings_result::failure("texture streamingMode must be a string");
        const auto parsed = parse_texture_streaming_mode(field->get<std::string>());
        if (!parsed) return texture_import_settings_result::failure("texture streamingMode is invalid");
        settings.streaming_mode = *parsed;
    }
    if (settings_version < 3) return texture_import_settings_result::success(settings);

    if (const auto field = document.find("preset"); field != document.end())
    {
        if (!field->is_string()) return texture_import_settings_result::failure("texture preset must be a string");
        const auto parsed = parse_texture_import_preset(field->get<std::string>());
        if (!parsed) return texture_import_settings_result::failure("texture preset is invalid");
        settings.preset = *parsed;
    }
    if (const auto field = document.find("semantic"); field != document.end())
    {
        if (!field->is_string()) return texture_import_settings_result::failure("texture semantic must be a string");
        const auto parsed = parse_texture_semantic(field->get<std::string>());
        if (!parsed) return texture_import_settings_result::failure("texture semantic is invalid");
        settings.semantic = *parsed;
    }
    if (const auto field = document.find("colorSpace"); field != document.end())
    {
        if (!field->is_string()) return texture_import_settings_result::failure("texture colorSpace must be a string");
        const auto parsed = parse_texture_color_space(field->get<std::string>());
        if (!parsed) return texture_import_settings_result::failure("texture colorSpace is invalid");
        settings.color_space = *parsed;
    }
    return texture_import_settings_result::success(settings);
}

std::string serialize_texture_import_settings(const texture_import_settings& settings)
{
    return json{{"colorSpace", texture_color_space_name(settings.color_space)},
                {"preset", texture_import_preset_name(settings.preset)},
                {"semantic", texture_semantic_name(settings.semantic)},
                {"streamingMode", texture_streaming_mode_name(settings.streaming_mode)}}
        .dump();
}

texture_cook_processor::texture_cook_processor()
{
    descriptor_ = {.id = assets::cook_processor_ids::texture,
                   .name = "ARC Texture Cooker",
                   .version = 3,
                   .schema = assets::artifact_schemas::texture,
                   .schema_version = texture_artifact_schema_version,
                   .affinity = jobs::job_affinity::any_worker,
                   .input_types = {assets::asset_types::texture_2d}};
}

const assets::asset_cook_processor_descriptor& texture_cook_processor::descriptor() const noexcept
{
    return descriptor_;
}

std::string texture_cook_processor::toolchain_fingerprint() const
{
    return "arc-texture-cooker-v3:arctex-v1:authored-semantic-color-space:rgba8-or-native-dds:no-recompression";
}

assets::asset_cook_result texture_cook_processor::cook(const assets::asset_cook_context& context)
{
    if (context.cancellation.stop_requested())
        return {.error = {.code = assets::asset_error_code::cancelled,
                          .guid = context.asset.guid,
                          .path = context.source.source_path,
                          .message = "texture cook was cancelled"}};
    const auto settings = parse_texture_import_settings(context.canonical_settings, context.settings_version);
    if (!settings) return cook_failure(context, settings.error());
    if (!is_supported_texture_asset(context.source.source_path) || context.source.source_path.extension() == ".hdr")
        return cook_failure(context, "streamable texture cooking supports DDS, PNG, JPEG, TGA, and BMP 2D sources");

    auto loaded = load_texture_asset_bytes(context.source.bytes, context.source.source_path);
    if (!loaded.succeeded() || loaded.texture.dimension != texture_dimension::texture_2d ||
        loaded.texture.array_layers != 1 || loaded.texture.mips.empty())
        return cook_failure(context, loaded.message.empty() ? "texture source could not be decoded" : loaded.message);

    loaded.texture.semantic = settings.value().semantic;
    loaded.texture.color_space = settings.value().color_space;
    loaded.texture.format = format_for_color_space(loaded.texture.format, settings.value().color_space);
    auto encoded = encode_texture_artifact(loaded.texture, settings.value().streaming_mode);
    if (!encoded) return cook_failure(context, encoded.error().message);

    assets::cooked_artifact artifact{.name = context.source.source_path.stem().string(),
                                     .extension = ".arctex",
                                     .schema = assets::artifact_schemas::texture,
                                     .schema_version = texture_artifact_schema_version,
                                     .gpu_compressed = loaded.texture.compressed,
                                     .bytes = std::move(encoded).value()};
    return {.artifacts = {std::move(artifact)}};
}

} // namespace arc::render::tools

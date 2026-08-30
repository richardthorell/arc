#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>

namespace arc::render::tools
{
namespace
{

using json = nlohmann::json;

std::string mode_name(texture_streaming_mode mode)
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

std::optional<texture_streaming_mode> parse_mode(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (value == "resident") return texture_streaming_mode::resident;
    if (value == "streamed_mips" || value == "streamedmips") return texture_streaming_mode::streamed_mips;
    if (value == "virtual_tiles" || value == "virtualtiles") return texture_streaming_mode::virtual_tiles;
    return std::nullopt;
}

assets::asset_cook_result cook_failure(const assets::asset_cook_context& context, std::string message)
{
    return {.error = {.code = assets::asset_error_code::import_failed,
                      .guid = context.asset.guid,
                      .path = context.source.source_path,
                      .message = std::move(message)}};
}

} // namespace

texture_import_settings_result parse_texture_import_settings(std::string_view canonical_json,
                                                              std::uint32_t settings_version)
{
    if (canonical_json.empty() || canonical_json == "{}" || settings_version < 2)
        return texture_import_settings_result::success({});
    const auto document = json::parse(canonical_json, nullptr, false);
    if (!document.is_object())
        return texture_import_settings_result::failure("texture import settings must be a JSON object");
    const auto field = document.find("streamingMode");
    if (field == document.end()) return texture_import_settings_result::success({});
    if (!field->is_string())
        return texture_import_settings_result::failure("texture streamingMode must be a string");
    const auto mode = parse_mode(field->get<std::string>());
    if (!mode)
        return texture_import_settings_result::failure(
            "texture streamingMode must be resident, streamed_mips, or virtual_tiles");
    return texture_import_settings_result::success({.streaming_mode = *mode});
}

std::string serialize_texture_import_settings(const texture_import_settings& settings)
{
    return json{{"streamingMode", mode_name(settings.streaming_mode)}}.dump();
}

texture_cook_processor::texture_cook_processor()
{
    descriptor_ = {.id = assets::cook_processor_ids::texture,
                   .name = "ARC Texture Cooker",
                   .version = 2,
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
    return "arc-texture-cooker-v2:arctex-v1:rgba8-or-native-dds:no-recompression";
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
    if (!is_supported_texture_asset(context.source.source_path) ||
        context.source.source_path.extension() == ".hdr")
        return cook_failure(context, "streamable texture cooking supports DDS, PNG, JPEG, TGA, and BMP 2D sources");

    auto loaded = load_texture_asset_bytes(context.source.bytes, context.source.source_path);
    if (!loaded.succeeded() || loaded.texture.dimension != texture_dimension::texture_2d ||
        loaded.texture.array_layers != 1 || loaded.texture.mips.empty())
        return cook_failure(context, loaded.message.empty() ? "texture source could not be decoded" : loaded.message);
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

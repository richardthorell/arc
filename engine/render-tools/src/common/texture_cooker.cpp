#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
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

template <class Enum>
std::optional<Enum> enum_value(std::string_view value, std::initializer_list<std::pair<std::string_view, Enum>> values)
{
    const auto text = lowercase(value);
    for (const auto& [name, result] : values)
        if (text == name) return result;
    return std::nullopt;
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

float srgb_to_linear_channel(float value) noexcept
{
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

float linear_to_srgb_channel(float value) noexcept
{
    value = std::clamp(value, 0.0f, 1.0f);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
}

std::byte byte_channel(float value) noexcept
{
    return static_cast<std::byte>(static_cast<std::uint8_t>(std::clamp(std::lround(value * 255.0f), 0l, 255l)));
}

std::uint32_t previous_power_of_two(std::uint32_t value) noexcept
{
    std::uint32_t result = 1;
    while (result <= value / 2u)
        result *= 2u;
    return result;
}

std::uint32_t next_power_of_two(std::uint32_t value) noexcept
{
    if (value <= 1u) return 1u;
    std::uint32_t result = 1u;
    while (result < value && result <= std::numeric_limits<std::uint32_t>::max() / 2u)
        result *= 2u;
    return result;
}

std::uint32_t platform_max_size(const assets::cook_target& target) noexcept
{
    switch (target.textures)
    {
        case assets::cook_texture_family::bc:
            return 16384;
        case assets::cook_texture_family::astc:
        case assets::cook_texture_family::etc2:
            return 8192;
        case assets::cook_texture_family::portable:
            return 4096;
    }
    return 4096;
}

std::vector<std::byte> resize_rgba8(std::span<const std::byte> source, std::uint32_t source_width,
                                    std::uint32_t source_height, std::uint32_t width, std::uint32_t height)
{
    std::vector<std::byte> result(static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y)
    {
        const auto source_y = std::min(
            source_height - 1u, static_cast<std::uint32_t>((static_cast<std::uint64_t>(y) * source_height) / height));
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const auto source_x = std::min(
                source_width - 1u, static_cast<std::uint32_t>((static_cast<std::uint64_t>(x) * source_width) / width));
            const auto from = (static_cast<std::size_t>(source_y) * source_width + source_x) * 4u;
            const auto to = (static_cast<std::size_t>(y) * width + x) * 4u;
            std::memcpy(result.data() + to, source.data() + from, 4u);
        }
    }
    return result;
}

float alpha_coverage(std::span<const std::byte> pixels, float threshold) noexcept
{
    if (pixels.empty()) return 0.0f;
    std::size_t covered{};
    const auto cutoff = static_cast<std::uint8_t>(std::clamp(std::lround(threshold * 255.0f), 0l, 255l));
    for (std::size_t index = 3; index < pixels.size(); index += 4u)
        if (std::to_integer<std::uint8_t>(pixels[index]) >= cutoff) ++covered;
    return static_cast<float>(covered) / static_cast<float>(pixels.size() / 4u);
}

void preserve_alpha_coverage(std::vector<std::byte>& pixels, float threshold, float target) noexcept
{
    if (pixels.empty() || target <= 0.0f || target >= 1.0f) return;
    auto coverage_for_scale = [&](float scale)
    {
        std::size_t covered{};
        for (std::size_t index = 3; index < pixels.size(); index += 4u)
        {
            const auto alpha = static_cast<float>(std::to_integer<std::uint8_t>(pixels[index])) / 255.0f;
            if (std::min(1.0f, alpha * scale) >= threshold) ++covered;
        }
        return static_cast<float>(covered) / static_cast<float>(pixels.size() / 4u);
    };
    float low = 0.0f;
    float high = 8.0f;
    for (std::uint32_t iteration = 0; iteration < 14u; ++iteration)
    {
        const float middle = (low + high) * 0.5f;
        if (coverage_for_scale(middle) < target)
            low = middle;
        else
            high = middle;
    }
    const float scale = (low + high) * 0.5f;
    for (std::size_t index = 3; index < pixels.size(); index += 4u)
    {
        const auto alpha = static_cast<float>(std::to_integer<std::uint8_t>(pixels[index])) / 255.0f;
        pixels[index] = byte_channel(std::min(1.0f, alpha * scale));
    }
}

std::vector<std::byte> downsample_rgba8(std::span<const std::byte> source, std::uint32_t width, std::uint32_t height,
                                        std::uint32_t next_width, std::uint32_t next_height,
                                        const texture_import_settings& settings)
{
    std::vector<std::byte> next(static_cast<std::size_t>(next_width) * next_height * 4u);
    for (std::uint32_t y = 0; y < next_height; ++y)
        for (std::uint32_t x = 0; x < next_width; ++x)
        {
            const auto destination = (static_cast<std::size_t>(y) * next_width + x) * 4u;
            if (settings.mip_generation_filter == texture_mip_generation_filter::nearest)
            {
                const auto source_x = std::min(width - 1u, x * 2u);
                const auto source_y = std::min(height - 1u, y * 2u);
                const auto from = (static_cast<std::size_t>(source_y) * width + source_x) * 4u;
                std::memcpy(next.data() + destination, source.data() + from, 4u);
                continue;
            }
            float accumulated[4]{};
            std::uint32_t samples{};
            for (std::uint32_t oy = 0; oy < 2u; ++oy)
                for (std::uint32_t ox = 0; ox < 2u; ++ox)
                {
                    const auto source_x = std::min(width - 1u, x * 2u + ox);
                    const auto source_y = std::min(height - 1u, y * 2u + oy);
                    const auto source_index = (static_cast<std::size_t>(source_y) * width + source_x) * 4u;
                    for (std::uint32_t channel = 0; channel < 4u; ++channel)
                    {
                        float value =
                            static_cast<float>(std::to_integer<std::uint8_t>(source[source_index + channel])) / 255.0f;
                        if (settings.semantic == texture_semantic::normal && channel < 3u)
                            value = value * 2.0f - 1.0f;
                        else if (settings.color_space == texture_color_space::srgb && channel < 3u)
                            value = srgb_to_linear_channel(value);
                        accumulated[channel] += value;
                    }
                    ++samples;
                }
            for (auto& value : accumulated)
                value /= static_cast<float>(samples);
            if (settings.semantic == texture_semantic::normal)
            {
                const float length = std::sqrt(accumulated[0] * accumulated[0] + accumulated[1] * accumulated[1] +
                                               accumulated[2] * accumulated[2]);
                if (length > 0.00001f)
                    for (std::uint32_t channel = 0; channel < 3u; ++channel)
                        accumulated[channel] /= length;
                else
                {
                    accumulated[0] = 0.0f;
                    accumulated[1] = 0.0f;
                    accumulated[2] = 1.0f;
                }
                next[destination] = byte_channel(accumulated[0] * 0.5f + 0.5f);
                next[destination + 1u] = byte_channel(accumulated[1] * 0.5f + 0.5f);
                next[destination + 2u] = byte_channel(accumulated[2] * 0.5f + 0.5f);
            }
            else
                for (std::uint32_t channel = 0; channel < 3u; ++channel)
                    next[destination + channel] = byte_channel(settings.color_space == texture_color_space::srgb
                                                                   ? linear_to_srgb_channel(accumulated[channel])
                                                                   : accumulated[channel]);
            next[destination + 3u] = byte_channel(accumulated[3]);
        }
    return next;
}

void rebuild_rgba8_mips(texture_data& texture, const texture_import_settings& settings,
                        std::vector<std::byte> base_level)
{
    texture.pixels.clear();
    texture.encoded.clear();
    texture.mips.clear();
    texture.compressed = false;
    texture.dds = false;
    auto level = std::move(base_level);
    auto width = texture.width;
    auto height = texture.height;
    const float target_coverage =
        settings.preserve_alpha_coverage ? alpha_coverage(level, settings.alpha_coverage_threshold) : 0.0f;
    while (true)
    {
        const auto offset = texture.pixels.size();
        texture.pixels.insert(texture.pixels.end(), level.begin(), level.end());
        texture.mips.push_back({.width = width, .height = height, .offset = offset, .size = level.size()});
        if (!settings.generate_mips || (width == 1u && height == 1u)) break;
        const auto next_width = std::max(1u, width / 2u);
        const auto next_height = std::max(1u, height / 2u);
        auto next = downsample_rgba8(level, width, height, next_width, next_height, settings);
        if (settings.preserve_alpha_coverage)
            preserve_alpha_coverage(next, settings.alpha_coverage_threshold, target_coverage);
        level = std::move(next);
        width = next_width;
        height = next_height;
    }
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
}

} // namespace

std::string_view texture_import_preset_name(texture_import_preset value) noexcept
{
    switch (value)
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
std::string_view texture_semantic_name(texture_semantic value) noexcept
{
    switch (value)
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
std::string_view texture_color_space_name(texture_color_space value) noexcept
{
    return value == texture_color_space::linear ? "linear" : "srgb";
}
std::string_view texture_streaming_mode_name(texture_streaming_mode value) noexcept
{
    switch (value)
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
std::string_view texture_compression_policy_name(texture_compression_policy value) noexcept
{
    switch (value)
    {
        case texture_compression_policy::automatic:
            return "automatic";
        case texture_compression_policy::color:
            return "color";
        case texture_compression_policy::normal:
            return "normal";
        case texture_compression_policy::mask:
            return "mask";
        case texture_compression_policy::hdr:
            return "hdr";
        case texture_compression_policy::uncompressed:
            return "uncompressed";
    }
    return "automatic";
}
std::string_view texture_power_of_two_policy_name(texture_power_of_two_policy value) noexcept
{
    switch (value)
    {
        case texture_power_of_two_policy::preserve:
            return "preserve";
        case texture_power_of_two_policy::resize_down:
            return "resize_down";
        case texture_power_of_two_policy::resize_up:
            return "resize_up";
    }
    return "preserve";
}
std::string_view texture_filter_mode_name(texture_filter_mode value) noexcept
{
    return value == texture_filter_mode::nearest ? "nearest" : "linear";
}
std::string_view texture_mip_filter_mode_name(texture_mip_filter_mode value) noexcept
{
    return value == texture_mip_filter_mode::nearest ? "nearest" : "linear";
}
std::string_view texture_address_mode_name(texture_address_mode value) noexcept
{
    switch (value)
    {
        case texture_address_mode::repeat:
            return "repeat";
        case texture_address_mode::clamp_to_edge:
            return "clamp_to_edge";
        case texture_address_mode::mirrored_repeat:
            return "mirrored_repeat";
    }
    return "repeat";
}
std::string_view texture_mip_generation_filter_name(texture_mip_generation_filter value) noexcept
{
    return value == texture_mip_generation_filter::nearest ? "nearest" : "box";
}

std::optional<texture_import_preset> parse_texture_import_preset(std::string_view value) noexcept
{
    return enum_value<texture_import_preset>(value, {{"custom", texture_import_preset::custom},
                                                     {"color", texture_import_preset::color},
                                                     {"default", texture_import_preset::color},
                                                     {"normal_map", texture_import_preset::normal_map},
                                                     {"normalmap", texture_import_preset::normal_map},
                                                     {"normal", texture_import_preset::normal_map},
                                                     {"data", texture_import_preset::data},
                                                     {"mask", texture_import_preset::data},
                                                     {"hdr", texture_import_preset::hdr},
                                                     {"ui", texture_import_preset::ui},
                                                     {"environment", texture_import_preset::environment}});
}
std::optional<texture_semantic> parse_texture_semantic(std::string_view value) noexcept
{
    return enum_value<texture_semantic>(value, {{"generic_color", texture_semantic::generic_color},
                                                {"genericcolor", texture_semantic::generic_color},
                                                {"base_color", texture_semantic::base_color},
                                                {"basecolor", texture_semantic::base_color},
                                                {"emissive", texture_semantic::emissive},
                                                {"normal", texture_semantic::normal},
                                                {"metallic_roughness", texture_semantic::metallic_roughness},
                                                {"metallicroughness", texture_semantic::metallic_roughness},
                                                {"occlusion", texture_semantic::occlusion},
                                                {"clear_coat", texture_semantic::clear_coat},
                                                {"clearcoat", texture_semantic::clear_coat},
                                                {"anisotropy", texture_semantic::anisotropy},
                                                {"thickness", texture_semantic::thickness},
                                                {"transmission", texture_semantic::transmission},
                                                {"lightmap", texture_semantic::lightmap},
                                                {"environment", texture_semantic::environment}});
}
std::optional<texture_color_space> parse_texture_color_space(std::string_view value) noexcept
{
    return enum_value<texture_color_space>(
        value, {{"linear", texture_color_space::linear}, {"srgb", texture_color_space::srgb}});
}
std::optional<texture_streaming_mode> parse_texture_streaming_mode(std::string_view value) noexcept
{
    return enum_value<texture_streaming_mode>(value, {{"resident", texture_streaming_mode::resident},
                                                      {"streamed_mips", texture_streaming_mode::streamed_mips},
                                                      {"streamedmips", texture_streaming_mode::streamed_mips},
                                                      {"virtual_tiles", texture_streaming_mode::virtual_tiles},
                                                      {"virtualtiles", texture_streaming_mode::virtual_tiles}});
}
std::optional<texture_compression_policy> parse_texture_compression_policy(std::string_view value) noexcept
{
    return enum_value<texture_compression_policy>(value, {{"automatic", texture_compression_policy::automatic},
                                                          {"color", texture_compression_policy::color},
                                                          {"normal", texture_compression_policy::normal},
                                                          {"mask", texture_compression_policy::mask},
                                                          {"hdr", texture_compression_policy::hdr},
                                                          {"uncompressed", texture_compression_policy::uncompressed}});
}
std::optional<texture_power_of_two_policy> parse_texture_power_of_two_policy(std::string_view value) noexcept
{
    return enum_value<texture_power_of_two_policy>(value, {{"preserve", texture_power_of_two_policy::preserve},
                                                           {"resize_down", texture_power_of_two_policy::resize_down},
                                                           {"resize_up", texture_power_of_two_policy::resize_up}});
}
std::optional<texture_filter_mode> parse_texture_filter_mode(std::string_view value) noexcept
{
    return enum_value<texture_filter_mode>(
        value, {{"nearest", texture_filter_mode::nearest}, {"linear", texture_filter_mode::linear}});
}
std::optional<texture_mip_filter_mode> parse_texture_mip_filter_mode(std::string_view value) noexcept
{
    return enum_value<texture_mip_filter_mode>(
        value, {{"nearest", texture_mip_filter_mode::nearest}, {"linear", texture_mip_filter_mode::linear}});
}
std::optional<texture_address_mode> parse_texture_address_mode(std::string_view value) noexcept
{
    return enum_value<texture_address_mode>(value, {{"repeat", texture_address_mode::repeat},
                                                    {"clamp_to_edge", texture_address_mode::clamp_to_edge},
                                                    {"clamp", texture_address_mode::clamp_to_edge},
                                                    {"mirrored_repeat", texture_address_mode::mirrored_repeat},
                                                    {"mirror", texture_address_mode::mirrored_repeat}});
}
std::optional<texture_mip_generation_filter> parse_texture_mip_generation_filter(std::string_view value) noexcept
{
    return enum_value<texture_mip_generation_filter>(
        value, {{"box", texture_mip_generation_filter::box}, {"nearest", texture_mip_generation_filter::nearest}});
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
            settings.compression = texture_compression_policy::color;
            break;
        case texture_import_preset::normal_map:
            settings.semantic = texture_semantic::normal;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            settings.compression = texture_compression_policy::normal;
            break;
        case texture_import_preset::data:
            settings.semantic = texture_semantic::metallic_roughness;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            settings.compression = texture_compression_policy::mask;
            settings.anisotropy = 4.0f;
            settings.max_size = 4096;
            break;
        case texture_import_preset::hdr:
            settings.semantic = texture_semantic::generic_color;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            settings.compression = texture_compression_policy::hdr;
            settings.max_size = 4096;
            break;
        case texture_import_preset::ui:
            settings.semantic = texture_semantic::generic_color;
            settings.color_space = texture_color_space::srgb;
            settings.streaming_mode = texture_streaming_mode::resident;
            settings.compression = texture_compression_policy::color;
            settings.wrap_u = texture_address_mode::clamp_to_edge;
            settings.wrap_v = texture_address_mode::clamp_to_edge;
            settings.generate_mips = false;
            settings.anisotropy = 1.0f;
            settings.max_size = 4096;
            break;
        case texture_import_preset::environment:
            settings.semantic = texture_semantic::environment;
            settings.color_space = texture_color_space::linear;
            settings.streaming_mode = texture_streaming_mode::streamed_mips;
            settings.compression = texture_compression_policy::hdr;
            settings.wrap_v = texture_address_mode::clamp_to_edge;
            settings.max_size = 4096;
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
    auto parse_string_field = [&](const char* name, auto parser, auto& target) -> std::optional<std::string>
    {
        const auto field = document.find(name);
        if (field == document.end()) return std::nullopt;
        if (!field->is_string()) return std::string{"texture "} + name + " must be a string";
        const auto value = parser(field->get<std::string>());
        if (!value) return std::string{"texture "} + name + " is invalid";
        target = *value;
        return std::nullopt;
    };
    if (const auto error = parse_string_field("streamingMode", parse_texture_streaming_mode, settings.streaming_mode))
        return texture_import_settings_result::failure(*error);
    if (settings_version < 3) return texture_import_settings_result::success(settings);
    if (const auto error = parse_string_field("preset", parse_texture_import_preset, settings.preset))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("semantic", parse_texture_semantic, settings.semantic))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("colorSpace", parse_texture_color_space, settings.color_space))
        return texture_import_settings_result::failure(*error);
    if (settings_version < 4) return texture_import_settings_result::success(settings);
    if (const auto error = parse_string_field("compression", parse_texture_compression_policy, settings.compression))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("powerOfTwo", parse_texture_power_of_two_policy, settings.power_of_two))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("minFilter", parse_texture_filter_mode, settings.min_filter))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("magFilter", parse_texture_filter_mode, settings.mag_filter))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("mipFilter", parse_texture_mip_filter_mode, settings.mip_filter))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("wrapU", parse_texture_address_mode, settings.wrap_u))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("wrapV", parse_texture_address_mode, settings.wrap_v))
        return texture_import_settings_result::failure(*error);
    if (const auto error = parse_string_field("mipGenerationFilter", parse_texture_mip_generation_filter,
                                              settings.mip_generation_filter))
        return texture_import_settings_result::failure(*error);
    settings.max_size = document.value("maxSize", settings.max_size);
    settings.anisotropy = document.value("anisotropy", settings.anisotropy);
    settings.lod_bias = document.value("lodBias", settings.lod_bias);
    settings.minimum_lod = document.value("minimumLod", settings.minimum_lod);
    settings.maximum_lod = document.value("maximumLod", settings.maximum_lod);
    settings.alpha_coverage_threshold = document.value("alphaCoverageThreshold", settings.alpha_coverage_threshold);
    settings.generate_mips = document.value("generateMips", settings.generate_mips);
    settings.preserve_alpha_coverage = document.value("preserveAlphaCoverage", settings.preserve_alpha_coverage);
    if (settings.max_size == 0 || settings.max_size > 32768 || !std::isfinite(settings.anisotropy) ||
        settings.anisotropy < 1.0f || settings.anisotropy > 16.0f || !std::isfinite(settings.lod_bias) ||
        !std::isfinite(settings.minimum_lod) || !std::isfinite(settings.maximum_lod) ||
        settings.maximum_lod < settings.minimum_lod || !std::isfinite(settings.alpha_coverage_threshold) ||
        settings.alpha_coverage_threshold < 0.0f || settings.alpha_coverage_threshold > 1.0f)
        return texture_import_settings_result::failure("texture import settings contain an invalid numeric value");
    return texture_import_settings_result::success(settings);
}

std::string serialize_texture_import_settings(const texture_import_settings& settings)
{
    return json{{"alphaCoverageThreshold", settings.alpha_coverage_threshold},
                {"anisotropy", settings.anisotropy},
                {"colorSpace", texture_color_space_name(settings.color_space)},
                {"compression", texture_compression_policy_name(settings.compression)},
                {"generateMips", settings.generate_mips},
                {"lodBias", settings.lod_bias},
                {"magFilter", texture_filter_mode_name(settings.mag_filter)},
                {"maxSize", settings.max_size},
                {"maximumLod", settings.maximum_lod},
                {"minFilter", texture_filter_mode_name(settings.min_filter)},
                {"minimumLod", settings.minimum_lod},
                {"mipFilter", texture_mip_filter_mode_name(settings.mip_filter)},
                {"mipGenerationFilter", texture_mip_generation_filter_name(settings.mip_generation_filter)},
                {"powerOfTwo", texture_power_of_two_policy_name(settings.power_of_two)},
                {"preset", texture_import_preset_name(settings.preset)},
                {"preserveAlphaCoverage", settings.preserve_alpha_coverage},
                {"semantic", texture_semantic_name(settings.semantic)},
                {"streamingMode", texture_streaming_mode_name(settings.streaming_mode)},
                {"wrapU", texture_address_mode_name(settings.wrap_u)},
                {"wrapV", texture_address_mode_name(settings.wrap_v)}}
        .dump();
}

texture_preprocess_result_type preprocess_texture_for_cook(texture_data texture,
                                                           const texture_import_settings& settings,
                                                           const assets::cook_target& target)
{
    texture_preprocess_result result;
    result.metadata.source_width = texture.width;
    result.metadata.source_height = texture.height;
    result.metadata.requested_max_size = settings.max_size;
    result.metadata.resolved_max_size = std::min(settings.max_size, platform_max_size(target));
    result.metadata.power_of_two = settings.power_of_two;
    result.metadata.compression = settings.compression;
    result.metadata.min_filter = settings.min_filter;
    result.metadata.mag_filter = settings.mag_filter;
    result.metadata.mip_filter = settings.mip_filter;
    result.metadata.wrap_u = settings.wrap_u;
    result.metadata.wrap_v = settings.wrap_v;
    result.metadata.anisotropy = settings.anisotropy;
    result.metadata.lod_bias = settings.lod_bias;
    result.metadata.minimum_lod = settings.minimum_lod;
    result.metadata.maximum_lod = settings.maximum_lod;
    result.metadata.alpha_coverage_threshold = settings.alpha_coverage_threshold;

    texture.semantic = settings.semantic;
    texture.color_space = settings.color_space;
    texture.format = format_for_color_space(texture.format, settings.color_space);
    if (texture.width == 0 || texture.height == 0 || texture.mips.empty())
        return texture_preprocess_result_type::failure("texture source has no mip payload");

    std::uint32_t desired_width = texture.width;
    std::uint32_t desired_height = texture.height;
    const auto largest = std::max(desired_width, desired_height);
    if (largest > result.metadata.resolved_max_size)
    {
        const float scale = static_cast<float>(result.metadata.resolved_max_size) / static_cast<float>(largest);
        desired_width = std::max(1u, static_cast<std::uint32_t>(std::floor(desired_width * scale)));
        desired_height = std::max(1u, static_cast<std::uint32_t>(std::floor(desired_height * scale)));
        result.metadata.resized = true;
    }
    if (settings.power_of_two != texture_power_of_two_policy::preserve)
    {
        const auto adjust =
            settings.power_of_two == texture_power_of_two_policy::resize_up ? next_power_of_two : previous_power_of_two;
        const auto adjusted_width = std::min(result.metadata.resolved_max_size, adjust(desired_width));
        const auto adjusted_height = std::min(result.metadata.resolved_max_size, adjust(desired_height));
        result.metadata.power_of_two_adjusted = adjusted_width != desired_width || adjusted_height != desired_height;
        desired_width = adjusted_width;
        desired_height = adjusted_height;
    }

    const bool rgba8 = texture.format == texture_format::rgba8_unorm || texture.format == texture_format::rgba8_srgb;
    if (!texture.compressed && rgba8 && texture.has_pixels())
    {
        const auto& base_mip = texture.mips.front();
        const auto base = std::span(texture.pixels).subspan(base_mip.offset, base_mip.size);
        auto level = desired_width == texture.width && desired_height == texture.height
                         ? std::vector<std::byte>(base.begin(), base.end())
                         : resize_rgba8(base, texture.width, texture.height, desired_width, desired_height);
        texture.width = desired_width;
        texture.height = desired_height;
        rebuild_rgba8_mips(texture, settings, std::move(level));
        result.metadata.generated_mips = settings.generate_mips && texture.mips.size() > 1u;
        result.metadata.normal_mips_renormalized =
            settings.semantic == texture_semantic::normal && result.metadata.generated_mips;
        result.metadata.alpha_coverage_preserved = settings.preserve_alpha_coverage && result.metadata.generated_mips;
    }
    else
    {
        if (desired_width != texture.width || desired_height != texture.height)
        {
            std::size_t selected{};
            while (selected + 1u < texture.mips.size() &&
                   (texture.mips[selected].width > desired_width || texture.mips[selected].height > desired_height))
                ++selected;
            if (selected == 0u || texture.mips[selected].width > desired_width ||
                texture.mips[selected].height > desired_height)
                return texture_preprocess_result_type::failure(
                    "compressed texture cannot satisfy the requested resize without a matching authored mip");
            auto storage = texture.has_encoded_mips() ? texture.encoded : texture.pixels;
            std::vector<std::byte> sliced;
            std::vector<texture_mip_data> mips;
            for (std::size_t index = selected; index < texture.mips.size(); ++index)
            {
                const auto& source = texture.mips[index];
                if (source.offset > storage.size() || source.size > storage.size() - source.offset)
                    return texture_preprocess_result_type::failure("compressed texture mip payload is invalid");
                const auto offset = sliced.size();
                sliced.insert(sliced.end(), storage.begin() + static_cast<std::ptrdiff_t>(source.offset),
                              storage.begin() + static_cast<std::ptrdiff_t>(source.offset + source.size));
                mips.push_back({.width = source.width, .height = source.height, .offset = offset, .size = source.size});
                if (!settings.generate_mips) break;
            }
            if (texture.has_encoded_mips())
                texture.encoded = std::move(sliced);
            else
                texture.pixels = std::move(sliced);
            texture.mips = std::move(mips);
            texture.width = texture.mips.front().width;
            texture.height = texture.mips.front().height;
            texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
        }
        else if (!settings.generate_mips && texture.mips.size() > 1u)
        {
            const auto source = texture.mips.front();
            auto& storage = texture.has_encoded_mips() ? texture.encoded : texture.pixels;
            std::vector<std::byte> base(storage.begin() + static_cast<std::ptrdiff_t>(source.offset),
                                        storage.begin() + static_cast<std::ptrdiff_t>(source.offset + source.size));
            storage = std::move(base);
            texture.mips = {{.width = source.width, .height = source.height, .offset = 0, .size = source.size}};
            texture.mip_levels = 1;
        }
        if (settings.semantic == texture_semantic::normal && texture.mips.size() > 1u)
            result.diagnostics.push_back({.severity = assets::asset_diagnostic_severity::warning,
                                          .category = "texture.import",
                                          .message = "Compressed normal-map mips are preserved as authored; per-mip "
                                                     "renormalization requires decoded source data"});
        if (settings.preserve_alpha_coverage && texture.mips.size() > 1u)
            result.diagnostics.push_back({.severity = assets::asset_diagnostic_severity::warning,
                                          .category = "texture.import",
                                          .message = "Compressed alpha mip coverage is preserved as authored"});
    }
    result.texture = std::move(texture);
    return texture_preprocess_result_type::success(std::move(result));
}

texture_cook_processor::texture_cook_processor()
{
    descriptor_ = {.id = assets::cook_processor_ids::texture,
                   .name = "ARC Texture Cooker",
                   .version = 4,
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
    return "arc-texture-cooker-v4:arctex-v2:deterministic-preprocessing:sampling-policy";
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
    auto processed = preprocess_texture_for_cook(std::move(loaded.texture), settings.value(), context.target);
    if (!processed) return cook_failure(context, processed.error());
    auto encoded =
        encode_texture_artifact(processed.value().texture, settings.value().streaming_mode, processed.value().metadata);
    if (!encoded) return cook_failure(context, encoded.error().message);
    assets::cooked_artifact artifact{.name = context.source.source_path.stem().string(),
                                     .extension = ".arctex",
                                     .schema = assets::artifact_schemas::texture,
                                     .schema_version = texture_artifact_schema_version,
                                     .gpu_compressed = processed.value().texture.compressed,
                                     .bytes = std::move(encoded).value()};
    return {.artifacts = {std::move(artifact)}, .diagnostics = std::move(processed).value().diagnostics};
}

} // namespace arc::render::tools

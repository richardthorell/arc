#include <arc/render_tools/texture_cooker.h>

#include <arc/render/texture.h>

#include <nlohmann/json.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#define STB_DXT_IMPLEMENTATION
#include <stb_dxt.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstring>
#include <limits>
#include <numbers>
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

bool rgba8_texture(const texture_data& texture) noexcept
{
    return texture.format == texture_format::rgba8_unorm || texture.format == texture_format::rgba8_srgb;
}

bool has_nonopaque_alpha(const texture_data& texture) noexcept
{
    if (!texture.has_pixels() || texture.mips.empty()) return false;
    const auto& mip = texture.mips.front();
    if (mip.offset > texture.pixels.size() || mip.size > texture.pixels.size() - mip.offset) return false;
    const auto pixels = std::span(texture.pixels).subspan(mip.offset, mip.size);
    for (std::size_t index = 3; index < pixels.size(); index += 4u)
        if (std::to_integer<std::uint8_t>(pixels[index]) != 255u) return true;
    return false;
}

texture_compression_policy resolved_compression_policy(const texture_import_settings& settings) noexcept
{
    if (settings.compression != texture_compression_policy::automatic) return settings.compression;
    switch (settings.semantic)
    {
        case texture_semantic::normal:
            return texture_compression_policy::normal;
        case texture_semantic::occlusion:
        case texture_semantic::metallic_roughness:
        case texture_semantic::clear_coat:
        case texture_semantic::anisotropy:
        case texture_semantic::thickness:
        case texture_semantic::transmission:
        case texture_semantic::lightmap:
            return texture_compression_policy::mask;
        case texture_semantic::environment:
        case texture_semantic::generic_color:
        case texture_semantic::base_color:
        case texture_semantic::emissive:
            return texture_compression_policy::color;
    }
    return texture_compression_policy::color;
}

std::array<std::uint8_t, 8> bc4_palette(std::uint8_t maximum, std::uint8_t minimum) noexcept
{
    std::array<std::uint8_t, 8> palette{};
    palette[0] = maximum;
    palette[1] = minimum;
    if (maximum > minimum)
    {
        for (std::uint32_t index = 1; index <= 6; ++index)
            palette[index + 1u] = static_cast<std::uint8_t>(((7u - index) * static_cast<std::uint32_t>(maximum) +
                                                             index * static_cast<std::uint32_t>(minimum) + 3u) /
                                                            7u);
    }
    else
    {
        for (std::uint32_t index = 1; index <= 4; ++index)
            palette[index + 1u] = static_cast<std::uint8_t>(((5u - index) * static_cast<std::uint32_t>(maximum) +
                                                             index * static_cast<std::uint32_t>(minimum) + 2u) /
                                                            5u);
        palette[6] = 0;
        palette[7] = 255;
    }
    return palette;
}

void encode_bc4_block(std::span<const std::uint8_t, 16> values, std::byte* output) noexcept
{
    const auto [minimum_it, maximum_it] = std::minmax_element(values.begin(), values.end());
    const std::uint8_t minimum = *minimum_it;
    const std::uint8_t maximum = *maximum_it;
    output[0] = static_cast<std::byte>(maximum);
    output[1] = static_cast<std::byte>(minimum);
    const auto palette = bc4_palette(maximum, minimum);
    std::uint64_t indices{};
    for (std::uint32_t texel = 0; texel < 16u; ++texel)
    {
        std::uint32_t best{};
        std::uint32_t best_error = 256u;
        for (std::uint32_t candidate = 0; candidate < palette.size(); ++candidate)
        {
            const auto error = static_cast<std::uint32_t>(
                std::abs(static_cast<int>(values[texel]) - static_cast<int>(palette[candidate])));
            if (error < best_error)
            {
                best = candidate;
                best_error = error;
            }
        }
        indices |= static_cast<std::uint64_t>(best) << (texel * 3u);
    }
    for (std::uint32_t byte = 0; byte < 6u; ++byte)
        output[2u + byte] = static_cast<std::byte>((indices >> (byte * 8u)) & 0xffu);
}

std::array<std::uint8_t, 64> gather_rgba_block(std::span<const std::byte> source, std::uint32_t width,
                                               std::uint32_t height, std::uint32_t block_x, std::uint32_t block_y)
{
    std::array<std::uint8_t, 64> block{};
    for (std::uint32_t y = 0; y < 4u; ++y)
        for (std::uint32_t x = 0; x < 4u; ++x)
        {
            const auto source_x = std::min(width - 1u, block_x * 4u + x);
            const auto source_y = std::min(height - 1u, block_y * 4u + y);
            const auto source_offset = (static_cast<std::size_t>(source_y) * width + source_x) * 4u;
            const auto destination = (y * 4u + x) * 4u;
            for (std::uint32_t channel = 0; channel < 4u; ++channel)
                block[destination + channel] = std::to_integer<std::uint8_t>(source[source_offset + channel]);
        }
    return block;
}

std::vector<std::byte> compress_bc_mip(std::span<const std::byte> source, std::uint32_t width, std::uint32_t height,
                                       texture_format format)
{
    const auto blocks_x = std::max(1u, (width + 3u) / 4u);
    const auto blocks_y = std::max(1u, (height + 3u) / 4u);
    const bool bc1 = format == texture_format::bc1_rgba_unorm || format == texture_format::bc1_rgba_srgb;
    const bool bc4 = format == texture_format::bc4_r_unorm;
    const bool bc5 = format == texture_format::bc5_rg_unorm;
    const std::uint32_t block_bytes = bc1 || bc4 ? 8u : 16u;
    std::vector<std::byte> result(static_cast<std::size_t>(blocks_x) * blocks_y * block_bytes);
    for (std::uint32_t block_y = 0; block_y < blocks_y; ++block_y)
        for (std::uint32_t block_x = 0; block_x < blocks_x; ++block_x)
        {
            const auto block = gather_rgba_block(source, width, height, block_x, block_y);
            auto* destination = result.data() + (static_cast<std::size_t>(block_y) * blocks_x + block_x) * block_bytes;
            if (bc1)
            {
                stb_compress_dxt_block(reinterpret_cast<unsigned char*>(destination), block.data(), 0,
                                       STB_DXT_HIGHQUAL);
            }
            else if (bc4 || bc5)
            {
                std::array<std::uint8_t, 16> channel{};
                for (std::uint32_t texel = 0; texel < 16u; ++texel)
                    channel[texel] = block[texel * 4u];
                encode_bc4_block(channel, destination);
                if (bc5)
                {
                    for (std::uint32_t texel = 0; texel < 16u; ++texel)
                        channel[texel] = block[texel * 4u + 1u];
                    encode_bc4_block(channel, destination + 8u);
                }
            }
            else
            {
                stb_compress_dxt_block(reinterpret_cast<unsigned char*>(destination), block.data(), 1,
                                       STB_DXT_HIGHQUAL);
            }
        }
    return result;
}

texture_format bc_format_for(const texture_data& texture, texture_compression_policy policy) noexcept
{
    switch (policy)
    {
        case texture_compression_policy::normal:
            return texture_format::bc5_rg_unorm;
        case texture_compression_policy::mask:
            if (texture.semantic == texture_semantic::occlusion) return texture_format::bc4_r_unorm;
            return texture.color_space == texture_color_space::srgb ? texture_format::bc3_rgba_srgb
                                                                    : texture_format::bc3_rgba_unorm;
        case texture_compression_policy::color:
        case texture_compression_policy::automatic:
            if (has_nonopaque_alpha(texture))
                return texture.color_space == texture_color_space::srgb ? texture_format::bc3_rgba_srgb
                                                                        : texture_format::bc3_rgba_unorm;
            return texture.color_space == texture_color_space::srgb ? texture_format::bc1_rgba_srgb
                                                                    : texture_format::bc1_rgba_unorm;
        case texture_compression_policy::hdr:
        case texture_compression_policy::uncompressed:
            return texture.format;
    }
    return texture.format;
}

bool compress_texture_for_target(texture_data& texture, const texture_import_settings& settings,
                                 const assets::cook_target& target, std::vector<assets::asset_diagnostic>& diagnostics,
                                 std::string& error)
{
    if (texture.compressed)
    {
        if (settings.compression == texture_compression_policy::uncompressed)
            diagnostics.push_back({.severity = assets::asset_diagnostic_severity::warning,
                                   .category = "texture.compression",
                                   .message = "Authored compressed texture is preserved because the cooker does not "
                                              "decode DDS blocks for uncompressed output"});
        return true;
    }
    const auto policy = resolved_compression_policy(settings);
    if (policy == texture_compression_policy::uncompressed) return true;
    if (target.textures != assets::cook_texture_family::bc)
    {
        diagnostics.push_back({.severity = assets::asset_diagnostic_severity::warning,
                               .category = "texture.compression",
                               .message = "Requested target texture family has no production encoder yet; keeping "
                                          "deterministic uncompressed RGBA payload"});
        return true;
    }
    if (policy == texture_compression_policy::hdr)
    {
        diagnostics.push_back({.severity = assets::asset_diagnostic_severity::warning,
                               .category = "texture.compression",
                               .message = "BC6H encoding is not available yet; HDR texture remains uncompressed"});
        return true;
    }
    if (!rgba8_texture(texture) || !texture.has_pixels())
    {
        error = "BC compression currently requires decoded RGBA8 source data";
        return false;
    }
    const auto target_format = bc_format_for(texture, policy);
    std::vector<std::byte> encoded;
    std::vector<texture_mip_data> mips;
    encoded.reserve(texture.pixels.size());
    mips.reserve(texture.mips.size());
    for (const auto& mip : texture.mips)
    {
        if (mip.offset > texture.pixels.size() || mip.size > texture.pixels.size() - mip.offset)
        {
            error = "decoded texture mip payload is invalid before BC compression";
            return false;
        }
        const auto source = std::span(texture.pixels).subspan(mip.offset, mip.size);
        auto compressed = compress_bc_mip(source, mip.width, mip.height, target_format);
        const auto offset = encoded.size();
        const auto size = compressed.size();
        encoded.insert(encoded.end(), compressed.begin(), compressed.end());
        mips.push_back({.width = mip.width, .height = mip.height, .offset = offset, .size = size});
    }
    texture.encoded = std::move(encoded);
    texture.pixels.clear();
    texture.mips = std::move(mips);
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
    texture.format = target_format;
    texture.compressed = true;
    texture.dds = false;
    return true;
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

float sinc(float value) noexcept
{
    if (std::abs(value) < 0.00001f) return 1.0f;
    const float x = std::numbers::pi_v<float> * value;
    return std::sin(x) / x;
}

float bessel_i0(float value) noexcept
{
    const float x = value * value * 0.25f;
    float sum = 1.0f;
    float term = 1.0f;
    for (std::uint32_t index = 1; index <= 8; ++index)
    {
        term *= x / static_cast<float>(index * index);
        sum += term;
    }
    return sum;
}

float mip_kernel(texture_mip_generation_filter filter, float distance) noexcept
{
    const float x = std::abs(distance);
    switch (filter)
    {
        case texture_mip_generation_filter::nearest:
            return x < 0.5f ? 1.0f : 0.0f;
        case texture_mip_generation_filter::box:
            return x <= 1.0f ? 1.0f : 0.0f;
        case texture_mip_generation_filter::bilinear:
            return std::max(0.0f, 1.0f - x);
        case texture_mip_generation_filter::bicubic:
        {
            if (x >= 2.0f) return 0.0f;
            if (x <= 1.0f) return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
            return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
        }
        case texture_mip_generation_filter::lanczos:
            return x < 3.0f ? sinc(x) * sinc(x / 3.0f) : 0.0f;
        case texture_mip_generation_filter::kaiser:
        {
            if (x >= 3.0f) return 0.0f;
            constexpr float beta = 4.0f;
            const float ratio = x / 3.0f;
            const float window = bessel_i0(beta * std::sqrt(std::max(0.0f, 1.0f - ratio * ratio))) / bessel_i0(beta);
            return sinc(x) * window;
        }
    }
    return 0.0f;
}

float mip_filter_radius(texture_mip_generation_filter filter) noexcept
{
    switch (filter)
    {
        case texture_mip_generation_filter::nearest:
            return 0.5f;
        case texture_mip_generation_filter::box:
        case texture_mip_generation_filter::bilinear:
            return 1.0f;
        case texture_mip_generation_filter::bicubic:
            return 2.0f;
        case texture_mip_generation_filter::lanczos:
        case texture_mip_generation_filter::kaiser:
            return 3.0f;
    }
    return 1.0f;
}

float decode_mip_channel(std::span<const std::byte> source, std::size_t index, std::uint32_t channel,
                         const texture_import_settings& settings) noexcept
{
    float value = static_cast<float>(std::to_integer<std::uint8_t>(source[index + channel])) / 255.0f;
    if (settings.semantic == texture_semantic::normal && channel < 3u) return value * 2.0f - 1.0f;
    if (settings.color_space == texture_color_space::srgb && channel < 3u) return srgb_to_linear_channel(value);
    return value;
}

std::vector<std::byte> downsample_rgba8(std::span<const std::byte> source, std::uint32_t width, std::uint32_t height,
                                        std::uint32_t next_width, std::uint32_t next_height,
                                        const texture_import_settings& settings)
{
    std::vector<std::byte> next(static_cast<std::size_t>(next_width) * next_height * 4u);
    const float scale_x = static_cast<float>(width) / static_cast<float>(next_width);
    const float scale_y = static_cast<float>(height) / static_cast<float>(next_height);
    const float radius = mip_filter_radius(settings.mip_generation_filter);
    for (std::uint32_t y = 0; y < next_height; ++y)
        for (std::uint32_t x = 0; x < next_width; ++x)
        {
            const auto destination = (static_cast<std::size_t>(y) * next_width + x) * 4u;
            const float center_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const float center_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
            const int minimum_x = static_cast<int>(std::floor(center_x - radius));
            const int maximum_x = static_cast<int>(std::ceil(center_x + radius));
            const int minimum_y = static_cast<int>(std::floor(center_y - radius));
            const int maximum_y = static_cast<int>(std::ceil(center_y + radius));
            float accumulated[4]{};
            float total_weight{};
            for (int source_y = minimum_y; source_y <= maximum_y; ++source_y)
                for (int source_x = minimum_x; source_x <= maximum_x; ++source_x)
                {
                    const float weight = mip_kernel(settings.mip_generation_filter, center_x - source_x) *
                                         mip_kernel(settings.mip_generation_filter, center_y - source_y);
                    if (std::abs(weight) < 0.000001f) continue;
                    const auto clamped_x =
                        static_cast<std::uint32_t>(std::clamp(source_x, 0, static_cast<int>(width) - 1));
                    const auto clamped_y =
                        static_cast<std::uint32_t>(std::clamp(source_y, 0, static_cast<int>(height) - 1));
                    const auto source_index = (static_cast<std::size_t>(clamped_y) * width + clamped_x) * 4u;
                    for (std::uint32_t channel = 0; channel < 4u; ++channel)
                        accumulated[channel] += decode_mip_channel(source, source_index, channel, settings) * weight;
                    total_weight += weight;
                }
            if (std::abs(total_weight) < 0.000001f) total_weight = 1.0f;
            for (auto& value : accumulated)
                value /= total_weight;
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
                for (std::uint32_t channel = 0; channel < 3u; ++channel)
                    next[destination + channel] = byte_channel(accumulated[channel] * 0.5f + 0.5f);
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

void sharpen_mip(std::vector<std::byte>& pixels, std::uint32_t width, std::uint32_t height,
                 const texture_import_settings& settings)
{
    if (settings.mip_sharpen <= 0.0f || settings.semantic == texture_semantic::normal || width < 2u || height < 2u)
        return;
    const auto source = pixels;
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
            for (std::uint32_t channel = 0; channel < 3u; ++channel)
            {
                const auto sample = [&](int sx, int sy)
                {
                    sx = std::clamp(sx, 0, static_cast<int>(width) - 1);
                    sy = std::clamp(sy, 0, static_cast<int>(height) - 1);
                    const auto index =
                        (static_cast<std::size_t>(sy) * width + static_cast<std::uint32_t>(sx)) * 4u + channel;
                    float value = static_cast<float>(std::to_integer<std::uint8_t>(source[index])) / 255.0f;
                    return settings.color_space == texture_color_space::srgb ? srgb_to_linear_channel(value) : value;
                };
                const float center = sample(static_cast<int>(x), static_cast<int>(y));
                const float blur = (sample(static_cast<int>(x) - 1, static_cast<int>(y)) +
                                    sample(static_cast<int>(x) + 1, static_cast<int>(y)) +
                                    sample(static_cast<int>(x), static_cast<int>(y) - 1) +
                                    sample(static_cast<int>(x), static_cast<int>(y) + 1)) *
                                   0.25f;
                float value = std::clamp(center + (center - blur) * settings.mip_sharpen, 0.0f, 1.0f);
                if (settings.color_space == texture_color_space::srgb) value = linear_to_srgb_channel(value);
                const auto index = (static_cast<std::size_t>(y) * width + x) * 4u + channel;
                pixels[index] = byte_channel(value);
            }
}

void deband_mip(std::vector<std::byte>& pixels, std::uint32_t width, std::uint32_t height,
                const texture_import_settings& settings)
{
    if (!settings.deband_mips || settings.deband_strength <= 0.0f || settings.semantic == texture_semantic::normal ||
        width < 2u || height < 2u)
        return;
    const auto source = pixels;
    const float threshold = 2.0f + settings.deband_strength * 10.0f;
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
            for (std::uint32_t channel = 0; channel < 3u; ++channel)
            {
                float minimum = 255.0f;
                float maximum = 0.0f;
                float sum{};
                std::uint32_t count{};
                for (int oy = -1; oy <= 1; ++oy)
                    for (int ox = -1; ox <= 1; ++ox)
                    {
                        const auto sx = static_cast<std::uint32_t>(
                            std::clamp(static_cast<int>(x) + ox, 0, static_cast<int>(width) - 1));
                        const auto sy = static_cast<std::uint32_t>(
                            std::clamp(static_cast<int>(y) + oy, 0, static_cast<int>(height) - 1));
                        const auto index = (static_cast<std::size_t>(sy) * width + sx) * 4u + channel;
                        const float value = static_cast<float>(std::to_integer<std::uint8_t>(source[index]));
                        minimum = std::min(minimum, value);
                        maximum = std::max(maximum, value);
                        sum += value;
                        ++count;
                    }
                if (maximum - minimum <= threshold)
                {
                    const auto index = (static_cast<std::size_t>(y) * width + x) * 4u + channel;
                    const float center = static_cast<float>(std::to_integer<std::uint8_t>(source[index]));
                    const float mixed =
                        std::lerp(center, sum / static_cast<float>(count), settings.deband_strength * 0.35f);
                    pixels[index] =
                        static_cast<std::byte>(static_cast<std::uint8_t>(std::clamp(std::lround(mixed), 0l, 255l)));
                }
            }
}

void dither_mip(std::vector<std::byte>& pixels, std::uint32_t width, std::uint32_t height,
                const texture_import_settings& settings)
{
    if (!settings.dither_mips || settings.semantic == texture_semantic::normal) return;
    static constexpr int bayer[4][4]{{0, 8, 2, 10}, {12, 4, 14, 6}, {3, 11, 1, 9}, {15, 7, 13, 5}};
    for (std::uint32_t y = 0; y < height; ++y)
        for (std::uint32_t x = 0; x < width; ++x)
        {
            const float noise = (static_cast<float>(bayer[y & 3u][x & 3u]) / 15.0f - 0.5f);
            const auto base = (static_cast<std::size_t>(y) * width + x) * 4u;
            for (std::uint32_t channel = 0; channel < 3u; ++channel)
            {
                const int value = static_cast<int>(std::to_integer<std::uint8_t>(pixels[base + channel]));
                pixels[base + channel] = static_cast<std::byte>(
                    static_cast<std::uint8_t>(std::clamp(value + static_cast<int>(std::round(noise)), 0, 255)));
            }
        }
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
        sharpen_mip(next, next_width, next_height, settings);
        deband_mip(next, next_width, next_height, settings);
        dither_mip(next, next_width, next_height, settings);
        if (settings.preserve_alpha_coverage)
            preserve_alpha_coverage(next, settings.alpha_coverage_threshold, target_coverage);
        level = std::move(next);
        width = next_width;
        height = next_height;
    }
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
}

float curve_slope(const texture_curve_point& left, const texture_curve_point& right) noexcept
{
    const float width = right.x - left.x;
    return width > 0.000001f ? (right.y - left.y) / width : 0.0f;
}

float automatic_curve_tangent(const texture_curve& curve, std::size_t index) noexcept
{
    if (curve.points.size() < 2u) return 1.0f;
    if (index == 0u) return curve_slope(curve.points[0], curve.points[1]);
    if (index + 1u >= curve.points.size()) return curve_slope(curve.points[index - 1u], curve.points[index]);
    const float left = curve_slope(curve.points[index - 1u], curve.points[index]);
    const float right = curve_slope(curve.points[index], curve.points[index + 1u]);
    if (left * right <= 0.0f) return 0.0f;
    return (left + right) * 0.5f;
}

float mapped_channel(const std::array<float, 4>& rgba, texture_channel_source source) noexcept
{
    switch (source)
    {
        case texture_channel_source::red:
            return rgba[0];
        case texture_channel_source::green:
            return rgba[1];
        case texture_channel_source::blue:
            return rgba[2];
        case texture_channel_source::alpha:
            return rgba[3];
        case texture_channel_source::zero:
            return 0.0f;
        case texture_channel_source::one:
            return 1.0f;
    }
    return 0.0f;
}

void apply_stage3_rgba8(std::vector<std::byte>& pixels, const texture_import_settings& settings) noexcept
{
    const texture_channel_source channels[4]{settings.channel_r, settings.channel_g, settings.channel_b,
                                             settings.channel_a};
    const bool invert[4]{settings.invert_r, settings.invert_g, settings.invert_b, settings.invert_a};
    const float level_range = std::max(0.0001f, settings.input_white - settings.input_black);
    const float exposure = std::exp2(settings.brightness);
    for (std::size_t offset = 0; offset + 3u < pixels.size(); offset += 4u)
    {
        std::array<float, 4> source{};
        for (std::size_t channel = 0; channel < 4u; ++channel)
            source[channel] = static_cast<float>(std::to_integer<std::uint8_t>(pixels[offset + channel])) / 255.0f;
        std::array<float, 4> value{};
        for (std::size_t channel = 0; channel < 4u; ++channel)
        {
            value[channel] = mapped_channel(source, channels[channel]);
            if (invert[channel]) value[channel] = 1.0f - value[channel];
        }

        if (settings.semantic != texture_semantic::normal)
        {
            for (std::size_t channel = 0; channel < 3u; ++channel)
            {
                if (settings.color_space == texture_color_space::srgb)
                    value[channel] = srgb_to_linear_channel(value[channel]);
                value[channel] = std::clamp((value[channel] - settings.input_black) / level_range, 0.0f, 1.0f);
                if (settings.curves_enabled)
                {
                    const texture_curve* channel_curve = channel == 0u   ? &settings.curve_r
                                                         : channel == 1u ? &settings.curve_g
                                                                         : &settings.curve_b;
                    value[channel] = evaluate_texture_curve(*channel_curve, value[channel]);
                    value[channel] = evaluate_texture_curve(settings.curve_master, value[channel]);
                }
                value[channel] = std::pow(std::clamp(value[channel], 0.0f, 1.0f), 1.0f / settings.gamma);
                value[channel] *= exposure;
                value[channel] = (value[channel] - 0.5f) * settings.contrast + 0.5f;
            }
            const float luminance = value[0] * 0.2126f + value[1] * 0.7152f + value[2] * 0.0722f;
            for (std::size_t channel = 0; channel < 3u; ++channel)
                value[channel] = luminance + (value[channel] - luminance) * settings.saturation;
            const float maximum = std::max({value[0], value[1], value[2]});
            const float minimum = std::min({value[0], value[1], value[2]});
            const float vibrance = 1.0f + settings.vibrance * (1.0f - std::clamp(maximum - minimum, 0.0f, 1.0f));
            for (std::size_t channel = 0; channel < 3u; ++channel)
                value[channel] = luminance + (value[channel] - luminance) * vibrance;
            value[0] *= settings.tint_r;
            value[1] *= settings.tint_g;
            value[2] *= settings.tint_b;
            for (std::size_t channel = 0; channel < 3u; ++channel)
            {
                value[channel] = settings.output_black + std::clamp(value[channel], 0.0f, 1.0f) *
                                                             (settings.output_white - settings.output_black);
                value[channel] = std::clamp(value[channel], 0.0f, 1.0f);
                if (settings.color_space == texture_color_space::srgb)
                    value[channel] = linear_to_srgb_channel(value[channel]);
            }
        }
        if (settings.curves_enabled) value[3] = evaluate_texture_curve(settings.curve_a, value[3]);
        for (std::size_t channel = 0; channel < 4u; ++channel)
            pixels[offset + channel] = byte_channel(value[channel]);
    }
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
    switch (value)
    {
        case texture_mip_generation_filter::nearest:
            return "nearest";
        case texture_mip_generation_filter::box:
            return "box";
        case texture_mip_generation_filter::bilinear:
            return "bilinear";
        case texture_mip_generation_filter::bicubic:
            return "bicubic";
        case texture_mip_generation_filter::lanczos:
            return "lanczos";
        case texture_mip_generation_filter::kaiser:
            return "kaiser";
    }
    return "box";
}
std::string_view texture_channel_source_name(texture_channel_source value) noexcept
{
    switch (value)
    {
        case texture_channel_source::red:
            return "red";
        case texture_channel_source::green:
            return "green";
        case texture_channel_source::blue:
            return "blue";
        case texture_channel_source::alpha:
            return "alpha";
        case texture_channel_source::zero:
            return "zero";
        case texture_channel_source::one:
            return "one";
    }
    return "zero";
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
    return enum_value<texture_mip_generation_filter>(value, {{"nearest", texture_mip_generation_filter::nearest},
                                                             {"box", texture_mip_generation_filter::box},
                                                             {"bilinear", texture_mip_generation_filter::bilinear},
                                                             {"bicubic", texture_mip_generation_filter::bicubic},
                                                             {"lanczos", texture_mip_generation_filter::lanczos},
                                                             {"kaiser", texture_mip_generation_filter::kaiser}});
}
std::optional<texture_channel_source> parse_texture_channel_source(std::string_view value) noexcept
{
    return enum_value<texture_channel_source>(value, {{"red", texture_channel_source::red},
                                                      {"green", texture_channel_source::green},
                                                      {"blue", texture_channel_source::blue},
                                                      {"alpha", texture_channel_source::alpha},
                                                      {"zero", texture_channel_source::zero},
                                                      {"one", texture_channel_source::one}});
}

std::string_view texture_curve_interpolation_name(texture_curve_interpolation value) noexcept
{
    switch (value)
    {
        case texture_curve_interpolation::constant:
            return "constant";
        case texture_curve_interpolation::linear:
            return "linear";
        case texture_curve_interpolation::smooth:
            return "smooth";
        case texture_curve_interpolation::manual:
            return "manual";
    }
    return "smooth";
}

core::result<texture_curve, std::string> parse_texture_curve(std::string_view canonical_json)
{
    const auto document = json::parse(canonical_json, nullptr, false);
    if (!document.is_array() || document.size() < 2u || document.size() > 32u)
        return core::result<texture_curve, std::string>::failure("texture curve must contain between 2 and 32 points");
    texture_curve curve;
    curve.points.clear();
    float previous_x = -1.0f;
    for (const auto& point : document)
    {
        if (!point.is_object())
            return core::result<texture_curve, std::string>::failure("texture curve point must be an object");
        texture_curve_point value;
        value.x = point.value("x", -1.0f);
        value.y = point.value("y", -1.0f);
        value.in_tangent = point.value("inTangent", 1.0f);
        value.out_tangent = point.value("outTangent", 1.0f);
        const auto interpolation = lowercase(point.value("interpolation", std::string{"smooth"}));
        if (interpolation == "constant")
            value.interpolation = texture_curve_interpolation::constant;
        else if (interpolation == "linear")
            value.interpolation = texture_curve_interpolation::linear;
        else if (interpolation == "smooth")
            value.interpolation = texture_curve_interpolation::smooth;
        else if (interpolation == "manual")
            value.interpolation = texture_curve_interpolation::manual;
        else
            return core::result<texture_curve, std::string>::failure("texture curve interpolation is invalid");
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.in_tangent) ||
            !std::isfinite(value.out_tangent) || value.x < 0.0f || value.x > 1.0f || value.y < 0.0f || value.y > 1.0f ||
            value.x <= previous_x || std::abs(value.in_tangent) > 16.0f || std::abs(value.out_tangent) > 16.0f)
            return core::result<texture_curve, std::string>::failure("texture curve point is invalid");
        previous_x = value.x;
        curve.points.push_back(value);
    }
    if (curve.points.front().x != 0.0f || curve.points.back().x != 1.0f)
        return core::result<texture_curve, std::string>::failure("texture curve endpoints must be at x=0 and x=1");
    return core::result<texture_curve, std::string>::success(std::move(curve));
}

std::string serialize_texture_curve(const texture_curve& curve)
{
    json result = json::array();
    for (const auto& point : curve.points)
        result.push_back({{"x", point.x},
                          {"y", point.y},
                          {"inTangent", point.in_tangent},
                          {"outTangent", point.out_tangent},
                          {"interpolation", texture_curve_interpolation_name(point.interpolation)}});
    return result.dump();
}

float evaluate_texture_curve(const texture_curve& curve, float input) noexcept
{
    if (curve.points.empty()) return std::clamp(input, 0.0f, 1.0f);
    const float value = std::clamp(input, 0.0f, 1.0f);
    if (value <= curve.points.front().x) return curve.points.front().y;
    if (value >= curve.points.back().x) return curve.points.back().y;
    auto right = std::upper_bound(curve.points.begin(), curve.points.end(), value,
                                  [](float x, const texture_curve_point& point) { return x < point.x; });
    const auto right_index = static_cast<std::size_t>(std::distance(curve.points.begin(), right));
    const auto left_index = right_index - 1u;
    const auto& left = curve.points[left_index];
    const auto& next = curve.points[right_index];
    const float width = next.x - left.x;
    const float t = width > 0.0f ? (value - left.x) / width : 0.0f;
    if (left.interpolation == texture_curve_interpolation::constant) return left.y;
    if (left.interpolation == texture_curve_interpolation::linear) return std::lerp(left.y, next.y, t);
    const float left_tangent = left.interpolation == texture_curve_interpolation::manual
                                   ? left.out_tangent
                                   : automatic_curve_tangent(curve, left_index);
    const float right_tangent = next.interpolation == texture_curve_interpolation::manual
                                    ? next.in_tangent
                                    : automatic_curve_tangent(curve, right_index);
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;
    const float h10 = t3 - 2.0f * t2 + t;
    const float h01 = -2.0f * t3 + 3.0f * t2;
    const float h11 = t3 - t2;
    return std::clamp(h00 * left.y + h10 * width * left_tangent + h01 * next.y + h11 * width * right_tangent, 0.0f,
                      1.0f);
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
    if (settings_version >= 5)
    {
        if (const auto error = parse_string_field("channelR", parse_texture_channel_source, settings.channel_r))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_string_field("channelG", parse_texture_channel_source, settings.channel_g))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_string_field("channelB", parse_texture_channel_source, settings.channel_b))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_string_field("channelA", parse_texture_channel_source, settings.channel_a))
            return texture_import_settings_result::failure(*error);
        settings.brightness = document.value("brightness", settings.brightness);
        settings.gamma = document.value("gamma", settings.gamma);
        settings.contrast = document.value("contrast", settings.contrast);
        settings.saturation = document.value("saturation", settings.saturation);
        settings.vibrance = document.value("vibrance", settings.vibrance);
        settings.tint_r = document.value("tintR", settings.tint_r);
        settings.tint_g = document.value("tintG", settings.tint_g);
        settings.tint_b = document.value("tintB", settings.tint_b);
        settings.input_black = document.value("inputBlack", settings.input_black);
        settings.input_white = document.value("inputWhite", settings.input_white);
        settings.output_black = document.value("outputBlack", settings.output_black);
        settings.output_white = document.value("outputWhite", settings.output_white);
        settings.invert_r = document.value("invertR", settings.invert_r);
        settings.invert_g = document.value("invertG", settings.invert_g);
        settings.invert_b = document.value("invertB", settings.invert_b);
        settings.invert_a = document.value("invertA", settings.invert_a);
    }
    if (settings_version >= 6)
    {
        settings.mip_sharpen = document.value("mipSharpen", settings.mip_sharpen);
        settings.dither_mips = document.value("ditherMips", settings.dither_mips);
        settings.deband_mips = document.value("debandMips", settings.deband_mips);
        settings.deband_strength = document.value("debandStrength", settings.deband_strength);
    }
    if (settings_version >= 7)
    {
        settings.curves_enabled = document.value("curvesEnabled", settings.curves_enabled);
        const auto parse_curve_field = [&](const char* name, texture_curve& target) -> std::optional<std::string>
        {
            const auto field = document.find(name);
            if (field == document.end()) return std::nullopt;
            const auto parsed = parse_texture_curve(field->dump());
            if (!parsed) return parsed.error();
            target = parsed.value();
            return std::nullopt;
        };
        if (const auto error = parse_curve_field("curveMaster", settings.curve_master))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_curve_field("curveR", settings.curve_r))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_curve_field("curveG", settings.curve_g))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_curve_field("curveB", settings.curve_b))
            return texture_import_settings_result::failure(*error);
        if (const auto error = parse_curve_field("curveA", settings.curve_a))
            return texture_import_settings_result::failure(*error);
    }
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
        settings.alpha_coverage_threshold < 0.0f || settings.alpha_coverage_threshold > 1.0f ||
        !std::isfinite(settings.brightness) || settings.brightness < -8.0f || settings.brightness > 8.0f ||
        !std::isfinite(settings.gamma) || settings.gamma < 0.05f || settings.gamma > 8.0f ||
        !std::isfinite(settings.contrast) || settings.contrast < 0.0f || settings.contrast > 4.0f ||
        !std::isfinite(settings.saturation) || settings.saturation < 0.0f || settings.saturation > 4.0f ||
        !std::isfinite(settings.vibrance) || settings.vibrance < -1.0f || settings.vibrance > 1.0f ||
        !std::isfinite(settings.tint_r) || !std::isfinite(settings.tint_g) || !std::isfinite(settings.tint_b) ||
        settings.tint_r < 0.0f || settings.tint_r > 4.0f || settings.tint_g < 0.0f || settings.tint_g > 4.0f ||
        settings.tint_b < 0.0f || settings.tint_b > 4.0f || !std::isfinite(settings.input_black) ||
        !std::isfinite(settings.input_white) || settings.input_black < 0.0f || settings.input_white > 1.0f ||
        settings.input_white <= settings.input_black || !std::isfinite(settings.output_black) ||
        !std::isfinite(settings.output_white) || settings.output_black < 0.0f || settings.output_white > 1.0f ||
        settings.output_white < settings.output_black || !std::isfinite(settings.mip_sharpen) ||
        settings.mip_sharpen < 0.0f || settings.mip_sharpen > 2.0f || !std::isfinite(settings.deband_strength) ||
        settings.deband_strength < 0.0f || settings.deband_strength > 1.0f)
        return texture_import_settings_result::failure("texture import settings contain an invalid numeric value");
    return texture_import_settings_result::success(settings);
}

std::string serialize_texture_import_settings(const texture_import_settings& settings)
{
    return json{{"alphaCoverageThreshold", settings.alpha_coverage_threshold},
                {"anisotropy", settings.anisotropy},
                {"brightness", settings.brightness},
                {"channelA", texture_channel_source_name(settings.channel_a)},
                {"channelB", texture_channel_source_name(settings.channel_b)},
                {"channelG", texture_channel_source_name(settings.channel_g)},
                {"channelR", texture_channel_source_name(settings.channel_r)},
                {"colorSpace", texture_color_space_name(settings.color_space)},
                {"compression", texture_compression_policy_name(settings.compression)},
                {"contrast", settings.contrast},
                {"curvesEnabled", settings.curves_enabled},
                {"curveMaster", json::parse(serialize_texture_curve(settings.curve_master))},
                {"curveR", json::parse(serialize_texture_curve(settings.curve_r))},
                {"curveG", json::parse(serialize_texture_curve(settings.curve_g))},
                {"curveB", json::parse(serialize_texture_curve(settings.curve_b))},
                {"curveA", json::parse(serialize_texture_curve(settings.curve_a))},
                {"gamma", settings.gamma},
                {"generateMips", settings.generate_mips},
                {"inputBlack", settings.input_black},
                {"inputWhite", settings.input_white},
                {"invertA", settings.invert_a},
                {"invertB", settings.invert_b},
                {"invertG", settings.invert_g},
                {"invertR", settings.invert_r},
                {"lodBias", settings.lod_bias},
                {"magFilter", texture_filter_mode_name(settings.mag_filter)},
                {"maxSize", settings.max_size},
                {"maximumLod", settings.maximum_lod},
                {"mipSharpen", settings.mip_sharpen},
                {"ditherMips", settings.dither_mips},
                {"debandMips", settings.deband_mips},
                {"debandStrength", settings.deband_strength},
                {"minFilter", texture_filter_mode_name(settings.min_filter)},
                {"minimumLod", settings.minimum_lod},
                {"mipFilter", texture_mip_filter_mode_name(settings.mip_filter)},
                {"mipGenerationFilter", texture_mip_generation_filter_name(settings.mip_generation_filter)},
                {"outputBlack", settings.output_black},
                {"outputWhite", settings.output_white},
                {"powerOfTwo", texture_power_of_two_policy_name(settings.power_of_two)},
                {"preset", texture_import_preset_name(settings.preset)},
                {"preserveAlphaCoverage", settings.preserve_alpha_coverage},
                {"saturation", settings.saturation},
                {"semantic", texture_semantic_name(settings.semantic)},
                {"streamingMode", texture_streaming_mode_name(settings.streaming_mode)},
                {"tintB", settings.tint_b},
                {"tintG", settings.tint_g},
                {"tintR", settings.tint_r},
                {"vibrance", settings.vibrance},
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
        apply_stage3_rgba8(level, settings);
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
                   .version = 9,
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
    return "arc-texture-cooker-v9:arctex-v3:checksummed-range-streaming:stb-dxt";
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
    std::string compression_error;
    if (!compress_texture_for_target(processed.value().texture, settings.value(), context.target,
                                     processed.value().diagnostics, compression_error))
        return cook_failure(context, std::move(compression_error));
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

#include <arc/render/texture.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>

#if defined(ARC_RENDER_HAS_TINYEXR) && defined(ARC_RENDER_HAS_STB)
#if defined(_MSC_VER)
#pragma warning(push)
// Third-party TinyEXR/STB code triggers unreachable-code and legacy sprintf warnings under MSVC /WX.
#pragma warning(disable : 4702 4996)
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#include <tinyexr.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
#endif

namespace arc::render
{
namespace
{
std::vector<std::byte> read_exr_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size <= 0) return {};
    stream.seekg(0, std::ios::beg);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

void append_rgba32f_mip(texture_data& texture, const float* pixels, std::uint32_t width, std::uint32_t height)
{
    const auto float_count = static_cast<std::size_t>(width) * height * 4u;
    const auto byte_count = float_count * sizeof(float);
    const auto offset = texture.pixels.size();
    texture.pixels.resize(offset + byte_count);
    std::memcpy(texture.pixels.data() + offset, pixels, byte_count);
    texture.mips.push_back({.width = width, .height = height, .offset = offset, .size = byte_count});
}

std::vector<float> downsample_rgba32f(const float* source, std::uint32_t width, std::uint32_t height,
                                     std::uint32_t next_width, std::uint32_t next_height)
{
    std::vector<float> result(static_cast<std::size_t>(next_width) * next_height * 4u);
    for (std::uint32_t y = 0; y < next_height; ++y)
    {
        const auto source_y_begin = std::min(y * 2u, height - 1u);
        const auto source_y_end = std::min(source_y_begin + 2u, height);
        for (std::uint32_t x = 0; x < next_width; ++x)
        {
            const auto source_x_begin = std::min(x * 2u, width - 1u);
            const auto source_x_end = std::min(source_x_begin + 2u, width);
            float sum[4]{};
            std::uint32_t samples{};
            for (auto source_y = source_y_begin; source_y < source_y_end; ++source_y)
            {
                for (auto source_x = source_x_begin; source_x < source_x_end; ++source_x)
                {
                    const auto source_index = (static_cast<std::size_t>(source_y) * width + source_x) * 4u;
                    for (std::size_t channel = 0; channel < 4u; ++channel)
                        sum[channel] += source[source_index + channel];
                    ++samples;
                }
            }
            const auto destination_index = (static_cast<std::size_t>(y) * next_width + x) * 4u;
            const auto sample_scale = samples > 0 ? 1.0f / static_cast<float>(samples) : 1.0f;
            for (std::size_t channel = 0; channel < 4u; ++channel)
                result[destination_index + channel] = sum[channel] * sample_scale;
        }
    }
    return result;
}

void store_rgba32f_mip_chain(texture_data& texture, const float* decoded, std::uint32_t width, std::uint32_t height)
{
    texture.pixels.clear();
    texture.mips.clear();
    append_rgba32f_mip(texture, decoded, width, height);

    auto current_width = width;
    auto current_height = height;
    const float* current_pixels = decoded;
    std::vector<float> current_storage;
    while (current_width > 1u || current_height > 1u)
    {
        const auto next_width = std::max(1u, current_width / 2u);
        const auto next_height = std::max(1u, current_height / 2u);
        auto next = downsample_rgba32f(current_pixels, current_width, current_height, next_width, next_height);
        append_rgba32f_mip(texture, next.data(), next_width, next_height);
        current_storage = std::move(next);
        current_pixels = current_storage.data();
        current_width = next_width;
        current_height = next_height;
    }
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
}
} // namespace

texture_load_result load_exr_texture_asset(const std::filesystem::path& path)
{
    auto bytes = read_exr_file(path);
    if (bytes.empty()) return {.message = "OpenEXR file could not be read"};
    return load_exr_texture_asset_bytes(std::move(bytes), path);
}

texture_load_result load_exr_texture_asset_bytes(std::vector<std::byte> bytes, const std::filesystem::path& source_path)
{
    if (bytes.empty()) return {.message = "OpenEXR payload is empty"};

#if defined(ARC_RENDER_HAS_TINYEXR) && defined(ARC_RENDER_HAS_STB)
    float* decoded{};
    int width{};
    int height{};
    const char* error{};
    const int result = LoadEXRFromMemory(&decoded, &width, &height,
                                         reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size(), &error);
    if (result != TINYEXR_SUCCESS || decoded == nullptr || width <= 0 || height <= 0)
    {
        std::string message = "OpenEXR decoding failed";
        if (error != nullptr)
        {
            message += ": ";
            message += error;
            FreeEXRErrorMessage(error);
        }
        if (decoded != nullptr) std::free(decoded);
        return {.message = std::move(message)};
    }

    texture_data texture;
    texture.name = source_path.filename().string();
    texture.source_path = source_path;
    texture.mime_type = "image/x-exr";
    texture.width = static_cast<std::uint32_t>(width);
    texture.height = static_cast<std::uint32_t>(height);
    texture.format = texture_format::rgba32f;
    texture.color_space = texture_color_space::linear;
    texture.semantic = texture_semantic::environment;

    // Environment roughness depends on real mip levels. A single-level HDRI
    // makes every material sample the sharp source image and can make even
    // rough dielectrics look mirror-like.
    store_rgba32f_mip_chain(texture, decoded, texture.width, texture.height);
    std::free(decoded);

    return {.texture = std::move(texture), .message = "loaded OpenEXR texture"};
#else
    (void)source_path;
    return {.message = "OpenEXR decoding requires TinyEXR and stb support"};
#endif
}

} // namespace arc::render

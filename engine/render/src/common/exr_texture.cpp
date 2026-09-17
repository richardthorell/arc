#include <arc/render/texture.h>

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
    texture.mip_levels = 1;

    const auto byte_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u * sizeof(float);
    texture.pixels.resize(byte_count);
    std::memcpy(texture.pixels.data(), decoded, byte_count);
    std::free(decoded);
    texture.mips.push_back({.width = texture.width, .height = texture.height, .offset = 0, .size = byte_count});

    return {.texture = std::move(texture), .message = "loaded OpenEXR texture"};
#else
    (void)source_path;
    return {.message = "OpenEXR decoding requires TinyEXR and stb support"};
#endif
}

} // namespace arc::render

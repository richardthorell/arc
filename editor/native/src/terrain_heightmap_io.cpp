#include <arc/editor/terrain_heightmap_io.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <vector>

#include <png.h>

namespace arc::editor
{
namespace
{

bool is_png(const std::filesystem::path& path)
{
    auto extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".png";
}

std::vector<std::byte> read_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const auto size = stream.tellg();
    if (size <= 0) return {};
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

terrain_heightmap_io_result write_atomic(const std::filesystem::path& path, std::span<const std::byte> bytes)
{
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    const auto temporary = path.string() + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) return {false, "could not create temporary heightmap"};
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) return {false, "could not write temporary heightmap"};
    }
    std::filesystem::rename(temporary, path, error);
    if (error)
    {
        std::filesystem::remove(path, error);
        error.clear();
        std::filesystem::rename(temporary, path, error);
    }
    return error ? terrain_heightmap_io_result{false, "could not atomically publish heightmap"}
                 : terrain_heightmap_io_result{true, {}};
}

} // namespace

terrain_heightmap_io_result load_terrain_heightmap(const std::filesystem::path& path, std::uint32_t raw_width,
                                                    std::uint32_t raw_height, scene::terrain_heightmap& output)
{
    const auto bytes = read_bytes(path);
    if (bytes.empty()) return {false, "heightmap file could not be read"};
    output = {};
    if (is_png(path))
    {
        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        if (!png_image_begin_read_from_memory(&image, bytes.data(), bytes.size()))
            return {false, "heightmap PNG header is invalid"};
        if ((image.format & PNG_FORMAT_FLAG_LINEAR) == 0u)
        {
            png_image_free(&image);
            return {false, "heightmap PNG must contain true 16-bit samples"};
        }
        image.format = PNG_FORMAT_LINEAR_Y;
        output.width = image.width;
        output.height = image.height;
        output.samples.resize(PNG_IMAGE_SIZE(image) / sizeof(std::uint16_t));
        if (!png_image_finish_read(&image, nullptr, output.samples.data(), 0, nullptr))
        {
            const std::string message = image.message;
            png_image_free(&image);
            return {false, "heightmap PNG decoding failed: " + message};
        }
        png_image_free(&image);
        return {true, {}};
    }
    if (raw_width < 2u || raw_height < 2u ||
        static_cast<std::uint64_t>(raw_width) * raw_height > std::numeric_limits<std::size_t>::max() / 2u)
        return {false, "RAW R16 import requires explicit valid width and height"};
    const auto sample_count = static_cast<std::size_t>(raw_width) * raw_height;
    if (bytes.size() != sample_count * 2u) return {false, "RAW R16 byte count does not match its dimensions"};
    output.width = raw_width;
    output.height = raw_height;
    output.samples.resize(sample_count);
    for (std::size_t index = 0; index < sample_count; ++index)
        output.samples[index] = static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[index * 2u]) |
                                                           std::to_integer<std::uint8_t>(bytes[index * 2u + 1u]) << 8u);
    return {true, {}};
}

terrain_heightmap_io_result save_terrain_heightmap(const std::filesystem::path& path,
                                                    const scene::terrain_heightmap& heightmap)
{
    if (heightmap.width < 2u || heightmap.height < 2u ||
        heightmap.samples.size() != static_cast<std::size_t>(heightmap.width) * heightmap.height)
        return {false, "heightmap sample payload is invalid"};
    std::vector<std::byte> bytes;
    if (is_png(path))
    {
        png_image image{};
        image.version = PNG_IMAGE_VERSION;
        image.width = heightmap.width;
        image.height = heightmap.height;
        image.format = PNG_FORMAT_LINEAR_Y;
        png_alloc_size_t byte_count{};
        if (!png_image_write_get_memory_size(image, byte_count, 0, heightmap.samples.data(), 0, nullptr))
            return {false, "heightmap PNG size calculation failed"};
        bytes.resize(byte_count);
        if (!png_image_write_to_memory(&image, bytes.data(), &byte_count, 0, heightmap.samples.data(), 0, nullptr))
            return {false, "heightmap PNG encoding failed"};
        bytes.resize(byte_count);
    }
    else
    {
        bytes.resize(heightmap.samples.size() * 2u);
        for (std::size_t index = 0; index < heightmap.samples.size(); ++index)
        {
            bytes[index * 2u] = static_cast<std::byte>(heightmap.samples[index] & 0xffu);
            bytes[index * 2u + 1u] = static_cast<std::byte>(heightmap.samples[index] >> 8u);
        }
    }
    const auto written = write_atomic(path, bytes);
    if (!written.succeeded) return written;
    if (!heightmap.encoded_minimum_elevation || !heightmap.encoded_maximum_elevation) return {true, {}};
    const auto metadata_path = path.string() + ".arcmeta.json";
    const auto metadata = std::string{"{\n  \"format\": \"arc.terrain.heightmap\",\n  \"width\": "} +
                          std::to_string(heightmap.width) + ",\n  \"height\": " +
                          std::to_string(heightmap.height) + ",\n  \"minimumElevation\": " +
                          std::to_string(*heightmap.encoded_minimum_elevation) +
                          ",\n  \"maximumElevation\": " + std::to_string(*heightmap.encoded_maximum_elevation) + "\n}\n";
    return write_atomic(metadata_path, std::as_bytes(std::span(metadata.data(), metadata.size())));
}

} // namespace arc::editor

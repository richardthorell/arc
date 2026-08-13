#pragma once

#include <arc/scene/terrain.h>

#include <filesystem>
#include <string>

namespace arc::editor
{

/** @brief Result of decoding or encoding an editor terrain heightmap. */
struct terrain_heightmap_io_result
{
    bool succeeded{};
    std::string message;
};

/** @brief Decode a 16-bit grayscale PNG or headerless little-endian R16 file. */
[[nodiscard]] terrain_heightmap_io_result load_terrain_heightmap(const std::filesystem::path& path,
                                                                 std::uint32_t raw_width,
                                                                 std::uint32_t raw_height,
                                                                 scene::terrain_heightmap& output);

/** @brief Atomically encode a 16-bit grayscale PNG or R16 file and its elevation metadata. */
[[nodiscard]] terrain_heightmap_io_result save_terrain_heightmap(const std::filesystem::path& path,
                                                                 const scene::terrain_heightmap& heightmap);

} // namespace arc::editor

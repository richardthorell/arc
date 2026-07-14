#pragma once

#include <arc/render/material.h>

#include <filesystem>
#include <string>
#include <vector>

namespace arc::render
{

/**
 * @brief Result from loading a texture asset into renderer-owned CPU metadata.
 */
struct texture_load_result
{
    texture_data texture;
    std::string message;

    /**
     * @brief Return whether the texture contains usable decoded or encoded data.
     */
    bool succeeded() const noexcept
    {
        return texture.has_pixels() || texture.has_encoded_mips() || !texture.encoded.empty();
    }
};

/**
 * @brief Parse DDS bytes and preserve compressed/uncompressed mip payloads.
 */
texture_load_result parse_dds_texture(
    const std::vector<std::byte>& bytes,
    std::string name = {});

/**
 * @brief Load a texture asset by extension.
 *
 * DDS files are parsed for format and mip metadata. Common material map names
 * are used to infer sRGB versus linear upload formats for imported assets.
 * Other image formats are decoded to upload-ready pixels when stb is available;
 * builds without a decoder preserve the encoded bytes for later processing.
 */
texture_load_result load_texture_asset(const std::filesystem::path& path);

/**
 * @brief Return whether an extension is accepted by the renderer texture loader.
 */
bool is_supported_texture_asset(const std::filesystem::path& path);

} // namespace arc::render

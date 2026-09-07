#pragma once

#include <arc/core/core.h>
#include <arc/scene/terrain_asset.h>

#include <memory>
#include <string>
#include <string_view>

namespace arc::scene
{

enum class terrain_asset_io_error_code : std::uint8_t
{
    invalid_document,
    unsupported_format_version,
    unsupported_schema_version,
    invalid_asset
};

struct terrain_asset_io_error
{
    terrain_asset_io_error_code code{terrain_asset_io_error_code::invalid_document};
    std::string message;
};

using terrain_asset_json_result = core::result<std::string, terrain_asset_io_error>;
using terrain_asset_decode_result = core::result<terrain_asset, terrain_asset_io_error>;

/** @brief Serialize a unified terrain asset to the versioned `.terrain` JSON document format. */
[[nodiscard]] terrain_asset_json_result write_terrain_asset_json(const terrain_asset& asset, bool pretty = true);

/** @brief Parse and validate a versioned `.terrain` JSON document. */
[[nodiscard]] terrain_asset_decode_result read_terrain_asset_json(std::string_view text);

/** @brief Create the asset-manager importer that materializes `.terrain` sources as `terrain_asset` payloads. */
[[nodiscard]] std::unique_ptr<assets::asset_importer> make_terrain_asset_importer();

/** @brief Register the terrain importer with an existing asset manager. */
bool register_terrain_asset_importer(assets::asset_manager& manager);

} // namespace arc::scene

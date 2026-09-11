#pragma once

#include <arc/assets/water_types.h>
#include <arc/core/core.h>
#include <arc/water/water_asset.h>

#include <memory>
#include <string>
#include <string_view>

namespace arc::water
{

enum class water_asset_io_error_code : std::uint8_t
{
    invalid_document,
    unsupported_format_version,
    unsupported_schema_version,
    invalid_asset
};

struct water_asset_io_error
{
    water_asset_io_error_code code{water_asset_io_error_code::invalid_document};
    std::string message;
};

using water_asset_json_result = core::result<std::string, water_asset_io_error>;
using water_asset_decode_result = core::result<water_preset, water_asset_io_error>;

/** @brief Serialize a Water preset to the versioned `.arcwater` JSON format. */
[[nodiscard]] water_asset_json_result write_water_asset_json(const water_preset& preset, bool pretty = true);

/** @brief Parse and validate a versioned `.arcwater` JSON document. */
[[nodiscard]] water_asset_decode_result read_water_asset_json(std::string_view text);

[[nodiscard]] std::unique_ptr<assets::asset_importer> make_water_asset_importer();
bool register_water_asset_importer(assets::asset_manager& manager);

} // namespace arc::water

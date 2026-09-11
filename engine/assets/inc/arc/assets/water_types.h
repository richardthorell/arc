#pragma once

#include <arc/assets/assets.h>

namespace arc::assets
{
namespace asset_types
{
/** @brief Authored Water preset consumed by the Water runtime. */
inline constexpr asset_type_id water_preset{0xa7ca55e700000001ull, 0x0000000000000010ull};
} // namespace asset_types

namespace importer_ids
{
/** @brief Versioned JSON importer for `.arcwater` preset assets. */
inline constexpr asset_importer_id water_preset{0xa7ca55e700000002ull, 0x0000000000000011ull};
} // namespace importer_ids
} // namespace arc::assets

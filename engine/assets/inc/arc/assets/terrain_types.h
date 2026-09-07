#pragma once

#include <arc/assets/assets.h>

namespace arc::assets
{
namespace asset_types
{
/** @brief Unified authored terrain definition. */
inline constexpr asset_type_id terrain{0xa7ca55e700000001ull, 0x000000000000000full};
} // namespace asset_types

namespace importer_ids
{
/** @brief Versioned JSON importer for unified terrain assets. */
inline constexpr asset_importer_id terrain{0xa7ca55e700000002ull, 0x0000000000000010ull};
} // namespace importer_ids
} // namespace arc::assets

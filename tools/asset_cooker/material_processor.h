#pragma once

#include <arc/assets/cook.h>

#include <memory>

namespace arc::tools
{

/** Create the first-class material cooker registered by arc_asset_cooker. */
[[nodiscard]] std::unique_ptr<assets::asset_cook_processor> make_material_processor();

} // namespace arc::tools

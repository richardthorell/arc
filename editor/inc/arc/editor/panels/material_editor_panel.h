#pragma once

#include <arc/editor/editor_state.h>

namespace arc::editor
{

void draw_material_editor_panel(
    editor_scene_state& editor_scene,
    const editor_asset_state& editor_assets,
    render::renderer* renderer);

} // namespace arc::editor

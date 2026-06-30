#pragma once

#include <arc/editor/editor_state.h>

namespace arc::editor
{

void draw_content_browser_panel(
    const editor_asset_state& editor_assets,
    editor_scene_state& editor_scene,
    render::renderer* renderer,
    editor_scene_import_state* import_state = nullptr);

} // namespace arc::editor

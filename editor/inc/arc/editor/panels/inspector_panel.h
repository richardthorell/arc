#pragma once

#include <arc/editor/arc_host.h>
#include <arc/editor/editor_state.h>

namespace arc::editor
{

void draw_inspector_panel(
    arc_host& host,
    const editor_asset_state& editor_assets,
    render::renderer* renderer = nullptr);

} // namespace arc::editor

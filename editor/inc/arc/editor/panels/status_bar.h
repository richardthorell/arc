#pragma once

#include <arc/editor/editor_state.h>
#include <arc/framework/runtime.h>

namespace arc::editor
{

void draw_status_bar(
    arc::runtime& runtime,
    const editor_build_state& build,
    const editor_source_control_state& source_control);

} // namespace arc::editor

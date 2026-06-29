#include <arc/editor/panels/render_graph_panel.h>

#include <arc/editor/ui/widgets.h>

#include <imgui.h>

namespace arc::editor
{

void draw_render_graph_panel()
{
    if (!ui::begin_panel("Render Graph"))
    {
        ui::end_panel();
        return;
    }

    ui::empty_state("Render graph view pending", "Named passes are visible in captures; the interactive graph view will land with the deferred renderer.");
    ImGui::Separator();
    ui::muted_text("Current frame path: viewport clear -> scene draw -> imgui -> present");

    ui::end_panel();
}

} // namespace arc::editor

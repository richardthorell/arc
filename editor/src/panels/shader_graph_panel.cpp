#include <arc/editor/panels/shader_graph_panel.h>

#include <arc/editor/ui/widgets.h>

namespace arc::editor
{

void draw_shader_graph_panel()
{
    if (!ui::begin_panel("Shader Graph"))
    {
        ui::end_panel();
        return;
    }

    ui::empty_state("Shader graph not available", "Material graph editing will appear here once shader nodes are implemented.");

    ui::end_panel();
}

} // namespace arc::editor

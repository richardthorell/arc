#include <arc/editor/panels/profiler_panel.h>

#include <arc/editor/ui/widgets.h>

#include <imgui.h>

namespace arc::editor
{

void draw_profiler_panel(const render::renderer& renderer)
{
    if (!ui::begin_panel("Profiler"))
    {
        ui::end_panel();
        return;
    }

    const auto profile = renderer.last_frame_profile();
    if (profile.summary.empty())
    {
        ui::empty_state("No GPU capture yet", "Run a frame with GPU timing enabled to inspect pass timings.");
    }
    else
    {
        if (ui::section_header("Frame", true))
        {
            ImGui::Text("Frame: %llu", static_cast<unsigned long long>(profile.frame_index));
            ImGui::TextWrapped("%s", profile.summary.c_str());
            ImGui::TreePop();
        }
        if (ui::section_header("Pass Timings", true))
        {
            if (profile.pass_timings.empty())
                ui::muted_text("GPU timestamp results pending.");
            else
            {
                for (const auto& timing : profile.pass_timings)
                    ImGui::Text("%s: %.3f ms", timing.name.c_str(), timing.milliseconds);
            }
            ImGui::TreePop();
        }
    }

    ui::end_panel();
}

} // namespace arc::editor

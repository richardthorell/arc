#include <arc/editor/panels/status_bar.h>

#include <arc/editor/ui/theme.h>

#include <imgui.h>

namespace arc::editor
{

void draw_status_bar(
    arc::runtime& runtime,
    const editor_build_state& build,
    const editor_source_control_state& source_control)
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float height = ui::scaled(30.0f);
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, height));
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ui::colors::shell_bg);
    ImGui::Begin("ARC Editor Status", nullptr, flags);
    ImGui::PopStyleColor();
    ImGui::PopStyleVar(2);

    ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text_muted);
    ImGui::TextUnformatted("Ready");
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::TextUnformatted("|");
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::Text("Platform: %s", build.platform.c_str());
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::TextUnformatted("|");
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::Text("Build: %s", build.configuration.c_str());
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::TextUnformatted("|");
    ImGui::SameLine(0.0f, ui::scaled(18.0f));
    ImGui::Text("Workers: %zu", runtime.jobs().worker_count());
    if (source_control.available)
    {
        ImGui::SameLine(0.0f, ui::scaled(18.0f));
        ImGui::TextUnformatted("|");
        ImGui::SameLine(0.0f, ui::scaled(18.0f));
        ImGui::Text("Branch: %s%s", source_control.branch.c_str(), source_control.has_changes ? " (modified)" : "");
    }
    ImGui::PopStyleColor();
    ImGui::End();
}

} // namespace arc::editor

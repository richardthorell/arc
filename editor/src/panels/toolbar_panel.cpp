#include <arc/editor/panels/toolbar_panel.h>

#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>

#include <imgui.h>

namespace arc::editor
{
namespace
{

void same_group_spacing()
{
    ImGui::SameLine(0.0f, ui::scaled(6.0f));
}

} // namespace

void draw_toolbar(editor_ui_state& state)
{
    const ImVec2 band_min = ImGui::GetCursorScreenPos();
    const float toolbar_height = ui::scaled(48.0f);
    ImGui::GetWindowDrawList()->AddRectFilled(
        band_min,
        ImVec2(band_min.x + ImGui::GetContentRegionAvail().x, band_min.y + toolbar_height),
        ImGui::ColorConvertFloat4ToU32(ui::colors::shell_bg));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ui::scaled(4.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ui::scaled(10.0f), ui::scaled(6.0f)));
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ui::scaled(8.0f));

    const ImVec2 tool_size(ui::scaled(78.0f), ui::scaled(31.0f));
    const auto tool_button = [&](editor_tool tool, const char* label) {
        if (ui::toolbar_button(label, state.active_tool == tool, tool_size))
            state.active_tool = tool;
        same_group_spacing();
    };

    tool_button(editor_tool::select, "Select");
    tool_button(editor_tool::translate, "Move");
    tool_button(editor_tool::rotate, "Rotate");
    tool_button(editor_tool::scale, "Scale");
    ui::toolbar_separator();

    ImGui::PushStyleColor(ImGuiCol_Button, ui::colors::success);
    ImGui::Button("Play", ImVec2(ui::scaled(58.0f), ui::scaled(31.0f)));
    ImGui::PopStyleColor();
    same_group_spacing();
    ImGui::Button("Pause", ImVec2(ui::scaled(66.0f), ui::scaled(31.0f)));
    same_group_spacing();
    ImGui::Button("Step", ImVec2(ui::scaled(56.0f), ui::scaled(31.0f)));
    ui::toolbar_separator();

    static const char* apis[]{ "Vulkan" };
    static int selected_api{};
    ImGui::SetNextItemWidth(ui::scaled(122.0f));
    ImGui::Combo("##toolbar-api", &selected_api, apis, 1);
    same_group_spacing();

    static const char* modes[]{ "Debug" };
    static int selected_mode{};
    ImGui::SetNextItemWidth(ui::scaled(112.0f));
    ImGui::Combo("##toolbar-mode", &selected_mode, modes, 1);
    ui::toolbar_separator();

    if (ui::toolbar_button("Grid", state.show_world_grid, ImVec2(ui::scaled(58.0f), ui::scaled(31.0f))))
        state.show_world_grid = !state.show_world_grid;
    same_group_spacing();
    if (ui::toolbar_button("Shadows", state.show_shadows, ImVec2(ui::scaled(88.0f), ui::scaled(31.0f))))
        state.show_shadows = !state.show_shadows;
    same_group_spacing();
    if (ui::toolbar_button("Sky", state.show_sky, ImVec2(ui::scaled(50.0f), ui::scaled(31.0f))))
        state.show_sky = !state.show_sky;
    same_group_spacing();
    if (ui::toolbar_button("Fog", state.show_fog, ImVec2(ui::scaled(50.0f), ui::scaled(31.0f))))
        state.show_fog = !state.show_fog;
    same_group_spacing();
    if (ui::toolbar_button("Terrain", state.show_terrain, ImVec2(ui::scaled(78.0f), ui::scaled(31.0f))))
        state.show_terrain = !state.show_terrain;
    same_group_spacing();
    if (ui::toolbar_button("Water", state.show_water, ImVec2(ui::scaled(66.0f), ui::scaled(31.0f))))
        state.show_water = !state.show_water;
    same_group_spacing();
    if (ui::toolbar_button("Veg", state.show_vegetation, ImVec2(ui::scaled(52.0f), ui::scaled(31.0f))))
        state.show_vegetation = !state.show_vegetation;
    same_group_spacing();
    if (ui::toolbar_button("Decals", state.show_decals, ImVec2(ui::scaled(70.0f), ui::scaled(31.0f))))
        state.show_decals = !state.show_decals;
    same_group_spacing();
    ui::toolbar_button("Snap", false, ImVec2(ui::scaled(58.0f), ui::scaled(31.0f)));
    ui::toolbar_separator();

    ImGui::TextDisabled("Camera");
    same_group_spacing();
    const char* camera_modes[]{ "Perspective", "Orthographic" };
    int selected_camera = state.viewport_camera == viewport_camera_mode::orthographic ? 1 : 0;
    ImGui::SetNextItemWidth(ui::scaled(150.0f));
    if (ImGui::Combo("##toolbar-camera", &selected_camera, camera_modes, 2))
    {
        state.viewport_camera = selected_camera == 1
            ? viewport_camera_mode::orthographic
            : viewport_camera_mode::perspective;
    }
    same_group_spacing();
    if (ui::toolbar_button("Layout", false, ImVec2(ui::scaled(72.0f), ui::scaled(31.0f))))
        ImGui::OpenPopup("layout-presets");
    if (ImGui::BeginPopup("layout-presets"))
    {
        if (ImGui::MenuItem("Reset Default Layout"))
        {
            state.layout_preset = editor_layout_preset::default_layout;
            state.show_bottom_panels = true;
            state.reset_layout = true;
        }
        if (ImGui::MenuItem("Scene Editing"))
        {
            state.layout_preset = editor_layout_preset::scene_editing;
            state.show_bottom_panels = true;
            state.reset_layout = true;
        }
        if (ImGui::MenuItem("Profiling"))
        {
            state.layout_preset = editor_layout_preset::profiling;
            state.show_bottom_panels = true;
            state.reset_layout = true;
        }
        if (ImGui::MenuItem("Minimal"))
        {
            state.layout_preset = editor_layout_preset::minimal;
            state.show_bottom_panels = false;
            state.reset_layout = true;
        }
        ImGui::EndPopup();
    }

    ImGui::PopStyleVar(2);
    ImGui::Dummy(ImVec2(1.0f, ui::scaled(8.0f)));
}

} // namespace arc::editor

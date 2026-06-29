#include <arc/editor/ui/theme.h>

namespace arc::editor::ui
{

float scaled(float value) noexcept
{
    return value * ui_scale;
}

void apply_dark_theme()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.ChildRounding = 4.0f;
    style.FrameRounding = 4.0f;
    style.PopupRounding = 4.0f;
    style.ScrollbarRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;

    style.WindowBorderSize = 0.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 0.0f;
    style.TabBorderSize = 0.0f;

    style.WindowPadding = ImVec2(10.0f, 8.0f);
    style.FramePadding = ImVec2(9.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.ScrollbarSize = 12.0f;
    style.ScaleAllSizes(ui_scale);

    ImVec4* c = style.Colors;
    c[ImGuiCol_Text] = colors::text;
    c[ImGuiCol_TextDisabled] = colors::text_muted;
    c[ImGuiCol_WindowBg] = colors::panel_bg;
    c[ImGuiCol_ChildBg] = colors::child_bg;
    c[ImGuiCol_PopupBg] = colors::panel_bg_alt;
    c[ImGuiCol_Border] = colors::border;
    c[ImGuiCol_FrameBg] = ImVec4(0.083f, 0.094f, 0.106f, 1.0f);
    c[ImGuiCol_FrameBgHovered] = ImVec4(0.112f, 0.132f, 0.152f, 1.0f);
    c[ImGuiCol_FrameBgActive] = ImVec4(0.140f, 0.178f, 0.220f, 1.0f);
    c[ImGuiCol_TitleBg] = colors::shell_bg;
    c[ImGuiCol_TitleBgActive] = colors::shell_bg;
    c[ImGuiCol_MenuBarBg] = colors::shell_bg;
    c[ImGuiCol_Header] = ImVec4(0.110f, 0.190f, 0.290f, 0.88f);
    c[ImGuiCol_HeaderHovered] = ImVec4(0.145f, 0.260f, 0.390f, 0.96f);
    c[ImGuiCol_HeaderActive] = ImVec4(0.175f, 0.340f, 0.520f, 1.0f);
    c[ImGuiCol_Button] = ImVec4(0.082f, 0.096f, 0.110f, 1.0f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.125f, 0.150f, 0.178f, 1.0f);
    c[ImGuiCol_ButtonActive] = ImVec4(0.150f, 0.240f, 0.350f, 1.0f);
    c[ImGuiCol_Tab] = colors::child_bg;
    c[ImGuiCol_TabHovered] = ImVec4(0.130f, 0.230f, 0.340f, 1.0f);
    c[ImGuiCol_TabSelected] = ImVec4(0.105f, 0.160f, 0.225f, 1.0f);
    c[ImGuiCol_DockingPreview] = ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.65f);
    c[ImGuiCol_Separator] = colors::border_subtle;
    c[ImGuiCol_CheckMark] = colors::accent_hover;
    c[ImGuiCol_SliderGrab] = colors::accent;
    c[ImGuiCol_SliderGrabActive] = colors::accent_hover;
    c[ImGuiCol_ResizeGrip] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    c[ImGuiCol_ResizeGripHovered] = ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.45f);
    c[ImGuiCol_ResizeGripActive] = ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.75f);
}

} // namespace arc::editor::ui

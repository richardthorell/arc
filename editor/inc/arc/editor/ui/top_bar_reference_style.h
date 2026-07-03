#pragma once

// This header is force-included for editor/src/main.cpp so the window chrome can be
// styled without moving the current dockspace implementation while the editor UI is
// still being iterated on.

#include <arc/editor/ui/theme.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_sdl3.h>
#include <ImGuizmo.h>

#if !defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
#include <imgui_impl_sdlrenderer3.h>
#endif

#include <array>
#include <cstdarg>
#include <cstring>

namespace arc::editor::ui::top_bar_reference
{

inline bool is_primary_menu(const char* label) noexcept
{
    return label && (
        std::strcmp(label, "File") == 0 ||
        std::strcmp(label, "Edit") == 0 ||
        std::strcmp(label, "View") == 0 ||
        std::strcmp(label, "Tools") == 0 ||
        std::strcmp(label, "Window") == 0 ||
        std::strcmp(label, "Help") == 0);
}

inline bool should_hide_menu(const char* label) noexcept
{
    if (!label)
        return false;

    // The reference top bar uses a blue Build action button instead of a Build menu,
    // and no longer shows the project selector/search area in the title bar.
    return std::strcmp(label, "Build") == 0 || !is_primary_menu(label);
}

inline ImU32 color_u32(const ImVec4& color) noexcept
{
    return ImGui::ColorConvertFloat4ToU32(color);
}

inline void add_quad(ImDrawList& draw_list, const std::array<ImVec2, 4>& points, ImU32 color)
{
    draw_list.AddConvexPolyFilled(points.data(), static_cast<int>(points.size()), color);
}

inline void draw_arc_mark(ImDrawList& draw_list, ImVec2 origin, float scale, ImU32 color)
{
    const auto point = [&](float x, float y) {
        return ImVec2(origin.x + x * scale, origin.y + y * scale);
    };

    add_quad(draw_list, { point(1.0f, 11.0f), point(7.0f, 4.0f), point(11.0f, 8.0f), point(5.0f, 15.0f) }, color);
    add_quad(draw_list, { point(8.0f, 3.0f), point(13.0f, 0.0f), point(22.0f, 9.0f), point(18.0f, 13.0f) }, color);
    add_quad(draw_list, { point(12.0f, 12.0f), point(17.0f, 13.0f), point(23.0f, 20.0f), point(17.0f, 20.0f) }, color);
}

inline void draw_brand(const ImRect& bar)
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec4 shell = colors::shell_bg;
    draw_list->AddRectFilled(
        ImVec2(bar.Min.x, bar.Min.y),
        ImVec2(bar.Min.x + scaled(98.0f), bar.Max.y),
        color_u32(shell));

    draw_arc_mark(
        *draw_list,
        ImVec2(bar.Min.x + scaled(12.0f), bar.Min.y + scaled(9.0f)),
        scaled(0.92f),
        IM_COL32(232, 238, 244, 255));

    draw_list->AddText(
        ImVec2(bar.Min.x + scaled(42.0f), bar.Min.y + scaled(10.0f)),
        IM_COL32(232, 238, 244, 255),
        "arc");
}

inline void draw_scene_title(const ImRect& bar)
{
    static constexpr const char* scene_name = "MountainVillage.arcscene*";
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 text_size = ImGui::CalcTextSize(scene_name);
    const float x = bar.Min.x + (bar.GetWidth() - text_size.x) * 0.5f;
    const float y = bar.Min.y + (bar.GetHeight() - text_size.y) * 0.5f;
    const ImU32 text_color = IM_COL32(157, 166, 176, 255);

    draw_list->AddText(ImVec2(x, y), text_color, scene_name);

    const float chevron_x = x + text_size.x + scaled(8.0f);
    const float chevron_y = y + text_size.y * 0.55f;
    draw_list->AddLine(
        ImVec2(chevron_x, chevron_y - scaled(2.0f)),
        ImVec2(chevron_x + scaled(4.0f), chevron_y + scaled(2.0f)),
        text_color,
        scaled(1.2f));
    draw_list->AddLine(
        ImVec2(chevron_x + scaled(4.0f), chevron_y + scaled(2.0f)),
        ImVec2(chevron_x + scaled(8.0f), chevron_y - scaled(2.0f)),
        text_color,
        scaled(1.2f));
}

inline void draw_build_button(const ImRect& bar)
{
    const ImVec2 size(scaled(78.0f), scaled(30.0f));
    const float controls_width = scaled(156.0f);
    const ImVec2 pos(
        bar.Max.x - controls_width - size.x - scaled(14.0f),
        bar.Min.y + (bar.GetHeight() - size.y) * 0.5f);

    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, scaled(3.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(14.0f), scaled(6.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.075f, 0.285f, 0.690f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.095f, 0.360f, 0.820f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.060f, 0.235f, 0.600f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.950f, 0.975f, 1.000f, 1.0f));
    ImGui::Button("Build", size);
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
}

inline void draw_bottom_border(const ImRect& bar)
{
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(bar.Min.x, bar.Max.y - 1.0f),
        ImVec2(bar.Max.x, bar.Max.y - 1.0f),
        IM_COL32(29, 36, 44, 255),
        1.0f);
}

inline void draw_window_button_icon(const char* label, const ImVec2& min, const ImVec2& max, bool hovered, bool active)
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const bool close_button = label && std::strcmp(label, "X") == 0;
    const ImU32 normal_bg = IM_COL32(0, 0, 0, 0);
    const ImU32 hover_bg = close_button ? IM_COL32(194, 42, 42, 255) : IM_COL32(36, 45, 55, 255);
    const ImU32 active_bg = close_button ? IM_COL32(158, 28, 28, 255) : IM_COL32(45, 58, 72, 255);
    const ImU32 icon_color = IM_COL32(204, 213, 222, 255);
    const float center_x = (min.x + max.x) * 0.5f;
    const float center_y = (min.y + max.y) * 0.5f;
    const float icon = scaled(10.0f);

    draw_list->AddRectFilled(min, max, active ? active_bg : hovered ? hover_bg : normal_bg);

    if (close_button)
    {
        draw_list->AddLine(ImVec2(center_x - icon * 0.45f, center_y - icon * 0.45f), ImVec2(center_x + icon * 0.45f, center_y + icon * 0.45f), icon_color, scaled(1.35f));
        draw_list->AddLine(ImVec2(center_x + icon * 0.45f, center_y - icon * 0.45f), ImVec2(center_x - icon * 0.45f, center_y + icon * 0.45f), icon_color, scaled(1.35f));
        return;
    }

    if (label && std::strcmp(label, "-") == 0)
    {
        draw_list->AddLine(ImVec2(center_x - icon * 0.50f, center_y + scaled(3.0f)), ImVec2(center_x + icon * 0.50f, center_y + scaled(3.0f)), icon_color, scaled(1.35f));
        return;
    }

    const bool restore = label && std::strcmp(label, "[]") == 0;
    if (restore)
    {
        draw_list->AddRect(ImVec2(center_x - icon * 0.25f, center_y - icon * 0.55f), ImVec2(center_x + icon * 0.45f, center_y + icon * 0.15f), icon_color, 0.0f, 0, scaled(1.15f));
        draw_list->AddRect(ImVec2(center_x - icon * 0.45f, center_y - icon * 0.20f), ImVec2(center_x + icon * 0.25f, center_y + icon * 0.50f), icon_color, 0.0f, 0, scaled(1.15f));
    }
    else
    {
        draw_list->AddRect(ImVec2(center_x - icon * 0.45f, center_y - icon * 0.45f), ImVec2(center_x + icon * 0.45f, center_y + icon * 0.45f), icon_color, 0.0f, 0, scaled(1.15f));
    }
}

} // namespace arc::editor::ui::top_bar_reference

namespace ImGui
{

inline bool Arc_BeginMenu(const char* label, bool enabled = true)
{
    using namespace arc::editor::ui::top_bar_reference;

    if (label && std::strcmp(label, "Tools") == 0)
    {
        if (ImGui::BeginMenu("Scene"))
        {
            ImGui::MenuItem("New Scene", "Ctrl+N", false, false);
            ImGui::MenuItem("Save Scene", "Ctrl+S", false, false);
            ImGui::Separator();
            ImGui::MenuItem("Import Into Current...", nullptr, false, false);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Render"))
        {
            ImGui::MenuItem("Lighting", nullptr, false, false);
            ImGui::MenuItem("Render Graph", nullptr, false, false);
            ImGui::MenuItem("Profiler", nullptr, false, false);
            ImGui::EndMenu();
        }
    }

    if (should_hide_menu(label))
        return false;

    return ImGui::BeginMenu(label, enabled);
}

inline void Arc_EndMenuBar()
{
    using namespace arc::editor::ui::top_bar_reference;

    if (ImGuiWindow* window = ImGui::GetCurrentWindowRead())
    {
        const ImRect bar = window->MenuBarRect();
        draw_build_button(bar);
        draw_brand(bar);
        draw_scene_title(bar);
        draw_bottom_border(bar);
    }

    ImGui::EndMenuBar();
}

inline bool Arc_Button(const char* label, const ImVec2& size_arg = ImVec2(0.0f, 0.0f))
{
    using namespace arc::editor::ui::top_bar_reference;

    const bool window_button = label && (
        std::strcmp(label, "-") == 0 ||
        std::strcmp(label, "[ ]") == 0 ||
        std::strcmp(label, "[]") == 0 ||
        std::strcmp(label, "X") == 0);

    if (!window_button)
        return ImGui::Button(label, size_arg);

    ImGui::PushID(label);
    const bool pressed = ImGui::InvisibleButton("arc-window-button", size_arg);
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    draw_window_button_icon(label, min, max, ImGui::IsItemHovered(), ImGui::IsItemActive());
    ImGui::PopID();
    return pressed;
}

inline void Arc_TextDisabled(const char* fmt, ...)
{
    if (fmt && std::strcmp(fmt, "Project:") == 0)
        return;

    va_list args;
    va_start(args, fmt);
    ImGui::TextDisabledV(fmt, args);
    va_end(args);
}

} // namespace ImGui

#define BeginMenu Arc_BeginMenu
#define EndMenuBar Arc_EndMenuBar
#define Button Arc_Button
#define TextDisabled Arc_TextDisabled

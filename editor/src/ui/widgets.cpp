#include <arc/editor/ui/widgets.h>

#include <arc/editor/ui/theme.h>

#include <cstdio>

namespace arc::editor::ui
{

bool toolbar_button(const char* label, bool active, ImVec2 size)
{
    if (active)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, colors::accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, colors::accent_hover);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, colors::accent_hover);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.96f, 0.98f, 1.0f, 1.0f));
    }

    const bool clicked = ImGui::Button(label, size);

    if (active)
        ImGui::PopStyleColor(4);
    return clicked;
}

void toolbar_separator()
{
    ImGui::SameLine(0.0f, scaled(8.0f));
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const float height = scaled(24.0f);
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(pos.x, pos.y + scaled(3.0f)),
        ImVec2(pos.x, pos.y + height),
        ImGui::ColorConvertFloat4ToU32(colors::border_subtle),
        1.0f);
    ImGui::Dummy(ImVec2(1.0f, height));
    ImGui::SameLine(0.0f, scaled(8.0f));
}

bool search_box(const char* id, const char* hint, char* buffer, std::size_t buffer_size)
{
    ImGui::PushStyleColor(ImGuiCol_FrameBg, colors::child_bg);
    const bool changed = ImGui::InputTextWithHint(id, hint, buffer, buffer_size);
    ImGui::PopStyleColor();
    return changed;
}

bool begin_panel(const char* name, bool* open, ImGuiWindowFlags flags)
{
    ImGui::PushStyleColor(ImGuiCol_WindowBg, colors::panel_bg);
    const bool visible = ImGui::Begin(name, open, flags);
    ImGui::PopStyleColor();
    return visible;
}

void end_panel()
{
    ImGui::End();
}

bool section_header(const char* label, bool default_open)
{
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (default_open)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    return ImGui::TreeNodeEx(label, flags);
}

bool component_card_begin(const char* label, bool default_open)
{
    ImGui::PushStyleColor(ImGuiCol_Header, colors::panel_bg_alt);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.22f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.30f));
    const bool open = section_header(label, default_open);
    ImGui::PopStyleColor(3);
    if (open)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(8.0f), scaled(4.0f)));
        ImGui::Indent(scaled(8.0f));
    }
    return open;
}

void component_card_end()
{
    ImGui::Unindent(scaled(8.0f));
    ImGui::PopStyleVar();
    ImGui::TreePop();
    ImGui::Dummy(ImVec2(0.0f, scaled(4.0f)));
}

bool entity_row(const char* label, const char* icon, bool selected, bool visible, bool locked)
{
    ImGui::PushID(label);
    if (!visible)
        ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.55f);

    const ImVec2 min = ImGui::GetCursorScreenPos();
    const float height = scaled(26.0f);
    const float width = ImGui::GetContentRegionAvail().x;
    const bool clicked = ImGui::InvisibleButton("entity-row", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 max(min.x + width, min.y + height);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    if (selected)
        draw_list->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32(ImVec4(colors::accent.x, colors::accent.y, colors::accent.z, 0.24f)), scaled(4.0f));
    else if (hovered)
        draw_list->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.045f)), scaled(4.0f));

    const ImU32 icon_color = ImGui::ColorConvertFloat4ToU32(selected ? colors::text : colors::text_muted);
    const ImU32 label_color = ImGui::ColorConvertFloat4ToU32(colors::text);
    const ImVec2 text_pos(min.x + scaled(8.0f), min.y + scaled(5.0f));
    draw_list->AddText(text_pos, icon_color, icon);
    draw_list->AddText(ImVec2(text_pos.x + scaled(24.0f), text_pos.y), label_color, label);
    if (locked)
        draw_list->AddText(ImVec2(max.x - scaled(34.0f), text_pos.y), icon_color, "lock");

    if (!visible)
        ImGui::PopStyleVar();
    ImGui::PopID();
    return clicked;
}

bool toggle_chip(const char* label, bool* value, int count)
{
    char text[96]{};
    if (count >= 0)
        std::snprintf(text, sizeof(text), "%s %d", label, count);
    else
        std::snprintf(text, sizeof(text), "%s", label);

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, scaled(12.0f));
    const bool clicked = toolbar_button(text, value && *value, ImVec2(0.0f, scaled(25.0f)));
    ImGui::PopStyleVar();
    if (clicked && value)
        *value = !*value;
    return clicked;
}

void log_level_chip(const char* label, bool* value, int count, ImVec4 color)
{
    if (value && *value)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(color.x, color.y, color.z, 0.80f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.x, color.y, color.z, 0.95f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, color);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.035f, 0.041f, 0.047f, 1.0f));
        toggle_chip(label, value, count);
        ImGui::PopStyleColor(4);
    }
    else
    {
        toggle_chip(label, value, count);
    }
}

bool vec3_field(const char* label, float values[3], float speed, float min_value, float max_value)
{
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(scaled(92.0f));
    ImGui::SetNextItemWidth(-1.0f);
    const bool changed = ImGui::DragFloat3("##value", values, speed, min_value, max_value, "%.3f");
    ImGui::PopID();
    return changed;
}

bool float_field(const char* label, float* value, float speed)
{
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(scaled(92.0f));
    ImGui::SetNextItemWidth(-1.0f);
    const bool changed = ImGui::DragFloat("##value", value, speed, 0.0f, 0.0f, "%.3f");
    ImGui::PopID();
    return changed;
}

bool color_field3(const char* label, float values[3])
{
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(scaled(92.0f));
    ImGui::SetNextItemWidth(-1.0f);
    const bool changed = ImGui::ColorEdit3("##value", values);
    ImGui::PopID();
    return changed;
}

bool checkbox_field(const char* label, bool* value)
{
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(scaled(92.0f));
    const bool changed = ImGui::Checkbox("##value", value);
    ImGui::PopID();
    return changed;
}

void empty_state(const char* title, const char* message)
{
    ImGui::Dummy(ImVec2(0.0f, scaled(10.0f)));
    ImGui::PushStyleColor(ImGuiCol_Text, colors::text_muted);
    ImGui::TextUnformatted(title);
    ImGui::TextWrapped("%s", message);
    ImGui::PopStyleColor();
}

void muted_text(const char* text)
{
    ImGui::PushStyleColor(ImGuiCol_Text, colors::text_muted);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void panel_toolbar_begin()
{
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(scaled(6.0f), scaled(4.0f)));
}

void panel_toolbar_end()
{
    ImGui::PopStyleVar();
    ImGui::Separator();
}

} // namespace arc::editor::ui

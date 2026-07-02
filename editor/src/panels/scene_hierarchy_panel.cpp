#include <arc/editor/panels/scene_hierarchy_panel.h>

#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/diagnostics/diagnostics.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <string>

namespace arc::editor
{
namespace
{

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool matches_filter(const char* label, const char* filter)
{
    if (!filter || filter[0] == '\0')
        return true;
    return lowercase(label).find(lowercase(filter)) != std::string::npos;
}

const char* icon_for_kind(host_entity_kind kind) noexcept
{
    switch (kind)
    {
    case host_entity_kind::camera:
        return "C";
    case host_entity_kind::light:
        return "*";
    case host_entity_kind::environment:
        return "E";
    case host_entity_kind::mesh:
    case host_entity_kind::imported:
        return "M";
    case host_entity_kind::primitive:
        return "P";
    case host_entity_kind::unknown:
        break;
    }
    return "?";
}

void log_command_failure(const host_response& result)
{
    if (result.succeeded)
        return;
    arc::error("editor.scene", result.error.empty() ? "Host command failed" : result.error);
}

void draw_entity(arc_host& host, const host_scene_entity_snapshot& entity, const char* filter)
{
    if (!matches_filter(entity.name.c_str(), filter))
        return;

    ImGui::PushID(static_cast<int>(entity.entity.index));
    if (ui::entity_row(entity.name.c_str(), icon_for_kind(entity.kind), entity.selected, entity.active))
        log_command_failure(host.execute(host_select_entity_command{ .entity = entity.entity }));

    if (ImGui::BeginPopupContextItem("entity-context"))
    {
        static std::array<char, 96> rename_buffer{};
        if (ImGui::IsWindowAppearing())
            std::snprintf(rename_buffer.data(), rename_buffer.size(), "%s", entity.name.c_str());

        ImGui::SetNextItemWidth(ui::scaled(180.0f));
        const bool rename_entered = ImGui::InputText(
            "##rename",
            rename_buffer.data(),
            rename_buffer.size(),
            ImGuiInputTextFlags_EnterReturnsTrue);
        if (rename_entered || ImGui::MenuItem("Rename"))
        {
            const std::string new_name{ rename_buffer.data() };
            if (!new_name.empty())
                log_command_failure(host.execute(host_rename_entity_command{ .entity = entity.entity, .name = new_name }));
            ImGui::CloseCurrentPopup();
        }
        if (ImGui::MenuItem("Delete"))
        {
            log_command_failure(host.execute(host_delete_entity_command{ .entity = entity.entity }));
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    ImGui::PopID();
}

} // namespace

void draw_scene_hierarchy_panel(arc_host& host)
{
    if (!ui::begin_panel("Scene##Hierarchy"))
    {
        ui::end_panel();
        return;
    }

    static char scene_search[128]{};
    ui::panel_toolbar_begin();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - ui::scaled(34.0f));
    ui::search_box("##scene-search", "Search entities...", scene_search, sizeof(scene_search));
    ImGui::SameLine();
    if (ImGui::Button("+", ImVec2(ui::scaled(28.0f), 0.0f)))
        ImGui::OpenPopup("create-entity-menu");
    if (ImGui::BeginPopup("create-entity-menu"))
    {
        ImGui::TextUnformatted("Add Primitive");
        ImGui::Separator();
        const auto add_entity = [&](const char* label, host_create_entity_kind kind) {
            if (ImGui::MenuItem(label))
            {
                const auto result = host.execute(host_create_entity_command{ .kind = kind });
                log_command_failure(result);
            }
        };
        add_entity("Plane", host_create_entity_kind::plane);
        add_entity("Cube", host_create_entity_kind::cube);
        add_entity("Sphere", host_create_entity_kind::sphere);
        add_entity("Cylinder", host_create_entity_kind::cylinder);
        ImGui::Separator();
        ImGui::TextUnformatted("Add Environment");
        add_entity("World Environment", host_create_entity_kind::world_environment);
        add_entity("Terrain", host_create_entity_kind::terrain);
        add_entity("Water Plane", host_create_entity_kind::water);
        add_entity("Grass Patch", host_create_entity_kind::grass_patch);
        add_entity("Decal", host_create_entity_kind::decal);
        ImGui::EndPopup();
    }
    ui::panel_toolbar_end();

    ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text_muted);
    ImGui::TextUnformatted("World");
    ImGui::PopStyleColor();
    ImGui::Indent(ui::scaled(8.0f));
    const auto snapshot = host.scene_snapshot();
    for (const auto& entity : snapshot.entities)
        draw_entity(host, entity, scene_search);
    ImGui::Unindent(ui::scaled(8.0f));

    ui::end_panel();
}

} // namespace arc::editor

#include <arc/editor/panels/scene_hierarchy_panel.h>

#include <arc/editor/editor_interaction.h>
#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/diagnostics/diagnostics.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
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

const char* entity_label(const editor_scene_state& state, scene::entity entity, const char* fallback)
{
    const auto* name = state.scene.try_get<scene::name_component>(entity);
    return name ? name->value.c_str() : fallback;
}

void draw_entity(
    editor_scene_state& state,
    scene::entity entity,
    const char* fallback,
    const char* icon,
    const char* filter)
{
    if (!entity.valid() || !state.scene.alive(entity))
        return;

    const char* label = entity_label(state, entity, fallback);
    if (!matches_filter(label, filter))
        return;

    const auto* active = state.scene.try_get<scene::active_component>(entity);
    const bool visible = !active || active->active;
    const bool selected = entity == state.selected_entity;
    if (ui::entity_row(label, icon, selected, visible))
        select_entity(state.scene, entity, state.selected_entity);
}

} // namespace

void draw_scene_hierarchy_panel(editor_scene_state& editor_scene, render::renderer& renderer)
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
        const auto add_primitive = [&](const char* label, editor_primitive_type type) {
            if (ImGui::MenuItem(label))
            {
                const auto entity = add_primitive_to_scene(editor_scene, renderer, type);
                if (entity.valid())
                    arc::info("editor.scene", std::string("Added primitive: ") + label);
                else
                    arc::error("editor.scene", std::string("Failed to add primitive: ") + label);
            }
        };
        add_primitive("Plane", editor_primitive_type::plane);
        add_primitive("Cube", editor_primitive_type::cube);
        add_primitive("Sphere", editor_primitive_type::sphere);
        add_primitive("Cylinder", editor_primitive_type::cylinder);
        ImGui::Separator();
        ImGui::TextUnformatted("Add Environment");
        const auto add_environment = [&](const char* label, const auto& create) {
            if (ImGui::MenuItem(label))
            {
                const auto entity = create();
                if (entity.valid())
                    arc::info("editor.scene", std::string("Added environment entity: ") + label);
                else
                    arc::error("editor.scene", std::string("Failed to add environment entity: ") + label);
            }
        };
        add_environment("World Environment", [&] { return add_world_environment_to_scene(editor_scene); });
        add_environment("Terrain", [&] { return add_terrain_to_scene(editor_scene, renderer); });
        add_environment("Water Plane", [&] { return add_water_to_scene(editor_scene, renderer); });
        add_environment("Grass Patch", [&] { return add_grass_patch_to_scene(editor_scene, renderer); });
        add_environment("Decal", [&] { return add_decal_to_scene(editor_scene); });
        ImGui::EndPopup();
    }
    ui::panel_toolbar_end();

    ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text_muted);
    ImGui::TextUnformatted("World");
    ImGui::PopStyleColor();
    ImGui::Indent(ui::scaled(8.0f));
    draw_entity(editor_scene, editor_scene.sun_entity, "Sun", "*", scene_search);
    draw_entity(editor_scene, editor_scene.camera_entity, "Editor Camera", "C", scene_search);
    for (const auto entity : editor_scene.environment_entities)
        draw_entity(editor_scene, entity, "Environment", "E", scene_search);
    draw_entity(editor_scene, editor_scene.mesh_entity, "Default Mesh", "M", scene_search);
    for (const auto entity : editor_scene.imported_scene_entities)
        draw_entity(editor_scene, entity, "Imported Mesh", "M", scene_search);
    for (const auto entity : editor_scene.primitive_entities)
        draw_entity(editor_scene, entity, "Primitive", "P", scene_search);
    ImGui::Unindent(ui::scaled(8.0f));

    ui::end_panel();
}

} // namespace arc::editor

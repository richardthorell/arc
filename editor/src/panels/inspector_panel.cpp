#include <arc/editor/panels/inspector_panel.h>

#include <arc/editor/editor_interaction.h>
#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>

#include <imgui.h>

#include <array>
#include <cstdio>
#include <string>

namespace arc::editor
{
namespace
{

void draw_reset_button(const char* id, const char* tooltip, const auto& reset)
{
    ImGui::SameLine();
    if (ImGui::SmallButton(id))
        reset();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("%s", tooltip);
}

} // namespace

void draw_inspector_panel(editor_scene_state& editor_scene)
{
    if (!ui::begin_panel("Inspector"))
    {
        ui::end_panel();
        return;
    }

    if (!editor_scene.scene.alive(editor_scene.selected_entity))
    {
        ui::empty_state("No entity selected", "Select an entity in the hierarchy or viewport to edit its components.");
        ui::end_panel();
        return;
    }

    const auto* name = editor_scene.scene.try_get<scene::name_component>(editor_scene.selected_entity);
    ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text);
    ImGui::TextUnformatted(name ? name->value.c_str() : "Unnamed Entity");
    ImGui::PopStyleColor();

    bool active = true;
    if (auto* active_component = editor_scene.scene.try_get<scene::active_component>(editor_scene.selected_entity))
        active = active_component->active;
    ImGui::SameLine(ImGui::GetWindowWidth() - ui::scaled(112.0f));
    if (ImGui::Checkbox("Active", &active))
        editor_scene.scene.emplace<scene::active_component>(editor_scene.selected_entity, active);

    std::array<char, 96> tag_text{};
    if (const auto* tag = editor_scene.scene.try_get<scene::tag_component>(editor_scene.selected_entity))
        std::snprintf(tag_text.data(), tag_text.size(), "%s", tag->value.c_str());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##entity-tag", "Tag", tag_text.data(), tag_text.size()))
        editor_scene.scene.emplace<scene::tag_component>(editor_scene.selected_entity, std::string{ tag_text.data() });
    ImGui::Separator();

    if (auto* transform = editor_scene.scene.try_get<scene::transform_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Transform"))
        {
            float position[3]{ transform->position[0], transform->position[1], transform->position[2] };
            auto euler = euler_degrees_from_quaternion(transform->rotation);
            float rotation[3]{ euler[0], euler[1], euler[2] };
            float scale[3]{ transform->scale[0], transform->scale[1], transform->scale[2] };
            if (ui::vec3_field("Position", position, 0.01f))
                transform->set_position({ position[0], position[1], position[2] });
            draw_reset_button("Reset##position", "Reset position", [&] { transform->set_position({ 0.0f, 0.0f, 0.0f }); });
            if (ui::vec3_field("Rotation", rotation, 0.1f))
                transform->set_rotation(quaternion_from_euler_degrees({ rotation[0], rotation[1], rotation[2] }));
            draw_reset_button("Reset##rotation", "Reset rotation", [&] { transform->set_rotation({}); });
            if (ui::vec3_field("Scale", scale, 0.01f, 0.001f, 1000.0f))
                transform->set_scale({ scale[0], scale[1], scale[2] });
            draw_reset_button("Reset##scale", "Reset scale", [&] { transform->set_scale({ 1.0f, 1.0f, 1.0f }); });
            ui::component_card_end();
        }
    }

    if (auto* mesh = editor_scene.scene.try_get<scene::mesh_renderer_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Mesh Renderer"))
        {
            ImGui::TextUnformatted("Mesh");
            ImGui::SameLine(ui::scaled(92.0f));
            ImGui::Text("Default Mesh (%u:%u)", mesh->mesh.index, mesh->mesh.generation);
            ImGui::TextUnformatted("Material");
            ImGui::SameLine(ui::scaled(92.0f));
            if (mesh->material.valid())
                ImGui::Text("Material %u:%u", mesh->material.index, mesh->material.generation);
            else
                ui::muted_text("None");
            ui::checkbox_field("Visible", &mesh->visible);
            ui::muted_text("Cast shadows will be added with shadow passes.");
            ui::component_card_end();
        }
    }

    if (auto* light = editor_scene.scene.try_get<scene::directional_light_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Directional Light"))
        {
            float color[3]{ light->color[0], light->color[1], light->color[2] };
            if (ui::color_field3("Color", color))
                light->color = { color[0], color[1], color[2] };
            ui::float_field("Intensity", &light->intensity, 0.05f);
            ui::checkbox_field("Cast Shadows", &light->casts_shadows);
            ui::component_card_end();
        }
    }

    ui::end_panel();
}

} // namespace arc::editor

#include <arc/editor/panels/inspector_panel.h>

#include <arc/editor/asset_drag_drop.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/material_library.h>
#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/diagnostics/diagnostics.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
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

const char* unit_label(render::light_intensity_unit unit) noexcept
{
    switch (unit)
    {
    case render::light_intensity_unit::unitless:
        return "Unitless";
    case render::light_intensity_unit::lumen:
        return "Lumen";
    case render::light_intensity_unit::candela:
        return "Candela";
    case render::light_intensity_unit::lux:
        return "Lux";
    }
    return "Unitless";
}

void draw_unit_combo(render::light_intensity_unit& unit)
{
    int selected = static_cast<int>(unit);
    const char* labels[]{ "Unitless", "Lumen", "Candela", "Lux" };
    ImGui::TextUnformatted("Unit");
    ImGui::SameLine(ui::scaled(92.0f));
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##light-unit", &selected, labels, 4))
        unit = static_cast<render::light_intensity_unit>(selected);
}

void draw_shadow_filter_combo(render::shadow_filter& filter)
{
    int selected = static_cast<int>(filter);
    const char* labels[]{ "None", "PCF 3x3", "PCF 5x5", "PCSS" };
    ImGui::TextUnformatted("Filter");
    ImGui::SameLine(ui::scaled(92.0f));
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::Combo("##shadow-filter", &selected, labels, 4))
        filter = static_cast<render::shadow_filter>(selected);
}

void draw_shadow_settings(render::shadow_settings& shadow, bool rendered)
{
    if (!ui::component_card_begin("Shadows"))
        return;

    ui::checkbox_field("Enabled", &shadow.enabled);
    int resolution = static_cast<int>(shadow.resolution);
    if (ImGui::DragInt("Resolution", &resolution, 16.0f, 256, 8192))
        shadow.resolution = static_cast<std::uint32_t>(std::max(256, resolution));
    ui::float_field("Strength", &shadow.strength, 0.01f);
    ui::float_field("Bias", &shadow.bias, 0.0001f);
    ui::float_field("Normal Bias", &shadow.normal_bias, 0.001f);
    draw_shadow_filter_combo(shadow.filter);
    if (!rendered)
        ui::muted_text("Shadow data is scaffolded; rendering arrives in a later pass.");

    ui::component_card_end();
}

template <class Light>
void draw_common_light_fields(Light& light)
{
    ui::checkbox_field("Enabled", &light.enabled);
    float color[3]{ light.color[0], light.color[1], light.color[2] };
    if (ui::color_field3("Color", color))
        light.color = { color[0], color[1], color[2] };
    ui::checkbox_field("Temperature", &light.use_color_temperature);
    if (light.use_color_temperature)
        ui::float_field("Kelvin", &light.temperature_kelvin, 25.0f);
    draw_unit_combo(light.intensity_unit);
    ui::float_field("Intensity", &light.intensity, 0.05f);
    ui::checkbox_field("Cast Shadows", &light.casts_shadows);
    ImGui::TextUnformatted("Cookie");
    ImGui::SameLine(ui::scaled(92.0f));
    ui::muted_text(light.cookie_texture.valid() ? "Assigned" : "None");
}

void sync_mesh_tint(editor_scene_state& editor_scene, const math::vector3f& color, float alpha = 1.0f)
{
    if (auto* mesh = editor_scene.scene.try_get<scene::mesh_renderer_component>(editor_scene.selected_entity))
        mesh->base_color_tint = { color[0], color[1], color[2], alpha };
}

} // namespace

void draw_inspector_panel(
    editor_scene_state& editor_scene,
    const editor_asset_state& editor_assets,
    render::renderer* renderer)
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
            const auto material_name = [&]() -> const char* {
                for (const auto& record : editor_scene.material_library.materials)
                {
                    if (record.material == mesh->material)
                        return record.asset.name.c_str();
                }
                return nullptr;
            }();
            if (material_name)
                ImGui::TextUnformatted(material_name);
            else if (mesh->material.valid())
                ImGui::Text("Material %u:%u", mesh->material.index, mesh->material.generation);
            else
                ui::muted_text("None");
            editor_asset_payload payload;
            if (accept_asset_drag_drop(payload))
            {
                if (payload.kind != editor_asset_kind::material)
                {
                    arc::warn("editor.materials", "Only material assets can be dropped onto mesh renderer material slots");
                }
                else if (!renderer)
                {
                    arc::warn("editor.materials", "No renderer is available for material assignment");
                }
                else
                {
                    std::string message;
                    if (apply_material_asset_to_entity(
                            editor_scene.material_library,
                            *renderer,
                            editor_assets.root,
                            payload.relative_path,
                            editor_scene.scene,
                            editor_scene.selected_entity,
                            &message))
                        arc::info("editor.materials", message);
                    else
                        arc::warn("editor.materials", message.empty() ? "Material drop was rejected" : message);
                }
            }
            if (editor_scene.material_editor.open && editor_scene.material_editor.material.valid())
            {
                if (ImGui::Button("Apply Open Material"))
                    apply_material_to_selected(editor_scene.scene, editor_scene.selected_entity, editor_scene.material_editor.material);
                ImGui::SameLine();
                ui::muted_text(editor_scene.material_editor.working.name.c_str());
            }
            float tint[3]{ mesh->base_color_tint[0], mesh->base_color_tint[1], mesh->base_color_tint[2] };
            if (ui::color_field3("Color", tint))
                mesh->base_color_tint = { tint[0], tint[1], tint[2], mesh->base_color_tint[3] };
            ui::checkbox_field("Visible", &mesh->visible);
            ui::component_card_end();
        }
    }

    if (auto* light = editor_scene.scene.try_get<scene::directional_light_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Directional Light"))
        {
            draw_common_light_fields(*light);
            ui::component_card_end();
        }
        draw_shadow_settings(light->shadow, true);
    }

    if (auto* light = editor_scene.scene.try_get<scene::point_light_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Point Light"))
        {
            draw_common_light_fields(*light);
            ui::float_field("Range", &light->range, 0.05f);
            ui::component_card_end();
        }
        draw_shadow_settings(light->shadow, false);
    }

    if (auto* light = editor_scene.scene.try_get<scene::spot_light_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Spot Light"))
        {
            draw_common_light_fields(*light);
            ui::float_field("Range", &light->range, 0.05f);
            ui::float_field("Inner", &light->inner_angle, 0.01f);
            ui::float_field("Outer", &light->outer_angle, 0.01f);
            ui::component_card_end();
        }
        draw_shadow_settings(light->shadow, false);
    }

    if (auto* sky = editor_scene.scene.try_get<scene::sky_atmosphere_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Sky Atmosphere"))
        {
            ui::checkbox_field("Enabled", &sky->enabled);
            float tint[3]{ sky->tint[0], sky->tint[1], sky->tint[2] };
            if (ui::color_field3("Tint", tint))
                sky->tint = { tint[0], tint[1], tint[2] };
            ui::float_field("Exposure", &sky->exposure, 0.01f);
            ui::float_field("Rayleigh", &sky->rayleigh_strength, 0.01f);
            ui::float_field("Mie", &sky->mie_strength, 0.01f);
            ui::float_field("Ozone", &sky->ozone_strength, 0.01f);
            ui::float_field("Sun Size", &sky->sun_disk_size, 0.001f);
            ui::float_field("Sun Power", &sky->sun_disk_intensity, 0.01f);
            ui::component_card_end();
        }
    }

    if (auto* fog = editor_scene.scene.try_get<scene::height_fog_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Height Fog"))
        {
            ui::checkbox_field("Enabled", &fog->enabled);
            float color[3]{ fog->color[0], fog->color[1], fog->color[2] };
            if (ui::color_field3("Color", color))
                fog->color = { color[0], color[1], color[2] };
            ui::float_field("Density", &fog->density, 0.001f);
            ui::float_field("Falloff", &fog->height_falloff, 0.001f);
            ui::float_field("Start", &fog->start_distance, 0.05f);
            ui::float_field("Opacity", &fog->max_opacity, 0.01f);
            ui::float_field("Sun Scatter", &fog->sun_scattering_strength, 0.01f);
            ui::component_card_end();
        }
    }

    if (auto* terrain = editor_scene.scene.try_get<scene::terrain_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Terrain"))
        {
            ui::checkbox_field("Enabled", &terrain->enabled);
            ui::float_field("Size", &terrain->size, 0.1f);
            int subdivisions = static_cast<int>(terrain->subdivisions);
            ImGui::TextUnformatted("Subdivisions");
            ImGui::SameLine(ui::scaled(92.0f));
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::DragInt("##terrain-subdivisions", &subdivisions, 1.0f, 1, 512))
                terrain->subdivisions = static_cast<std::uint32_t>(std::max(1, subdivisions));
            ui::float_field("Height", &terrain->height_scale, 0.01f);
            float color[3]{ terrain->base_color[0], terrain->base_color[1], terrain->base_color[2] };
            if (ui::color_field3("Color", color))
            {
                terrain->base_color = { color[0], color[1], color[2] };
                sync_mesh_tint(editor_scene, terrain->base_color);
            }
            ui::checkbox_field("Shadows", &terrain->receive_shadows);
            ui::muted_text("Size/subdivision edits update scene data; mesh regeneration will follow terrain tooling.");
            ui::component_card_end();
        }
    }

    if (auto* water = editor_scene.scene.try_get<scene::water_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Water"))
        {
            ui::checkbox_field("Enabled", &water->enabled);
            ui::float_field("Size", &water->size, 0.1f);
            float color[3]{ water->color[0], water->color[1], water->color[2] };
            if (ui::color_field3("Color", color))
            {
                water->color = { color[0], color[1], color[2] };
                sync_mesh_tint(editor_scene, water->color, std::clamp(1.0f - water->transparency, 0.05f, 1.0f));
            }
            ui::float_field("Roughness", &water->roughness, 0.01f);
            ui::float_field("Wave Scale", &water->wave_scale, 0.001f);
            ui::float_field("Wave Speed", &water->wave_speed, 0.01f);
            ui::float_field("Transparency", &water->transparency, 0.01f);
            sync_mesh_tint(editor_scene, water->color, std::clamp(1.0f - water->transparency, 0.05f, 1.0f));
            ui::component_card_end();
        }
    }

    if (auto* vegetation = editor_scene.scene.try_get<scene::vegetation_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Vegetation"))
        {
            ui::checkbox_field("Enabled", &vegetation->enabled);
            int density = static_cast<int>(vegetation->density);
            ImGui::TextUnformatted("Density");
            ImGui::SameLine(ui::scaled(92.0f));
            ImGui::SetNextItemWidth(-1.0f);
            if (ImGui::DragInt("##vegetation-density", &density, 1.0f, 1, 10000))
                vegetation->density = static_cast<std::uint32_t>(std::max(1, density));
            ui::float_field("Patch Size", &vegetation->patch_size, 0.1f);
            float color[3]{ vegetation->color[0], vegetation->color[1], vegetation->color[2] };
            if (ui::color_field3("Color", color))
            {
                vegetation->color = { color[0], color[1], color[2] };
                sync_mesh_tint(editor_scene, vegetation->color);
            }
            ui::float_field("Wind", &vegetation->wind_strength, 0.01f);
            ui::float_field("Wind Speed", &vegetation->wind_speed, 0.01f);
            ui::component_card_end();
        }
    }

    if (auto* decal = editor_scene.scene.try_get<scene::decal_component>(editor_scene.selected_entity))
    {
        if (ui::component_card_begin("Decal"))
        {
            ui::checkbox_field("Enabled", &decal->enabled);
            float color[3]{ decal->color[0], decal->color[1], decal->color[2] };
            if (ui::color_field3("Color", color))
                decal->color = { color[0], color[1], color[2], decal->color[3] };
            ui::float_field("Opacity", &decal->opacity, 0.01f);
            decal->color[3] = decal->opacity;
            ui::muted_text("Projection rendering is scaffolded for a later decal pass.");
            ui::component_card_end();
        }
    }

    ui::end_panel();
}

} // namespace arc::editor

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
#include <utility>

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

void draw_world_environment_inspector(arc_host& host, host_world_environment_snapshot environment)
{
    bool changed{};
    const auto check = [&](bool value) { changed = value || changed; };
    const auto preset = [&](const char* label, host_world_environment_preset value) {
        if (ImGui::SmallButton(label))
            host.execute(host_apply_world_environment_preset_command{ environment.entity, value });
    };

    if (ui::component_card_begin("World Environment"))
    {
        check(ui::checkbox_field("Enabled", &environment.enabled));
        check(ui::checkbox_field("Sky Visible", &environment.sky_visible));
        check(ui::checkbox_field("Affect Lighting", &environment.affect_lighting));
        int source = static_cast<int>(environment.sky_source);
        const char* labels[]{ "Physical Atmosphere", "HDRI", "Solid Color" };
        if (ImGui::Combo("Sky Source", &source, labels, 3))
        {
            environment.sky_source = static_cast<host_sky_source>(source);
            changed = true;
        }
        check(ui::float_field("Radiance", &environment.radiance_intensity, 0.02f));
        check(ui::float_field("HDRI Rotation", &environment.hdri_rotation_degrees, 1.0f));
        float solid[3]{ environment.solid_color.x, environment.solid_color.y, environment.solid_color.z };
        if (ui::color_field3("Solid Color", solid))
        {
            environment.solid_color = { solid[0], solid[1], solid[2] };
            changed = true;
        }
        preset("Clear", host_world_environment_preset::clear_day); ImGui::SameLine();
        preset("Alpine", host_world_environment_preset::alpine_late_morning); ImGui::SameLine();
        preset("Golden", host_world_environment_preset::golden_hour);
        preset("Overcast", host_world_environment_preset::overcast); ImGui::SameLine();
        preset("Night", host_world_environment_preset::night); ImGui::SameLine();
        preset("Indoor", host_world_environment_preset::indoor_neutral);
        ui::component_card_end();
    }

    if (ui::component_card_begin("Sun, Moon & Time"))
    {
        int sun_mode = static_cast<int>(environment.sun_mode);
        const char* sun_modes[]{ "Manual Light", "Geographic" };
        if (ImGui::Combo("Sun Position", &sun_mode, sun_modes, 2))
        {
            environment.sun_mode = static_cast<host_sun_position_mode>(sun_mode);
            changed = true;
        }
        int time_mode = static_cast<int>(environment.time_mode);
        const char* time_modes[]{ "Fixed", "Simulated", "System Clock" };
        if (ImGui::Combo("Clock", &time_mode, time_modes, 3))
        {
            environment.time_mode = static_cast<host_celestial_time_mode>(time_mode);
            changed = true;
        }
        check(ui::checkbox_field("Playing", &environment.playing));
        check(ui::float_field("Time of Day", &environment.local_time_hours, 0.05f));
        check(ui::float_field("Time Scale", &environment.time_scale, 1.0f));
        check(ui::float_field("Latitude", &environment.latitude_degrees, 0.1f));
        check(ui::float_field("Longitude", &environment.longitude_degrees, 0.1f));
        check(ui::float_field("UTC Offset", &environment.utc_offset_hours, 0.25f));
        check(ui::float_field("North Offset", &environment.north_offset_degrees, 1.0f));
        check(ui::checkbox_field("Loop Day", &environment.loop_day));
        check(ui::checkbox_field("Automatic Sun", &environment.automatic_sun_light));
        check(ui::float_field("Sun Intensity", &environment.sun_intensity_multiplier, 0.05f));
        check(ui::float_field("Sun Temperature", &environment.sun_temperature_multiplier, 0.05f));
        check(ui::checkbox_field("Moon", &environment.moon_enabled));
        check(ui::checkbox_field("Automatic Phase", &environment.automatic_moon_phase));
        check(ui::float_field("Moon Phase", &environment.moon_phase, 0.01f));
        check(ui::float_field("Moon Brightness", &environment.moon_intensity, 0.01f));
        check(ui::float_field("Moon Angular Radius", &environment.moon_angular_radius_degrees, 0.01f));
        check(ui::checkbox_field("Stars", &environment.stars_enabled));
        check(ui::float_field("Star Density", &environment.star_density, 0.01f));
        check(ui::float_field("Star Intensity", &environment.star_intensity, 0.05f));
        check(ui::float_field("Star Twinkle", &environment.star_twinkle, 0.01f));
        ui::component_card_end();
    }

    if (ui::component_card_begin("Sky Atmosphere"))
    {
        float tint[3]{ environment.atmosphere_tint.x, environment.atmosphere_tint.y, environment.atmosphere_tint.z };
        if (ui::color_field3("Tint", tint))
        {
            environment.atmosphere_tint = { tint[0], tint[1], tint[2] };
            changed = true;
        }
        check(ui::float_field("Exposure", &environment.exposure, 0.01f));
        check(ui::float_field("Rayleigh", &environment.rayleigh_strength, 0.01f));
        check(ui::float_field("Mie", &environment.mie_strength, 0.01f));
        check(ui::float_field("Ozone", &environment.ozone_strength, 0.01f));
        check(ui::float_field("Mie Anisotropy", &environment.mie_anisotropy, 0.01f));
        check(ui::float_field("Rayleigh Height", &environment.rayleigh_scale_height, 0.1f));
        check(ui::float_field("Mie Height", &environment.mie_scale_height, 0.1f));
        check(ui::float_field("Multi Scattering", &environment.multi_scattering_factor, 0.05f));
        check(ui::float_field("Planet Radius", &environment.planet_radius, 1.0f));
        check(ui::float_field("Atmosphere Radius", &environment.atmosphere_radius, 1.0f));
        check(ui::float_field("Sun Size", &environment.sun_disk_size, 0.001f));
        check(ui::float_field("Sun Power", &environment.sun_disk_intensity, 0.01f));
        ui::component_card_end();
    }

    if (ui::component_card_begin("Cloud Layers"))
    {
        check(ui::checkbox_field("Enabled", &environment.clouds_enabled));
        check(ui::checkbox_field("Cloud Shadows", &environment.cloud_shadows));
        check(ui::checkbox_field("Cumulus", &environment.cumulus.enabled));
        check(ui::float_field("Cumulus Coverage", &environment.cumulus.coverage, 0.01f));
        check(ui::float_field("Cumulus Density", &environment.cumulus.density, 0.01f));
        check(ui::float_field("Cumulus Wind", &environment.cumulus.wind_speed, 0.1f));
        check(ui::checkbox_field("Cirrus", &environment.cirrus.enabled));
        check(ui::float_field("Cirrus Coverage", &environment.cirrus.coverage, 0.01f));
        check(ui::float_field("Cirrus Density", &environment.cirrus.density, 0.01f));
        check(ui::float_field("Cirrus Wind", &environment.cirrus.wind_speed, 0.1f));
        ui::component_card_end();
    }

    if (ui::component_card_begin("Environment Lighting"))
    {
        check(ui::checkbox_field("Enabled", &environment.lighting_enabled));
        int source = static_cast<int>(environment.lighting_source);
        const char* labels[]{ "Follow Sky", "HDRI", "Constant Color" };
        if (ImGui::Combo("Source", &source, labels, 3))
        {
            environment.lighting_source = static_cast<host_environment_lighting_source>(source);
            changed = true;
        }
        float color[3]{ environment.lighting_color.x, environment.lighting_color.y, environment.lighting_color.z };
        if (ui::color_field3("Ambient", color))
        {
            environment.lighting_color = { color[0], color[1], color[2] };
            changed = true;
        }
        check(ui::float_field("Diffuse", &environment.diffuse_intensity, 0.02f));
        check(ui::float_field("Specular", &environment.specular_intensity, 0.02f));
        ui::component_card_end();
    }

    if (ui::component_card_begin("Height Fog"))
    {
        check(ui::checkbox_field("Enabled", &environment.fog_enabled));
        float color[3]{ environment.fog_color.x, environment.fog_color.y, environment.fog_color.z };
        if (ui::color_field3("Color", color))
        {
            environment.fog_color = { color[0], color[1], color[2] };
            changed = true;
        }
        check(ui::float_field("Density", &environment.fog_density, 0.001f));
        check(ui::float_field("Falloff", &environment.fog_height_falloff, 0.001f));
        check(ui::float_field("Start", &environment.fog_start_distance, 0.05f));
        check(ui::float_field("Opacity", &environment.fog_max_opacity, 0.01f));
        check(ui::float_field("Sun Scatter", &environment.fog_sun_scattering, 0.01f));
        ui::component_card_end();
    }

    if (changed)
        host.execute(host_set_world_environment_command{ std::move(environment) });
}

} // namespace

void draw_inspector_panel(
    arc_host& host,
    const editor_asset_state& editor_assets,
    render::renderer* renderer)
{
    editor_scene_state& editor_scene = host.scene_state();

    if (!ui::begin_panel("Inspector"))
    {
        ui::end_panel();
        return;
    }

    const auto selected = host.selected_entity_snapshot();
    if (!selected.entity.valid())
    {
        ui::empty_state("No entity selected", "Select an entity in the hierarchy or viewport to edit its components.");
        ui::end_panel();
        return;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text);
    ImGui::TextUnformatted(selected.name.empty() ? "Unnamed Entity" : selected.name.c_str());
    ImGui::PopStyleColor();

    bool active = selected.active;
    ImGui::SameLine(ImGui::GetWindowWidth() - ui::scaled(112.0f));
    if (ImGui::Checkbox("Active", &active))
    {
        const auto result = host.execute(host_set_active_command{ .entity = selected.entity, .active = active });
        if (!result.succeeded)
            arc::error("editor.inspector", result.error);
    }

    std::array<char, 96> tag_text{};
    std::snprintf(tag_text.data(), tag_text.size(), "%s", selected.tag.c_str());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##entity-tag", "Tag", tag_text.data(), tag_text.size()))
    {
        const auto result = host.execute(host_set_tag_command{ .entity = selected.entity, .tag = std::string{ tag_text.data() } });
        if (!result.succeeded)
            arc::error("editor.inspector", result.error);
    }
    ImGui::Separator();

    if (selected.transform)
    {
        auto transform = *selected.transform;
        if (ui::component_card_begin("Transform"))
        {
            bool transform_changed{};
            float position[3]{ transform.position.x, transform.position.y, transform.position.z };
            math::quatf rotation_quat{ transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w };
            auto euler = euler_degrees_from_quaternion(rotation_quat);
            float rotation[3]{ euler[0], euler[1], euler[2] };
            float scale[3]{ transform.scale.x, transform.scale.y, transform.scale.z };
            if (ui::vec3_field("Position", position, 0.01f))
            {
                transform.position = { position[0], position[1], position[2] };
                transform_changed = true;
            }
            draw_reset_button("Reset##position", "Reset position", [&] {
                transform.position = {};
                transform_changed = true;
            });
            if (ui::vec3_field("Rotation", rotation, 0.1f))
            {
                const auto updated = quaternion_from_euler_degrees({ rotation[0], rotation[1], rotation[2] });
                transform.rotation = { updated[0], updated[1], updated[2], updated[3] };
                transform_changed = true;
            }
            draw_reset_button("Reset##rotation", "Reset rotation", [&] {
                transform.rotation = {};
                transform_changed = true;
            });
            if (ui::vec3_field("Scale", scale, 0.01f, 0.001f, 1000.0f))
            {
                transform.scale = { scale[0], scale[1], scale[2] };
                transform_changed = true;
            }
            draw_reset_button("Reset##scale", "Reset scale", [&] {
                transform.scale = { 1.0f, 1.0f, 1.0f };
                transform_changed = true;
            });
            if (transform_changed)
            {
                const auto result = host.execute(host_set_transform_command{
                    .entity = selected.entity,
                    .transform = transform });
                if (!result.succeeded)
                    arc::error("editor.inspector", result.error);
            }
            ui::component_card_end();
        }
    }

    // TODO(host): migrate the remaining rich component editors through command/snapshot data.
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

    if (auto environment = host.world_environment_snapshot(selected.entity))
        draw_world_environment_inspector(host, std::move(*environment));

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

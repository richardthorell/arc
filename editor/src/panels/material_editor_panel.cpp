#include <arc/editor/panels/material_editor_panel.h>

#include <arc/editor/asset_drag_drop.h>
#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/diagnostics/diagnostics.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>

namespace arc::editor
{
namespace
{

void mark_dirty(material_editor_state& state)
{
    state.dirty = true;
}

bool color4_field(const char* label, math::vector4f& color)
{
    float values[4]{ color[0], color[1], color[2], color[3] };
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(ui::scaled(118.0f));
    ImGui::SetNextItemWidth(-1.0f);
    const bool changed = ImGui::ColorEdit4("##value", values);
    ImGui::PopID();
    if (changed)
        color = { values[0], values[1], values[2], values[3] };
    return changed;
}

bool texture_path_field(
    const char* label,
    material_editor_state& state,
    editor_material_library& library,
    render::renderer* renderer,
    const std::filesystem::path& asset_root,
    material_texture_slot slot,
    std::string& value)
{
    std::array<char, 256> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%s", value.c_str());
    ImGui::PushID(label);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(ui::scaled(118.0f));
    ImGui::SetNextItemWidth(-ui::scaled(58.0f));
    const bool changed = ImGui::InputText("##path", buffer.data(), buffer.size());
    ImGui::SameLine();
    bool cleared = false;
    if (ImGui::SmallButton("Clear"))
    {
        value.clear();
        cleared = true;
    }
    editor_asset_payload payload;
    bool dropped = false;
    if (accept_asset_drag_drop(payload))
    {
        if (!is_texture_asset_kind(payload.kind))
        {
            arc::warn("editor.materials", "Only texture assets can be dropped into material texture slots");
        }
        else
        {
            std::string message;
            if (assign_texture_to_material_slot(state, slot, asset_root, payload.relative_path, &message))
            {
                dropped = true;
                if (renderer)
                    update_material_editor_live_material(state, library, *renderer, asset_root);
                arc::info("editor.materials", message);
            }
            else
            {
                arc::warn("editor.materials", message.empty() ? "Texture drop was rejected" : message);
            }
        }
    }
    ImGui::PopID();
    if (changed)
        value = buffer.data();
    return changed || cleared || dropped;
}

void draw_preview(const material_asset& asset)
{
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float size = std::min(avail.x, ui::scaled(220.0f));
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 center(start.x + size * 0.5f, start.y + size * 0.5f);
    const float radius = size * 0.42f;

    draw->AddRectFilled(start, ImVec2(start.x + size, start.y + size), ImGui::ColorConvertFloat4ToU32(ui::colors::child_bg), ui::scaled(6.0f));
    const auto base = asset.material.base_color;
    const float rough = std::clamp(asset.material.roughness, 0.04f, 1.0f);
    const float metal = std::clamp(asset.material.metallic, 0.0f, 1.0f);
    const ImVec4 base_color(base[0], base[1], base[2], 1.0f);
    for (int ring = 24; ring >= 0; --ring)
    {
        const float t = static_cast<float>(ring) / 24.0f;
        const float shade = 0.18f + 0.70f * (1.0f - t * 0.75f);
        const float spec = std::pow(1.0f - t, 8.0f + rough * 30.0f) * (0.18f + metal * 0.45f);
        const ImVec4 color(
            std::min(base_color.x * shade + spec, 1.0f),
            std::min(base_color.y * shade + spec, 1.0f),
            std::min(base_color.z * shade + spec, 1.0f),
            1.0f);
        draw->AddCircleFilled(
            ImVec2(center.x - radius * 0.12f * t, center.y - radius * 0.18f * t),
            radius * t,
            ImGui::ColorConvertFloat4ToU32(color),
            64);
    }
    draw->AddCircle(center, radius, ImGui::ColorConvertFloat4ToU32(ui::colors::border), 64, 1.0f);
    ImGui::Dummy(ImVec2(size, size));
    ui::muted_text("Preview uses the material values; dedicated GPU preview target is next.");
}

} // namespace

void draw_material_editor_panel(
    editor_scene_state& editor_scene,
    const editor_asset_state& editor_assets,
    render::renderer* renderer)
{
    auto& state = editor_scene.material_editor;
    if (!state.open)
        return;

    if (!ui::begin_panel("Material Editor", &state.open))
    {
        ui::end_panel();
        return;
    }

    auto& asset = state.working;
    std::array<char, 128> name{};
    std::snprintf(name.data(), name.size(), "%s", asset.name.c_str());
    ImGui::TextUnformatted(asset.path.empty() ? "Unsaved Material" : asset.path.filename().string().c_str());
    ImGui::SameLine();
    ui::muted_text(state.dirty ? "Modified" : "Saved");
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##material-name", "Material name", name.data(), name.size()))
    {
        asset.name = name.data();
        asset.material.name = asset.name;
        mark_dirty(state);
    }

    if (ImGui::Button("Save"))
    {
        std::string message;
        if (renderer && save_material_editor(state, editor_scene.material_library, *renderer, editor_assets.root, message))
            arc::info("editor.materials", message);
        else
            arc::error("editor.materials", message.empty() ? "failed to save material" : message);
    }
    ImGui::SameLine();
    if (ImGui::Button("Revert"))
    {
        state.working = state.saved;
        state.dirty = false;
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply To Selected"))
    {
        if (apply_material_to_selected(editor_scene.scene, editor_scene.selected_entity, state.material))
            arc::info("editor.materials", "Applied material to selected entity");
    }

    if (ui::component_card_begin("Surface"))
    {
        if (color4_field("Base Color", asset.material.base_color))
            mark_dirty(state);
        if (ui::float_field("Metallic", &asset.material.metallic, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Roughness", &asset.material.roughness, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Alpha Cutoff", &asset.material.alpha_cutoff, 0.01f))
            mark_dirty(state);
        int blend = static_cast<int>(asset.material.alpha_mode);
        const char* blend_items[]{ "Opaque", "Masked", "Blend" };
        ImGui::TextUnformatted("Blend");
        ImGui::SameLine(ui::scaled(118.0f));
        ImGui::SetNextItemWidth(-1.0f);
        if (ImGui::Combo("##blend-mode", &blend, blend_items, 3))
        {
            asset.material.alpha_mode = static_cast<render::material_alpha_mode>(blend);
            mark_dirty(state);
        }
        if (ui::checkbox_field("Double Sided", &asset.material.double_sided))
            mark_dirty(state);
        ui::component_card_end();
    }

    if (ui::component_card_begin("Textures"))
    {
        if (texture_path_field("Base Color", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::base_color, asset.textures.base_color))
            mark_dirty(state);
        if (texture_path_field("Metal/Rough", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::metallic_roughness, asset.textures.metallic_roughness))
            mark_dirty(state);
        if (texture_path_field("Normal", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::normal, asset.textures.normal))
            mark_dirty(state);
        if (texture_path_field("AO", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::ao, asset.textures.ao))
            mark_dirty(state);
        if (texture_path_field("Emissive", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::emissive, asset.textures.emissive))
            mark_dirty(state);
        if (texture_path_field("Height", state, editor_scene.material_library, renderer, editor_assets.root, material_texture_slot::height, asset.textures.height))
            mark_dirty(state);
        ui::component_card_end();
    }

    if (ui::component_card_begin("Advanced", false))
    {
        if (ui::float_field("Normal Scale", &asset.material.normal_scale, 0.01f))
            mark_dirty(state);
        if (ui::float_field("AO Strength", &asset.material.occlusion_strength, 0.01f))
            mark_dirty(state);
        float emissive[3]{ asset.material.emissive_factor[0], asset.material.emissive_factor[1], asset.material.emissive_factor[2] };
        if (ui::color_field3("Emissive", emissive))
        {
            asset.material.emissive_factor = { emissive[0], emissive[1], emissive[2] };
            mark_dirty(state);
        }
        if (ui::float_field("Emissive Power", &asset.material.emissive_strength, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Clear Coat", &asset.material.clear_coat_factor, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Sheen", &asset.material.sheen_factor, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Transmission", &asset.material.transmission_factor, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Subsurface", &asset.material.subsurface_factor, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Anisotropy", &asset.material.anisotropy_factor, 0.01f))
            mark_dirty(state);
        if (ui::float_field("Parallax", &asset.material.parallax_height_scale, 0.001f))
            mark_dirty(state);
        ui::component_card_end();
    }

    if (ui::component_card_begin("Preview"))
    {
        const char* shapes[]{ "Sphere", "Plane", "Cube" };
        ImGui::SetNextItemWidth(ui::scaled(160.0f));
        ImGui::Combo("Shape", &state.preview_shape, shapes, 3);
        draw_preview(asset);
        ui::component_card_end();
    }

    ui::end_panel();
}

} // namespace arc::editor

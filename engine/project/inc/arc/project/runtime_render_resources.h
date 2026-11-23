#pragma once

/** @file
 * @brief CPU-side decoding of renderer resources referenced by packaged scenes.
 */

#include <arc/project/runtime_catalog.h>
#include <arc/render/cooked_material.h>
#include <arc/render/cooked_material_instance.h>
#include <arc/render/cooked_material_texture_bindings.h>
#include <arc/render/mesh.h>
#include <arc/render/texture.h>
#include <arc/scene/runtime_scene.h>

#include <optional>
#include <utility>
#include <vector>

namespace arc::project
{
struct decoded_scene_render_resource
{
    ecs::entity_guid entity{};
    render::mesh_data mesh;
    bool procedural_primitive{};
    assets::asset_guid material_guid{};
    assets::asset_guid shader_material_guid{};
    std::optional<render::cooked_material_instance_values> instance;
    std::optional<render::cooked_material_metadata> material;
    std::optional<render::cooked_material_texture_manifest> material_textures;
    std::vector<render::cooked_material_shader> material_shaders;
    std::vector<std::pair<assets::asset_guid, render::texture_data>> textures;
};

using decoded_scene_render_resources = std::vector<decoded_scene_render_resource>;
using decoded_scene_render_result = core::result<decoded_scene_render_resources, std::string>;

/** Decode active-provider artifacts without allocating renderer-local handles. */
[[nodiscard]] decoded_scene_render_result
decode_scene_render_resources(runtime_catalog_session& catalog, const scene::cooked_scene_mesh_bindings& bindings);
} // namespace arc::project

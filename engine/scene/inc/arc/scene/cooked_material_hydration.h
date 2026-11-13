#pragma once

/** @file
 * @brief Attach executable cooked material programs to persistent ECS entities.
 * Material handles belong to one renderer generation, not to cooked archives.
 */

#include <arc/assets/assets.h>
#include <arc/ecs/identity.h>
#include <arc/ecs/world.h>
#include <arc/render/renderer.h>

#include <span>
#include <string>
#include <vector>

namespace arc::scene
{
struct cooked_material_upload
{
    ecs::entity_guid entity{};
    assets::asset_guid material{};
    /** GPU shader package owner; differs from material GUID for parent-reuse instances. */
    assets::asset_guid program_asset{};
    render::material_descriptor descriptor;
};

struct cooked_material_residency
{
    ecs::entity entity{};
    render::material_handle handle{};
};

using cooked_material_residencies = std::vector<cooked_material_residency>;

/** Validate all material programs and owning entities before committing ECS.
 * Duplicate material asset GUIDs share a renderer material handle.
 */
[[nodiscard]] std::string hydrate_cooked_scene_materials(ecs::world& scene, render::renderer& renderer,
                                                         std::span<const cooked_material_upload> uploads,
                                                         cooked_material_residencies& residency);

/** Clear only ECS material handles still owned by this renderer generation.
 * The renderer itself owns material allocation and release on destruction.
 */
void unbind_cooked_scene_materials(ecs::world& scene, cooked_material_residencies& residency) noexcept;

} // namespace arc::scene

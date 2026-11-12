#pragma once

/** @file
 * @brief Bind decoded cooked conventional meshes to a renderer generation.
 * Authored asset GUIDs remain in the runtime catalog; ECS receives only
 * renderer-local handles and those handles are cleared before renderer teardown.
 */

#include <arc/ecs/identity.h>
#include <arc/ecs/world.h>
#include <arc/render/renderer.h>

#include <span>
#include <string>
#include <vector>

namespace arc::scene
{
struct cooked_mesh_upload
{
    ecs::entity_guid entity{};
    render::mesh_data mesh;
};

struct cooked_mesh_residency
{
    ecs::entity entity{};
    render::mesh_handle mesh{};
};

using cooked_mesh_residencies = std::vector<cooked_mesh_residency>;

/** Validate every owner first, then upload all meshes; a failed upload rolls
 * back all newly allocated handles without altering ECS. Requires that the
 * caller has already cleared any older renderer generation's bindings.
 * Empty uploads succeed and leave existing ECS state untouched.
 */
[[nodiscard]] std::string hydrate_cooked_scene_meshes(ecs::world& scene, render::renderer& renderer,
                                                      std::span<const cooked_mesh_upload> uploads,
                                                      cooked_mesh_residencies& residency);

/** Unbind handles before destroying the corresponding renderer generation.
 * Never clears handles which no longer match the recorded residency.
 */
void release_cooked_scene_meshes(ecs::world& scene, render::renderer& renderer,
                                 cooked_mesh_residencies& residency) noexcept;

} // namespace arc::scene

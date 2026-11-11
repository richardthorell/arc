#pragma once

/** @file
 * @brief Materialize validated cooked scene archives into a runtime ECS world.
 */

#include <arc/assets/assets.h>
#include <arc/ecs/world.h>

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace arc::scene
{

/** Stable authored references from the cooked scene dependency manifest.
 * Renderer resource handles are allocated separately for each renderer generation.
 */
struct cooked_scene_mesh_binding
{
    ecs::entity_guid entity{};
    assets::asset_reference source;
    assets::asset_reference material;
    std::string subresource;
};

using cooked_scene_mesh_bindings = std::vector<cooked_scene_mesh_binding>;

/**
 * Decode the same tagged-binary scene archive emitted by the ARC cooker.
 * Returns an empty string on success, or an actionable failure.
 *
 * The destination must initially be empty. Component and hierarchy failures
 * roll back created entities. Unsupported component types fail closed rather
 * than silently producing an incomplete packaged scene.
 */
[[nodiscard]] std::string materialize_cooked_scene(ecs::world& destination, std::span<const std::byte> bytes,
                                                  cooked_scene_mesh_bindings* bindings = nullptr);

} // namespace arc::scene

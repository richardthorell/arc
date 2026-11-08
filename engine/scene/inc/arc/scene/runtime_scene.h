#pragma once

/** @file
 * @brief Materialize validated cooked scene archives into a runtime ECS world.
 */

#include <arc/ecs/world.h>

#include <cstddef>
#include <span>
#include <string>

namespace arc::scene
{

/**
 * Decode the same tagged-binary scene archive emitted by the ARC cooker.
 * Returns an empty string on success, or an actionable failure.
 *
 * The destination must initially be empty. Component and hierarchy failures
 * roll back created entities. Unsupported component types fail closed rather
 * than silently producing an incomplete packaged scene.
 */
[[nodiscard]] std::string materialize_cooked_scene(ecs::world& destination, std::span<const std::byte> bytes);

} // namespace arc::scene

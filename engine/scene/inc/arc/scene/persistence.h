#pragma once

#include <arc/persistence/persistence.h>

namespace arc::scene
{

/**
 * Registers every scene-owned reflected component with the generic persistence
 * layer. Editor documents and the cooker call this same function so readable
 * component names, stable IDs, and schema versions cannot drift.
 */
bool register_persistence_components(persistence::component_persistence_registry& registry);

/**
 * Registers the consecutive migrations currently required by scene and prefab
 * authoring documents.
 */
[[nodiscard]] persistence::persistence_status
register_persistence_migrations(persistence::schema_migration_registry& registry);

} // namespace arc::scene

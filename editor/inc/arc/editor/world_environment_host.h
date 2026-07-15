#pragma once

#include <arc/editor/host_protocol.h>
#include <arc/scene/environment.h>

#include <filesystem>

namespace arc::editor
{

host_world_environment_snapshot to_host_world_environment_snapshot(
    host_entity_id entity,
    const scene::world_environment_settings& settings,
    const std::filesystem::path& hdri_path);

scene::world_environment_settings apply_host_world_environment_snapshot(
    const host_world_environment_snapshot& snapshot,
    const scene::world_environment_settings& current);

} // namespace arc::editor

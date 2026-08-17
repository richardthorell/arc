#include <arc/editor/arc_host.h>

#include <arc/diagnostics/diagnostics.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_history.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/prefab_document.h>
#include <arc/editor/scene_document.h>
#include <arc/editor/terrain_heightmap_io.h>
#include <arc/editor/viewport_render_stats.h>
#include "project_module_loader.h"
#include <arc/editor/world_environment_host.h>
#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/geometric/box.h>
#include <arc/framework/framework.h>
#include <arc/project/project.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

// Keep the existing host implementation intact while adding narrow editor-only
// mesh assignment and viewport telemetry paths. Base entry points are renamed
// so public wrappers can extend the protocol without duplicating host logic.
#define execute execute_base
#define query(...) query_base(__VA_ARGS__)
#define has_material \
    has_material = mesh_renderer.material.valid(); \
    snapshot.has_mesh = mesh_renderer.mesh.valid(); \
    if (const auto* arc_mesh_binding = find_asset_binding(state, entity_guid_of(state, entity)); arc_mesh_binding) \
    { \
        snapshot.asset_backed_mesh = \
            arc_mesh_binding->source.guid.valid() || !arc_mesh_binding->source.path_hint.empty(); \
        if (!arc_mesh_binding->source.path_hint.empty()) \
        { \
            auto arc_mesh_path = std::filesystem::path{arc_mesh_binding->source.path_hint}; \
            if (arc_mesh_path.is_absolute()) arc_mesh_path = arc_mesh_path.lexically_relative(project_root); \
            snapshot.mesh_path = arc::assets::normalize_asset_path(arc_mesh_path); \
        } \
        snapshot.mesh_name = !arc_mesh_binding->subresource.empty() \
                                 ? arc_mesh_binding->subresource \
                                 : std::filesystem::path{snapshot.mesh_path}.stem().string(); \
    } \
    snapshot.has_material
#include "arc_host_base.inc"
#undef has_material
#undef query
#undef execute

namespace arc::editor
{
namespace
{
constexpr std::string_view mesh_assignment_prefix = "__arc_mesh__/";

std::optional<std::filesystem::path> mesh_assignment_path(const host_set_entity_material_command& command)
{
    const std::string encoded = command.path.generic_string();
    if (!encoded.starts_with(mesh_assignment_prefix)) return std::nullopt;
    const std::string reference = encoded.substr(mesh_assignment_prefix.size());
    return reference.empty() ? std::nullopt : std::optional<std::filesystem::path>{reference};
}

geometric::box3f assigned_mesh_bounds(const render::mesh_data& mesh)
{
    if (mesh.vertices.empty())
        return geometric::box3f{geometric::point3f{-0.5f, -0.5f, -0.5f}, geometric::point3f{0.5f, 0.5f, 0.5f}};

    math::vector3f minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    math::vector3f maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};
    for (const auto& vertex : mesh.vertices)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], vertex.position[axis]);
            maximum[axis] = std::max(maximum[axis], vertex.position[axis]);
        }
    }
    return geometric::box3f{geometric::point3f{minimum}, geometric::point3f{maximum}};
}
} // namespace

host_response arc_host::execute(host_command_payload command)
{
    return execute(host_command_envelope{.command_type = command_type(command), .payload = std::move(command)});
}

host_response arc_host::execute(const host_command_envelope& command)
{
    const auto* material_command = std::get_if<host_set_entity_material_command>(&command.payload);
    const auto mesh_reference = material_command ? mesh_assignment_path(*material_command) : std::nullopt;
    if (!material_command || !mesh_reference) return execute_base(command);

    const auto response_with_revisions = [this](host_response response)
    {
        response.scene_revision = state_->scene_revision;
        response.world_epoch = state_->world_epoch;
        response.frame_revision = state_->viewport_frame_index;
        return response;
    };
    const auto fail = [this, &command, &response_with_revisions](std::string message, ecs::entity entity = {})
    {
        arc::diagnostics::warn("editor.host", message);
        push_event(state_->events, state_->event_sequence, host_event_type::command_failed, message, entity);
        return response_with_revisions(
            {.request_id = command.request_id, .succeeded = false, .error = std::move(message)});
    };

    if (state_->project_read_only) return fail("The project is open read-only");
    if (!state_->project_open) return fail("Cannot assign a mesh before a project is open");
    if (command.expected_scene_revision && *command.expected_scene_revision != state_->scene_revision)
        return fail("Scene revision is stale");
    if (command.edit) return fail("Mesh asset assignment does not use a continuous edit transaction");

    const auto entity = to_scene_entity(material_command->entity);
    const auto targets = edit_targets(state_->scene.scene, entity, material_command->apply_to_selection);
    if (targets.empty()) return fail("Cannot edit a missing or unselected mesh renderer", entity);
    if (std::any_of(targets.begin(), targets.end(), [&](ecs::entity target)
                    { return !state_->scene.scene.has<scene::mesh_renderer_component>(target); }))
        return fail("Every selected entity must have an editable mesh renderer component", entity);

    const auto resolved =
        resolve_editor_asset(state_->assets, state_->asset_registry.get(), state_->project.root, *mesh_reference);
    if (!resolved) return fail("Mesh must be a project or built-in scene asset", entity);

    render::scene_import_options options;
    options.asset_root = resolved->asset_root;
    options.copy_assets = false;
    auto imported = render::load_scene_asset(resolved->path, options);
    if (!imported.succeeded())
        return fail(imported.message.empty() ? "Failed to load mesh asset" : imported.message, entity);

    std::size_t mesh_index = std::numeric_limits<std::size_t>::max();
    for (const auto& node : imported.nodes)
    {
        if (node.mesh_index < imported.meshes.size())
        {
            mesh_index = node.mesh_index;
            break;
        }
    }
    if (mesh_index == std::numeric_limits<std::size_t>::max() && !imported.meshes.empty()) mesh_index = 0;
    if (mesh_index >= imported.meshes.size()) return fail("Mesh asset contains no renderable geometry", entity);

    const auto& selected_mesh = imported.meshes[mesh_index];
    const auto mesh_handle = state_->renderer->create_mesh(selected_mesh);
    if (!mesh_handle.valid()) return fail("Renderer could not create the selected mesh", entity);
    const auto local_bounds = assigned_mesh_bounds(selected_mesh);

    auto before = state_->scene;
    ensure_scene_authoring_metadata(state_->scene);

    std::filesystem::path binding_path = *mesh_reference;
    if (binding_path.is_absolute()) binding_path = binding_path.lexically_relative(state_->project.root);
    const auto normalized_path = arc::assets::normalize_asset_path(binding_path);
    arc::assets::asset_reference source_reference{
        .expected_type = arc::assets::asset_types::imported_scene,
        .path_hint = normalized_path,
    };
    if (state_->asset_registry)
    {
        auto resolved_reference = state_->asset_registry->resolve(normalized_path, arc::assets::asset_types::imported_scene);
        if (resolved_reference.guid.valid() || !resolved_reference.path_hint.empty())
            source_reference = std::move(resolved_reference);
    }

    for (const auto target : targets)
    {
        auto& renderer = state_->scene.scene.get<scene::mesh_renderer_component>(target);
        renderer.mesh = mesh_handle;

        if (auto* bounds = state_->scene.scene.try_get<scene::bounds_component>(target))
        {
            bounds->local_bounds = local_bounds;
            bounds->dirty = true;
        }
        else
        {
            state_->scene.scene.emplace<scene::bounds_component>(target, local_bounds, local_bounds, true);
        }

        const auto guid = entity_guid_of(state_->scene, target);
        auto* binding = find_asset_binding(state_->scene, guid);
        if (!binding)
        {
            state_->scene.asset_bindings.push_back({.entity = guid});
            binding = &state_->scene.asset_bindings.back();
        }
        binding->source_kind = "imported";
        binding->source = source_reference;
        binding->subresource = selected_mesh.name.empty() ? resolved->path.stem().string() : selected_mesh.name;
    }

    ++state_->scene_revision;
    state_->history.record("Assign Mesh", std::move(before), state_->scene);
    push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Mesh asset assigned", entity);
    return response_with_revisions({.request_id = command.request_id,
                                    .succeeded = true,
                                    .payload_json = "{\"entity\":" + to_json(material_command->entity) + '}'});
}

host_response arc_host::query(const host_query_envelope& query) const
{
    auto response = query_base(query);
    if (!response.succeeded || !std::holds_alternative<host_viewport_state_query>(query.payload)) return response;

    auto payload = nlohmann::json::parse(response.payload_json, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) return response;

    const auto stats = collect_viewport_render_stats(state_->scene, *state_->renderer);
    payload["triangles"] = stats.triangles;
    payload["vertices"] = stats.vertices;
    if (stats.gpu_memory_available) payload["gpuMemoryBytes"] = stats.gpu_memory_used_bytes;
    if (stats.gpu_memory_budget_bytes != 0) payload["gpuMemoryBudgetBytes"] = stats.gpu_memory_budget_bytes;

    const double fps = payload.value("fps", 0.0);
    payload["frameIntervalMs"] = fps > 0.0 ? 1000.0 / fps : 0.0;
    payload["cpuRenderTimeMs"] = payload.value("frameTimeMs", 0.0);
    response.payload_json = payload.dump();
    return response;
}

host_response in_process_host_session::execute(const host_command_envelope& command)
{
    return host_->execute(command);
}

host_response in_process_host_session::query(const host_query_envelope& query)
{
    return host_->query(query);
}

} // namespace arc::editor

#include <arc/editor/arc_host.h>

#include <arc/diagnostics/diagnostics.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_gizmo.h>
#include <arc/editor/editor_history.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/material_preview.h>
#include <arc/editor/prefab_document.h>
#include <arc/editor/procedural_mesh.h>
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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>

namespace arc::editor
{
namespace
{
constexpr std::string_view mesh_assignment_prefix = "__arc_mesh__/";
constexpr std::string_view primitive_assignment_prefix = "__arc_primitive__/";
constexpr std::string_view primitive_parameter_prefix = "__arc_primitive_parameter__/";
constexpr std::string_view primitive_mesh_uri_prefix = "arc://primitive/";
constexpr std::string_view material_preview_viewport_prefix = "asset-preview-material-";
constexpr std::string_view shader_preview_viewport_prefix = "asset-preview-shader-";

enum class asset_preview_kind : std::uint8_t
{
    none,
    material,
    shader
};

struct asset_preview_identity
{
    asset_preview_kind kind{asset_preview_kind::none};
    assets::asset_guid guid{};
};

std::string primitive_mesh_uri(std::string_view name)
{
    std::string token{name};
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return std::string{primitive_mesh_uri_prefix} + token;
}

std::string normalized_viewport_id(std::string_view viewport_id)
{
    return viewport_id.empty() ? "viewport-1" : std::string{viewport_id};
}

asset_preview_identity asset_preview_from_viewport_id(std::string_view viewport_id)
{
    asset_preview_identity result;
    std::string_view guid_text;
    if (viewport_id.starts_with(material_preview_viewport_prefix))
    {
        result.kind = asset_preview_kind::material;
        guid_text = viewport_id.substr(material_preview_viewport_prefix.size());
    }
    else if (viewport_id.starts_with(shader_preview_viewport_prefix))
    {
        result.kind = asset_preview_kind::shader;
        guid_text = viewport_id.substr(shader_preview_viewport_prefix.size());
    }
    else
    {
        return result;
    }

    if (const auto guid = assets::parse_asset_guid(guid_text)) result.guid = *guid;
    return result;
}

const char* asset_preview_kind_name(asset_preview_kind kind) noexcept
{
    switch (kind)
    {
        case asset_preview_kind::material:
            return "material";
        case asset_preview_kind::shader:
            return "shader";
        case asset_preview_kind::none:
            break;
    }
    return "none";
}
} // namespace

struct viewport_surface_registry
{
    struct surface_state
    {
        host_viewport_request options;
        std::chrono::steady_clock::time_point last_request_time{};
        double fps{};
        double frame_time_ms{};
        std::uint32_t draw_calls{};
        std::uint64_t local_frame_index{};
        bool submitted{};

        asset_preview_kind preview_kind{asset_preview_kind::none};
        assets::asset_guid preview_guid{};
        std::unique_ptr<editor_scene_state> preview_scene;
        editor_camera_controller preview_camera;
        ecs::entity preview_entity{};
        render::mesh_handle preview_mesh{};
        assets::asset_handle<render::material_handle> preview_material;
        std::uint64_t preview_material_generation{};
        std::string preview_error;
    };

    viewport_surface_registry()
    {
        (void)ensure("viewport-1");
    }

    surface_state* find(std::string_view viewport_id)
    {
        const auto found = surfaces.find(normalized_viewport_id(viewport_id));
        return found == surfaces.end() ? nullptr : &found->second;
    }

    const surface_state* find(std::string_view viewport_id) const
    {
        const auto found = surfaces.find(normalized_viewport_id(viewport_id));
        return found == surfaces.end() ? nullptr : &found->second;
    }

    surface_state& ensure(std::string_view viewport_id)
    {
        const auto id = normalized_viewport_id(viewport_id);
        auto [found, inserted] = surfaces.try_emplace(id);
        if (inserted)
        {
            auto& surface = found->second;
            surface.options.viewport_id = id;
            const auto preview = asset_preview_from_viewport_id(id);
            surface.preview_kind = preview.kind;
            surface.preview_guid = preview.guid;
            if (surface.preview_kind != asset_preview_kind::none)
            {
                surface.options.overlay = host_overlay_mode::none;
                surface.options.grid = false;
                surface.options.shadows = true;
                surface.options.realtime = true;
                surface.options.environment.sky = true;
                surface.options.environment.fog = false;
                surface.options.environment.terrain = false;
                surface.options.environment.water = false;
                surface.options.environment.vegetation = false;
                surface.options.environment.decals = false;
            }
        }
        return found->second;
    }

    surface_state& primary()
    {
        return ensure("viewport-1");
    }

    std::unordered_map<std::string, surface_state> surfaces;
    std::optional<std::string> pending_pick_surface;
    std::uint64_t next_renderer_frame_index{};
};

namespace
{
template <class Variant> std::optional<std::string> viewport_id_from(const Variant& payload)
{
    return std::visit(
        [](const auto& value) -> std::optional<std::string>
        {
            if constexpr (requires { value.viewport_id; })
                return normalized_viewport_id(value.viewport_id);
            else
                return std::nullopt;
        },
        payload);
}

bool creates_viewport_surface(const host_command_payload& payload)
{
    return std::holds_alternative<host_viewport_attach_command>(payload) ||
           std::holds_alternative<host_viewport_create_command>(payload);
}

template <class HostState>
void activate_viewport_surface(HostState& host, const viewport_surface_registry::surface_state& surface)
{
    host.active_viewport_id = surface.options.viewport_id;
    host.viewport_options = surface.options;
    host.viewport_fps = surface.fps;
    host.viewport_frame_ms = surface.frame_time_ms;
    host.viewport_draw_calls = surface.draw_calls;
    host.viewport_submitted = surface.submitted;
}

template <class HostState>
void capture_viewport_surface(const HostState& host, viewport_surface_registry::surface_state& surface)
{
    const auto id = surface.options.viewport_id;
    surface.options = host.viewport_options;
    surface.options.viewport_id = id;
    surface.frame_time_ms = host.viewport_frame_ms;
    surface.draw_calls = host.viewport_draw_calls;
    surface.submitted = host.viewport_submitted;
}
} // namespace
} // namespace arc::editor

// Keep the existing host implementation intact while adding narrow editor-only
// mesh assignment and viewport telemetry paths. Base entry points are renamed
// so public wrappers can extend the protocol without duplicating host logic.
#define execute execute_base
#define query(...) query_base(__VA_ARGS__)
#define request_viewport request_viewport_base
#define has_material                                                                                                   \
    has_material = mesh_renderer.material.valid();                                                                     \
    snapshot.has_mesh = mesh_renderer.mesh.valid();                                                                    \
    if (const auto* arc_mesh_binding = find_asset_binding(state, entity_guid_of(state, entity)); arc_mesh_binding)     \
    {                                                                                                                  \
        if (arc_mesh_binding->source_kind == "primitive" && !arc_mesh_binding->subresource.empty())                    \
        {                                                                                                              \
            snapshot.asset_backed_mesh = false;                                                                        \
            snapshot.mesh_name = arc_mesh_binding->subresource;                                                        \
            snapshot.mesh_path = arc::editor::primitive_mesh_uri(arc_mesh_binding->subresource);                       \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            snapshot.asset_backed_mesh =                                                                               \
                arc_mesh_binding->source.guid.valid() || !arc_mesh_binding->source.path_hint.empty();                  \
            if (!arc_mesh_binding->source.path_hint.empty())                                                           \
            {                                                                                                          \
                auto arc_mesh_path = std::filesystem::path{arc_mesh_binding->source.path_hint};                        \
                if (arc_mesh_path.is_absolute()) arc_mesh_path = arc_mesh_path.lexically_relative(project_root);       \
                snapshot.mesh_path = arc::assets::normalize_asset_path(arc_mesh_path);                                 \
            }                                                                                                          \
            snapshot.mesh_name = !arc_mesh_binding->subresource.empty()                                                \
                                     ? arc_mesh_binding->subresource                                                   \
                                     : std::filesystem::path{snapshot.mesh_path}.stem().string();                      \
        }                                                                                                              \
    }                                                                                                                  \
    snapshot.has_material
#include "arc_host_base.inc"
#undef has_material
#undef request_viewport
#undef query
#undef execute

namespace arc::editor
{
namespace
{
struct primitive_parameter_assignment
{
    std::string parameter;
    double value{};
};

std::optional<std::filesystem::path> mesh_assignment_path(const host_set_entity_material_command& command)
{
    const std::string encoded = command.path.generic_string();
    if (!encoded.starts_with(mesh_assignment_prefix)) return std::nullopt;
    const std::string reference = encoded.substr(mesh_assignment_prefix.size());
    return reference.empty() ? std::nullopt : std::optional<std::filesystem::path>{reference};
}

std::optional<editor_primitive_type> primitive_assignment_type(const host_set_entity_material_command& command)
{
    const std::string encoded = command.path.generic_string();
    if (!encoded.starts_with(primitive_assignment_prefix)) return std::nullopt;
    return procedural_mesh_type_from_token(encoded.substr(primitive_assignment_prefix.size()));
}

std::optional<primitive_parameter_assignment>
primitive_parameter_assignment_from(const host_set_entity_material_command& command)
{
    const std::string encoded = command.path.generic_string();
    if (!encoded.starts_with(primitive_parameter_prefix)) return std::nullopt;
    const std::string_view payload{encoded.data() + primitive_parameter_prefix.size(),
                                   encoded.size() - primitive_parameter_prefix.size()};
    const auto separator = payload.find('/');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 >= payload.size()) return std::nullopt;

    try
    {
        std::size_t consumed{};
        const std::string value_text{payload.substr(separator + 1)};
        const double value = std::stod(value_text, &consumed);
        if (consumed != value_text.size() || !std::isfinite(value)) return std::nullopt;
        return primitive_parameter_assignment{std::string{payload.substr(0, separator)}, value};
    }
    catch (...)
    {
        return std::nullopt;
    }
}

bool primitive_create_command(const host_command_payload& payload)
{
    const auto* create = std::get_if<host_create_entity_command>(&payload);
    if (!create) return false;
    switch (create->kind)
    {
        case host_create_entity_kind::plane:
        case host_create_entity_kind::cube:
        case host_create_entity_kind::sphere:
        case host_create_entity_kind::cylinder:
        case host_create_entity_kind::cone:
        case host_create_entity_kind::capsule:
            return true;
        default:
            return false;
    }
}

bool should_synchronize_procedural_meshes(const host_command_payload& payload)
{
    return primitive_create_command(payload) || std::holds_alternative<host_open_project_command>(payload) ||
           std::holds_alternative<host_open_scene_command>(payload) ||
           std::holds_alternative<host_open_recovery_scene_command>(payload) ||
           std::holds_alternative<host_duplicate_entity_command>(payload) ||
           std::holds_alternative<host_instantiate_prefab_command>(payload) ||
           std::holds_alternative<host_revert_prefab_command>(payload) ||
           std::holds_alternative<host_history_undo_command>(payload) ||
           std::holds_alternative<host_history_redo_command>(payload) ||
           std::holds_alternative<host_history_cancel_transaction_command>(payload) ||
           std::holds_alternative<host_runtime_restore_snapshot_command>(payload);
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

template <class HostState>
bool refresh_asset_preview_material(HostState& host, viewport_surface_registry::surface_state& surface)
{
    if (surface.preview_kind != asset_preview_kind::material || !surface.preview_scene) return true;
    if (!surface.preview_guid.valid())
    {
        surface.preview_error = "Material preview viewport has an invalid asset GUID";
        return false;
    }
    if (!host.asset_registry)
    {
        surface.preview_error = "Material preview is waiting for the project asset registry";
        return false;
    }

    if (!surface.preview_material.valid())
    {
        assets::asset_load_request request;
        request.reference.guid = surface.preview_guid;
        request.reference.expected_type = assets::asset_types::material;
        request.priority = assets::asset_streaming_priority::high;
        request.residency = assets::asset_residency::device;
        request.allow_fallback = true;
        auto loaded = host.asset_registry->template load<render::material_handle>(std::move(request)).get();
        if (!loaded.asset.valid())
        {
            surface.preview_error =
                loaded.error.message.empty() ? "Material preview asset could not be loaded" : loaded.error.message;
            return false;
        }
        surface.preview_material = std::move(loaded.asset);
        surface.preview_material_generation = 0;
    }

    const auto generation = surface.preview_material.generation();
    if (generation == surface.preview_material_generation) return true;
    const auto* material = surface.preview_material.get();
    if (!material || !material->valid())
    {
        surface.preview_error = "Material preview resolved to an invalid renderer material";
        return false;
    }
    auto* mesh_renderer = surface.preview_scene->scene.try_get<scene::mesh_renderer_component>(surface.preview_entity);
    if (!mesh_renderer)
    {
        surface.preview_error = "Material preview sphere has no mesh renderer";
        return false;
    }
    mesh_renderer->material = *material;
    surface.preview_material_generation = generation;
    if (surface.preview_material.using_fallback())
    {
        surface.preview_error = "Material preview is using the renderer error material";
    }
    else
    {
        surface.preview_error.clear();
    }
    return true;
}

template <class HostState>
bool ensure_asset_preview_scene(HostState& host, viewport_surface_registry::surface_state& surface)
{
    if (surface.preview_kind == asset_preview_kind::none) return true;
    if (surface.preview_scene) return refresh_asset_preview_material(host, surface);
    if (!host.renderer)
    {
        surface.preview_error = "Asset preview renderer is unavailable";
        return false;
    }

    auto preview = std::make_unique<editor_scene_state>(create_blank_scene(*host.renderer, false, nullptr));
    preview->scene_name = std::string{"Asset Preview: "} + asset_preview_kind_name(surface.preview_kind);
    preview->primitive_material =
        host.scene.primitive_material.valid() ? host.scene.primitive_material : host.scene.default_material;
    const auto sphere = add_primitive_to_scene(*preview, *host.renderer, editor_primitive_type::sphere);
    if (!preview->scene.alive(sphere))
    {
        surface.preview_error = "Renderer could not create the asset preview sphere";
        return false;
    }
    if (auto* name = preview->scene.template try_get<scene::name_component>(sphere))
    {
        name->value = "Asset Preview Sphere";
    }
    clear_selection(preview->scene, preview->selected_entity);

    surface.preview_camera = {};
    (void)surface.preview_camera.place({1.65f, 0.55f, 2.25f}, {0.0f, 0.0f, 0.0f});
    if (auto* camera_transform = preview->scene.template try_get<scene::transform_component>(preview->camera_entity))
    {
        surface.preview_camera.apply_to(*camera_transform);
    }

    surface.preview_entity = sphere;
    if (const auto* mesh_renderer =
            std::as_const(preview->scene).template try_get<scene::mesh_renderer_component>(sphere))
        surface.preview_mesh = mesh_renderer->mesh;
    surface.preview_scene = std::move(preview);

    if (surface.preview_kind == asset_preview_kind::shader)
    {
        if (!surface.preview_guid.valid())
            surface.preview_error = "Shader preview viewport has an invalid asset GUID";
        else if (host.asset_registry)
        {
            const auto shader = host.asset_registry->find(surface.preview_guid);
            if (!shader || shader->type != assets::asset_types::shader)
                surface.preview_error = "Shader preview asset is not registered as a shader";
            else
                surface.preview_error.clear();
        }
        return true;
    }
    return refresh_asset_preview_material(host, surface);
}

template <class HostState> class asset_preview_scene_scope
{
public:
    asset_preview_scene_scope(HostState& host, viewport_surface_registry::surface_state& surface)
        : host_(host), surface_(surface)
    {
        if (!surface_.preview_scene) return;
        std::swap(host_.scene, *surface_.preview_scene);
        std::swap(host_.camera_controller, surface_.preview_camera);
        active_ = true;
    }

    ~asset_preview_scene_scope()
    {
        if (!active_) return;
        std::swap(host_.camera_controller, surface_.preview_camera);
        std::swap(host_.scene, *surface_.preview_scene);
    }

    asset_preview_scene_scope(const asset_preview_scene_scope&) = delete;
    asset_preview_scene_scope& operator=(const asset_preview_scene_scope&) = delete;

private:
    HostState& host_;
    viewport_surface_registry::surface_state& surface_;
    bool active_{};
};

void cleanup_asset_preview_surface(viewport_surface_registry::surface_state& surface, render::renderer& renderer)
{
    if (!surface.preview_scene) return;
    surface.preview_scene->terrain_render_proxies.clear(renderer);
    if (surface.preview_mesh.valid()) (void)renderer.destroy_mesh(surface.preview_mesh);
    if (surface.preview_scene->environment_lighting_resource.valid())
        (void)renderer.destroy_environment(surface.preview_scene->environment_lighting_resource);
    surface.preview_scene.reset();
    surface.preview_entity = {};
    surface.preview_mesh = {};
    surface.preview_material = {};
    surface.preview_material_generation = 0;
}

void reset_asset_preview_scenes(viewport_surface_registry& surfaces, render::renderer& renderer)
{
    for (auto& [_, surface] : surfaces.surfaces)
    {
        if (surface.preview_kind != asset_preview_kind::none) cleanup_asset_preview_surface(surface, renderer);
    }
}
} // namespace

host_response arc_host::execute(host_command_payload command)
{
    return execute(host_command_envelope{.command_type = command_type(command), .payload = std::move(command)});
}

host_response arc_host::execute(const host_command_envelope& command)
{
    if (!viewport_surfaces_) viewport_surfaces_ = std::make_unique<viewport_surface_registry>();
    auto& surfaces = *viewport_surfaces_;
    const auto viewport_id = viewport_id_from(command.payload);
    viewport_surface_registry::surface_state* viewport_surface{};
    if (viewport_id)
    {
        viewport_surface =
            creates_viewport_surface(command.payload) ? &surfaces.ensure(*viewport_id) : surfaces.find(*viewport_id);
        if (!viewport_surface)
        {
            host_response response{
                .request_id = command.request_id, .succeeded = false, .error = "Viewport is not attached"};
            response.scene_revision = state_->scene_revision;
            response.world_epoch = state_->world_epoch;
            response.frame_revision = state_->viewport_frame_index;
            return response;
        }
    }
    else
    {
        viewport_surface = &surfaces.primary();
    }
    if (viewport_surface->preview_kind != asset_preview_kind::none)
        (void)ensure_asset_preview_scene(*state_, *viewport_surface);
    activate_viewport_surface(*state_, *viewport_surface);

    const auto* material_command = std::get_if<host_set_entity_material_command>(&command.payload);
    const auto mesh_reference = material_command ? mesh_assignment_path(*material_command) : std::nullopt;
    const auto primitive_type = material_command ? primitive_assignment_type(*material_command) : std::nullopt;
    const auto parameter_assignment =
        material_command ? primitive_parameter_assignment_from(*material_command) : std::nullopt;
    if (!material_command || (!mesh_reference && !primitive_type && !parameter_assignment))
    {
        host_response response;
        {
            asset_preview_scene_scope preview_scope(*state_, *viewport_surface);
            response = execute_base(command);
        }
        capture_viewport_surface(*state_, *viewport_surface);
        if (response.succeeded && std::holds_alternative<host_viewport_pick_command>(command.payload))
            surfaces.pending_pick_surface = viewport_surface->options.viewport_id;
        if (std::holds_alternative<host_viewport_detach_command>(command.payload) &&
            surfaces.pending_pick_surface == viewport_surface->options.viewport_id)
            surfaces.pending_pick_surface.reset();
        if (response.succeeded && state_->project_open && should_synchronize_procedural_meshes(command.payload))
        {
            synchronize_procedural_mesh_components(state_->scene, *state_->renderer);
            if (primitive_create_command(command.payload) && state_->scene.scene.alive(state_->scene.selected_entity))
                persist_procedural_mesh_component(state_->scene, state_->scene.selected_entity);
        }
        if (response.succeeded && (std::holds_alternative<host_open_project_command>(command.payload) ||
                                   std::holds_alternative<host_close_project_command>(command.payload)))
            reset_asset_preview_scenes(surfaces, *state_->renderer);
        if (response.succeeded && std::holds_alternative<host_viewport_detach_command>(command.payload) &&
            viewport_surface->preview_kind != asset_preview_kind::none)
        {
            const auto detached_id = viewport_surface->options.viewport_id;
            cleanup_asset_preview_surface(*viewport_surface, *state_->renderer);
            surfaces.surfaces.erase(detached_id);
        }
        return response;
    }

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

    const auto entity = to_scene_entity(material_command->entity);
    const auto targets = edit_targets(state_->scene.scene, entity, material_command->apply_to_selection);
    if (targets.empty()) return fail("Cannot edit a missing or unselected mesh renderer", entity);
    if (std::any_of(targets.begin(), targets.end(), [&](ecs::entity target)
                    { return !state_->scene.scene.has<scene::mesh_renderer_component>(target); }))
        return fail("Every selected entity must have an editable mesh renderer component", entity);

    if (parameter_assignment)
    {
        for (const auto target : targets)
        {
            auto* component = ensure_procedural_mesh_component(state_->scene, target);
            if (!component) return fail("Procedural mesh parameters are only available for procedural meshes", target);
            auto validation = *component;
            if (!set_procedural_mesh_parameter(validation, parameter_assignment->parameter,
                                               parameter_assignment->value))
                return fail("The selected procedural mesh does not support parameter '" +
                                parameter_assignment->parameter + "'",
                            target);
        }

        if (command.edit && command.edit->phase == host_edit_phase::cancel)
        {
            if (!state_->history.cancel(command.edit->id, state_->scene))
                return fail("Edit transaction is not active", entity);
            synchronize_procedural_mesh_components(state_->scene, *state_->renderer);
            ++state_->scene_revision;
            return response_with_revisions({.request_id = command.request_id, .succeeded = true});
        }

        std::optional<editor_scene_state> before;
        if (!command.edit)
            before = state_->scene;
        else if (command.edit->phase == host_edit_phase::begin &&
                 !state_->history.begin(command.edit->id,
                                        command.edit->label.empty() ? "Edit Procedural Mesh" : command.edit->label,
                                        state_->scene))
            return fail("Could not begin procedural mesh edit transaction", entity);
        else if (command.edit->phase != host_edit_phase::begin &&
                 !state_->history.transaction_matches(command.edit->id))
            return fail("Edit transaction is not active", entity);

        for (const auto target : targets)
        {
            auto* component = ensure_procedural_mesh_component(state_->scene, target);
            if (!component ||
                !set_procedural_mesh_parameter(*component, parameter_assignment->parameter,
                                               parameter_assignment->value) ||
                !regenerate_procedural_mesh(state_->scene, *state_->renderer, target))
            {
                if (command.edit)
                {
                    (void)state_->history.cancel(command.edit->id, state_->scene);
                    synchronize_procedural_mesh_components(state_->scene, *state_->renderer);
                }
                else if (before)
                {
                    state_->scene = std::move(*before);
                }
                return fail("Renderer could not update the procedural mesh", target);
            }
            persist_procedural_mesh_component(state_->scene, target);
        }

        if (!command.edit)
            state_->history.record("Edit Procedural Mesh", std::move(*before), state_->scene);
        else if (command.edit->phase == host_edit_phase::commit &&
                 !state_->history.commit(command.edit->id, state_->scene))
            return fail("Could not commit procedural mesh edit transaction", entity);

        ++state_->scene_revision;
        push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                   "Procedural mesh parameters updated", entity);
        return response_with_revisions({.request_id = command.request_id,
                                        .succeeded = true,
                                        .payload_json = "{\"entity\":" + to_json(material_command->entity) + '}'});
    }

    if (command.edit) return fail("Mesh assignment does not use a continuous edit transaction");

    if (primitive_type)
    {
        auto before = state_->scene;
        ensure_scene_authoring_metadata(state_->scene);
        for (const auto target : targets)
        {
            state_->scene.scene.emplace<procedural_mesh_component>(
                target, procedural_mesh_component{default_procedural_mesh_parameters(*primitive_type)});
            if (!regenerate_procedural_mesh(state_->scene, *state_->renderer, target))
            {
                state_->scene = std::move(before);
                return fail("Renderer could not create the procedural mesh", target);
            }

            const auto guid = entity_guid_of(state_->scene, target);
            auto* binding = find_asset_binding(state_->scene, guid);
            if (!binding)
            {
                state_->scene.asset_bindings.push_back({.entity = guid});
                binding = &state_->scene.asset_bindings.back();
            }
            binding->source_kind = "primitive";
            binding->source = {};
            binding->subresource = primitive_type_name(*primitive_type);
            persist_procedural_mesh_component(state_->scene, target);
        }

        ++state_->scene_revision;
        state_->history.record("Assign Procedural Mesh", std::move(before), state_->scene);
        push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                   "Procedural mesh assigned", entity);
        return response_with_revisions({.request_id = command.request_id,
                                        .succeeded = true,
                                        .payload_json = "{\"entity\":" + to_json(material_command->entity) + '}'});
    }

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
        auto resolved_reference =
            state_->asset_registry->resolve(normalized_path, arc::assets::asset_types::imported_scene);
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

        clear_procedural_mesh_component(state_->scene, target);
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
    push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Mesh asset assigned",
               entity);
    return response_with_revisions({.request_id = command.request_id,
                                    .succeeded = true,
                                    .payload_json = "{\"entity\":" + to_json(material_command->entity) + '}'});
}

host_response arc_host::query(const host_query_envelope& query) const
{
    if (!viewport_surfaces_) viewport_surfaces_ = std::make_unique<viewport_surface_registry>();
    auto& surfaces = *viewport_surfaces_;
    const auto viewport_id = viewport_id_from(query.payload);
    viewport_surface_registry::surface_state* viewport_surface =
        viewport_id ? surfaces.find(*viewport_id) : &surfaces.primary();
    if (!viewport_surface)
    {
        host_response response{.request_id = query.request_id, .succeeded = false, .error = "Viewport is not attached"};
        response.scene_revision = state_->scene_revision;
        response.world_epoch = state_->world_epoch;
        response.frame_revision = state_->viewport_frame_index;
        return response;
    }
    if (viewport_surface->preview_kind != asset_preview_kind::none)
        (void)ensure_asset_preview_scene(*state_, *viewport_surface);
    activate_viewport_surface(*state_, *viewport_surface);

    host_response response;
    {
        asset_preview_scene_scope preview_scope(*state_, *viewport_surface);
        response = query_base(query);
    }
    if (!response.succeeded) return response;

    if (std::holds_alternative<host_selected_entity_query>(query.payload))
    {
        auto payload = nlohmann::json::parse(response.payload_json, nullptr, false);
        if (!payload.is_discarded() && payload.is_object())
        {
            const auto& query_scene =
                viewport_surface->preview_scene ? *viewport_surface->preview_scene : state_->scene;
            const auto entity = query_scene.selected_entity;
            if (query_scene.scene.alive(entity))
            {
                const auto& const_query_scene = std::as_const(query_scene.scene);
                if (const auto* procedural = const_query_scene.try_get<procedural_mesh_component>(entity))
                {
                    auto procedural_json =
                        nlohmann::json::parse(procedural_mesh_snapshot_json(*procedural), nullptr, false);
                    if (!procedural_json.is_discarded()) payload["proceduralMesh"] = std::move(procedural_json);
                }
            }
            response.payload_json = payload.dump();
        }
        return response;
    }

    if (!std::holds_alternative<host_viewport_state_query>(query.payload)) return response;

    auto payload = nlohmann::json::parse(response.payload_json, nullptr, false);
    if (payload.is_discarded() || !payload.is_object()) return response;

    payload["viewportId"] = viewport_surface->options.viewport_id;
    payload["width"] = viewport_surface->options.width;
    payload["height"] = viewport_surface->options.height;
    payload["fps"] = viewport_surface->fps;
    payload["frameTimeMs"] = viewport_surface->frame_time_ms;
    payload["drawCalls"] = viewport_surface->draw_calls;
    payload["frameIndex"] = viewport_surface->local_frame_index;
    payload["submitted"] = viewport_surface->submitted;
    if (viewport_surface->preview_kind != asset_preview_kind::none)
    {
        payload["assetPreviewKind"] = asset_preview_kind_name(viewport_surface->preview_kind);
        payload["assetPreviewGuid"] = assets::to_string(viewport_surface->preview_guid);
        if (!viewport_surface->preview_error.empty()) payload["assetPreviewError"] = viewport_surface->preview_error;
    }

    const auto& telemetry_scene = viewport_surface->preview_scene ? *viewport_surface->preview_scene : state_->scene;
    const auto stats = collect_viewport_render_stats(telemetry_scene, *state_->renderer);
    payload["viewportTelemetryVersion"] = viewport_render_stats_schema_version;
    payload["triangles"] = stats.triangles;
    payload["verticesComplete"] = stats.vertices_complete;
    if (stats.vertices_complete) payload["vertices"] = stats.vertices;
    if (stats.gpu_memory_available) payload["gpuMemoryBytes"] = stats.gpu_memory_used_bytes;
    if (stats.gpu_memory_budget_bytes != 0) payload["gpuMemoryBudgetBytes"] = stats.gpu_memory_budget_bytes;

    const double fps = payload.value("fps", 0.0);
    payload["frameIntervalMs"] = fps > 0.0 ? 1000.0 / fps : 0.0;
    payload["cpuRenderTimeMs"] = payload.value("frameTimeMs", 0.0);
    response.payload_json = payload.dump();
    return response;
}

host_viewport_frame arc_host::request_viewport(const host_viewport_request& request)
{
    if (!viewport_surfaces_) viewport_surfaces_ = std::make_unique<viewport_surface_registry>();
    auto& surfaces = *viewport_surfaces_;
    const auto viewport_id = normalized_viewport_id(request.viewport_id);
    auto* viewport_surface = surfaces.find(viewport_id);
    if (!viewport_surface) return {.message = "Viewport render skipped: viewport is not attached"};

    if (viewport_surface->preview_kind != asset_preview_kind::none &&
        !ensure_asset_preview_scene(*state_, *viewport_surface) && !viewport_surface->preview_scene)
        return {.message = viewport_surface->preview_error.empty() ? "Asset preview scene is unavailable"
                                                                   : viewport_surface->preview_error};

    activate_viewport_surface(*state_, *viewport_surface);
    host_viewport_request routed_request = request;
    routed_request.viewport_id = viewport_id;
    routed_request.frame_index = surfaces.next_renderer_frame_index++;

    std::optional<state::pending_viewport_pick> suspended_pick;
    if (state_->pending_pick && surfaces.pending_pick_surface && *surfaces.pending_pick_surface != viewport_id)
    {
        suspended_pick = std::move(state_->pending_pick);
        state_->pending_pick.reset();
    }

    const auto request_time = std::chrono::steady_clock::now();
    host_viewport_frame response;
    {
        asset_preview_scene_scope preview_scope(*state_, *viewport_surface);
        response = request_viewport_base(routed_request);
    }

    if (suspended_pick)
        state_->pending_pick = std::move(suspended_pick);
    else if (!state_->pending_pick && surfaces.pending_pick_surface == viewport_id)
        surfaces.pending_pick_surface.reset();

    capture_viewport_surface(*state_, *viewport_surface);
    viewport_surface->local_frame_index = request.frame_index;
    viewport_surface->options.frame_index = request.frame_index;
    if (viewport_surface->last_request_time.time_since_epoch().count() != 0)
    {
        const double interval =
            std::chrono::duration<double>(request_time - viewport_surface->last_request_time).count();
        if (interval > 0.0) viewport_surface->fps = 1.0 / interval;
    }
    viewport_surface->last_request_time = request_time;
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

host_viewport_frame in_process_host_session::request_viewport(const host_viewport_request& request)
{
    return host_->request_viewport(request);
}

} // namespace arc::editor
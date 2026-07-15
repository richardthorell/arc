#include <arc/editor/arc_host.h>

#include <arc/diagnostics/diagnostics.h>
#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/world_environment_host.h>
#include <arc/geometric/box.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace arc::editor
{
namespace
{

scene::entity to_scene_entity(host_entity_id entity) noexcept
{
    return { .index = entity.index, .generation = entity.generation };
}

host_entity_id to_host_entity(scene::entity entity) noexcept
{
    return { .index = entity.index, .generation = entity.generation };
}

host_vec3 to_host_vec3(const math::vector3f& value) noexcept
{
    return { value[0], value[1], value[2] };
}

math::vector3f to_math_vec3(host_vec3 value) noexcept
{
    return { value.x, value.y, value.z };
}

host_quat to_host_quat(const math::quatf& value) noexcept
{
    return { value[0], value[1], value[2], value[3] };
}

math::quatf to_math_quat(host_quat value) noexcept
{
    return { value.x, value.y, value.z, value.w };
}

host_vec4 to_host_vec4(const math::vector4f& value) noexcept
{
    return { value[0], value[1], value[2], value[3] };
}

math::vector4f to_math_vec4(host_vec4 value) noexcept
{
    return { value.x, value.y, value.z, value.w };
}

host_transform to_host_transform(const scene::transform_component& transform) noexcept
{
    return {
        .position = to_host_vec3(transform.position),
        .rotation = to_host_quat(transform.rotation),
        .scale = to_host_vec3(transform.scale)
    };
}

scene::transform_component to_scene_transform(const host_transform& transform) noexcept
{
    scene::transform_component result;
    result.position = to_math_vec3(transform.position);
    result.rotation = to_math_quat(transform.rotation);
    result.scale = to_math_vec3(transform.scale);
    result.mark_dirty();
    return result;
}

host_camera_projection to_host_projection(scene::camera_projection projection) noexcept
{
    return projection == scene::camera_projection::orthographic
        ? host_camera_projection::orthographic
        : host_camera_projection::perspective;
}

host_camera_snapshot to_host_camera(const scene::camera_component& camera) noexcept
{
    return {
        .projection = to_host_projection(camera.projection),
        .fov_y_degrees = math::to_degrees(camera.fov_y_radians),
        .orthographic_height = camera.orthographic_height,
        .near_plane = camera.near_plane,
        .far_plane = camera.far_plane,
        .active = camera.active,
        .clear_color = to_host_vec4(camera.clear_color)
    };
}

bool valid_camera(const host_camera_snapshot& camera) noexcept
{
    const auto finite = [](float value) { return std::isfinite(value); };
    return finite(camera.fov_y_degrees) && camera.fov_y_degrees > 1.0f && camera.fov_y_degrees < 179.0f &&
        finite(camera.orthographic_height) && camera.orthographic_height > 0.0f &&
        finite(camera.near_plane) && camera.near_plane > 0.0f &&
        finite(camera.far_plane) && camera.far_plane > camera.near_plane &&
        finite(camera.clear_color.x) && camera.clear_color.x >= 0.0f && camera.clear_color.x <= 1.0f &&
        finite(camera.clear_color.y) && camera.clear_color.y >= 0.0f && camera.clear_color.y <= 1.0f &&
        finite(camera.clear_color.z) && camera.clear_color.z >= 0.0f && camera.clear_color.z <= 1.0f &&
        finite(camera.clear_color.w) && camera.clear_color.w >= 0.0f && camera.clear_color.w <= 1.0f;
}

scene::camera_component to_scene_camera(const host_camera_snapshot& camera) noexcept
{
    return {
        .projection = camera.projection == host_camera_projection::orthographic
            ? scene::camera_projection::orthographic
            : scene::camera_projection::perspective,
        .fov_y_radians = math::to_radians(camera.fov_y_degrees),
        .near_plane = camera.near_plane,
        .far_plane = camera.far_plane,
        .orthographic_height = camera.orthographic_height,
        .active = camera.active,
        .clear_color = to_math_vec4(camera.clear_color)
    };
}

editor_primitive_type primitive_type_for(host_create_entity_kind kind) noexcept
{
    switch (kind)
    {
    case host_create_entity_kind::plane:
        return editor_primitive_type::plane;
    case host_create_entity_kind::cube:
        return editor_primitive_type::cube;
    case host_create_entity_kind::sphere:
        return editor_primitive_type::sphere;
    case host_create_entity_kind::cylinder:
        return editor_primitive_type::cylinder;
    default:
        return editor_primitive_type::cube;
    }
}

const char* create_entity_kind_label(host_create_entity_kind kind) noexcept
{
    switch (kind)
    {
    case host_create_entity_kind::plane:
        return "Plane";
    case host_create_entity_kind::cube:
        return "Cube";
    case host_create_entity_kind::sphere:
        return "Sphere";
    case host_create_entity_kind::cylinder:
        return "Cylinder";
    case host_create_entity_kind::world_environment:
        return "World Environment";
    case host_create_entity_kind::terrain:
        return "Terrain";
    case host_create_entity_kind::water:
        return "Water Plane";
    case host_create_entity_kind::grass_patch:
        return "Grass Patch";
    case host_create_entity_kind::decal:
        return "Decal";
    }
    return "Entity";
}

scene::camera_projection to_scene_projection(host_camera_projection projection) noexcept
{
    return projection == host_camera_projection::orthographic
        ? scene::camera_projection::orthographic
        : scene::camera_projection::perspective;
}

render::render_mode to_render_mode(host_render_mode mode) noexcept
{
    return mode == host_render_mode::wireframe ? render::render_mode::wireframe : render::render_mode::shaded;
}

render::mesh_visualization_mode to_visualization(host_visualization_mode mode) noexcept
{
    switch (mode)
    {
    case host_visualization_mode::albedo: return render::mesh_visualization_mode::albedo;
    case host_visualization_mode::opacity: return render::mesh_visualization_mode::opacity;
    case host_visualization_mode::world_normal: return render::mesh_visualization_mode::world_normal;
    case host_visualization_mode::specularity: return render::mesh_visualization_mode::specularity;
    case host_visualization_mode::gloss: return render::mesh_visualization_mode::gloss;
    case host_visualization_mode::metalness: return render::mesh_visualization_mode::metalness;
    case host_visualization_mode::ao: return render::mesh_visualization_mode::ao;
    case host_visualization_mode::emission: return render::mesh_visualization_mode::emission;
    case host_visualization_mode::lighting: return render::mesh_visualization_mode::lighting;
    case host_visualization_mode::uv0: return render::mesh_visualization_mode::uv0;
    case host_visualization_mode::cascade_debug: return render::mesh_visualization_mode::cascade_debug;
    case host_visualization_mode::shadow_mask: return render::mesh_visualization_mode::shadow_mask;
    case host_visualization_mode::light_complexity: return render::mesh_visualization_mode::light_complexity;
    case host_visualization_mode::cluster_debug: return render::mesh_visualization_mode::cluster_debug;
    case host_visualization_mode::standard:
        break;
    }
    return render::mesh_visualization_mode::standard;
}

render::editor_overlay_mode to_overlay(host_overlay_mode mode) noexcept
{
    switch (mode)
    {
    case host_overlay_mode::none: return render::editor_overlay_mode::none;
    case host_overlay_mode::all_wireframe: return render::editor_overlay_mode::all_wireframe;
    case host_overlay_mode::selected_wireframe:
        break;
    }
    return render::editor_overlay_mode::selected_wireframe;
}

scene::scene_render_visibility to_scene_visibility(host_environment_visibility visibility) noexcept
{
    return {
        .sky = visibility.sky,
        .fog = visibility.fog,
        .terrain = visibility.terrain,
        .water = visibility.water,
        .vegetation = visibility.vegetation,
        .decals = visibility.decals
    };
}

scene::world_environment_preset to_scene_preset(host_world_environment_preset preset) noexcept
{
    switch (preset)
    {
    case host_world_environment_preset::clear_day: return scene::world_environment_preset::clear_day;
    case host_world_environment_preset::golden_hour: return scene::world_environment_preset::golden_hour;
    case host_world_environment_preset::overcast: return scene::world_environment_preset::overcast;
    case host_world_environment_preset::night: return scene::world_environment_preset::night;
    case host_world_environment_preset::indoor_neutral: return scene::world_environment_preset::indoor_neutral;
    case host_world_environment_preset::alpine_late_morning: break;
    }
    return scene::world_environment_preset::alpine_late_morning;
}

std::string validation_message(const scene::environment_validation_result& validation)
{
    std::ostringstream message;
    message << "Invalid world environment";
    for (const auto& error : validation.errors)
        message << "; " << error;
    return message.str();
}

void remove_entity_ref(std::vector<scene::entity>& entities, scene::entity entity)
{
    entities.erase(
        std::remove(entities.begin(), entities.end(), entity),
        entities.end());
}

void forget_entity(editor_scene_state& scene, scene::entity entity)
{
    if (scene.camera_entity == entity)
        scene.camera_entity = {};
    if (scene.game_camera_entity == entity)
        scene.game_camera_entity = {};
    if (scene.sun_entity == entity)
        scene.sun_entity = {};
    if (scene.world_environment_entity == entity)
        scene.world_environment_entity = {};
    if (scene.mesh_entity == entity)
        scene.mesh_entity = {};
    if (scene.terrain_entity == entity)
        scene.terrain_entity = {};
    if (scene.water_entity == entity)
        scene.water_entity = {};
    if (scene.vegetation_entity == entity)
        scene.vegetation_entity = {};
    if (scene.selected_entity == entity)
        scene.selected_entity = {};
    remove_entity_ref(scene.primitive_entities, entity);
    remove_entity_ref(scene.imported_scene_entities, entity);
    remove_entity_ref(scene.world_feature_entities, entity);
}

std::string entity_name(const editor_scene_state& state, scene::entity entity, const char* fallback)
{
    if (const auto* name = state.scene.try_get<scene::name_component>(entity))
        return name->value;
    return fallback;
}

bool entity_active(const editor_scene_state& state, scene::entity entity)
{
    const auto* active = state.scene.try_get<scene::active_component>(entity);
    return !active || active->active;
}

void push_event(
    std::vector<host_event>& events,
    std::uint64_t& sequence,
    host_event_type type,
    std::string message,
    scene::entity entity = {},
    std::string payload_json = {})
{
    events.push_back({
        .sequence = ++sequence,
        .event_type = type,
        .entity = to_host_entity(entity),
        .message = std::move(message),
        .payload_json = std::move(payload_json)
    });
}

void add_component_snapshot(std::vector<host_component_snapshot>& components, host_component_kind kind, const char* label)
{
    components.push_back({ .kind = kind, .label = label, .editable = true });
}

editor_scene_state create_default_scene(const editor_asset_state& assets, render::renderer& renderer)
{
    editor_scene_state state;

    math::vector3f center{};
    math::vector3f local_min = defaults::fallback_mesh_bounds_min;
    math::vector3f local_max = defaults::fallback_mesh_bounds_max;
    float radius = defaults::fallback_mesh_radius;
    if (assets.default_mesh_loaded && !assets.default_mesh.vertices.empty())
    {
        local_min = math::vector3f{
            assets.default_mesh.vertices[0].position[0],
            assets.default_mesh.vertices[0].position[1],
            assets.default_mesh.vertices[0].position[2]
        };
        local_max = local_min;
        for (const auto& vertex : assets.default_mesh.vertices)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                local_min[axis] = std::min(local_min[axis], vertex.position[axis]);
                local_max[axis] = std::max(local_max[axis], vertex.position[axis]);
            }
        }

        center = math::mul(math::add(local_min, local_max), 0.5f);
        const auto span = math::sub(local_max, local_min);
        radius = std::max({ span[0], span[1], span[2], defaults::fallback_mesh_radius }) * 0.5f;
    }

    if (assets.default_mesh_loaded)
    {
        state.default_textures.reserve(assets.default_textures.size());
        for (const auto& texture : assets.default_textures)
            state.default_textures.push_back(renderer.create_texture(texture));

        render::material_desc material;
        material.name = assets.default_mesh.name + " Material";
        if (!assets.default_materials.empty())
        {
            const auto material_index = assets.default_mesh.material_index < assets.default_materials.size()
                ? assets.default_mesh.material_index
                : std::size_t{ 0 };
            const auto& imported = assets.default_materials[material_index];
            material = imported.material;

            const auto assign_texture = [&](std::size_t index, render::texture_handle& handle) {
                if (index != render::material_texture_indices::invalid && index < state.default_textures.size())
                    handle = state.default_textures[index];
            };

            assign_texture(imported.textures.base_color, material.base_color_texture);
            assign_texture(imported.textures.metallic_roughness, material.metallic_roughness_texture);
            assign_texture(imported.textures.normal, material.normal_texture);
            assign_texture(imported.textures.occlusion, material.occlusion_texture);
            assign_texture(imported.textures.emissive, material.emissive_texture);
        }
        state.default_material = renderer.create_material(material);
        state.default_mesh = renderer.create_mesh(assets.default_mesh);
        state.mesh_uploaded = state.default_mesh.valid();
    }

    const auto camera = state.scene.create();
    state.camera_entity = camera;
    scene::transform_component camera_transform;
    camera_transform.position = defaults::default_camera_position;
    state.scene.emplace<scene::name_component>(camera, "Editor Camera");
    state.scene.emplace<scene::tag_component>(camera, "Editor");
    state.scene.emplace<scene::active_component>(camera);
    state.scene.emplace<scene::transform_component>(camera, camera_transform);
    scene::camera_component editor_camera;
    editor_camera.near_plane = 0.1f;
    editor_camera.far_plane = 2000.0f;
    editor_camera.clear_color = math::vector4f{ 0.055f, 0.12f, 0.22f, 1.0f };
    state.scene.emplace<scene::camera_component>(camera, editor_camera);

    const auto game_camera = state.scene.create();
    state.game_camera_entity = game_camera;
    scene::transform_component game_camera_transform;
    game_camera_transform.position = defaults::default_camera_position;
    state.scene.emplace<scene::name_component>(game_camera, "Main Camera");
    state.scene.emplace<scene::tag_component>(game_camera, "Camera");
    state.scene.emplace<scene::active_component>(game_camera);
    state.scene.emplace<scene::transform_component>(game_camera, game_camera_transform);
    scene::camera_component main_camera;
    main_camera.active = false;
    main_camera.near_plane = 0.1f;
    main_camera.far_plane = 2000.0f;
    state.scene.emplace<scene::camera_component>(game_camera, main_camera);

    const auto sun = state.scene.create();
    state.sun_entity = sun;
    scene::transform_component sun_transform;
    sun_transform.rotation = quaternion_from_euler_degrees(defaults::default_sun_rotation_degrees);
    state.scene.emplace<scene::name_component>(sun, "Sun");
    state.scene.emplace<scene::tag_component>(sun, "Light");
    state.scene.emplace<scene::active_component>(sun);
    state.scene.emplace<scene::transform_component>(sun, sun_transform);
    state.scene.emplace<scene::directional_light_component>(
        sun,
        defaults::default_sun_color,
        defaults::default_sun_intensity,
        true);
    auto& sun_light = state.scene.get<scene::directional_light_component>(sun);
    sun_light.shadow.resolution = defaults::default_sun_shadow_resolution;
    sun_light.shadow.filter = defaults::default_sun_shadow_filter;
    sun_light.shadow.bias = defaults::default_sun_shadow_bias;
    sun_light.shadow.normal_bias = defaults::default_sun_shadow_normal_bias;

    add_world_environment_to_scene(state);

    render::environment_desc environment_lighting;
    environment_lighting.name = "Default Mountain Daylight";
    environment_lighting.fallback_color = math::vector3f{ 0.16f, 0.22f, 0.30f };
    environment_lighting.intensity = 1.1f;
    environment_lighting.diffuse_irradiance = math::vector3f{ 0.18f, 0.23f, 0.29f };
    environment_lighting.diffuse_intensity = 1.0f;
    state.environment_lighting_resource = renderer.create_environment(environment_lighting);
    if (auto* lighting = state.scene.try_get<scene::environment_lighting_component>(state.world_environment_entity))
        lighting->environment = state.environment_lighting_resource;

    render::material_handle terrain_material;
    if (!assets.root.empty())
    {
        terrain_material = load_material_for_editor(
            state.material_library,
            renderer,
            assets.root,
            assets.root / "materials" / "mountain_landscape.arcmat");
    }
    const auto terrain = add_terrain_to_scene(state, renderer, terrain_material);
    add_water_to_scene(state, renderer);
    add_grass_patch_to_scene(state, renderer);

    if (state.default_mesh.valid())
    {
        const auto mesh = state.scene.create();
        state.mesh_entity = mesh;
        const float scale = defaults::imported_mesh_fit_size / radius;
        scene::transform_component mesh_transform;
        mesh_transform.position = math::mul(center, -scale);
        constexpr float landmark_x = -3.0f;
        constexpr float landmark_z = 4.0f;
        const float landmark_height = render::sample_terrain_height(
            landmark_x,
            landmark_z,
            defaults::default_terrain_size,
            defaults::default_terrain_height_scale);
        mesh_transform.position = math::add(
            mesh_transform.position,
            math::vector3f{ landmark_x, landmark_height + defaults::imported_mesh_fit_size, landmark_z });
        mesh_transform.scale = math::vector3f{ scale, scale, scale };
        state.scene.emplace<scene::name_component>(mesh, assets.default_mesh.name);
        state.scene.emplace<scene::tag_component>(mesh, "Mesh");
        state.scene.emplace<scene::active_component>(mesh);
        state.scene.emplace<scene::selection_component>(mesh, true);
        state.scene.emplace<scene::bounds_component>(
            mesh,
            geometric::box3f{ geometric::point3f(local_min), geometric::point3f(local_max) },
            geometric::box3f{ geometric::point3f(local_min), geometric::point3f(local_max) },
            true);
        state.scene.emplace<scene::transform_component>(mesh, mesh_transform);
        state.scene.emplace<scene::mesh_renderer_component>(
            mesh,
            state.default_mesh,
            state.default_material,
            true);
        state.selected_entity = mesh;
    }

    if (state.scene.alive(terrain))
        select_entity(state.scene, terrain, state.selected_entity);

    return state;
}

} // namespace

struct arc_host::state
{
    explicit state(std::unique_ptr<render::renderer> renderer_value)
        : renderer(std::move(renderer_value))
    {
    }

    std::unique_ptr<render::renderer> renderer;
    editor_project_state project;
    editor_asset_state assets;
    editor_scene_state scene;
    editor_camera_controller camera_controller;
    host_viewport_request viewport_options;
    std::chrono::steady_clock::time_point last_viewport_frame_time{};
    double viewport_fps{};
    double viewport_frame_ms{};
    std::uint32_t viewport_draw_calls{};
    std::uint64_t viewport_frame_index{};
    bool viewport_submitted{};
    std::vector<host_event> events;
    std::uint64_t event_sequence{};
    bool project_open{};
};

arc_host::arc_host(std::unique_ptr<render::renderer> renderer)
    : state_(std::make_unique<state>(std::move(renderer)))
{
    arc::info("editor.host", "Arc Host started");
    push_event(state_->events, state_->event_sequence, host_event_type::host_started, "Arc Host started");
}

arc_host::~arc_host()
{
    if (state_)
    {
        arc::info("editor.host", "Arc Host shutdown");
        push_event(state_->events, state_->event_sequence, host_event_type::host_shutdown, "Arc Host shutdown");
    }
}

host_response arc_host::open_project(
    const host_open_project_command& command,
    const editor_asset_state& assets,
    std::uint64_t request_id)
{
    state_->assets = assets;
    state_->project.name = command.name.empty() ? "Arc Project" : command.name;
    state_->project.root = command.root;
    state_->scene = create_default_scene(assets, *state_->renderer);
    state_->camera_controller = {};
    state_->camera_controller.focus(defaults::default_camera_focus, defaults::default_camera_focus_radius);
    state_->camera_controller.orbit(defaults::default_camera_orbit_x, defaults::default_camera_orbit_y);
    if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
        state_->camera_controller.apply_to(*camera_transform);
    state_->project_open = true;

    const std::string message = "Opened project '" + state_->project.name + "'";
    arc::info("editor.host", message);
    push_event(state_->events, state_->event_sequence, host_event_type::project_opened, message);
    push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Default editor scene loaded");
    return {
        .request_id = request_id,
        .succeeded = true,
        .payload_json = "{\"entity\":" + to_json(to_host_entity(state_->scene.selected_entity)) + '}'
    };
}

host_response arc_host::execute(host_command_payload command)
{
    return execute({ .command_type = command_type(command), .payload = std::move(command) });
}

host_response arc_host::execute(const host_command_envelope& command)
{
    return std::visit([this, request_id = command.request_id](const auto& payload) -> host_response {
        using command_type = std::decay_t<decltype(payload)>;
        const auto fail = [this, request_id](std::string message, scene::entity entity = {}) {
            arc::warn("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::command_failed, message, entity);
            return host_response{ .request_id = request_id, .succeeded = false, .error = std::move(message) };
        };
        const auto success = [request_id](std::string payload_json = "{}") {
            return host_response{ .request_id = request_id, .succeeded = true, .payload_json = std::move(payload_json) };
        };

        if constexpr (std::is_same_v<command_type, host_open_project_command>)
        {
            editor_asset_state empty_assets;
            const auto project_assets = payload.root / "assets";
            empty_assets.root = std::filesystem::is_directory(project_assets)
                ? project_assets
                : payload.root;
            return open_project(payload, empty_assets, request_id);
        }
        else if constexpr (std::is_same_v<command_type, host_close_project_command>)
        {
            if (!state_->project_open)
                return success("{\"message\":\"No project is open\"}");

            const std::string message = "Closed project '" + state_->project.name + "'";
            state_->scene = {};
            state_->project = {};
            state_->project_open = false;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::project_closed, message);
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_open_scene_command>)
        {
            if (!state_->project_open)
                return fail("Cannot open a scene before a project is open");

            const auto mode = payload.append ? editor_scene_open_mode::append : editor_scene_open_mode::replace;
            const auto asset_root = payload.path.is_absolute() ? payload.path.parent_path() : state_->assets.root;
            const auto result = open_scene_asset_in_editor(
                state_->scene,
                *state_->renderer,
                asset_root,
                payload.path,
                mode);
            if (!result.succeeded)
                return fail(result.message.empty() ? "Failed to open scene asset" : result.message);

            const std::string message =
                std::string(payload.append ? "Imported scene asset: " : "Opened scene asset: ") +
                payload.path.filename().string();
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, message, state_->scene.selected_entity);
            return success("{\"entityCount\":" + std::to_string(result.entity_count) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_create_entity_command>)
        {
            scene::entity created{};
            switch (payload.kind)
            {
            case host_create_entity_kind::plane:
            case host_create_entity_kind::cube:
            case host_create_entity_kind::sphere:
            case host_create_entity_kind::cylinder:
                created = add_primitive_to_scene(state_->scene, *state_->renderer, primitive_type_for(payload.kind));
                break;
            case host_create_entity_kind::world_environment:
                created = add_world_environment_to_scene(state_->scene);
                break;
            case host_create_entity_kind::terrain:
                created = add_terrain_to_scene(state_->scene, *state_->renderer);
                break;
            case host_create_entity_kind::water:
                created = add_water_to_scene(state_->scene, *state_->renderer);
                break;
            case host_create_entity_kind::grass_patch:
                created = add_grass_patch_to_scene(state_->scene, *state_->renderer);
                break;
            case host_create_entity_kind::decal:
                created = add_decal_to_scene(state_->scene);
                break;
            }

            const std::string label = create_entity_kind_label(payload.kind);
            if (!created.valid())
                return fail("Failed to create entity: " + label);

            const std::string message = "Created entity: " + label;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_created, message, created);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Selected entity", created);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", created);
            return success("{\"entity\":" + to_json(to_host_entity(created)) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_delete_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot delete a missing entity", entity);

            const std::string name = entity_name(state_->scene, entity, "Entity");
            state_->scene.scene.destroy(entity);
            forget_entity(state_->scene, entity);
            const std::string message = "Deleted entity: " + name;
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_deleted, message, entity);
            push_event(state_->events, state_->event_sequence, host_event_type::scene_changed, "Scene changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_rename_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot rename a missing entity", entity);

            state_->scene.scene.emplace<scene::name_component>(entity, payload.name);
            const std::string message = "Renamed entity to '" + payload.name + "'";
            arc::info("editor.host", message);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, message, entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_select_entity_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!select_entity(state_->scene.scene, entity, state_->scene.selected_entity))
                return fail("Cannot select a missing entity", entity);

            const std::string message = "Selected entity: " + entity_name(state_->scene, entity, "Entity");
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, message, entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_clear_selection_command>)
        {
            clear_selection(state_->scene.scene, state_->scene.selected_entity);
            push_event(state_->events, state_->event_sequence, host_event_type::entity_selected, "Cleared entity selection");
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_set_active_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::active_component>(entity, payload.active);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity active state changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_tag_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::tag_component>(entity, payload.tag);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity tag changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_transform_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);

            state_->scene.scene.emplace<scene::transform_component>(entity, to_scene_transform(payload.transform));
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity transform changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_render_layer_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            if (!state_->scene.scene.alive(entity))
                return fail("Cannot edit a missing entity", entity);
            if (payload.render_layer_mask == 0u)
                return fail("Render layer mask must contain at least one layer", entity);

            state_->scene.scene.emplace<scene::render_layer_component>(entity, payload.render_layer_mask);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity render layer changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_camera_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* camera = state_->scene.scene.try_get<scene::camera_component>(entity);
            if (!camera)
                return fail("Entity does not have an editable camera component", entity);
            if (!valid_camera(payload.camera))
                return fail("Camera values are outside their valid authored ranges", entity);

            *camera = to_scene_camera(payload.camera);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Entity camera changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_world_environment_command>)
        {
            const auto entity = to_scene_entity(payload.environment.entity);
            auto settings = scene::read_world_environment_settings(state_->scene.scene, entity);
            if (!settings)
                return fail("Cannot edit a missing or incomplete world environment", entity);

            const auto next_settings = apply_host_world_environment_snapshot(payload.environment, *settings);
            const auto validation = scene::validate_world_environment(next_settings);
            if (!validation.valid)
                return fail(validation_message(validation), entity);
            if (!scene::set_world_environment_settings(state_->scene.scene, entity, next_settings))
                return fail("World environment update could not be applied", entity);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                "World environment changed", entity);
            return success("{\"entity\":" + to_json(payload.environment.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_apply_world_environment_preset_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto settings = scene::read_world_environment_settings(state_->scene.scene, entity);
            if (!settings)
                return fail("Cannot apply a preset to a missing world environment", entity);

            scene::apply_world_environment_preset(to_scene_preset(payload.preset), *settings);
            scene::set_world_environment_settings(state_->scene.scene, entity, *settings);
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                "World environment preset applied", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_environment_hdri_command>)
        {
            const auto entity = to_scene_entity(payload.entity);
            auto* world = state_->scene.scene.try_get<scene::world_environment_component>(entity);
            auto* lighting = state_->scene.scene.try_get<scene::environment_lighting_component>(entity);
            if (!world || !lighting)
                return fail("Cannot assign an HDRI to a missing world environment", entity);

            auto path = payload.path;
            if (path.is_relative())
                path = state_->assets.root / path;
            if (!render::is_supported_texture_asset(path))
                return fail("Unsupported environment texture: " + path.extension().string(), entity);
            const auto loaded = render::load_texture_asset(path);
            if (!loaded.succeeded())
                return fail(loaded.message.empty() ? "Failed to load environment texture" : loaded.message, entity);

            auto texture = loaded.texture;
            texture.name = path.filename().string();
            if (world->hdri_texture.valid())
                state_->renderer->update_texture(world->hdri_texture, std::move(texture));
            else
                world->hdri_texture = state_->renderer->create_texture(std::move(texture));
            lighting->hdri_texture = world->hdri_texture;
            world->source = scene::sky_source::hdri;
            state_->scene.world_environment_hdri_path = path;

            if (lighting->environment.valid())
            {
                render::environment_desc environment;
                environment.name = "World Environment HDRI";
                environment.equirectangular_texture = world->hdri_texture;
                environment.fallback_color = world->solid_color;
                environment.intensity = world->radiance_intensity;
                environment.diffuse_irradiance = lighting->constant_color;
                environment.diffuse_intensity = lighting->diffuse_intensity;
                state_->renderer->update_environment(lighting->environment, std::move(environment));
            }
            push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                "World environment HDRI changed", entity);
            return success("{\"entity\":" + to_json(payload.entity) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_set_camera_projection_command>)
        {
            if (auto* camera = state_->scene.scene.try_get<scene::camera_component>(state_->scene.camera_entity))
            {
                camera->projection = to_scene_projection(payload.projection);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed, "Camera projection changed", state_->scene.camera_entity);
                return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
            }
            return fail("No editor camera is available", state_->scene.camera_entity);
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_attach_command>)
        {
            state_->viewport_options.width = payload.width;
            state_->viewport_options.height = payload.height;
            return success("{\"nativeHandle\":" + std::to_string(payload.native_handle) + '}');
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_resize_command>)
        {
            state_->viewport_options.width = payload.width;
            state_->viewport_options.height = payload.height;
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_set_camera_mode_command>)
        {
            return execute(host_command_envelope{
                .request_id = request_id,
                .command_type = "camera.setProjection",
                .payload = host_set_camera_projection_command{ .projection = payload.projection } });
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_set_render_options_command>)
        {
            state_->viewport_options.render_mode = payload.render_mode;
            state_->viewport_options.visualization = payload.visualization;
            state_->viewport_options.overlay = payload.overlay;
            state_->viewport_options.shadows = payload.shadows;
            state_->viewport_options.environment = payload.environment;
            return success();
        }
        else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)
        {
            if (payload.focus_selected)
                focus_selected_entity(state_->scene.scene, state_->scene.selected_entity, state_->camera_controller);
            if (payload.orbit_x != 0.0f || payload.orbit_y != 0.0f)
                state_->camera_controller.orbit(payload.orbit_x, payload.orbit_y);
            if (payload.pan_x != 0.0f || payload.pan_y != 0.0f)
                state_->camera_controller.pan(payload.pan_x, payload.pan_y);
            if (payload.forward != 0.0f)
                state_->camera_controller.move_forward(payload.forward);
            if (payload.zoom != 0.0f)
                state_->camera_controller.zoom(payload.zoom);
            if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            {
                state_->camera_controller.apply_to(*camera_transform);
                push_event(
                    state_->events,
                    state_->event_sequence,
                    host_event_type::component_changed,
                    "Viewport camera changed",
                    state_->scene.camera_entity);
                return success("{\"entity\":" + to_json(to_host_entity(state_->scene.camera_entity)) + '}');
            }
            return fail("No editor camera is available", state_->scene.camera_entity);
        }

        return fail("Unsupported host command");
    }, command.payload);
}

host_response arc_host::query(const host_query_envelope& query) const
{
    return std::visit([this, request_id = query.request_id](const auto& payload) -> host_response {
        using query_type = std::decay_t<decltype(payload)>;
        if constexpr (std::is_same_v<query_type, host_scene_hierarchy_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(scene_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_selected_entity_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(selected_entity_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_project_assets_query>)
        {
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(project_assets_snapshot()) };
        }
        else if constexpr (std::is_same_v<query_type, host_viewport_state_query>)
        {
            return {
                .request_id = request_id,
                .succeeded = true,
                .payload_json = "{\"width\":" + std::to_string(state_->viewport_options.width) +
                    ",\"height\":" + std::to_string(state_->viewport_options.height) +
                    ",\"fps\":" + std::to_string(state_->viewport_fps) +
                    ",\"frameTimeMs\":" + std::to_string(state_->viewport_frame_ms) +
                    ",\"drawCalls\":" + std::to_string(state_->viewport_draw_calls) +
                    ",\"frameIndex\":" + std::to_string(state_->viewport_frame_index) +
                    ",\"submitted\":" + std::string(state_->viewport_submitted ? "true" : "false") + '}'
            };
        }
        else if constexpr (std::is_same_v<query_type, host_world_environment_query>)
        {
            const auto snapshot = world_environment_snapshot(payload.entity);
            if (!snapshot)
                return { .request_id = request_id, .succeeded = false, .error = "World environment is missing" };
            return { .request_id = request_id, .succeeded = true, .payload_json = to_json(*snapshot) };
        }

        return { .request_id = request_id, .succeeded = false, .error = "Unsupported host query" };
    }, query.payload);
}

host_scene_snapshot arc_host::scene_snapshot() const
{
    host_scene_snapshot snapshot;
    const auto add = [&](scene::entity entity, const char* fallback, host_entity_kind kind) {
        if (!entity.valid() || !state_->scene.scene.alive(entity))
            return;
        snapshot.entities.push_back({
            .entity = to_host_entity(entity),
            .name = entity_name(state_->scene, entity, fallback),
            .kind = kind,
            .active = entity_active(state_->scene, entity),
            .selected = entity == state_->scene.selected_entity
        });
    };

    add(state_->scene.sun_entity, "Sun", host_entity_kind::light);
    add(state_->scene.game_camera_entity, "Main Camera", host_entity_kind::camera);
    for (const auto entity : state_->scene.world_feature_entities)
        add(entity, "Environment", host_entity_kind::environment);
    add(state_->scene.mesh_entity, "Default Mesh", host_entity_kind::mesh);
    for (const auto entity : state_->scene.imported_scene_entities)
        add(entity, "Imported Mesh", host_entity_kind::imported);
    for (const auto entity : state_->scene.primitive_entities)
        add(entity, "Primitive", host_entity_kind::primitive);
    return snapshot;
}

host_selected_entity_snapshot arc_host::selected_entity_snapshot() const
{
    host_selected_entity_snapshot snapshot;
    const auto selected = state_->scene.selected_entity;
    if (!state_->scene.scene.alive(selected))
        return snapshot;

    snapshot.entity = to_host_entity(selected);
    snapshot.name = entity_name(state_->scene, selected, "Unnamed Entity");
    if (const auto* tag = state_->scene.scene.try_get<scene::tag_component>(selected))
        snapshot.tag = tag->value;
    snapshot.active = entity_active(state_->scene, selected);
    if (const auto* layer = state_->scene.scene.try_get<scene::render_layer_component>(selected))
        snapshot.render_layer_mask = layer->mask;
    if (const auto* transform = state_->scene.scene.try_get<scene::transform_component>(selected))
    {
        snapshot.transform = to_host_transform(*transform);
        add_component_snapshot(snapshot.components, host_component_kind::transform, "Transform");
    }
    if (const auto* camera = state_->scene.scene.try_get<scene::camera_component>(selected))
    {
        snapshot.camera = to_host_camera(*camera);
        add_component_snapshot(snapshot.components, host_component_kind::camera, "Camera");
    }
    if (state_->scene.scene.has<scene::mesh_renderer_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::mesh_renderer, "Mesh Renderer");
    if (state_->scene.scene.has<scene::directional_light_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::directional_light, "Directional Light");
    if (state_->scene.scene.has<scene::point_light_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::point_light, "Point Light");
    if (state_->scene.scene.has<scene::spot_light_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::spot_light, "Spot Light");
    if (state_->scene.scene.has<scene::sky_atmosphere_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::sky_atmosphere, "Sky Atmosphere");
    if (state_->scene.scene.has<scene::world_environment_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::world_environment, "World Environment");
    if (state_->scene.scene.has<scene::celestial_sky_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::celestial_sky, "Sun, Moon & Time");
    if (state_->scene.scene.has<scene::cloud_layers_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::cloud_layers, "Cloud Layers");
    if (state_->scene.scene.has<scene::environment_lighting_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::environment_lighting, "Environment Lighting");
    if (state_->scene.scene.has<scene::height_fog_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::height_fog, "Height Fog");
    if (state_->scene.scene.has<scene::terrain_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::terrain, "Terrain");
    if (state_->scene.scene.has<scene::water_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::water, "Water");
    if (state_->scene.scene.has<scene::vegetation_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::vegetation, "Vegetation");
    if (state_->scene.scene.has<scene::decal_component>(selected))
        add_component_snapshot(snapshot.components, host_component_kind::decal, "Decal");
    return snapshot;
}

std::optional<host_world_environment_snapshot> arc_host::world_environment_snapshot(host_entity_id host_entity) const
{
    const auto settings = scene::read_world_environment_settings(
        state_->scene.scene,
        to_scene_entity(host_entity));
    if (!settings)
        return std::nullopt;
    return to_host_world_environment_snapshot(
        host_entity,
        *settings,
        state_->scene.world_environment_hdri_path);
}

host_project_assets_snapshot arc_host::project_assets_snapshot() const
{
    host_project_assets_snapshot snapshot;
    snapshot.project_name = state_->project.name;
    snapshot.project_root = state_->project.root;
    snapshot.asset_root = state_->assets.root;
    snapshot.default_mesh_path = state_->assets.default_mesh_path.generic_string();
    snapshot.default_mesh_loaded = state_->assets.default_mesh_loaded;
    snapshot.default_mesh_message = state_->assets.default_mesh_message;
    if (!state_->assets.default_mesh_path.empty())
    {
        snapshot.assets.push_back({
            .path = state_->assets.default_mesh_path.generic_string(),
            .kind = "scene",
            .imported = state_->assets.default_mesh_loaded,
            .import_running = false
        });
    }
    for (const auto& material : state_->scene.material_library.materials)
    {
        snapshot.assets.push_back({
            .path = material.asset.path.generic_string(),
            .kind = "material",
            .imported = true,
            .import_running = false
        });
    }
    if (!state_->assets.root.empty())
    {
        std::error_code error;
        for (std::filesystem::recursive_directory_iterator iterator(
                 state_->assets.root,
                 std::filesystem::directory_options::skip_permission_denied,
                 error), end;
             iterator != end && !error;
             iterator.increment(error))
        {
            if (!iterator->is_regular_file(error))
                continue;
            auto extension = iterator->path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            if (extension != ".hdr")
                continue;
            snapshot.assets.push_back({
                .path = std::filesystem::relative(iterator->path(), state_->assets.root, error).generic_string(),
                .kind = "environment",
                .imported = true,
                .import_running = false
            });
        }
    }
    return snapshot;
}

std::vector<host_event> arc_host::poll_events()
{
    auto events = std::move(state_->events);
    state_->events.clear();
    return events;
}

host_viewport_frame arc_host::request_viewport(const host_viewport_request& request)
{
    const auto frame_start = std::chrono::steady_clock::now();
    float delta_seconds = 0.0f;
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
        delta_seconds = std::chrono::duration<float>(frame_start - state_->last_viewport_frame_time).count();
    state_->viewport_options = request;
    state_->viewport_frame_index = request.frame_index;
    if (!state_->renderer->backend())
    {
        const std::string message = "Viewport render skipped: no render backend attached";
        state_->viewport_submitted = false;
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error, message);
        return { .message = message };
    }

    if (state_->scene.focus_imported_scene_requested)
    {
        focus_selected_entity(state_->scene.scene, state_->scene.selected_entity, state_->camera_controller);
        if (auto* camera_transform = state_->scene.scene.try_get<scene::transform_component>(state_->scene.camera_entity))
            state_->camera_controller.apply_to(*camera_transform);
        state_->scene.focus_imported_scene_requested = false;
    }

    state_->scene.last_render = scene::render_scene(
        state_->scene.scene,
        *state_->renderer,
        request.width,
        request.height,
        to_render_mode(request.render_mode),
        to_visualization(request.visualization),
        to_overlay(request.overlay),
        request.shadows,
        to_scene_visibility(request.environment),
        delta_seconds);

    const auto submit_result = state_->renderer->render_frame(
        request.frame_index,
        render::make_scene_draw_graph(
            "viewport",
            state_->renderer->resolved_config(),
            true,
            state_->scene.last_render.environment));
    const auto frame_end = std::chrono::steady_clock::now();
    state_->viewport_frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    if (state_->last_viewport_frame_time.time_since_epoch().count() != 0)
    {
        const double delta_seconds = std::chrono::duration<double>(frame_end - state_->last_viewport_frame_time).count();
        if (delta_seconds > 0.0)
            state_->viewport_fps = 1.0 / delta_seconds;
    }
    state_->last_viewport_frame_time = frame_end;
    state_->viewport_draw_calls = state_->scene.last_render.submitted_draw_count;
    state_->viewport_submitted = submit_result.submitted;
    if (!submit_result.submitted && !submit_result.message.empty())
    {
        arc::error("editor.host", submit_result.message);
        push_event(state_->events, state_->event_sequence, host_event_type::viewport_error, submit_result.message);
    }

    return {
        .submitted = submit_result.submitted,
        .message = submit_result.message,
        .payload_json = "{\"drawCalls\":" + std::to_string(state_->scene.last_render.submitted_draw_count) + '}'
    };
}

render::renderer& arc_host::renderer_service() noexcept
{
    return *state_->renderer;
}

const render::renderer& arc_host::renderer_service() const noexcept
{
    return *state_->renderer;
}

editor_scene_state& arc_host::scene_state() noexcept
{
    return state_->scene;
}

const editor_scene_state& arc_host::scene_state() const noexcept
{
    return state_->scene;
}

in_process_host_session::in_process_host_session(std::shared_ptr<arc_host> host)
    : host_(std::move(host))
{
}

host_response in_process_host_session::execute(const host_command_envelope& command)
{
    return host_->execute(command);
}

host_response in_process_host_session::query(const host_query_envelope& query)
{
    return host_->query(query);
}

std::vector<host_event> in_process_host_session::poll_events()
{
    return host_->poll_events();
}

host_viewport_frame in_process_host_session::request_viewport(const host_viewport_request& request)
{
    return host_->request_viewport(request);
}

std::string stdio_host_session::command_line(const host_command_envelope& command)
{
    return to_json(command) + '\n';
}

std::string stdio_host_session::query_line(const host_query_envelope& query)
{
    return to_json(query) + '\n';
}

std::shared_ptr<arc_host> arc_host_manager::acquire(std::unique_ptr<render::renderer> renderer)
{
    if (auto existing = host_.lock())
    {
        arc::info("editor.host", "Acquired existing Arc Host");
        return existing;
    }

    auto created = std::make_shared<arc_host>(std::move(renderer));
    host_ = created;
    return created;
}

} // namespace arc::editor

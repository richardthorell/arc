#include <arc/editor/editor_state.h>

#include <arc/editor/editor_defaults.h>
#include <arc/editor/editor_interaction.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/geometric/box.h>
#include <arc/render/primitives.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>

namespace arc::editor
{
namespace
{

render::mesh_data make_primitive_mesh(editor_primitive_type type)
{
    switch (type)
    {
    case editor_primitive_type::plane:
        return render::make_plane_mesh(4.0f);
    case editor_primitive_type::cube:
        return render::make_cube_mesh(1.0f);
    case editor_primitive_type::sphere:
        return render::make_uv_sphere_mesh(0.5f, 32, 16);
    case editor_primitive_type::cylinder:
        return render::make_cylinder_mesh(0.5f, 1.0f, 32);
    }
    return render::make_plane_mesh(4.0f);
}

geometric::box3f bounds_for_mesh(const render::mesh_data& mesh)
{
    if (mesh.vertices.empty())
        return geometric::box3f{ geometric::point3f{ -0.5f, -0.5f, -0.5f }, geometric::point3f{ 0.5f, 0.5f, 0.5f } };

    math::vector3f min_value{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()
    };
    math::vector3f max_value{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()
    };

    for (const auto& vertex : mesh.vertices)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            min_value[axis] = std::min(min_value[axis], vertex.position[axis]);
            max_value[axis] = std::max(max_value[axis], vertex.position[axis]);
        }
    }

    return geometric::box3f{ geometric::point3f{ min_value }, geometric::point3f{ max_value } };
}

render::material_handle ensure_default_material(editor_scene_state& scene, render::renderer& renderer)
{
    if (scene.primitive_material.valid())
        return scene.primitive_material;

    render::material_desc material;
    material.name = "Default Primitive Material";
    material.base_color = math::vector4f{ 0.76f, 0.79f, 0.83f, 1.0f };
    material.roughness = 0.62f;
    scene.primitive_material = renderer.create_material(material);
    return scene.primitive_material;
}

render::material_handle ensure_terrain_material(editor_scene_state& scene, render::renderer& renderer)
{
    if (scene.terrain_material.valid())
        return scene.terrain_material;

    render::material_desc material;
    material.name = "Default Terrain Material";
    material.base_color = math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f };
    material.roughness = 0.82f;
    scene.terrain_material = renderer.create_material(material);
    return scene.terrain_material;
}

render::material_handle ensure_water_material(editor_scene_state& scene, render::renderer& renderer)
{
    if (scene.water_material.valid())
        return scene.water_material;

    render::material_desc material;
    material.name = "Default Water Material";
    material.base_color = math::vector4f{ 0.16f, 0.35f, 0.48f, 0.55f };
    material.roughness = 0.18f;
    material.alpha_mode = render::material_alpha_mode::blend;
    scene.water_material = renderer.create_material(material);
    return scene.water_material;
}

render::material_handle ensure_vegetation_material(editor_scene_state& scene, render::renderer& renderer)
{
    if (scene.vegetation_material.valid())
        return scene.vegetation_material;

    render::material_desc material;
    material.name = "Default Vegetation Material";
    material.base_color = math::vector4f{ 0.22f, 0.46f, 0.18f, 1.0f };
    material.roughness = 0.9f;
    material.double_sided = true;
    scene.vegetation_material = renderer.create_material(material);
    return scene.vegetation_material;
}

void add_selectable_common(editor_scene_state& scene, scene::entity entity, const char* name, const char* tag)
{
    scene.scene.emplace<scene::name_component>(entity, name);
    scene.scene.emplace<scene::tag_component>(entity, tag);
    scene.scene.emplace<scene::active_component>(entity);
    scene.scene.emplace<scene::selection_component>(entity, true);
    select_entity(scene.scene, entity, scene.selected_entity);
}

void destroy_entity_if_alive(editor_scene_state& state, scene::entity& entity)
{
    if (state.scene.alive(entity))
        state.scene.destroy(entity);
    entity = {};
}

void destroy_entities(editor_scene_state& state, std::vector<scene::entity>& entities)
{
    for (auto entity : entities)
    {
        if (state.scene.alive(entity))
            state.scene.destroy(entity);
    }
    entities.clear();
}

void clear_imported_content(editor_scene_state& state)
{
    destroy_entity_if_alive(state, state.mesh_entity);
    destroy_entity_if_alive(state, state.terrain_entity);
    destroy_entity_if_alive(state, state.water_entity);
    destroy_entity_if_alive(state, state.vegetation_entity);
    destroy_entities(state, state.primitive_entities);
    destroy_entities(state, state.imported_scene_entities);
    state.selected_entity = {};
}

render::material_handle material_from_import(
    editor_scene_state& state,
    render::renderer& renderer,
    const render::material_import& imported,
    const std::vector<render::texture_handle>& textures)
{
    auto material = imported.material;
    const auto assign_texture = [&](std::size_t index, render::texture_handle& handle) {
        if (index != render::material_texture_indices::invalid && index < textures.size())
            handle = textures[index];
    };
    assign_texture(imported.textures.base_color, material.base_color_texture);
    assign_texture(imported.textures.metallic_roughness, material.metallic_roughness_texture);
    assign_texture(imported.textures.normal, material.normal_texture);
    assign_texture(imported.textures.occlusion, material.occlusion_texture);
    assign_texture(imported.textures.emissive, material.emissive_texture);
    if (material.name.empty())
        material.name = "Imported Material";
    const auto handle = renderer.create_material(material);
    return handle.valid() ? handle : ensure_default_material(state, renderer);
}

} // namespace

const char* selected_entity_name(const editor_scene_state& scene, const char* fallback)
{
    if (!scene.scene.alive(scene.selected_entity))
        return fallback;

    const auto* name = scene.scene.try_get<arc::scene::name_component>(scene.selected_entity);
    return name ? name->value.c_str() : fallback;
}

const char* primitive_type_name(editor_primitive_type type) noexcept
{
    switch (type)
    {
    case editor_primitive_type::plane:
        return "Plane";
    case editor_primitive_type::cube:
        return "Cube";
    case editor_primitive_type::sphere:
        return "Sphere";
    case editor_primitive_type::cylinder:
        return "Cylinder";
    }
    return "Primitive";
}

scene::entity add_primitive_to_scene(
    editor_scene_state& scene,
    render::renderer& renderer,
    editor_primitive_type type)
{
    auto mesh = make_primitive_mesh(type);
    const auto local_bounds = bounds_for_mesh(mesh);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid())
        return {};

    const auto material = ensure_default_material(scene, renderer);
    const auto entity = scene.scene.create();
    char name[64]{};
    const std::uint32_t serial = ++scene.primitive_serial;
    std::snprintf(name, sizeof(name), "%s %u", primitive_type_name(type), serial);

    scene::transform_component transform;
    if (type == editor_primitive_type::plane)
        transform.position = math::vector3f{ 0.0f, -0.01f, 0.0f };

    scene.scene.emplace<scene::name_component>(entity, name);
    scene.scene.emplace<scene::tag_component>(entity, "Primitive");
    scene.scene.emplace<scene::active_component>(entity);
    scene.scene.emplace<scene::selection_component>(entity, true);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity, transform);
    scene.scene.emplace<scene::mesh_renderer_component>(entity, mesh_handle, material, true);
    scene.primitive_entities.push_back(entity);
    select_entity(scene.scene, entity, scene.selected_entity);
    return entity;
}

scene::entity add_world_environment_to_scene(editor_scene_state& scene)
{
    if (scene.scene.alive(scene.world_environment_entity))
    {
        select_entity(scene.scene, scene.world_environment_entity, scene.selected_entity);
        return scene.world_environment_entity;
    }

    const auto entity = scene.scene.create();
    scene.world_environment_entity = entity;
    add_selectable_common(scene, entity, "World Environment", "Environment");
    scene::sky_atmosphere_component sky;
    sky.rayleigh_strength = 1.18f;
    sky.mie_strength = 0.24f;
    sky.ozone_strength = 0.22f;
    sky.tint = math::vector3f{ 0.48f, 0.68f, 1.0f };
    sky.exposure = 1.08f;
    sky.sun_disk_size = 0.012f;
    sky.sun_disk_intensity = 2.4f;
    scene.scene.emplace<scene::sky_atmosphere_component>(entity, sky);

    scene::height_fog_component fog;
    fog.color = math::vector3f{ 0.62f, 0.71f, 0.80f };
    fog.density = 0.008f;
    fog.height_falloff = 0.055f;
    fog.start_distance = 34.0f;
    fog.max_opacity = 0.42f;
    fog.sun_scattering_strength = 0.32f;
    scene.scene.emplace<scene::height_fog_component>(entity, fog);
    scene.environment_entities.push_back(entity);
    return entity;
}

scene::entity add_terrain_to_scene(
    editor_scene_state& scene,
    render::renderer& renderer,
    render::material_handle material)
{
    auto mesh = render::make_terrain_grid_mesh(
        defaults::default_terrain_size,
        defaults::default_terrain_subdivisions,
        defaults::default_terrain_height_scale);
    const auto local_bounds = bounds_for_mesh(mesh);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid())
        return {};

    if (!material.valid())
        material = ensure_terrain_material(scene, renderer);
    else
        scene.terrain_material = material;
    const auto entity = scene.scene.create();
    scene.terrain_entity = entity;
    add_selectable_common(scene, entity, "Terrain", "Environment");
    scene::terrain_component terrain;
    terrain.size = defaults::default_terrain_size;
    terrain.subdivisions = defaults::default_terrain_subdivisions;
    terrain.height_scale = defaults::default_terrain_height_scale;
    terrain.material = material;
    scene.scene.emplace<scene::terrain_component>(entity, terrain);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity);
    scene.scene.emplace<scene::mesh_renderer_component>(
        entity,
        mesh_handle,
        material,
        true,
        math::vector4f{ 1.0f, 1.0f, 1.0f, 1.0f });
    scene.environment_entities.push_back(entity);
    return entity;
}

scene::entity add_water_to_scene(editor_scene_state& scene, render::renderer& renderer)
{
    auto mesh = render::make_plane_mesh(defaults::default_water_size);
    const auto local_bounds = bounds_for_mesh(mesh);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid())
        return {};

    const auto material = ensure_water_material(scene, renderer);
    const auto entity = scene.scene.create();
    scene.water_entity = entity;
    add_selectable_common(scene, entity, "Water Plane", "Environment");
    scene::transform_component transform;
    transform.position = defaults::default_water_position;
    scene::water_component water;
    water.size = defaults::default_water_size;
    water.color = math::vector3f{ 0.08f, 0.30f, 0.42f };
    water.roughness = 0.12f;
    water.wave_scale = 0.14f;
    water.transparency = 0.34f;
    scene.scene.emplace<scene::water_component>(entity, water);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity, transform);
    scene.scene.emplace<scene::mesh_renderer_component>(
        entity,
        mesh_handle,
        material,
        true,
        math::vector4f{ 0.7f, 0.9f, 1.0f, 0.55f });
    scene.environment_entities.push_back(entity);
    return entity;
}

scene::entity add_grass_patch_to_scene(editor_scene_state& scene, render::renderer& renderer)
{
    auto mesh = render::make_grass_patch_mesh(20.0f, 320, 0.85f);
    const auto local_bounds = bounds_for_mesh(mesh);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid())
        return {};

    const auto material = ensure_vegetation_material(scene, renderer);
    const auto entity = scene.scene.create();
    scene.vegetation_entity = entity;
    add_selectable_common(scene, entity, "Grass Patch", "Environment");
    scene::transform_component transform;
    transform.position = defaults::default_grass_position;
    transform.position[1] = render::sample_terrain_height(
        transform.position[0],
        transform.position[2],
        defaults::default_terrain_size,
        defaults::default_terrain_height_scale) + 0.02f;
    scene::vegetation_component vegetation;
    vegetation.patch_size = 20.0f;
    vegetation.density = 320;
    vegetation.color = math::vector3f{ 0.20f, 0.48f, 0.16f };
    vegetation.wind_strength = 0.28f;
    scene.scene.emplace<scene::vegetation_component>(entity, vegetation);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity, transform);
    scene.scene.emplace<scene::mesh_renderer_component>(entity, mesh_handle, material, true);
    scene.environment_entities.push_back(entity);
    return entity;
}

scene::entity add_decal_to_scene(editor_scene_state& scene)
{
    const auto entity = scene.scene.create();
    add_selectable_common(scene, entity, "Decal", "Environment");
    scene::transform_component transform;
    transform.position = math::vector3f{ 0.0f, 0.05f, 0.0f };
    transform.scale = math::vector3f{ 1.5f, 0.05f, 1.5f };
    const geometric::box3f local_bounds{
        geometric::point3f{ -0.5f, -0.5f, -0.5f },
        geometric::point3f{ 0.5f, 0.5f, 0.5f }
    };
    scene.scene.emplace<scene::decal_component>(entity);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity, transform);
    scene.environment_entities.push_back(entity);
    return entity;
}

editor_scene_open_result open_scene_asset_in_editor(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    editor_scene_open_mode mode)
{
    const auto resolved_path = path.is_absolute() ? path : asset_root / path;
    render::scene_import_options options;
    options.asset_root = asset_root;
    auto imported = render::load_scene_asset(resolved_path, options);
    return apply_scene_import_result_to_editor(scene, renderer, resolved_path, std::move(imported), mode);
}

editor_scene_open_result apply_scene_import_result_to_editor(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& source_path,
    render::scene_import_result imported,
    editor_scene_open_mode mode)
{
    if (!imported.succeeded())
    {
        for (const auto& diagnostic : imported.diagnostics)
            arc::warn("editor.assets", diagnostic);
        return {
            .message = imported.message.empty() ? "scene asset could not be imported" : imported.message
        };
    }

    if (mode == editor_scene_open_mode::replace)
        clear_imported_content(scene);

    std::vector<render::texture_handle> textures;
    textures.reserve(imported.textures.size());
    for (auto& texture : imported.textures)
        textures.push_back(renderer.create_texture(std::move(texture)));

    std::vector<render::material_handle> materials;
    materials.reserve(imported.materials.size());
    for (const auto& material : imported.materials)
        materials.push_back(material_from_import(scene, renderer, material, textures));
    if (materials.empty())
        materials.push_back(ensure_default_material(scene, renderer));

    std::vector<render::mesh_handle> meshes;
    meshes.reserve(imported.meshes.size());
    std::vector<geometric::box3f> bounds;
    bounds.reserve(imported.meshes.size());
    for (const auto& mesh : imported.meshes)
    {
        meshes.push_back(renderer.create_mesh(mesh));
        bounds.push_back(bounds_for_mesh(mesh));
    }

    std::size_t created{};
    scene::entity first_entity{};
    for (const auto& node : imported.nodes)
    {
        if (node.mesh_index >= meshes.size() || !meshes[node.mesh_index].valid())
            continue;

        const auto entity = scene.scene.create();
        if (!first_entity.valid())
            first_entity = entity;

        const auto material_index = node.material_index < materials.size() ? node.material_index : std::size_t{ 0 };
        scene::transform_component transform;
        transform.position = node.position;
        transform.rotation = node.rotation;
        transform.scale = node.scale;

        scene.scene.emplace<scene::name_component>(entity, node.name.empty() ? "Imported Mesh" : node.name);
        scene.scene.emplace<scene::tag_component>(entity, "Imported");
        scene.scene.emplace<scene::active_component>(entity);
        scene.scene.emplace<scene::selection_component>(entity, false);
        scene.scene.emplace<scene::bounds_component>(entity, bounds[node.mesh_index], bounds[node.mesh_index], true);
        scene.scene.emplace<scene::transform_component>(entity, transform);
        scene.scene.emplace<scene::mesh_renderer_component>(
            entity,
            meshes[node.mesh_index],
            materials[material_index],
            true);
        scene.imported_scene_entities.push_back(entity);
        ++created;
    }

    if (first_entity.valid())
    {
        select_entity(scene.scene, first_entity, scene.selected_entity);
        scene.focus_imported_scene_requested = true;
    }

    arc::info(
        "editor.assets",
        std::string(mode == editor_scene_open_mode::replace ? "Opened scene asset '" : "Imported scene asset '") +
            source_path.filename().string() + "' with " + std::to_string(created) + " entity(s)");

    return {
        .succeeded = created != 0,
        .entity_count = created,
        .message = imported.message
    };
}

bool start_scene_import(
    editor_scene_import_state& state,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    editor_scene_open_mode mode)
{
    if (state.status == editor_scene_import_status::running)
        return false;

    reset_scene_import(state);
    const auto resolved_path = path.is_absolute() ? path : asset_root / path;
    auto shared = std::make_shared<editor_scene_import_shared_state>();
    shared->message = "Queued import";

    render::scene_import_options options;
    options.asset_root = asset_root;
    options.cancel_requested = &shared->cancel_requested;

    state.status = editor_scene_import_status::running;
    state.mode = mode;
    state.source_path = resolved_path;
    state.shared = shared;
    state.modal_open = true;
    state.task = std::async(std::launch::async, [resolved_path, options, shared]() mutable {
        return render::load_scene_asset(resolved_path, options, [shared](const render::scene_import_progress& progress) {
            {
                std::scoped_lock lock(shared->mutex);
                shared->progress = std::clamp(progress.progress, 0.0f, 1.0f);
                shared->stage = progress.stage;
                shared->message = progress.message;
            }
            return !shared->cancel_requested.load();
        });
    });
    return true;
}

bool poll_scene_import(editor_scene_import_state& state)
{
    if (state.status != editor_scene_import_status::running || !state.task.valid())
        return false;

    if (state.task.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return false;

    state.result = state.task.get();
    if (state.shared)
    {
        std::scoped_lock lock(state.shared->mutex);
        state.shared->progress = 1.0f;
        state.shared->stage = render::scene_import_stage::finalizing;
        state.shared->message = state.result.message;
        state.shared->diagnostics = state.result.diagnostics;
    }

    const bool cancelled = state.result.message.find("cancelled") != std::string::npos;
    state.status = state.result.succeeded()
        ? editor_scene_import_status::succeeded
        : (cancelled ? editor_scene_import_status::cancelled : editor_scene_import_status::failed);
    state.result_ready = true;
    state.modal_open = state.status != editor_scene_import_status::succeeded;
    return true;
}

void reset_scene_import(editor_scene_import_state& state)
{
    if (state.status == editor_scene_import_status::running && state.shared)
        state.shared->cancel_requested = true;
    if (state.task.valid())
        state.task.wait();
    state = {};
}

const char* scene_import_stage_label(render::scene_import_stage stage) noexcept
{
    switch (stage)
    {
    case render::scene_import_stage::loading:
        return "Loading";
    case render::scene_import_stage::extracting_textures:
        return "Extracting textures";
    case render::scene_import_stage::building_materials:
        return "Building materials";
    case render::scene_import_stage::building_meshes:
        return "Building meshes";
    case render::scene_import_stage::finalizing:
        return "Finalizing";
    }
    return "Importing";
}

} // namespace arc::editor

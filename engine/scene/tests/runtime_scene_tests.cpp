#include <arc/scene/runtime_scene.h>

#include <arc/ecs/hierarchy.h>
#include <arc/persistence/persistence.h>
#include <arc/scene/components.h>
#include <arc/scene/persistence.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

arc::persistence::archive_document authored_scene(bool with_child)
{
    namespace persistence = arc::persistence;
    persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));

    persistence::archive_document document;
    document.id = {1, 2};
    document.name = "Packaged Startup";

    persistence::archive_entity_record parent;
    parent.id = {3, 4};
    const arc::scene::name_component name{"Root Entity"};
    persistence::archive_component_record encoded;
    REQUIRE(registry.encode(arc::ecs::component_metadata<arc::scene::name_component>().id, &name, encoded));
    parent.components.push_back(std::move(encoded));
    document.entities.push_back(std::move(parent));

    if (with_child)
    {
        persistence::archive_entity_record child;
        child.id = {5, 6};
        child.parent = {3, 4};
        child.sibling_order = 0;
        document.entities.push_back(std::move(child));
    }
    return document;
}

std::vector<std::byte> cooked_bytes(const arc::persistence::archive_document& document)
{
    const auto result = arc::persistence::write_tagged_binary(document, "windows-x64-vulkan12");
    REQUIRE(result);
    return result.value();
}

} // namespace

TEST_CASE("D1.7 cooked scene materializes persisted entity IDs, components and hierarchy")
{
    const auto data = cooked_bytes(authored_scene(true));
    arc::ecs::world world;
    CHECK(arc::scene::materialize_cooked_scene(world, data).empty());
    REQUIRE(world.live_count() == 2);

    arc::ecs::entity root;
    arc::ecs::entity child;
    for (auto entity : world.entities())
    {
        const auto id = world.get<arc::ecs::persistent_id_component>(entity).value;
        if (id == arc::ecs::entity_guid{3, 4}) root = entity;
        if (id == arc::ecs::entity_guid{5, 6}) child = entity;
    }
    REQUIRE(root.valid());
    REQUIRE(child.valid());
    CHECK(world.get<arc::scene::name_component>(root).value == "Root Entity");
    REQUIRE(world.has<arc::ecs::hierarchy_component>(child));
    CHECK(world.get<arc::ecs::hierarchy_component>(child).parent == root);
}

TEST_CASE("D1.7 cooked scene refuses damaged archive without creating ECS entities")
{
    auto data = cooked_bytes(authored_scene(false));
    REQUIRE_FALSE(data.empty());
    data.back() ^= std::byte{1};
    arc::ecs::world world;
    CHECK_FALSE(arc::scene::materialize_cooked_scene(world, data).empty());
    CHECK(world.live_count() == 0);
}

TEST_CASE("D1.7 cooked scene rolls back partially materialized worlds")
{
    auto document = authored_scene(true);
    arc::persistence::archive_component_record unsupported;
    unsupported.type = {0x11111111, 0x22222222};
    unsupported.name = "Unregistered Gameplay Component";
    document.entities.back().components.push_back(std::move(unsupported));

    const auto data = cooked_bytes(document);
    arc::ecs::world world;
    const auto error = arc::scene::materialize_cooked_scene(world, data);
    CHECK(error.find("Unregistered Gameplay Component") != std::string::npos);
    CHECK(world.live_count() == 0);
}

namespace
{
arc::persistence::archive_document authored_mesh_scene()
{
    auto document = authored_scene(false);
    arc::persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));

    arc::scene::mesh_renderer_component renderer{};
    renderer.visible = false;
    renderer.forced_lod = 2;
    renderer.mesh = arc::render::mesh_handle{3, 7};
    renderer.material = arc::render::material_handle{8, 9};
    arc::persistence::archive_component_record encoded;
    const auto component = arc::ecs::component_metadata<arc::scene::mesh_renderer_component>().id;
    REQUIRE(registry.encode(component, &renderer, encoded));
    document.entities.front().components.push_back(std::move(encoded));

    arc::persistence::archive_value subresource;
    subresource.kind = arc::persistence::archive_value_kind::string;
    subresource.string = "Body_LOD0";
    arc::persistence::archive_value binding;
    binding.kind = arc::persistence::archive_value_kind::object;
    binding.object.emplace_back("subresource", std::move(subresource));
    document.entities.front().extensions.kind = arc::persistence::archive_value_kind::object;
    document.entities.front().extensions.object.emplace_back("assetBinding", std::move(binding));

    arc::persistence::dependency_manifest_entry source;
    source.reference = {.guid = {20, 30},
                        .expected_type = arc::assets::asset_types::imported_scene,
                        .path_hint = "Content/vehicle.glb"};
    source.owner_entity = {3, 4};
    source.owner_component = component;
    source.owner_field = 1;
    source.required = true;
    document.dependencies.push_back(std::move(source));

    arc::persistence::dependency_manifest_entry material;
    material.reference.guid = {40, 50};
    material.reference.expected_type = arc::assets::asset_types::material;
    material.reference.path_hint = "Content/paint.arcmat";
    material.owner_entity = {3, 4};
    material.owner_component = component;
    material.owner_field = 2;
    material.required = false;
    document.dependencies.push_back(std::move(material));
    return document;
}
} // namespace

TEST_CASE("V3.4b cooked scene preserves mesh and material GUIDs and model subresource")
{
    auto document = authored_mesh_scene();
    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings;
    CHECK(arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings).empty());
    REQUIRE(world.live_count() == 1);
    REQUIRE(bindings.size() == 1);

    CHECK((bindings[0].entity == arc::ecs::entity_guid{3, 4}));
    CHECK((bindings[0].source.guid == arc::assets::asset_guid{20, 30}));
    CHECK(bindings[0].source.expected_type == arc::assets::asset_types::imported_scene);
    CHECK(bindings[0].source.path_hint == "Content/vehicle.glb");
    CHECK((bindings[0].material.guid == arc::assets::asset_guid{40, 50}));
    CHECK(bindings[0].material.expected_type == arc::assets::asset_types::material);
    CHECK(bindings[0].subresource == "Body_LOD0");

    const auto entity = world.entities().front();
    const auto& mesh = world.get<arc::scene::mesh_renderer_component>(entity);
    CHECK_FALSE(mesh.mesh.valid());
    CHECK_FALSE(mesh.material.valid());
    CHECK_FALSE(mesh.visible);
    CHECK(mesh.forced_lod == 2);
}

TEST_CASE("V3.4b cooked scene rejects duplicate mesh bindings and rolls back")
{
    auto document = authored_mesh_scene();
    document.dependencies.push_back(document.dependencies.front());
    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings{{.entity = {10, 11}}};
    const auto error = arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings);
    CHECK(error.find("duplicate") != std::string::npos);
    CHECK(world.live_count() == 0);
    CHECK(bindings.empty());
}

TEST_CASE("V3.4b cooked scene rejects missing mesh binding owners")
{
    auto document = authored_mesh_scene();
    document.dependencies.front().owner_entity = {99, 99};
    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings;
    const auto error = arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings);
    // Archive validation may reject the missing owner before ECS materialization.
    CHECK_FALSE(error.empty());
    CHECK(world.live_count() == 0);
    CHECK(bindings.empty());
}

TEST_CASE("V3.4b cooked scene rejects incompatible mesh asset types")
{
    auto document = authored_mesh_scene();
    document.dependencies.front().reference.expected_type = arc::assets::asset_types::texture_2d;
    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings;
    const auto error = arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings);
    CHECK(error.find("incompatible") != std::string::npos);
    CHECK(world.live_count() == 0);
    CHECK(bindings.empty());
}

TEST_CASE("V3.4b legacy scenes without mesh dependencies remain loadable")
{
    auto document = authored_mesh_scene();
    document.dependencies.clear();
    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings;
    CHECK(arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings).empty());
    CHECK(world.live_count() == 1);
    CHECK(bindings.empty());
}

TEST_CASE("Cooked scene retains renderer-owned primitive bindings")
{
    auto document = authored_mesh_scene();
    document.dependencies.clear();
    auto& asset_binding = document.entities.front().extensions.object.front().value.object;
    asset_binding.front().value.string = "Plane";
    arc::persistence::archive_value kind;
    kind.kind = arc::persistence::archive_value_kind::string;
    kind.string = "primitive";
    asset_binding.emplace_back("kind", std::move(kind));

    arc::ecs::world world;
    arc::scene::cooked_scene_mesh_bindings bindings;
    CHECK(arc::scene::materialize_cooked_scene(world, cooked_bytes(document), &bindings).empty());
    REQUIRE(bindings.size() == 1);
    CHECK(bindings.front().procedural_primitive);
    CHECK(bindings.front().subresource == "Plane");
    CHECK_FALSE(bindings.front().source.resolved());
}

TEST_CASE("Cooked default environment expands authored sky policy and light components")
{
    auto document = authored_scene(false);
    arc::persistence::archive_component_record environment;
    environment.type = arc::ecs::component_metadata<arc::scene::world_environment_component>().id;
    environment.name = "WorldEnvironment";
    environment.schema_version =
        arc::ecs::component_metadata<arc::scene::world_environment_component>().schema_version;
    arc::persistence::archive_value sky_visible;
    sky_visible.kind = arc::persistence::archive_value_kind::boolean;
    sky_visible.boolean = false;
    environment.fields.push_back({0, "skyVisible", std::move(sky_visible), false});
    document.entities.front().components.push_back(std::move(environment));

    arc::persistence::archive_component_record light;
    light.type = arc::ecs::component_metadata<arc::scene::directional_light_component>().id;
    light.name = "DirectionalLight";
    light.schema_version = arc::ecs::component_metadata<arc::scene::directional_light_component>().schema_version;
    arc::persistence::archive_value intensity;
    intensity.kind = arc::persistence::archive_value_kind::floating_point;
    intensity.floating_point = 85000.0;
    light.fields.push_back({0, "intensity", std::move(intensity), false});
    document.entities.front().components.push_back(std::move(light));

    arc::ecs::world world;
    REQUIRE(arc::scene::materialize_cooked_scene(world, cooked_bytes(document)).empty());
    const auto entity = world.entities().front();
    CHECK_FALSE(world.get<arc::scene::world_environment_component>(entity).sky_visible);
    CHECK(world.has<arc::scene::sky_atmosphere_component>(entity));
    CHECK(world.has<arc::scene::celestial_sky_component>(entity));
    CHECK(world.has<arc::scene::cloud_layers_component>(entity));
    CHECK(world.has<arc::scene::environment_lighting_component>(entity));
    CHECK(world.has<arc::scene::height_fog_component>(entity));
    CHECK(world.get<arc::scene::directional_light_component>(entity).intensity == 85000.0f);
}

TEST_CASE("V3.4b editor v4 MeshRenderer field names migrate to the runtime schema")
{
    auto document = authored_mesh_scene();
    auto& component = document.entities.front().components.back();
    component.schema_version = 4;
    auto field = std::find_if(component.fields.begin(), component.fields.end(),
                              [](const auto& value) { return value.name == "casts_shadows"; });
    REQUIRE(field != component.fields.end());
    field->name = "castsShadows";
    field->id = 0;
    field->known = false;

    arc::persistence::schema_migration_registry migrations;
    REQUIRE(arc::scene::register_persistence_migrations(migrations));
    const auto result = migrations.migrate(component, 5);
    REQUIRE(result);
    CHECK(component.schema_version == 5);
    CHECK(field->name == "casts_shadows");
    CHECK(field->id == 5u);
    CHECK(field->known);
}

TEST_CASE("Authored component names repair provisional stable type IDs before migration")
{
    auto document = authored_mesh_scene();
    document.dependencies.clear();
    auto& component = document.entities.front().components.back();
    component.type = *arc::ecs::parse_component_type_id("a7c0000000000001000000000000000b");
    component.name = "MeshRenderer";
    component.schema_version = 4;

    const auto json = arc::persistence::write_reflected_json(document, false);
    REQUIRE(json);
    arc::persistence::component_persistence_registry components;
    arc::persistence::schema_migration_registry migrations;
    REQUIRE(arc::scene::register_persistence_components(components));
    REQUIRE(arc::scene::register_persistence_migrations(migrations));
    const auto restored = arc::persistence::read_reflected_json(json.value(), components, &migrations);
    REQUIRE(restored.succeeded());
    REQUIRE(restored.document.entities.front().components.size() == 2);
    const auto repaired = std::find_if(restored.document.entities.front().components.begin(),
                                       restored.document.entities.front().components.end(),
                                       [](const auto& value) { return value.name == "MeshRenderer"; });
    REQUIRE(repaired != restored.document.entities.front().components.end());
    CHECK(repaired->type == arc::ecs::component_metadata<arc::scene::mesh_renderer_component>().id);
    CHECK(repaired->schema_version == 5);
}

TEST_CASE("Active value fields migrate to the canonical runtime schema")
{
    arc::persistence::archive_component_record component;
    component.type = arc::ecs::component_metadata<arc::scene::active_component>().id;
    component.name = "Active";
    component.schema_version = 1;
    arc::persistence::archive_value value;
    value.kind = arc::persistence::archive_value_kind::boolean;
    value.boolean = false;
    component.fields.push_back({0, "value", std::move(value), false});

    arc::persistence::schema_migration_registry migrations;
    REQUIRE(arc::scene::register_persistence_migrations(migrations));
    REQUIRE(migrations.migrate(component, 2));
    REQUIRE(component.fields.size() == 1);
    CHECK(component.schema_version == 2);
    CHECK(component.fields.front().name == "active");
    CHECK(component.fields.front().id == 1u);
    CHECK(component.fields.front().known);
}

TEST_CASE("Android device smoke Camera codec preserves projection exposure and rejects unknown fields")
{
    arc::persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));
    arc::scene::camera_component input;
    input.projection = arc::scene::camera_projection::orthographic;
    input.orthographic_height = 8.0f;
    input.clear_color = {0.2f, 0.3f, 0.4f, 1.0f};
    input.exposure.mode = arc::render::exposure_mode::manual;
    input.exposure.manual_ev100 = 12.0f;

    const auto id = arc::ecs::component_metadata<arc::scene::camera_component>().id;
    arc::persistence::archive_component_record encoded;
    REQUIRE(registry.encode(id, &input, encoded));
    arc::scene::camera_component restored;
    REQUIRE(registry.decode(id, &restored, encoded));
    CHECK(restored.projection == input.projection);
    CHECK(restored.orthographic_height == 8.0f);
    CHECK(restored.clear_color[1] == 0.3f);
    CHECK(restored.exposure.mode == arc::render::exposure_mode::manual);
    CHECK(restored.exposure.manual_ev100 == 12.0f);

    auto invalid = encoded;
    invalid.fields.push_back({99u, "unexpected", {}, true});
    CHECK_FALSE(registry.decode(id, &restored, invalid));
    CHECK(restored.exposure.manual_ev100 == 12.0f);
}

TEST_CASE("Authored Camera field names bind to stable cooked runtime fields")
{
    arc::persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));
    arc::scene::camera_component input;
    input.fov_y_radians = 0.9f;
    input.near_plane = 0.25f;
    input.far_plane = 800.0f;
    input.orthographic_height = 12.0f;
    input.clear_color = {0.15f, 0.25f, 0.35f, 1.0f};

    const auto id = arc::ecs::component_metadata<arc::scene::camera_component>().id;
    arc::persistence::archive_component_record authored;
    REQUIRE(registry.encode(id, &input, authored));
    for (auto& field : authored.fields)
    {
        field.known = false;
        field.id = 0;
        if (field.name == "fov_y_radians")
            field.name = "fovY";
        else if (field.name == "near_plane")
            field.name = "near";
        else if (field.name == "far_plane")
            field.name = "far";
        else if (field.name == "orthographic_height")
            field.name = "orthographicHeight";
        else if (field.name == "clear_color")
            field.name = "clearColor";
        else if (field.name == "anti_aliasing")
            field.name = "antiAliasing";
    }

    arc::scene::camera_component restored;
    REQUIRE(registry.decode(id, &restored, authored));
    CHECK(restored.fov_y_radians == input.fov_y_radians);
    CHECK(restored.near_plane == input.near_plane);
    CHECK(restored.far_plane == input.far_plane);
    CHECK(restored.orthographic_height == input.orthographic_height);
    CHECK(restored.clear_color[0] == input.clear_color[0]);
    CHECK(restored.clear_color[1] == input.clear_color[1]);
    CHECK(restored.clear_color[2] == input.clear_color[2]);
    CHECK(restored.clear_color[3] == input.clear_color[3]);
}

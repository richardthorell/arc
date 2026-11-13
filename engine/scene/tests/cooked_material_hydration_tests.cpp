#include <arc/scene/cooked_material_hydration.h>

#include <arc/ecs/hierarchy.h>
#include <arc/render/cooked_material.h>
#include <arc/render/material_pass.h>
#include <arc/scene/components.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <memory>

namespace
{
arc::render::cooked_material_metadata metadata()
{
    arc::render::cooked_material_metadata material;
    material.compiled.package = {.high = 9, .low = 10};
    arc::render::material_pass_binding binding;
    binding.pass = arc::render::material_pass::forward;
    binding.permutation = arc::render::shader_permutation_id{3};
    binding.entry_point = arc::render::shader_entry_point_id{4};
    binding.build_hash.bytes[0] = std::byte{0x5a};
    material.compiled.passes.push_back(binding);
    return material;
}

arc::render::cooked_material_shader shader()
{
    arc::render::cooked_material_shader cooked;
    cooked.artifact_name = "test.forward";
    auto& package = cooked.package;
    package.id = {.high = 9, .low = 10};
    package.generation = arc::render::shader_generation_id{1};
    package.permutation = arc::render::shader_permutation_id{3};
    package.target = arc::render::shader_target::spirv;
    package.compiled.bytecode = {1, 2, 3, 4};
    package.compiled.build_hash.bytes[0] = std::byte{0x5a};
    package.compiled.reflection.entry_points.push_back(
        {.id = arc::render::shader_entry_point_id{4}, .name = "main", .stage = arc::render::shader_stage::fragment});
    package.compiled.reflection.passes.push_back(
        {.pass = arc::render::material_pass::forward, .entry_point = arc::render::shader_entry_point_id{4}});
    return cooked;
}

arc::render::material_descriptor descriptor(const arc::render::material_runtime_program& program)
{
    arc::render::material_descriptor result;
    result.shader_package = {.high = 9, .low = 10};
    result.runtime_program = std::make_shared<arc::render::material_runtime_program>(program);
    result.render_path = arc::render::material_render_path::clustered_forward;
    return result;
}
} // namespace

TEST_CASE("V3.4d.3 validates cooked shader identities and builds executable material passes")
{
    const auto material = metadata();
    const std::array shaders{shader()};
    const auto result = arc::render::realize_cooked_material_program(material, shaders);
    REQUIRE(result);
    REQUIRE(result.value().passes.size() == 1);
    CHECK(result.value().passes.front().permutation == arc::render::shader_permutation_id{3});
    CHECK(result.value().passes.front().compiled.bytecode.size() == 4);

    auto mismatch = shader();
    mismatch.package.compiled.build_hash.bytes[0] ^= std::byte{1};
    const std::array corrupted{mismatch};
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, corrupted));

    auto wrong_entry = shader();
    wrong_entry.package.compiled.reflection.entry_points.front().id = arc::render::shader_entry_point_id{8};
    const std::array incompatible{wrong_entry};
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, incompatible));
}

TEST_CASE("V3.4d.3 binds shared material assets and unbinds stale ECS handles")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    const auto one = scene.create();
    const auto two = scene.create();
    scene.emplace<arc::ecs::persistent_id_component>(one, arc::ecs::entity_guid{1, 2});
    scene.emplace<arc::ecs::persistent_id_component>(two, arc::ecs::entity_guid{3, 4});
    scene.emplace<arc::scene::mesh_renderer_component>(one);
    scene.emplace<arc::scene::mesh_renderer_component>(two);
    const std::array shaders{shader()};
    const auto compiled = arc::render::realize_cooked_material_program(metadata(), shaders);
    REQUIRE(compiled);
    const auto upload = descriptor(compiled.value());
    std::vector<arc::scene::cooked_material_upload> uploads;
    uploads.push_back({.entity = {1, 2}, .material = {9, 10}, .descriptor = upload});
    uploads.push_back({.entity = {3, 4}, .material = {9, 10}, .descriptor = upload});
    arc::scene::cooked_material_residencies residency;
    CHECK(arc::scene::hydrate_cooked_scene_materials(scene, renderer, uploads, residency).empty());
    REQUIRE(residency.size() == 2);
    const auto handle = residency.front().handle;
    CHECK(renderer.material_alive(handle));
    CHECK(residency.back().handle == handle);
    CHECK(scene.get<arc::scene::mesh_renderer_component>(one).material == handle);
    CHECK(scene.get<arc::scene::mesh_renderer_component>(two).material == handle);
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_materials(scene, renderer, uploads, residency).empty());
    arc::scene::unbind_cooked_scene_materials(scene, residency);
    CHECK(residency.empty());
    CHECK_FALSE(scene.get<arc::scene::mesh_renderer_component>(one).material.valid());
    CHECK_FALSE(scene.get<arc::scene::mesh_renderer_component>(two).material.valid());
}

TEST_CASE("V3.4d.3 refuses invalid bindings without changing ECS")
{
    arc::ecs::world scene;
    arc::render::renderer renderer;
    auto entity = scene.create();
    scene.emplace<arc::ecs::persistent_id_component>(entity, arc::ecs::entity_guid{1, 2});
    scene.emplace<arc::scene::mesh_renderer_component>(entity);
    arc::scene::cooked_material_residencies residency;
    const std::array invalid{arc::scene::cooked_material_upload{.entity = {1, 2}, .material = {9, 10}}};
    CHECK_FALSE(arc::scene::hydrate_cooked_scene_materials(scene, renderer, invalid, residency).empty());
    CHECK(residency.empty());
    CHECK_FALSE(scene.get<arc::scene::mesh_renderer_component>(entity).material.valid());
}

TEST_CASE("V3.4d.6 retains authored scalar defaults from cooked shader reflection")
{
    auto material = metadata();
    arc::render::shader_parameter_descriptor roughness;
    roughness.id = arc::render::shader_parameter_id{11};
    roughness.name = "roughness";
    roughness.type = arc::render::shader_parameter_type::float32;
    roughness.offset = 0;
    roughness.size = 4;
    material.parameters.push_back(roughness);

    auto cooked = shader();
    cooked.package.compiled.reflection.parameter_block_size = 16;
    roughness.default_value = {std::byte{0x00}, std::byte{0x00}, std::byte{0x40}, std::byte{0x3f}};
    cooked.package.compiled.reflection.parameters.push_back(roughness);

    const std::array shaders{cooked};
    const auto realized = arc::render::realize_cooked_material_program(material, shaders);
    REQUIRE(realized);
    REQUIRE(realized.value().parameter_defaults.size() == 16);
    for (std::size_t index = 0; index < roughness.default_value.size(); ++index)
        CHECK(realized.value().parameter_defaults[index] == roughness.default_value[index]);
    for (std::size_t index = roughness.default_value.size(); index < 16; ++index)
        CHECK(realized.value().parameter_defaults[index] == std::byte{});
}

TEST_CASE("V3.4d.6 rejects incompatible parameter blocks and defaults before material allocation")
{
    auto material = metadata();
    arc::render::shader_parameter_descriptor parameter;
    parameter.id = arc::render::shader_parameter_id{11};
    parameter.name = "roughness";
    parameter.type = arc::render::shader_parameter_type::float32;
    parameter.size = 4;
    material.parameters.push_back(parameter);

    auto forward = shader();
    forward.package.compiled.reflection.parameter_block_size = 16;
    parameter.default_value = {std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0x3f}};
    forward.package.compiled.reflection.parameters.push_back(parameter);

    auto shadow = forward;
    shadow.artifact_name = "test.shadow";
    shadow.package.permutation = arc::render::shader_permutation_id{5};
    shadow.package.compiled.build_hash.bytes[0] = std::byte{0x44};
    shadow.package.compiled.reflection.passes.front().pass = arc::render::material_pass::shadow;
    material.compiled.passes.push_back({.pass = arc::render::material_pass::shadow,
                                        .permutation = arc::render::shader_permutation_id{5},
                                        .entry_point = arc::render::shader_entry_point_id{4},
                                        .build_hash = shadow.package.compiled.build_hash});

    std::array shaders{forward, shadow};
    REQUIRE(arc::render::realize_cooked_material_program(material, shaders));

    shaders[1].package.compiled.reflection.parameters.front().default_value[3] = std::byte{0x40};
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, shaders));
    shaders[1] = shadow;
    shaders[1].package.compiled.reflection.parameter_block_size = 32;
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, shaders));
    shaders[1] = shadow;
    shaders[1].package.compiled.reflection.parameters.front().offset = 14;
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, shaders));
    shaders[1] = shadow;
    shaders[1].package.compiled.reflection.parameters.front().default_value.push_back(std::byte{0});
    CHECK_FALSE(arc::render::realize_cooked_material_program(material, shaders));
}

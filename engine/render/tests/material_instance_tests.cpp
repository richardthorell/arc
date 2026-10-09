#include <arc/render/material_instance.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstring>

namespace
{
using namespace arc::render;

material_parameter_override make_override(std::uint64_t id, float value)
{
    return {shader_parameter_id{id}, "Parameter", value};
}
} // namespace

TEST_CASE("material instance requires a valid parent", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{7};

    REQUIRE(validate_material_instance(instance).valid());

    instance.parent = {};
    REQUIRE(validate_material_instance(instance).error == material_instance_validation_error::missing_parent);
}

TEST_CASE("material instance accepts unique parameter overrides", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{11};
    instance.overrides = {make_override(101, 0.25f), make_override(202, 0.75f)};

    REQUIRE(validate_material_instance(instance).valid());
}

TEST_CASE("material instance rejects invalid and duplicate parameter ids", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{13};
    instance.overrides = {make_override(0, 0.5f)};

    auto validation = validate_material_instance(instance);
    REQUIRE(validation.error == material_instance_validation_error::invalid_parameter_id);

    instance.overrides = {make_override(303, 0.25f), make_override(303, 0.75f)};
    validation = validate_material_instance(instance);
    REQUIRE(validation.error == material_instance_validation_error::duplicate_parameter_id);
    REQUIRE(validation.parameter_id == shader_parameter_id{303});
}

TEST_CASE("material instance override editing preserves stable parameter identity", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{17};
    instance.overrides = {make_override(101, 0.25f), make_override(202, 0.75f)};

    REQUIRE(is_material_instance_parameter_overridden(instance, shader_parameter_id{101}));
    REQUIRE_FALSE(is_material_instance_parameter_overridden(instance, shader_parameter_id{303}));

    set_material_instance_override(instance, make_override(101, 0.5f));
    REQUIRE(instance.overrides.size() == 2);
    REQUIRE(instance.overrides[0].id == shader_parameter_id{101});
    REQUIRE(std::get<float>(instance.overrides[0].value) == 0.5f);
    REQUIRE(instance.overrides[1].id == shader_parameter_id{202});

    set_material_instance_override(instance, make_override(303, 1.0f));
    REQUIRE(instance.overrides.size() == 3);
    REQUIRE(instance.overrides[2].id == shader_parameter_id{303});
    REQUIRE(find_material_instance_override(instance, shader_parameter_id{303}) != nullptr);

    REQUIRE(reset_material_instance_override(instance, shader_parameter_id{202}));
    REQUIRE_FALSE(reset_material_instance_override(instance, shader_parameter_id{202}));
    REQUIRE(instance.overrides.size() == 2);
    REQUIRE(instance.overrides[0].id == shader_parameter_id{101});
    REQUIRE(instance.overrides[1].id == shader_parameter_id{303});
    REQUIRE(validate_material_instance(instance).valid());
}

TEST_CASE("material instance can reset multiple overrides without reordering survivors", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{19};
    instance.overrides = {make_override(101, 0.1f), make_override(202, 0.2f), make_override(303, 0.3f),
                          make_override(404, 0.4f)};

    const std::array reset_ids{shader_parameter_id{303}, shader_parameter_id{101}, shader_parameter_id{303},
                               shader_parameter_id{999}, shader_parameter_id{}};
    REQUIRE(reset_material_instance_overrides(instance, reset_ids) == 2);
    REQUIRE(instance.overrides.size() == 2);
    REQUIRE(instance.overrides[0].id == shader_parameter_id{202});
    REQUIRE(instance.overrides[1].id == shader_parameter_id{404});
    REQUIRE(validate_material_instance(instance).valid());

    REQUIRE(reset_material_instance_overrides(instance, std::span<const shader_parameter_id>{}) == 0);
    REQUIRE(instance.overrides.size() == 2);
}

TEST_CASE("material instance specialization validates its runtime payload", "[render][material-instance]")
{
    material_instance_descriptor instance;
    instance.parent = material_handle{23};
    instance.function_specialization_key = 17;

    REQUIRE(validate_material_instance(instance).error == material_instance_validation_error::invalid_specialization);

    instance.specialized_runtime_program = std::make_shared<material_runtime_program>();
    instance.specialized_parameter_layout.push_back(
        {.id = shader_parameter_id{701}, .name = "Checker / Scale", .type = shader_parameter_type::float32});
    REQUIRE(validate_material_instance(instance).valid());
}

TEST_CASE("material instance applies texture overrides to runtime texture bindings", "[render][material-instance]")
{
    material_definition_descriptor parent;
    parent.material.handle = material_handle{29};
    auto runtime_program = std::make_shared<material_runtime_program>();
    runtime_program->texture_bindings.push_back(
        {.slot = 2, .parameter_id = shader_parameter_id{909}, .type = shader_parameter_type::texture_2d});
    parent.material.runtime_program = std::move(runtime_program);
    parent.material.runtime_textures.resize(3);
    parent.parameter_layout.push_back(
        {.id = shader_parameter_id{909}, .name = "Base Color Texture", .type = shader_parameter_type::texture_2d});

    material_instance_descriptor instance;
    instance.parent = material_handle{29};
    instance.overrides.push_back(
        {.id = shader_parameter_id{909}, .name = "Base Color Texture", .value = resource_handle{77, 3}});

    const auto resolved = resolve_material_instance(parent, instance);
    REQUIRE(resolved);
    REQUIRE(resolved.value().runtime_textures.size() == 3);
    CHECK(resolved.value().runtime_textures[2] == resource_handle{77, 3});
}

TEST_CASE("material instance resolves overrides against specialized reflected parameters",
          "[render][material-instance]")
{
    material_definition_descriptor parent;
    parent.material.handle = material_handle{31};
    parent.parameter_layout.push_back(
        {.id = shader_parameter_id{101}, .name = "Parent", .type = shader_parameter_type::float32});

    material_instance_descriptor instance;
    instance.parent = material_handle{31};
    instance.function_specialization_key = 33;
    instance.specialized_runtime_program = std::make_shared<material_runtime_program>();
    instance.specialized_parameter_layout.push_back(
        {.id = shader_parameter_id{701}, .name = "Checker / Scale", .type = shader_parameter_type::float32});
    instance.overrides = {make_override(701, 2.0f)};

    const auto resolved = resolve_material_instance(parent, instance);
    REQUIRE(resolved);
    REQUIRE(resolved.value().runtime_program == instance.specialized_runtime_program);
    REQUIRE(resolved.value().parameters.size() == 1);
    CHECK(resolved.value().parameters.front().id == shader_parameter_id{701});
    CHECK(std::get<float>(resolved.value().parameters.front().value) == 2.0f);
}

TEST_CASE("emissive overrides update both compiled and bindless material representations",
          "[render][material-instance][emissive]")
{
    material_definition_descriptor parent;
    parent.material.handle = material_handle{43};
    parent.material.name = "Standard Lit";
    auto runtime_program = std::make_shared<material_runtime_program>();
    runtime_program->parameter_block_size = 32u;
    runtime_program->parameter_defaults.resize(32u);
    runtime_program->parameters = {
        {.id = shader_parameter_id{801},
         .name = "Emissive Color",
         .type = shader_parameter_type::float4,
         .offset = 0,
         .size = 16},
        {.id = shader_parameter_id{802},
         .name = "Emissive Strength",
         .type = shader_parameter_type::float32,
         .offset = 16,
         .size = 4},
    };
    const std::array<float, 4> default_color{1.0f, 1.0f, 1.0f, 1.0f};
    std::memcpy(runtime_program->parameter_defaults.data(), default_color.data(), sizeof(default_color));
    parent.parameter_layout = runtime_program->parameters;
    parent.material.runtime_program = std::move(runtime_program);

    material_instance_descriptor instance;
    instance.parent = parent.material.handle;
    instance.overrides = {
        {.id = shader_parameter_id{801}, .name = "Emissive Color", .value = math::vector4f{0.0f, 0.0f, 1.0f, 1.0f}},
        {.id = shader_parameter_id{802}, .name = "Emissive Strength", .value = 5.557f},
    };
    const auto bright = resolve_material_instance(parent, instance);
    REQUIRE(bright);
    CHECK(bright.value().emissive_factor[0] == 0.0f);
    CHECK(bright.value().emissive_factor[1] == 0.0f);
    CHECK(bright.value().emissive_factor[2] == 1.0f);
    CHECK(bright.value().emissive_strength == 5.557f);
    REQUIRE(bright.value().runtime_program);
    float compiled_intensity{};
    std::memcpy(&compiled_intensity, bright.value().runtime_program->parameter_defaults.data() + 16,
                sizeof(compiled_intensity));
    CHECK(compiled_intensity == 5.557f);

    instance.overrides = {
        {.id = shader_parameter_id{802}, .name = "Emissive Strength", .value = 5.0f},
    };
    const auto strength_only = resolve_material_instance(parent, instance);
    REQUIRE(strength_only);
    CHECK(strength_only.value().emissive_factor[0] == 1.0f);
    CHECK(strength_only.value().emissive_factor[1] == 1.0f);
    CHECK(strength_only.value().emissive_factor[2] == 1.0f);
    CHECK(strength_only.value().emissive_strength == 5.0f);

    instance.overrides = {
        {.id = shader_parameter_id{801}, .name = "Emissive Color", .value = math::vector4f{0.0f, 0.0f, 1.0f, 1.0f}},
        {.id = shader_parameter_id{802}, .name = "Emissive Strength", .value = 0.0f},
    };
    const auto disabled = resolve_material_instance(parent, instance);
    REQUIRE(disabled);
    CHECK(disabled.value().emissive_strength == 0.0f);
}

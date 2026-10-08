#include <arc/render/material_instance.h>

#include <catch2/catch_test_macros.hpp>

#include <array>

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

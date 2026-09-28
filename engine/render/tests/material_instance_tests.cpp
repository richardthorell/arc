#include <arc/render/material_instance.h>

#include <catch2/catch_test_macros.hpp>

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

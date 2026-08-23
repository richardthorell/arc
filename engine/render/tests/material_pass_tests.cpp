#include <arc/render/material_pass.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("material pass permutations are stable and pass-specific")
{
    arc::render::material_descriptor material;
    material.normal_texture = {.index = 3, .generation = 1};
    material.alpha_mode = arc::render::material_alpha_mode::masked;
    material.double_sided = true;

    const auto gbuffer =
        arc::render::make_material_pass_permutation_key(material, arc::render::material_pass::gbuffer, 2, false);
    const auto gbuffer_again =
        arc::render::make_material_pass_permutation_key(material, arc::render::material_pass::gbuffer, 2, false);
    const auto shadow =
        arc::render::make_material_pass_permutation_key(material, arc::render::material_pass::shadow, 2, false);

    REQUIRE(gbuffer == gbuffer_again);
    REQUIRE(arc::render::hash_material_pass_permutation_key(gbuffer) ==
            arc::render::hash_material_pass_permutation_key(gbuffer_again));
    REQUIRE(arc::render::make_material_pass_permutation_id(gbuffer) ==
            arc::render::make_material_pass_permutation_id(gbuffer_again));
    REQUIRE(arc::render::make_material_pass_permutation_id(gbuffer) !=
            arc::render::make_material_pass_permutation_id(shadow));
    REQUIRE(gbuffer.evaluates_material);
    REQUIRE(gbuffer.writes_motion);
    REQUIRE(shadow.evaluates_material);
    REQUIRE_FALSE(shadow.writes_motion);
}

TEST_CASE("material pass eligibility follows alpha and render routing")
{
    arc::render::material_descriptor material;

    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::depth));
    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::shadow));
    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::gbuffer));
    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::forward));
    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::motion));
    REQUIRE_FALSE(arc::render::material_pass_evaluates_surface(arc::render::material_pass::depth,
                                                               arc::render::material_alpha_mode::opaque));

    material.alpha_mode = arc::render::material_alpha_mode::masked;
    REQUIRE(arc::render::material_pass_evaluates_surface(arc::render::material_pass::depth,
                                                         arc::render::material_alpha_mode::masked));
    REQUIRE(arc::render::material_pass_evaluates_surface(arc::render::material_pass::shadow,
                                                         arc::render::material_alpha_mode::masked));

    material.alpha_mode = arc::render::material_alpha_mode::blend;
    REQUIRE_FALSE(arc::render::material_supports_pass(material, arc::render::material_pass::depth));
    REQUIRE_FALSE(arc::render::material_supports_pass(material, arc::render::material_pass::shadow));
    REQUIRE_FALSE(arc::render::material_supports_pass(material, arc::render::material_pass::gbuffer));
    REQUIRE(arc::render::material_supports_pass(material, arc::render::material_pass::forward));
}

TEST_CASE("compiled material programs never silently fall back when a pass is unavailable")
{
    arc::render::material_compiled_program program;
    REQUIRE_FALSE(arc::render::material_program_supports_pass(program, arc::render::material_pass::gbuffer));

    program.package = {.high = 1, .low = 2};
    program.passes.push_back({.pass = arc::render::material_pass::gbuffer,
                              .permutation = {17},
                              .entry_point = arc::render::make_shader_entry_point_id(
                                  "arc_material_gbuffer", arc::render::shader_stage::fragment)});

    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::gbuffer));
    REQUIRE_FALSE(arc::render::material_program_supports_pass(program, arc::render::material_pass::shadow));

    program.contract_version = arc::render::material_pass_contract_version + 1;
    REQUIRE_FALSE(arc::render::material_program_supports_pass(program, arc::render::material_pass::gbuffer));
}

TEST_CASE("compiled routing accepts every raster material pass produced by the cooker")
{
    arc::render::material_compiled_program program;
    program.package = {.high = 5, .low = 6};

    const auto add_pass = [&](arc::render::material_pass pass, std::uint64_t permutation)
    {
        program.passes.push_back({.pass = pass,
                                  .permutation = {permutation},
                                  .entry_point = arc::render::make_shader_entry_point_id(
                                      "arc_material_pass", arc::render::shader_stage::fragment)});
    };
    add_pass(arc::render::material_pass::depth, 1);
    add_pass(arc::render::material_pass::shadow, 2);
    add_pass(arc::render::material_pass::gbuffer, 3);
    add_pass(arc::render::material_pass::forward, 4);
    add_pass(arc::render::material_pass::motion, 5);
    add_pass(arc::render::material_pass::object_id, 6);
    add_pass(arc::render::material_pass::selection, 7);

    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::depth));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::shadow));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::gbuffer));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::forward));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::motion));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::object_id));
    REQUIRE(arc::render::material_program_supports_pass(program, arc::render::material_pass::selection));
}

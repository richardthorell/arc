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

TEST_CASE("compiled materials are the default and never silently fall back to legacy")
{
    arc::render::material_descriptor material;
    REQUIRE(material.pipeline == arc::render::material_pipeline::compiled);

    const auto unavailable =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::gbuffer, nullptr);
    REQUIRE_FALSE(unavailable.use_legacy);
    REQUIRE_FALSE(unavailable.use_compiled);
    REQUIRE_FALSE(unavailable.compare);

    arc::render::material_compiled_program program;
    program.package = {.high = 1, .low = 2};
    program.passes.push_back({.pass = arc::render::material_pass::gbuffer,
                              .permutation = {17},
                              .entry_point = arc::render::make_shader_entry_point_id(
                                  "arc_material_gbuffer", arc::render::shader_stage::fragment)});

    const auto compiled =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::gbuffer, &program);
    REQUIRE_FALSE(compiled.use_legacy);
    REQUIRE(compiled.use_compiled);
    REQUIRE_FALSE(compiled.compare);

    const auto missing_shadow =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::shadow, &program);
    REQUIRE_FALSE(missing_shadow.use_legacy);
    REQUIRE_FALSE(missing_shadow.use_compiled);
    REQUIRE_FALSE(missing_shadow.compare);

    program.contract_version = arc::render::material_pass_contract_version + 1;
    const auto incompatible =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::gbuffer, &program);
    REQUIRE_FALSE(incompatible.use_legacy);
    REQUIRE_FALSE(incompatible.use_compiled);
    REQUIRE_FALSE(incompatible.compare);
}

TEST_CASE("explicit legacy and compare modes remain available during migration")
{
    arc::render::material_descriptor material;
    arc::render::material_compiled_program program;
    program.package = {.high = 3, .low = 4};
    program.passes.push_back({.pass = arc::render::material_pass::forward,
                              .permutation = {23},
                              .entry_point = arc::render::make_shader_entry_point_id(
                                  "arc_material_forward", arc::render::shader_stage::fragment)});

    material.pipeline = arc::render::material_pipeline::legacy;
    const auto legacy = arc::render::resolve_material_pipeline(material, arc::render::material_pass::forward, &program);
    REQUIRE(legacy.use_legacy);
    REQUIRE_FALSE(legacy.use_compiled);
    REQUIRE_FALSE(legacy.compare);

    material.pipeline = arc::render::material_pipeline::compare;
    const auto compare =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::forward, &program);
    REQUIRE(compare.use_legacy);
    REQUIRE(compare.use_compiled);
    REQUIRE(compare.compare);

    const auto compare_missing =
        arc::render::resolve_material_pipeline(material, arc::render::material_pass::shadow, &program);
    REQUIRE(compare_missing.use_legacy);
    REQUIRE_FALSE(compare_missing.use_compiled);
    REQUIRE_FALSE(compare_missing.compare);
}

TEST_CASE("compiled routing accepts every raster material pass produced by the cooker")
{
    arc::render::material_descriptor material;
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

    const auto require_compiled = [&](arc::render::material_pass pass)
    {
        const auto resolved = arc::render::resolve_material_pipeline(material, pass, &program);
        REQUIRE_FALSE(resolved.use_legacy);
        REQUIRE(resolved.use_compiled);
        REQUIRE_FALSE(resolved.compare);
    };
    require_compiled(arc::render::material_pass::depth);
    require_compiled(arc::render::material_pass::shadow);
    require_compiled(arc::render::material_pass::gbuffer);
    require_compiled(arc::render::material_pass::forward);
    require_compiled(arc::render::material_pass::motion);
    require_compiled(arc::render::material_pass::object_id);
    require_compiled(arc::render::material_pass::selection);
}

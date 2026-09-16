#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <memory>
#include <vector>

#if !defined(ARC_RENDER_TEST_ASSET_ROOT)
#define ARC_RENDER_TEST_ASSET_ROOT "assets"
#endif

#include "render_test_support.h"

using arc::render::tests::counting_shader_compiler;

TEST_CASE("shader library cache reuses unchanged source requests")
{
    const auto path = std::filesystem::temp_directory_path() / "arc_shader_cache_test.slang";
    {
        std::ofstream file(path);
        file << "float4 main() : SV_Target { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = path.string(),
                                                .entry_point = "main",
                                                .profile = "fragment",
                                                .target = arc::render::shader_target::spirv};

    const auto first = cache.compile_or_get(compiler, request);
    const auto second = cache.compile_or_get(compiler, request);

    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(compiler.count == 1);
    REQUIRE(cache.size() == 1);
    REQUIRE_FALSE(cache.source_changed(request));

    std::filesystem::remove(path);
}

TEST_CASE("shader cache invalidates when an include changes")
{
    const auto directory = std::filesystem::temp_directory_path() / "arc_shader_include_cache_test";
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "main.slang";
    const auto include_path = directory / "common.slang";
    {
        std::ofstream source(source_path);
        source << "#include \"common.slang\"\nfloat4 main() : SV_Target { return color(); }";
        std::ofstream include(include_path);
        include << "float4 color() { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = source_path.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .include_directories = {directory}};
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE_FALSE(cache.source_changed(request));
    {
        std::ofstream include(include_path, std::ios::trunc);
        include << "float4 color() { return 0; }";
    }
    REQUIRE(cache.source_changed(request));
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 2);
    std::filesystem::remove_all(directory);
}

TEST_CASE("shader cache tracks Slang module imports and shader library versions")
{
    const auto directory = std::filesystem::temp_directory_path() / "arc_shader_import_cache_test";
    std::filesystem::create_directories(directory);
    const auto source_path = directory / "main.slang";
    const auto module_path = directory / "arc_surface.slang";
    {
        std::ofstream source(source_path);
        source << "import arc.surface;\nfloat4 main() : SV_Target { return arcColor(); }";
        std::ofstream module(module_path);
        module << "module arc.surface;\nfloat4 arcColor() { return 1; }";
    }

    counting_shader_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = source_path.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .include_directories = {directory}};
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE_FALSE(cache.source_changed(request));
    {
        std::ofstream module(module_path, std::ios::trunc);
        module << "module arc.surface;\nfloat4 arcColor() { return 0; }";
    }
    REQUIRE(cache.source_changed(request));
    REQUIRE(cache.compile_or_get(compiler, request));

    request.library_version = "arc-shader-library/2";
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 3);
    std::filesystem::remove_all(directory);
}

TEST_CASE("shader packages round trip reflection and reject corruption")
{
    arc::render::shader_package package{
        .id = {.high = 1, .low = 2},
        .generation = {3},
        .target = arc::render::shader_target::spirv,
        .permutation = {4},
        .compiled = {.bytecode = {3, 2, 35, 7},
                     .reflection = {.domain = arc::render::shader_domain::surface,
                                    .entry_points = {{.id = arc::render::make_shader_entry_point_id(
                                                          "main", arc::render::shader_stage::fragment),
                                                      .name = "main",
                                                      .stage = arc::render::shader_stage::fragment,
                                                      .profile = "spirv_1_5"}},
                                    .parameters = {{.id = arc::render::make_shader_parameter_id("baseColor"),
                                                    .name = "baseColor",
                                                    .type = arc::render::shader_parameter_type::float4,
                                                    .size = 16}},
                                    .passes = {{.pass = arc::render::material_pass::forward,
                                                .entry_point = arc::render::make_shader_entry_point_id(
                                                    "main", arc::render::shader_stage::fragment)}}},
                     .build_hash = {.bytes = {std::byte{1}}},
                     .dependencies = {{.path = "main.slang", .content_hash = {.bytes = {std::byte{2}}}}},
                     .source_map = {{.generated_line = 12,
                                     .source = {.path = "material.arcmat", .graph_node_id = "base-color"}}},
                     .diagnostics = {{.severity = arc::render::shader_diagnostic_severity::warning,
                                      .code = "W001",
                                      .message = "test warning",
                                      .location = {.path = "main.slang", .line = 4, .column = 2},
                                      .include_stack = {{.path = "shared.slang", .line = 1, .column = 1}},
                                      .permutation = arc::render::shader_permutation_id{4}}},
                     .compiler_fingerprint = "slang/2026.14.1"}};
    const auto encoded = arc::render::serialize_shader_package(package);
    REQUIRE(encoded);
    const auto decoded = arc::render::deserialize_shader_package(encoded.value());
    REQUIRE(decoded);
    REQUIRE(decoded.value().id == package.id);
    REQUIRE(decoded.value().compiled.bytecode == package.compiled.bytecode);
    REQUIRE(decoded.value().compiled.reflection.parameters.front().name == "baseColor");
    REQUIRE(decoded.value().compiled.source_map.front().source.graph_node_id == "base-color");
    REQUIRE(decoded.value().compiled.diagnostics.front().include_stack.front().path == "shared.slang");

    auto corrupt = encoded.value();
    corrupt.pop_back();
    REQUIRE_FALSE(arc::render::deserialize_shader_package(corrupt));
}

TEST_CASE("shader package publication preserves last good generations")
{
    arc::render::shader_package_library library;
    arc::render::shader_package first{
        .id = {.high = 7, .low = 9},
        .generation = {1},
        .permutation = {4},
        .compiled = {.bytecode = {1, 2, 3}},
    };
    first.compiled.build_hash.bytes[0] = std::byte{1};
    REQUIRE(library.publish(first, 3) == arc::render::shader_publication_status::published);

    auto conflicting = first;
    conflicting.compiled.build_hash.bytes[0] = std::byte{9};
    REQUIRE(library.publish(std::move(conflicting), 3) ==
            arc::render::shader_publication_status::rejected_stale_generation);

    auto second = first;
    second.generation = arc::render::shader_generation_id{2};
    second.compiled.build_hash.bytes[0] = std::byte{2};
    REQUIRE(library.publish(second, 8) == arc::render::shader_publication_status::published);
    REQUIRE(library.retired_count() == 1);
    REQUIRE(library.find(first.id, first.permutation)->generation == second.generation);

    library.report_failure(
        first.id, first.permutation,
        {.code = arc::render::shader_compile_error_code::compilation_failed, .message = "transient edit failed"});
    REQUIRE(library.snapshot(first.id, first.permutation)->last_error.has_value());
    REQUIRE(library.find(first.id, first.permutation)->generation == second.generation);

    library.collect(7);
    REQUIRE(library.retired_count() == 1);
    library.collect(8);
    REQUIRE(library.retired_count() == 0);
    REQUIRE(library.publish(first, 9) == arc::render::shader_publication_status::rejected_stale_generation);
}

TEST_CASE("material instances validate stable parameter overrides without changing permutations")
{
    const auto tint = arc::render::make_shader_parameter_id("tint");
    arc::render::material_definition_descriptor definition{
        .material = {.name = "Base", .shader_permutation = {99}},
        .parameter_layout = {
            {.id = tint, .name = "tint", .type = arc::render::shader_parameter_type::float3, .size = 12}}};
    arc::render::material_instance_descriptor instance{
        .parent = {.index = 1, .generation = 1},
        .name = "Blue",
        .overrides = {{.id = tint, .name = "tint", .value = arc::math::vector3f{0.1f, 0.2f, 1.0f}}}};
    const auto resolved = arc::render::resolve_material_instance(definition, instance);
    REQUIRE(resolved);
    REQUIRE(resolved.value().name == "Blue");
    REQUIRE(resolved.value().shader_permutation == definition.material.shader_permutation);
    REQUIRE(resolved.value().parameters.size() == 1);

    instance.overrides.front().value = 1.0f;
    const auto invalid = arc::render::resolve_material_instance(definition, instance);
    REQUIRE_FALSE(invalid);
    REQUIRE(invalid.error().code == arc::render::material_instance_error_code::incompatible_type);
}

TEST_CASE("material routing keeps standard surfaces deferred and custom contracts forward")
{
    arc::render::material_descriptor material;
    REQUIRE(arc::render::resolve_material_render_path(material) == arc::render::material_render_path::deferred);
    material.deferred_compatible = false;
    REQUIRE(arc::render::resolve_material_render_path(material) ==
            arc::render::material_render_path::clustered_forward);
    material.deferred_compatible = true;
    material.shading_model = arc::render::material_shading_model::unlit;
    REQUIRE(arc::render::resolve_material_render_path(material) ==
            arc::render::material_render_path::clustered_forward);
}

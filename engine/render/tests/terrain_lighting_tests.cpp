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

TEST_CASE("HZB min max reduction is conservative for odd extents")
{
    using namespace arc::render;
    REQUIRE(hzb_mip_count(1, 1) == 1);
    REQUIRE(hzb_mip_count(7, 3) == 3);
    REQUIRE(hzb_mip_count(1920, 1080) == 11);

    const auto reduced =
        reduce_hzb_depth(reduce_hzb_depth({0.2f, 0.8f}, {0.1f, 0.7f}), reduce_hzb_depth({0.4f, 0.9f}, {0.3f, 0.6f}));
    REQUIRE(reduced.nearest == Catch::Approx(0.1f));
    REQUIRE(reduced.farthest == Catch::Approx(0.9f));
}

TEST_CASE("anti aliasing policy resolves path aware auto and explicit fallbacks")
{
    using namespace arc::render;
    render_capabilities capabilities{};
    capabilities.fxaa = true;
    capabilities.temporal_resolve = true;
    capabilities.temporal_upscale = true;

    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::forward_plus, 1.0f, capabilities) ==
            anti_aliasing_method::fxaa);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::taa);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 0.75f, capabilities) ==
            anti_aliasing_method::taau);
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::disabled, render_path::deferred, 0.5f, capabilities) ==
            anti_aliasing_method::disabled);

    capabilities.temporal_upscale = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::taau, render_path::deferred, 0.5f, capabilities) ==
            anti_aliasing_method::taa);
    capabilities.temporal_resolve = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::taa, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::fxaa);
    capabilities.fxaa = false;
    REQUIRE(resolve_anti_aliasing(anti_aliasing_method::auto_select, render_path::deferred, 1.0f, capabilities) ==
            anti_aliasing_method::disabled);
}

TEST_CASE("terrain hierarchy is deterministic monotonic and incrementally updated")
{
    constexpr std::uint32_t resolution = 65u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] =
                std::sin(static_cast<float>(x) * 0.11f) * std::cos(static_cast<float>(z) * 0.07f);
    const auto first = arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 16u});
    auto second = arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 16u});
    REQUIRE(first.nodes.size() == second.nodes.size());
    REQUIRE(first.leaf_count == 16u);
    REQUIRE(first.nodes[first.root].geometric_error == Catch::Approx(second.nodes[second.root].geometric_error));
    for (const auto& node : first.nodes)
        for (const auto child : node.children)
            if (child != arc::render::invalid_terrain_node)
                REQUIRE(node.geometric_error >= first.nodes[child].geometric_error);

    const auto root_before = second.nodes[second.root].maximum_height;
    heights[32u * resolution + 32u] += 20.0f;
    REQUIRE(arc::render::update_terrain_hierarchy(second, heights, resolution, 64.0f, 64.0f, {31u, 31u, 33u, 33u},
                                                  {.patch_quads = 16u}));
    REQUIRE(second.nodes[second.root].maximum_height > root_before);
}

TEST_CASE("terrain stitched topology variants remain valid and deterministic")
{
    for (std::uint8_t mask = 0u; mask < 16u; ++mask)
    {
        const auto first = arc::render::make_terrain_patch_indices(32u, mask);
        const auto second = arc::render::make_terrain_patch_indices(32u, mask);
        REQUIRE(first == second);
        REQUIRE_FALSE(first.empty());
        REQUIRE(first.size() % 3u == 0u);
        for (const auto index : first)
            REQUIRE(index < 33u * 33u);
        for (std::size_t triangle = 0; triangle < first.size(); triangle += 3u)
        {
            REQUIRE(first[triangle] != first[triangle + 1u]);
            REQUIRE(first[triangle + 1u] != first[triangle + 2u]);
            REQUIRE(first[triangle] != first[triangle + 2u]);
        }
    }
}

TEST_CASE("terrain GPU hierarchy packing preserves deterministic std430 records")
{
    constexpr std::uint32_t resolution = 65u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] = static_cast<float>(x + z) * 0.125f;
    const auto hierarchy =
        arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 16u});
    const auto first = arc::render::make_terrain_gpu_hierarchy(hierarchy);
    const auto second = arc::render::make_terrain_gpu_hierarchy(hierarchy);
    REQUIRE(first.valid());
    REQUIRE(first.root == hierarchy.root);
    REQUIRE(first.nodes.size() == hierarchy.nodes.size());
    REQUIRE(std::memcmp(first.nodes.data(), second.nodes.data(),
                        first.nodes.size() * sizeof(arc::render::gpu_terrain_node_record)) == 0);
    const auto& source = hierarchy.nodes[first.root];
    const auto& packed = first.nodes[first.root];
    REQUIRE(packed.samples[0] == source.samples.min_x);
    REQUIRE(packed.samples[3] == source.samples.max_z);
    REQUIRE(packed.bounds_min[3] == Catch::Approx(source.minimum_height));
    REQUIRE(packed.bounds_max[3] == Catch::Approx(source.maximum_height));
    REQUIRE(packed.leaf == (source.leaf() ? 1u : 0u));
}

TEST_CASE("bounded terrain traversal discards partial output on overflow")
{
    constexpr std::uint32_t resolution = 65u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] =
                std::sin(static_cast<float>(x) * 0.31f) * std::cos(static_cast<float>(z) * 0.27f);
    const auto hierarchy = arc::render::build_terrain_hierarchy(heights, resolution, 64.0f, 64.0f, {.patch_quads = 8u});
    arc::render::render_camera camera;
    camera.view_projection = arc::math::identity<float, 4>();
    camera.render_width = 1920u;
    camera.render_height = 1080u;
    const auto unbounded = arc::render::select_terrain_patches({.index = 1u, .generation = 1u}, hierarchy,
                                                               arc::math::identity<float, 4>(), camera, 0.0f);
    REQUIRE(unbounded.patches.size() > 1u);
    const auto overflow = arc::render::select_terrain_patches_bounded(
        {.index = 1u, .generation = 1u}, hierarchy, arc::math::identity<float, 4>(), camera, 0.0f, 1u);
    REQUIRE(overflow.overflowed);
    REQUIRE(overflow.use_conventional_fallback);
    REQUIRE(overflow.selection.patches.empty());
    const auto complete = arc::render::select_terrain_patches_bounded(
        {.index = 1u, .generation = 1u}, hierarchy, arc::math::identity<float, 4>(), camera, 0.0f,
        static_cast<std::uint32_t>(unbounded.patches.size()));
    REQUIRE_FALSE(complete.overflowed);
    REQUIRE(complete.selection.patches.size() == unbounded.patches.size());
}

TEST_CASE("terrain selection responds to projected error and balances neighboring LODs")
{
    constexpr std::uint32_t resolution = 129u;
    std::vector<float> heights(static_cast<std::size_t>(resolution) * resolution);
    for (std::uint32_t z = 0; z < resolution; ++z)
        for (std::uint32_t x = 0; x < resolution; ++x)
            heights[static_cast<std::size_t>(z) * resolution + x] =
                5.0f * std::sin(static_cast<float>(x) * 0.17f) * std::cos(static_cast<float>(z) * 0.13f);
    const auto hierarchy =
        arc::render::build_terrain_hierarchy(heights, resolution, 128.0f, 128.0f, {.patch_quads = 16u});
    arc::render::render_camera camera;
    camera.position = {0.0f, 5.0f, 96.0f};
    camera.render_width = 1920u;
    camera.render_height = 1080u;
    const float near_plane = 0.1f;
    const float far_plane = 500.0f;
    const float inverse_tangent = 1.0f / std::tan(arc::math::to_radians(60.0f) * 0.5f);
    camera.projection = {};
    camera.projection(0, 0) = inverse_tangent / (16.0f / 9.0f);
    camera.projection(1, 1) = inverse_tangent;
    camera.projection(2, 2) = far_plane / (near_plane - far_plane);
    camera.projection(2, 3) = far_plane * near_plane / (near_plane - far_plane);
    camera.projection(3, 2) = -1.0f;
    camera.view = arc::math::identity<float, 4>();
    camera.view(0, 3) = -camera.position[0];
    camera.view(1, 3) = -camera.position[1];
    camera.view(2, 3) = -camera.position[2];
    camera.view_projection = arc::math::matmul(camera.projection, camera.view);
    arc::render::terrain_selection_scratch scratch;
    const auto detailed = arc::render::select_terrain_patches({1u, 1u}, hierarchy, arc::math::identity<float, 4>(),
                                                              camera, 0.25f, 1.0f, &scratch);
    const auto coarse =
        arc::render::select_terrain_patches({1u, 1u}, hierarchy, arc::math::identity<float, 4>(), camera, 10000.0f);
    REQUIRE(detailed.patches.size() > coarse.patches.size());
    REQUIRE(detailed.statistics.rendered_triangles > coarse.statistics.rendered_triangles);
    for (std::size_t a = 0; a < detailed.patches.size(); ++a)
        for (std::size_t b = a + 1u; b < detailed.patches.size(); ++b)
        {
            const auto& left = detailed.patches[a];
            const auto& right = detailed.patches[b];
            const bool vertical =
                (left.samples.max_x == right.samples.min_x || right.samples.max_x == left.samples.min_x) &&
                left.samples.min_z < right.samples.max_z && right.samples.min_z < left.samples.max_z;
            const bool horizontal =
                (left.samples.max_z == right.samples.min_z || right.samples.max_z == left.samples.min_z) &&
                left.samples.min_x < right.samples.max_x && right.samples.min_x < left.samples.max_x;
            if (vertical || horizontal)
                REQUIRE(std::abs(static_cast<int>(left.lod) - static_cast<int>(right.lod)) <= 1);
        }
}

TEST_CASE("lighting scene emits precise incremental updates and rejects stale world generations")
{
    using namespace arc::render;
    lighting_scene scene;
    lighting_scene_instance instance{.stable_id = 42,
                                     .geometry = {2, 1},
                                     .material = {3, 1},
                                     .world_bounds = {{-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}},
                                     .transform_revision = 1,
                                     .material_revision = 1};
    auto update = scene.synchronize(7, 1, 1, std::span(&instance, 1));
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::upsert);
    REQUIRE(update.updates.front().geometry_dirty);

    update = scene.synchronize(7, 1, 2, std::span(&instance, 1));
    REQUIRE(update.updates.empty());

    instance.transform_revision = 2;
    update = scene.synchronize(7, 1, 3, std::span(&instance, 1));
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().transform_dirty);
    REQUIRE_FALSE(update.updates.front().material_dirty);

    update = scene.synchronize(7, 2, 4, std::span(&instance, 1));
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::reset);
    REQUIRE(scene.snapshot().world_epoch == 2);

    update = scene.synchronize(7, 2, 5, {});
    REQUIRE(update.updates.size() == 1);
    REQUIRE(update.updates.front().kind == lighting_scene_update_kind::destroy);
}

TEST_CASE("dynamic indirect lighting graph selects the resolved screen software and hardware hierarchy")
{
    using namespace arc::render;
    renderer_config renderer_settings;
    renderer_settings.quality = render_quality_tier::ultra;
    render_capabilities capabilities;
    capabilities.compute_shaders = true;
    capabilities.storage_buffers = true;
    capabilities.storage_images = true;
    capabilities.hzb_occlusion = true;
    capabilities.temporal_resolve = true;
    capabilities.screen_space_indirect_lighting = true;
    capabilities.surface_cache = true;
    capabilities.radiance_cache = true;
    capabilities.software_ray_tracing = true;
    capabilities.hardware_ray_query = true;
    capabilities.ray_tracing = true;
    const auto config = resolve_render_config(renderer_settings, capabilities);
    REQUIRE(config.indirect_lighting_path == lighting_trace_path::hybrid_hardware);

    world_environment_data environment;
    environment.enabled = true;
    environment.indirect_lighting.enabled = true;
    environment.indirect_lighting.method = indirect_lighting_method::auto_select;
    const auto graph = make_scene_draw_graph("gi-test", config, false, environment).compile().value();
    const auto contains = [&](builtin_render_pass pass)
    {
        return std::ranges::any_of(graph.passes,
                                   [pass](const compiled_render_pass& candidate) { return candidate.builtin == pass; });
    };
    REQUIRE(contains(builtin_render_pass::screen_space_gi));
    REQUIRE(contains(builtin_render_pass::software_gi_trace));
    REQUIRE(contains(builtin_render_pass::hardware_gi_trace));
    REQUIRE(contains(builtin_render_pass::screen_space_reflections));
    REQUIRE(contains(builtin_render_pass::software_reflections));
    REQUIRE(contains(builtin_render_pass::hardware_reflections));
    REQUIRE(contains(builtin_render_pass::indirect_lighting_temporal));
    REQUIRE(contains(builtin_render_pass::reflection_temporal));
    REQUIRE(contains(builtin_render_pass::indirect_lighting_composite));
}

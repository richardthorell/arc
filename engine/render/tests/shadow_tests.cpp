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

TEST_CASE("directional shadow cascade splits are deterministic and ordered")
{
    const auto splits = arc::render::cascade_splits(0.1f, 100.0f, 0.65f);

    REQUIRE(splits[0] > 0.1f);
    REQUIRE(splits[0] < splits[1]);
    REQUIRE(splits[1] < splits[2]);
    REQUIRE(splits[2] < splits[3]);
    REQUIRE(splits[3] == Catch::Approx(100.0f));
}

TEST_CASE("stable directional shadow fitting produces ordered blended cascades")
{
    arc::render::directional_shadow_camera camera{};
    camera.near_plane = 0.1f;
    camera.far_plane = 1000.0f;
    camera.inverse_view_projection = arc::math::identity<float, 4>();
    arc::render::directional_shadow_settings settings{};
    settings.cascade_count = 4;
    settings.maximum_distance = 200.0f;
    settings.blend_fraction = 0.1f;

    const auto first = arc::render::fit_directional_shadow_cascades(camera, {0.35f, -0.85f, -0.4f}, settings, 2048);
    const auto second = arc::render::fit_directional_shadow_cascades(camera, {0.35f, -0.85f, -0.4f}, settings, 2048);

    REQUIRE(first.cascade_count == 4);
    REQUIRE(first.cascades[3].split_depth == Catch::Approx(200.0f));
    for (std::uint32_t index = 0; index < first.cascade_count; ++index)
    {
        const auto& cascade = first.cascades[index];
        REQUIRE(cascade.radius > 0.0f);
        REQUIRE(cascade.texel_world_size > 0.0f);
        REQUIRE(cascade.blend_start_depth < cascade.split_depth);
        REQUIRE(std::memcmp(cascade.light_view_projection.data(), second.cascades[index].light_view_projection.data(),
                            sizeof(float) * 16u) == 0);
        if (index > 0) REQUIRE(cascade.split_depth > first.cascades[index - 1].split_depth);

        const auto light_direction = arc::math::normalize(arc::math::vector3f{0.35f, -0.85f, -0.4f});
        const auto light_right =
            arc::math::normalize(arc::math::cross(light_direction, arc::math::vector3f{0.0f, 1.0f, 0.0f}));
        const auto light_up = arc::math::cross(light_right, light_direction);
        const float snapped_x = arc::math::dot(light_right, cascade.center) / cascade.texel_world_size;
        const float snapped_y = arc::math::dot(light_up, cascade.center) / cascade.texel_world_size;
        REQUIRE(snapped_x == Catch::Approx(std::round(snapped_x)).margin(0.0001f));
        REQUIRE(snapped_y == Catch::Approx(std::round(snapped_y)).margin(0.0001f));
    }
}

TEST_CASE("shadow atlas allocates point faces atomically and invalidates released handles")
{
    arc::render::shadow_atlas_allocator atlas(2048, 128, 2);
    const auto point = atlas.allocate({.kind = arc::render::shadow_light_kind::point,
                                       .light_key = 42,
                                       .requested_resolution = 512,
                                       .minimum_resolution = 256,
                                       .priority = 200,
                                       .frame_index = 1});
    REQUIRE(point);
    REQUIRE(point->face_count == arc::render::point_shadow_face_count);
    REQUIRE(point->resolved_resolution == 512);
    for (const auto& face : point->faces)
        REQUIRE(face.valid());

    const auto handle = point->handle;
    REQUIRE(atlas.find(handle) != nullptr);
    REQUIRE(atlas.release(handle));
    REQUIRE(atlas.find(handle) == nullptr);
    REQUIRE_FALSE(atlas.release(handle));
}

TEST_CASE("shadow atlas reduces resolution and evicts lower priority allocations")
{
    arc::render::shadow_atlas_allocator atlas(512, 128, 2);
    for (std::uint64_t light = 1; light <= 4; ++light)
    {
        REQUIRE(atlas.allocate({.kind = arc::render::shadow_light_kind::spot,
                                .light_key = light,
                                .requested_resolution = 240,
                                .minimum_resolution = 120,
                                .priority = 10,
                                .frame_index = light}));
    }
    const auto important = atlas.allocate({.kind = arc::render::shadow_light_kind::spot,
                                           .light_key = 99,
                                           .requested_resolution = 240,
                                           .minimum_resolution = 120,
                                           .priority = 250,
                                           .frame_index = 10});
    REQUIRE(important);
    REQUIRE(atlas.statistics().eviction_count >= 1);
}

TEST_CASE("virtual shadow address spaces normalize light topology and invalidate generations")
{
    arc::render::virtual_shadow_cache cache(16ull * 1024ull * 1024ull);
    const auto directional = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::directional,
                                                         .light_key = 11,
                                                         .level_count = 2,
                                                         .face_count = 3});
    REQUIRE(directional);
    const auto* directional_descriptor = cache.address_space(*directional);
    REQUIRE(directional_descriptor != nullptr);
    REQUIRE(directional_descriptor->level_count == arc::render::virtual_shadow_directional_clip_levels);
    REQUIRE(directional_descriptor->face_count == 1);

    const auto point = cache.create_address_space(
        {.light_kind = arc::render::shadow_light_kind::point, .light_key = 12, .level_count = 4});
    REQUIRE(point);
    REQUIRE(cache.address_space(*point)->face_count == arc::render::point_shadow_face_count);

    const auto stale = *directional;
    REQUIRE(cache.destroy_address_space(*directional));
    REQUIRE(cache.address_space(stale) == nullptr);
    const auto replacement =
        cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot, .light_key = 13});
    REQUIRE(replacement);
    REQUIRE(replacement->index == stale.index);
    REQUIRE(replacement->generation != stale.generation);
}

TEST_CASE("virtual shadow page requests are deterministic and retain coarse fallback")
{
    constexpr std::uint64_t one_d16_page_pair =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_page_texels +
                                   arc::render::virtual_shadow_page_guard_texels * 2u) *
        (arc::render::virtual_shadow_page_texels + arc::render::virtual_shadow_page_guard_texels * 2u) * 4u;
    arc::render::virtual_shadow_cache cache(one_d16_page_pair * 4u);
    const auto light = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .light_key = 42,
                                                   .virtual_resolution = 2048,
                                                   .level_count = 5});
    REQUIRE(light);
    const arc::render::virtual_shadow_page_key root{.address_space = *light,
                                                    .coordinate = {.x = 0, .y = 0, .level = 4, .face = 0}};
    const arc::render::virtual_shadow_page_key child{.address_space = *light,
                                                     .coordinate = {.x = 2, .y = 2, .level = 2, .face = 0}};
    const std::array requests{
        arc::render::virtual_shadow_page_request{
            .key = child, .frame_index = 1, .content_revision = 7, .projected_coverage = 10.0f, .light_priority = 200},
        arc::render::virtual_shadow_page_request{.key = root,
                                                 .frame_index = 1,
                                                 .content_revision = 7,
                                                 .projected_coverage = 1.0f,
                                                 .light_priority = 200,
                                                 .coarse_page = true}};
    const auto first = cache.resolve_requests(requests, 1);
    REQUIRE(first.render_pages.size() == 2);
    REQUIRE(first.render_pages.front().key == root);
    REQUIRE(cache.publish(root, 7));
    REQUIRE(cache.publish(child, 7));
    REQUIRE(cache.find_resident_or_ancestor(
                {.address_space = *light, .coordinate = {.x = 5, .y = 5, .level = 1, .face = 0}}) != nullptr);

    const auto second = cache.resolve_requests(requests, 2);
    REQUIRE(second.cache_hits == 2);
    REQUIRE(second.render_pages.empty());
    REQUIRE(cache.invalidate(*light, arc::render::virtual_shadow_invalidation_reason::material_alpha) == 2);
    REQUIRE(cache.statistics().dirty_pages == 2);
}

TEST_CASE("virtual shadow cache protects recent and pinned pages under pressure")
{
    constexpr std::uint64_t one_d16_page_pair =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_page_texels +
                                   arc::render::virtual_shadow_page_guard_texels * 2u) *
        (arc::render::virtual_shadow_page_texels + arc::render::virtual_shadow_page_guard_texels * 2u) * 4u;
    arc::render::virtual_shadow_cache cache(one_d16_page_pair * 4u);
    const auto light = cache.create_address_space(
        {.light_kind = arc::render::shadow_light_kind::spot, .light_key = 71, .level_count = 5});
    REQUIRE(light);
    const auto request = [&](std::uint16_t x, std::uint64_t frame, bool coarse)
    {
        const arc::render::virtual_shadow_page_request value{
            .key = {.address_space = *light, .coordinate = {.x = x, .y = 0, .level = 0, .face = 0}},
            .frame_index = frame,
            .content_revision = 1,
            .projected_coverage = 1.0f,
            .light_priority = 1,
            .coarse_page = coarse};
        return cache.resolve_requests(std::span{&value, 1}, frame);
    };
    REQUIRE(request(0, 1, true).render_pages.size() == 1);
    REQUIRE(request(1, 1, false).render_pages.size() == 1);
    REQUIRE(request(2, 1, false).render_pages.size() == 1);
    REQUIRE(request(3, 1, false).render_pages.size() == 1);
    REQUIRE(request(4, 2, false).failed_requests == 1);
    REQUIRE(request(4, 64, false).render_pages.size() == 1);
    REQUIRE(cache.statistics().eviction_count == 1);
    REQUIRE(cache.statistics().pinned_pages == 1);
}

TEST_CASE("virtual shadow clipmap origins snap at physical page granularity")
{
    const auto snapped = arc::render::snap_virtual_shadow_clipmap_origin({13.2f, -4.1f}, 0.125f);
    REQUIRE(snapped[0] == Catch::Approx(0.0f));
    REQUIRE(snapped[1] == Catch::Approx(-16.0f));
    const arc::render::virtual_shadow_address_space_descriptor directional{
        .light_kind = arc::render::shadow_light_kind::directional, .virtual_resolution = 16384};
    const arc::render::virtual_shadow_address_space_descriptor local{
        .light_kind = arc::render::shadow_light_kind::spot, .virtual_resolution = 16384, .level_count = 5};
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(directional, 0) == 128);
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(directional, 4) == 128);
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(local, 0) == 128);
    REQUIRE(arc::render::virtual_shadow_pages_per_axis(local, 4) == 8);
    REQUIRE(arc::render::virtual_shadow_parent_page({6, 10, 1, 2}) ==
            arc::render::virtual_shadow_page_coordinate{3, 5, 2, 2});
}

TEST_CASE("virtual shadow pool layout honors paired atlas budget and device limits")
{
    constexpr std::uint64_t d16_pair_bytes =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_physical_page_texels) *
        arc::render::virtual_shadow_physical_page_texels * 4u;
    const auto d16 = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes * 10u, arc::render::virtual_shadow_physical_page_texels * 8u,
        {.d16_unorm = true, .d32_float = true});
    REQUIRE(d16.valid());
    REQUIRE(d16.format == arc::render::virtual_shadow_depth_format::d16_unorm);
    REQUIRE(d16.pages_per_axis == 3);
    REQUIRE(d16.physical_page_capacity == 9);
    REQUIRE(d16.allocated_bytes == d16_pair_bytes * 9u);

    const auto dimension_limited = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes * 100u, arc::render::virtual_shadow_physical_page_texels * 2u, {.d16_unorm = true});
    REQUIRE(dimension_limited.pages_per_axis == 2);
    REQUIRE(dimension_limited.physical_page_capacity == 4);

    const auto d32 = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes * 8u, arc::render::virtual_shadow_physical_page_texels * 8u, {.d32_float = true});
    REQUIRE(d32.format == arc::render::virtual_shadow_depth_format::d32_float);
    REQUIRE(d32.pages_per_axis == 2);
    REQUIRE_FALSE(arc::render::resolve_virtual_shadow_physical_pool(
                      d16_pair_bytes, arc::render::virtual_shadow_physical_page_texels - 1u, {.d16_unorm = true})
                      .valid());
}

TEST_CASE("virtual shadow dense tables are deterministic and layer independent")
{
    constexpr std::uint64_t d16_pair_bytes =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_physical_page_texels) *
        arc::render::virtual_shadow_physical_page_texels * 4u;
    const auto pool = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes * 4u, arc::render::virtual_shadow_physical_page_texels * 2u, {.d16_unorm = true});
    arc::render::virtual_shadow_cache cache(
        {.physical_pool = pool, .page_table_entry_capacity = 16, .view_capacity = 8});
    const auto light = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .light_key = 91,
                                                   .virtual_resolution = 256,
                                                   .level_count = 2});
    REQUIRE(light);
    const auto* descriptor = cache.address_space(*light);
    REQUIRE(descriptor != nullptr);
    REQUIRE(arc::render::virtual_shadow_page_table_entry_count(*descriptor) == 5);
    REQUIRE(arc::render::virtual_shadow_dense_page_offset(*descriptor, {0, 0, 0, 0}) == 0u);
    REQUIRE(arc::render::virtual_shadow_dense_page_offset(*descriptor, {1, 1, 0, 0}) == 3u);
    REQUIRE(arc::render::virtual_shadow_dense_page_offset(*descriptor, {0, 0, 1, 0}) == 4u);
    REQUIRE_FALSE(arc::render::virtual_shadow_dense_page_offset(*descriptor, {1, 0, 1, 0}));

    const arc::render::virtual_shadow_page_key static_key{.address_space = *light,
                                                          .coordinate = {1, 1, 0, 0},
                                                          .layer =
                                                              arc::render::virtual_shadow_page_layer::static_depth};
    const arc::render::virtual_shadow_page_key dynamic_key{.address_space = *light,
                                                           .coordinate = {1, 1, 0, 0},
                                                           .layer =
                                                               arc::render::virtual_shadow_page_layer::dynamic_depth};
    const std::array requests{
        arc::render::virtual_shadow_page_request{.key = static_key, .frame_index = 1, .content_revision = 11},
        arc::render::virtual_shadow_page_request{.key = dynamic_key, .frame_index = 1, .content_revision = 12}};
    REQUIRE(cache.resolve_requests(requests, 1).render_pages.size() == 2);
    REQUIRE(cache.publish(static_key, 11));
    REQUIRE(cache.publish(dynamic_key, 12));
    const auto dense_index = cache.dense_page_index(static_key);
    REQUIRE(dense_index == cache.dense_page_index(dynamic_key));
    const auto snapshot = cache.gpu_snapshot();
    REQUIRE(snapshot.page_table[*dense_index].static_depth.physical_page != arc::render::invalid_virtual_shadow_index);
    REQUIRE(snapshot.page_table[*dense_index].dynamic_depth.physical_page != arc::render::invalid_virtual_shadow_index);
    REQUIRE(snapshot.page_table[*dense_index].static_depth.content_revision_low == 11);
    REQUIRE(snapshot.page_table[*dense_index].dynamic_depth.content_revision_low == 12);
}

TEST_CASE("virtual shadow dense ranges recycle without reviving stale identities")
{
    constexpr std::uint64_t d16_pair_bytes =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_physical_page_texels) *
        arc::render::virtual_shadow_physical_page_texels * 4u;
    const auto pool = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes, arc::render::virtual_shadow_physical_page_texels, {.d16_unorm = true});
    arc::render::virtual_shadow_cache cache(
        {.physical_pool = pool, .page_table_entry_capacity = 1, .view_capacity = 1});
    const auto first = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .virtual_resolution = arc::render::virtual_shadow_page_texels,
                                                   .level_count = 1});
    REQUIRE(first);
    REQUIRE_FALSE(cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                              .virtual_resolution = arc::render::virtual_shadow_page_texels,
                                              .level_count = 1}));
    const auto stale = *first;
    REQUIRE(cache.destroy_address_space(*first));
    const auto replacement = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                         .virtual_resolution = arc::render::virtual_shadow_page_texels,
                                                         .level_count = 1});
    REQUIRE(replacement);
    REQUIRE(replacement->index == stale.index);
    REQUIRE(replacement->generation != stale.generation);
    REQUIRE_FALSE(cache.dense_page_index({.address_space = stale, .coordinate = {0, 0, 0, 0}}));
    REQUIRE(cache.dense_page_index({.address_space = *replacement, .coordinate = {0, 0, 0, 0}}) == 0u);
}

TEST_CASE("directional virtual shadow views are stable equal-grid clip levels")
{
    const arc::render::virtual_shadow_address_space_descriptor descriptor{
        .light_kind = arc::render::shadow_light_kind::directional, .virtual_resolution = 1024};
    arc::render::directional_shadow_camera camera{};
    camera.inverse_view_projection = arc::math::identity<float, 4>();
    const auto first = arc::render::make_directional_virtual_shadow_views(descriptor, camera, {0.1f, 0.0f, 0.0f},
                                                                          {0.0f, -1.0f, 0.0f}, 200.0f);
    const auto within_page = arc::render::make_directional_virtual_shadow_views(descriptor, camera, {1.0f, 0.0f, 0.0f},
                                                                                {0.0f, -1.0f, 0.0f}, 200.0f);
    const auto crossed_page = arc::render::make_directional_virtual_shadow_views(descriptor, camera, {4.0f, 0.0f, 0.0f},
                                                                                 {0.0f, -1.0f, 0.0f}, 200.0f);
    REQUIRE(first.size() == arc::render::virtual_shadow_directional_clip_levels);
    for (std::uint32_t axis = 0; axis < 3; ++axis)
        REQUIRE(within_page[0].snapped_origin[axis] == Catch::Approx(first[0].snapped_origin[axis]));
    const auto crossed_delta = arc::math::sub(crossed_page[0].snapped_origin, first[0].snapped_origin);
    REQUIRE(arc::math::length_squared(crossed_delta) > 0.0f);
    for (std::size_t level = 0; level < first.size(); ++level)
    {
        REQUIRE(first[level].pages_per_axis == 8);
        if (level != 0)
            REQUIRE(first[level].world_units_per_texel == Catch::Approx(first[level - 1].world_units_per_texel * 2.0f));
        for (std::uint32_t row = 0; row < 4; ++row)
            for (std::uint32_t column = 0; column < 4; ++column)
                REQUIRE(std::isfinite(first[level].world_to_shadow_clip(row, column)));
    }
}

TEST_CASE("virtual shadow requests always retain a conventional executable fallback")
{
    using arc::render::resolve_shadow_map_method;
    using arc::render::shadow_map_method;

    REQUIRE(resolve_shadow_map_method(shadow_map_method::auto_select, true) == shadow_map_method::virtualized);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, true) == shadow_map_method::virtualized);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::conventional, true) == shadow_map_method::conventional);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::auto_select, false) == shadow_map_method::conventional);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, false) == shadow_map_method::conventional);
}

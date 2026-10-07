#include <arc/render/render.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <array>
#include <bit>
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

TEST_CASE("virtual shadow GPU feedback is generation safe deterministic and bounded")
{
    constexpr std::uint64_t d16_pair_bytes =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_physical_page_texels) *
        arc::render::virtual_shadow_physical_page_texels * 4u;
    const auto pool = arc::render::resolve_virtual_shadow_physical_pool(
        d16_pair_bytes * 4u, arc::render::virtual_shadow_physical_page_texels * 2u, {.d16_unorm = true});
    arc::render::virtual_shadow_cache cache(
        {.physical_pool = pool, .page_table_entry_capacity = 16, .view_capacity = 8});
    const auto light = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .virtual_resolution = 256,
                                                   .level_count = 2,
                                                   .priority = 120});
    REQUIRE(light);
    REQUIRE(cache.update_address_space_request_metadata(*light, arc::render::render_mobility::stationary, 200));
    const auto snapshot = cache.gpu_snapshot();
    REQUIRE(snapshot.address_spaces[light->index].mobility ==
            static_cast<std::uint32_t>(arc::render::render_mobility::stationary));
    REQUIRE(snapshot.address_spaces[light->index].priority == 200);

    const arc::render::virtual_shadow_page_request request{
        .key = {.address_space = *light,
                .coordinate = {1, 1, 0, 0},
                .layer = arc::render::virtual_shadow_page_layer::dynamic_depth},
        .frame_index = 2,
        .content_revision = 11,
        .projected_coverage = 0.4f,
        .light_priority = 120};
    auto duplicate = arc::render::encode_virtual_shadow_gpu_request(request);
    duplicate.frame_low = 3;
    duplicate.content_revision_low = 12;
    duplicate.projected_coverage_bits = std::bit_cast<std::uint32_t>(0.8f);
    duplicate.light_priority = 200;
    auto stale = duplicate;
    ++stale.address_space_generation;
    auto invalid = duplicate;
    invalid.page_xy = 9u;
    auto future = duplicate;
    future.frame_low = 4;
    const std::array feedback{arc::render::encode_virtual_shadow_gpu_request(request), duplicate, stale, invalid,
                              future};
    const auto translated = cache.translate_gpu_feedback(feedback,
                                                         {.request_count = static_cast<std::uint32_t>(feedback.size()),
                                                          .raw_request_count = 7,
                                                          .duplicate_count = 1,
                                                          .overflow_count = 2},
                                                         3);
    REQUIRE(translated.requests.size() == 1);
    CHECK(translated.raw_requests == 7);
    CHECK(translated.duplicate_requests == 2);
    CHECK(translated.stale_requests == 1);
    CHECK(translated.invalid_requests == 2);
    CHECK(translated.overflow_requests == 2);
    CHECK(translated.requests[0].frame_index == 3);
    CHECK(translated.requests[0].content_revision == 12);
    CHECK(translated.requests[0].projected_coverage == Catch::Approx(0.8f));
    CHECK(translated.requests[0].light_priority == 200);
}

TEST_CASE("virtual shadow render completion rejects stale physical and work generations")
{
    constexpr std::uint64_t page_pair_bytes =
        static_cast<std::uint64_t>(arc::render::virtual_shadow_physical_page_texels) *
        arc::render::virtual_shadow_physical_page_texels * 4u;
    arc::render::virtual_shadow_cache cache(page_pair_bytes * 2u);
    const auto light = cache.create_address_space({.light_kind = arc::render::shadow_light_kind::spot,
                                                   .virtual_resolution = arc::render::virtual_shadow_page_texels,
                                                   .level_count = 1});
    REQUIRE(light);
    const arc::render::virtual_shadow_page_request request{
        .key = {.address_space = *light, .coordinate = {0, 0, 0, 0}}, .frame_index = 1, .content_revision = 7};
    auto result = cache.resolve_requests(std::span{&request, 1}, 1);
    REQUIRE(result.render_pages.size() == 1);
    const auto stale = arc::render::make_virtual_shadow_page_render_token(result.render_pages.front());
    REQUIRE(cache.set_in_flight(stale.key, true));
    REQUIRE(cache.invalidate(*light, arc::render::virtual_shadow_invalidation_reason::caster_transform) == 1);
    REQUIRE_FALSE(cache.complete_render(stale, true));
    REQUIRE(cache.find(stale.key)->dirty());

    result = cache.resolve_requests(std::span{&request, 1}, 2);
    REQUIRE(result.render_pages.size() == 1);
    const auto retry = arc::render::make_virtual_shadow_page_render_token(result.render_pages.front());
    REQUIRE(cache.set_in_flight(retry.key, true));
    REQUIRE(cache.complete_render(retry, false));
    REQUIRE(cache.find(retry.key)->dirty());
    REQUIRE_FALSE(cache.find(retry.key)->in_flight);

    result = cache.resolve_requests(std::span{&request, 1}, 3);
    REQUIRE(result.render_pages.size() == 1);
    const auto completed = arc::render::make_virtual_shadow_page_render_token(result.render_pages.front());
    REQUIRE(cache.set_in_flight(completed.key, true));
    REQUIRE(cache.complete_render(completed, true));
    REQUIRE(cache.find(completed.key)->resident);
    REQUIRE_FALSE(cache.find(completed.key)->dirty());
}

TEST_CASE("virtual shadow replacement retains published depth until raster and guards both complete")
{
    using namespace arc::render;
    constexpr std::uint64_t pair_bytes = virtual_shadow_physical_page_texels * virtual_shadow_physical_page_texels * 4u;
    virtual_shadow_cache cache(pair_bytes * 4u);
    const auto light = cache.create_address_space(
        {.light_kind = shadow_light_kind::spot, .virtual_resolution = virtual_shadow_page_texels, .level_count = 1});
    REQUIRE(light);
    virtual_shadow_page_request request{.key = {.address_space = *light}, .content_revision = 7};
    auto result = cache.resolve_requests(std::span{&request, 1}, 1);
    REQUIRE(result.render_pages.size() == 1);
    const auto original = result.render_pages[0].physical_page;
    REQUIRE(cache.publish(request.key, 7));
    const auto dense = cache.dense_page_index(request.key);
    REQUIRE(dense);
    const auto published = [&] { return cache.gpu_snapshot().page_table[*dense].static_depth; };

    request.content_revision = 8;
    result = cache.resolve_requests(std::span{&request, 1}, 2);
    REQUIRE(result.render_pages.size() == 1);
    const auto replacement = result.render_pages[0].physical_page;
    REQUIRE(replacement != original);
    auto token = make_virtual_shadow_page_render_token(result.render_pages[0]);
    CHECK(published().physical_page == original.index);
    REQUIRE(cache.set_in_flight(request.key, true));
    REQUIRE(cache.complete_render(token, true, false));
    CHECK(published().physical_page == original.index);
    CHECK(published().content_revision_low == 7);
    CHECK(cache.find(request.key)->dirty());
    CHECK(cache.find(request.key)->resident);

    result = cache.resolve_requests(std::span{&request, 1}, 3);
    REQUIRE(result.render_pages.size() == 1);
    CHECK(result.render_pages[0].physical_page == replacement);
    REQUIRE(cache.set_in_flight(request.key, true));
    REQUIRE(cache.complete_render(token, false, true));
    CHECK(published().physical_page == original.index);
    REQUIRE(cache.set_in_flight(request.key, true));
    REQUIRE(cache.complete_render(token, true, true));
    CHECK(published().physical_page == replacement.index);
    CHECK(published().content_revision_low == 8);
    CHECK_FALSE(cache.find(request.key)->retained_physical_page.valid());
    CHECK_FALSE(cache.complete_render(token, true, true));
}

TEST_CASE("virtual shadow directional lookup reprojects layers independently and bounds filter taps")
{
    using namespace arc::render;
    constexpr std::uint64_t pair_bytes = virtual_shadow_physical_page_texels * virtual_shadow_physical_page_texels * 4u;
    virtual_shadow_cache cache(pair_bytes * 4u);
    const auto light =
        cache.create_address_space({.light_kind = shadow_light_kind::directional, .virtual_resolution = 256});
    REQUIRE(light);
    std::array<virtual_shadow_view_descriptor, virtual_shadow_directional_clip_levels> views{};
    for (std::uint16_t level = 0; level < views.size(); ++level)
    {
        views[level].world_to_shadow_clip = arc::math::identity<float, 4>();
        views[level].pages_per_axis = 2;
        views[level].level = level;
    }
    views[1].world_to_shadow_clip(0, 3) = -1.0f;
    REQUIRE(cache.update_address_space_views(*light, views));
    const virtual_shadow_page_request fine{.key = {*light, {1, 1, 0, 0}, virtual_shadow_page_layer::static_depth},
                                           .content_revision = 1};
    const virtual_shadow_page_request coarse{.key = {*light, {0, 1, 1, 0}, virtual_shadow_page_layer::dynamic_depth},
                                             .content_revision = 2};
    const std::array requests{fine, coarse};
    REQUIRE(cache.resolve_requests(requests, 1).render_pages.size() == 2);
    REQUIRE(cache.publish(fine.key, 1));
    REQUIRE(cache.publish(coarse.key, 2));
    const auto sample = [&](virtual_shadow_page_layer layer, arc::math::vector2f tap = {})
    {
        return resolve_directional_virtual_shadow_sample(cache.gpu_snapshot(), *light, {0.1f, 0.1f, 0.5f}, layer,
                                                         cache.physical_pool_layout(), tap);
    };
    REQUIRE(sample(virtual_shadow_page_layer::static_depth));
    REQUIRE(sample(virtual_shadow_page_layer::dynamic_depth));
    CHECK(sample(virtual_shadow_page_layer::static_depth)->level == 0);
    CHECK(sample(virtual_shadow_page_layer::dynamic_depth)->level == 1);
    CHECK(sample(virtual_shadow_page_layer::dynamic_depth)->depth == Catch::Approx(0.5f));
    const auto pool = cache.physical_pool_layout();
    for (const auto layer : {virtual_shadow_page_layer::static_depth, virtual_shadow_page_layer::dynamic_depth})
    {
        for (float offset : {-1000.0f, -2.0f, 0.0f, 2.0f, 1000.0f})
        {
            const auto location = sample(layer, {offset, offset});
            REQUIRE(location);
            const auto physical = location->mapping.physical_page;
            CHECK(location->atlas_uv[0] * pool.atlas_extent >= (physical % pool.pages_per_axis) * 136u + 0.5f);
            CHECK(location->atlas_uv[0] * pool.atlas_extent <= (physical % pool.pages_per_axis + 1u) * 136u - 0.5f);
            CHECK(location->atlas_uv[1] * pool.atlas_extent >= (physical / pool.pages_per_axis) * 136u + 0.5f);
            CHECK(location->atlas_uv[1] * pool.atlas_extent <= (physical / pool.pages_per_axis + 1u) * 136u - 0.5f);
        }
    }
    auto stale = *light;
    ++stale.generation;
    CHECK_FALSE(resolve_directional_virtual_shadow_sample(cache.gpu_snapshot(), stale, {0.1f, 0.1f, 0.5f},
                                                          virtual_shadow_page_layer::static_depth, pool));
    // Any changed view invalidates depth and work made under the old projections.
    REQUIRE(cache.invalidate(*light, virtual_shadow_invalidation_reason::geometry) == 2);
    const auto retry = cache.resolve_requests(requests, 2);
    REQUIRE_FALSE(retry.render_pages.empty());
    const auto token = make_virtual_shadow_page_render_token(retry.render_pages.front());
    REQUIRE(cache.set_in_flight(token.key, true));
    views[0].world_to_shadow_clip(0, 3) += 0.1f;
    REQUIRE(cache.update_address_space_views(*light, views));
    CHECK_FALSE(cache.complete_render(token, true, true));
    CHECK_FALSE(sample(virtual_shadow_page_layer::static_depth));
    CHECK_FALSE(sample(virtual_shadow_page_layer::dynamic_depth));
}

TEST_CASE("virtual shadow refresh with no spare tile preserves the resident mapping")
{
    using namespace arc::render;
    constexpr std::uint64_t pair_bytes = virtual_shadow_physical_page_texels * virtual_shadow_physical_page_texels * 4u;
    virtual_shadow_cache cache(pair_bytes);
    const auto light = cache.create_address_space(
        {.light_kind = shadow_light_kind::spot, .virtual_resolution = virtual_shadow_page_texels, .level_count = 1});
    REQUIRE(light);
    virtual_shadow_page_request request{.key = {.address_space = *light}, .content_revision = 1};
    REQUIRE(cache.resolve_requests(std::span{&request, 1}, 1).render_pages.size() == 1);
    REQUIRE(cache.publish(request.key, 1));
    request.content_revision = 2;
    const auto result = cache.resolve_requests(std::span{&request, 1}, 2);
    CHECK(result.render_pages.empty());
    CHECK(result.failed_requests == 1);
    REQUIRE(cache.find_resident_or_ancestor(request.key));
    const auto dense = cache.dense_page_index(request.key);
    REQUIRE(dense);
    CHECK(cache.gpu_snapshot().page_table[*dense].static_depth.content_revision_low == 1);
}

TEST_CASE("virtual shadow render pages encode deterministic page projections and work ranges")
{
    arc::render::virtual_shadow_view_descriptor view{};
    view.world_to_shadow_clip = arc::math::identity<float, 4>();
    view.pages_per_axis = 4;
    view.face = 2;
    view.level = 1;
    const arc::render::virtual_shadow_page_mapping mapping{
        .key = {.address_space = {3, 9},
                .coordinate = {1, 2, 1, 2},
                .layer = arc::render::virtual_shadow_page_layer::dynamic_depth},
        .physical_page = {18, 4},
        .content_revision = 0x123456789abcdef0ull,
        .work_revision = 5};
    const auto encoded = arc::render::encode_virtual_shadow_render_page(mapping, view, 17, 8, 1024, 512, 33);
    CHECK(encoded.address_physical[0] == 3);
    CHECK(encoded.address_physical[1] == 9);
    CHECK(encoded.address_physical[2] == 18);
    CHECK(encoded.address_physical[3] == 4);
    CHECK(encoded.virtual_page[2] == 17);
    CHECK((encoded.virtual_page[3] & 0xffffu) == 2);
    CHECK((encoded.virtual_page[3] >> 16u) == 2);
    CHECK(encoded.work[0] == 1024);
    CHECK(encoded.work[1] == 512);
    CHECK(encoded.revision[0] == 0x9abcdef0u);
    CHECK(encoded.revision[1] == 0x12345678u);
    constexpr float guarded_scale = static_cast<float>(arc::render::virtual_shadow_page_texels) /
                                    static_cast<float>(arc::render::virtual_shadow_physical_page_texels);
    CHECK(encoded.world_to_page_clip[0] == Catch::Approx(4.0f * guarded_scale));
    CHECK(encoded.world_to_page_clip[3] == Catch::Approx(guarded_scale));
    CHECK(encoded.world_to_page_clip[5] == Catch::Approx(4.0f * guarded_scale));
    CHECK(encoded.world_to_page_clip[7] == Catch::Approx(-guarded_scale));
}

TEST_CASE("virtual shadow guarded page projection preserves the logical 128 texel region")
{
    arc::render::virtual_shadow_view_descriptor view{};
    view.world_to_shadow_clip = arc::math::identity<float, 4>();
    view.pages_per_axis = 2;
    const auto logical = arc::render::virtual_shadow_page_view_projection(view, {1, 0, 0, 0});
    const auto guarded = arc::render::virtual_shadow_guarded_page_view_projection(view, {1, 0, 0, 0});
    constexpr float scale = static_cast<float>(arc::render::virtual_shadow_page_texels) /
                            static_cast<float>(arc::render::virtual_shadow_physical_page_texels);
    for (std::uint32_t column = 0; column < 4; ++column)
    {
        CHECK(guarded(0, column) == Catch::Approx(logical(0, column) * scale));
        CHECK(guarded(1, column) == Catch::Approx(logical(1, column) * scale));
        CHECK(guarded(2, column) == Catch::Approx(logical(2, column)));
        CHECK(guarded(3, column) == Catch::Approx(logical(3, column)));
    }
    CHECK((1.0f - scale) * 0.5f * arc::render::virtual_shadow_physical_page_texels ==
          Catch::Approx(static_cast<float>(arc::render::virtual_shadow_page_guard_texels)));
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

TEST_CASE("point and spot virtual shadow views are deterministic finite quadtrees")
{
    const arc::render::virtual_shadow_address_space_descriptor point_descriptor{
        .light_kind = arc::render::shadow_light_kind::point, .virtual_resolution = 512, .level_count = 3};
    const auto point = arc::render::make_point_virtual_shadow_views(point_descriptor, {1.0f, 2.0f, 3.0f}, 50.0f);
    const auto point_again = arc::render::make_point_virtual_shadow_views(point_descriptor, {1.0f, 2.0f, 3.0f}, 50.0f);
    REQUIRE(point.size() == arc::render::point_shadow_face_count * point_descriptor.level_count);
    REQUIRE(point_again.size() == point.size());
    for (std::size_t index = 0; index < point.size(); ++index)
    {
        REQUIRE(point[index].face == point_again[index].face);
        REQUIRE(point[index].level == point_again[index].level);
        for (std::uint32_t row = 0; row < 4; ++row)
            for (std::uint32_t column = 0; column < 4; ++column)
                REQUIRE(point[index].world_to_shadow_clip(row, column) ==
                        point_again[index].world_to_shadow_clip(row, column));
    }

    const arc::render::virtual_shadow_address_space_descriptor spot_descriptor{
        .light_kind = arc::render::shadow_light_kind::spot, .virtual_resolution = 512, .level_count = 3};
    const auto spot = arc::render::make_spot_virtual_shadow_views(spot_descriptor, {1.0f, 2.0f, 3.0f},
                                                                  {0.0f, -1.0f, 0.0f}, 0.75f, 50.0f);
    REQUIRE(spot.size() == spot_descriptor.level_count);

    const auto require_finite_quadtree =
        [](const std::span<const arc::render::virtual_shadow_view_descriptor> views, std::uint8_t levels)
    {
        for (std::size_t index = 0; index < views.size(); ++index)
        {
            const auto level = static_cast<std::uint32_t>(index % levels);
            REQUIRE(views[index].level == level);
            REQUIRE(views[index].pages_per_axis == std::max(1u, 4u >> level));
            for (std::uint32_t row = 0; row < 4; ++row)
                for (std::uint32_t column = 0; column < 4; ++column)
                    REQUIRE(std::isfinite(views[index].world_to_shadow_clip(row, column)));
        }
    };
    require_finite_quadtree(point, point_descriptor.level_count);
    require_finite_quadtree(spot, spot_descriptor.level_count);
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

    const arc::render::virtual_shadow_light_support directional_only{.directional = true};
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, arc::render::shadow_light_kind::directional,
                                      directional_only) == shadow_map_method::virtualized);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, arc::render::shadow_light_kind::point,
                                      directional_only) == shadow_map_method::conventional);
    REQUIRE(resolve_shadow_map_method(shadow_map_method::virtualized, arc::render::shadow_light_kind::spot,
                                      directional_only) == shadow_map_method::conventional);
}

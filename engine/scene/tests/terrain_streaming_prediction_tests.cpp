#include <arc/render/renderer.h>
#include <arc/render/virtual_mesh.h>
#include <arc/scene/terrain_streaming_prediction.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace
{

arc::render::virtual_mesh_data make_streaming_region(float z_offset)
{
    using namespace arc::render;
    constexpr std::uint32_t side = 65u;
    mesh_data mesh;
    mesh.name = "M2.4 predictive terrain region";
    mesh.vertices.resize(static_cast<std::size_t>(side) * side);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t x = 0; x < side; ++x)
        {
            auto& vertex = mesh.vertices[static_cast<std::size_t>(z) * side + x];
            vertex.position[0] = static_cast<float>(x);
            vertex.position[1] = static_cast<float>((x * 3u + z * 5u) % 11u) * 0.1f;
            vertex.position[2] = z_offset + static_cast<float>(z);
            vertex.normal[1] = 1.0f;
            vertex.tangent[0] = 1.0f;
            vertex.tangent[3] = 1.0f;
        }
    for (std::uint32_t z = 0; z + 1u < side; ++z)
        for (std::uint32_t x = 0; x + 1u < side; ++x)
        {
            const auto i0 = z * side + x;
            const auto i1 = i0 + 1u;
            const auto i2 = i0 + side;
            const auto i3 = i2 + 1u;
            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    return build_virtual_mesh(mesh, {.build_conventional_lods = false});
}

arc::scene::terrain_render_region_proxy make_region(arc::render::renderer& renderer, arc::scene::terrain_region_id id,
                                                    float z_offset)
{
    arc::scene::terrain_render_region_proxy region;
    region.id = id;
    region.geometry = renderer.create_geometry_resource(make_streaming_region(z_offset));
    region.local_bounds = {{0.0f, 0.0f, z_offset}, {64.0f, 2.0f, z_offset + 64.0f}};
    return region;
}

} // namespace

TEST_CASE("M2.4 terrain prediction prioritizes detail in the direction of travel")
{
    using namespace arc;

    render::renderer renderer;
    scene::terrain_render_proxy proxy;
    const auto behind = make_region(renderer, {0, 0}, -512.0f);
    const auto ahead = make_region(renderer, {0, 1}, 512.0f);
    REQUIRE(behind.geometry.virtualized.valid());
    REQUIRE(ahead.geometry.virtualized.valid());
    proxy.regions = {behind, ahead};

    renderer.virtual_geometry_residency().begin_frame(1u);
    scene::terrain_streaming_predictor predictor({.prediction_horizon_seconds = 2.0f,
                                                  .prefetch_distance = 700.0f,
                                                  .hysteresis_distance = 100.0f,
                                                  .forward_bias = 1.5f,
                                                  .streaming_importance = 1.0f,
                                                  .maximum_prefetch_pages = 1u});
    const scene::terrain_streaming_prediction_view view{.camera_position = {32.0f, 4.0f, 0.0f},
                                                        .camera_velocity = {0.0f, 0.0f, 250.0f},
                                                        .camera_forward = {0.0f, 0.0f, 1.0f}};
    const auto prediction = predictor.update(proxy, renderer, view);
    REQUIRE(prediction.candidate_pages > 1u);
    REQUIRE(prediction.requested_pages == 1u);
    CHECK(prediction.predicted_camera_position[2] == 500.0f);

    const auto loads = renderer.take_virtual_geometry_page_loads();
    REQUIRE(loads.size() == 1u);
    CHECK(loads.front().resource == ahead.geometry.virtualized);
}

TEST_CASE("M2.4 camera cuts suppress velocity lookahead but keep nearby prefetch available")
{
    using namespace arc;

    render::renderer renderer;
    scene::terrain_render_proxy proxy;
    const auto far_region = make_region(renderer, {0, 0}, 900.0f);
    REQUIRE(far_region.geometry.virtualized.valid());
    proxy.regions.push_back(far_region);

    renderer.virtual_geometry_residency().begin_frame(1u);
    scene::terrain_streaming_predictor predictor({.prediction_horizon_seconds = 1.0f,
                                                  .prefetch_distance = 220.0f,
                                                  .hysteresis_distance = 64.0f,
                                                  .forward_bias = 1.0f,
                                                  .streaming_importance = 1.0f,
                                                  .maximum_prefetch_pages = 4u});
    scene::terrain_streaming_prediction_view view{.camera_position = {32.0f, 4.0f, 0.0f},
                                                  .camera_velocity = {0.0f, 0.0f, 900.0f},
                                                  .camera_forward = {0.0f, 0.0f, 1.0f},
                                                  .camera_cut = true};
    const auto cut = predictor.update(proxy, renderer, view);
    CHECK(cut.predicted_camera_position[2] == 0.0f);
    CHECK(cut.requested_pages == 0u);

    view.camera_cut = false;
    const auto moving = predictor.update(proxy, renderer, view);
    CHECK(moving.predicted_camera_position[2] == 900.0f);
    CHECK(moving.requested_pages > 0u);
}

TEST_CASE("M2.4 prefetch stays bounded independently of terrain page count")
{
    using namespace arc;

    render::renderer renderer;
    scene::terrain_render_proxy proxy;
    proxy.regions.push_back(make_region(renderer, {0, 0}, 64.0f));
    proxy.regions.push_back(make_region(renderer, {0, 1}, 160.0f));
    proxy.regions.push_back(make_region(renderer, {0, 2}, 256.0f));
    proxy.regions.push_back(make_region(renderer, {0, 3}, 352.0f));
    proxy.regions.push_back(make_region(renderer, {0, 4}, 448.0f));
    for (const auto& region : proxy.regions)
        REQUIRE(region.geometry.virtualized.valid());

    renderer.virtual_geometry_residency().begin_frame(1u);
    scene::terrain_streaming_predictor predictor({.prediction_horizon_seconds = 1.0f,
                                                  .prefetch_distance = 512.0f,
                                                  .hysteresis_distance = 64.0f,
                                                  .forward_bias = 1.0f,
                                                  .streaming_importance = 2.0f,
                                                  .maximum_prefetch_pages = 3u});
    const scene::terrain_streaming_prediction_view view{.camera_position = {32.0f, 4.0f, 0.0f},
                                                        .camera_velocity = {0.0f, 0.0f, 80.0f},
                                                        .camera_forward = {0.0f, 0.0f, 1.0f}};
    const auto prediction = predictor.update(proxy, renderer, view);
    REQUIRE(prediction.candidate_pages > 3u);
    CHECK(prediction.requested_pages == 3u);
    CHECK(renderer.take_virtual_geometry_page_loads().size() == 3u);
}
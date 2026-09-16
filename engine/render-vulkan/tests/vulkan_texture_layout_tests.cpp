#include "vulkan_texture_layout.h"

#include <arc/render/texture_artifact.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Vulkan texture topology accepts T2 resident texture dimensions")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::backend_detail;

    STATIC_REQUIRE(vulkan_texture_topology_supported(texture_dimension::texture_2d, 16, 8, 1, 1));
    STATIC_REQUIRE(vulkan_texture_topology_supported(texture_dimension::texture_2d, 16, 8, 1, 4));
    STATIC_REQUIRE(vulkan_texture_topology_supported(texture_dimension::texture_3d, 16, 8, 4, 1));
    STATIC_REQUIRE(vulkan_texture_topology_supported(texture_dimension::cube, 16, 16, 1, 1));

    STATIC_REQUIRE_FALSE(vulkan_texture_topology_supported(texture_dimension::texture_3d, 16, 8, 4, 2));
    STATIC_REQUIRE_FALSE(vulkan_texture_topology_supported(texture_dimension::cube, 16, 8, 1, 1));
    STATIC_REQUIRE_FALSE(vulkan_texture_topology_supported(texture_dimension::cube, 16, 16, 1, 2));
    STATIC_REQUIRE_FALSE(vulkan_texture_topology_supported(texture_dimension::texture_2d, 16, 8, 2, 1));
}

TEST_CASE("Vulkan streamed mip windows preserve cube topology")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::backend_detail;

    texture_artifact_index artifact;
    artifact.dimension = texture_dimension::cube;
    artifact.width = 8;
    artifact.height = 8;
    artifact.depth = 1;
    artifact.array_layers = 1;
    artifact.face_count = 6;
    artifact.mip_count = 4;
    artifact.mips = {{.width = 8, .height = 8, .depth = 1},
                     {.width = 4, .height = 4, .depth = 1},
                     {.width = 2, .height = 2, .depth = 1},
                     {.width = 1, .height = 1, .depth = 1}};

    streamed_texture_window_layout layout;
    REQUIRE(resolve_streamed_texture_window_layout(artifact, 1, layout));
    CHECK(layout.dimension == texture_dimension::cube);
    CHECK(layout.width == 4);
    CHECK(layout.height == 4);
    CHECK(layout.depth == 1);
    CHECK(layout.array_layers == 1);
    CHECK(layout.mip_levels == 3);
}

TEST_CASE("Vulkan streamed mip windows preserve volume depth")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::backend_detail;

    texture_artifact_index artifact;
    artifact.dimension = texture_dimension::texture_3d;
    artifact.width = 8;
    artifact.height = 4;
    artifact.depth = 4;
    artifact.array_layers = 1;
    artifact.face_count = 1;
    artifact.mip_count = 4;
    artifact.mips = {{.width = 8, .height = 4, .depth = 4},
                     {.width = 4, .height = 2, .depth = 2},
                     {.width = 2, .height = 1, .depth = 1},
                     {.width = 1, .height = 1, .depth = 1}};

    streamed_texture_window_layout layout;
    REQUIRE(resolve_streamed_texture_window_layout(artifact, 1, layout));
    CHECK(layout.dimension == texture_dimension::texture_3d);
    CHECK(layout.width == 4);
    CHECK(layout.height == 2);
    CHECK(layout.depth == 2);
    CHECK(layout.array_layers == 1);
    CHECK(layout.mip_levels == 3);
}

TEST_CASE("Vulkan streamed mip windows reject unsupported cube arrays")
{
    using namespace arc::render;
    using namespace arc::render::vulkan::backend_detail;

    texture_artifact_index artifact;
    artifact.dimension = texture_dimension::cube;
    artifact.width = 8;
    artifact.height = 8;
    artifact.depth = 1;
    artifact.array_layers = 2;
    artifact.face_count = 6;
    artifact.mip_count = 1;
    artifact.mips = {{.width = 8, .height = 8, .depth = 1}};

    streamed_texture_window_layout layout;
    CHECK_FALSE(resolve_streamed_texture_window_layout(artifact, 0, layout));
}

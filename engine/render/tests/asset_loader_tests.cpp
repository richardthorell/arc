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

using arc::render::tests::make_dds_header;
using arc::render::tests::write_triangle_glb;

TEST_CASE("GLB mesh loader reads static triangle geometry")
{
    const auto path = write_triangle_glb();
    const auto result = arc::render::load_gltf_mesh(path);

    REQUIRE(result.succeeded());
    REQUIRE(result.mesh.vertices.size() == 3);
    REQUIRE(result.mesh.indices == std::vector<std::uint32_t>{0, 1, 2});
    REQUIRE(result.mesh.vertices[0].position[1] == 0.5f);
    REQUIRE(result.mesh.vertices[0].normal[2] == 1.0f);
    REQUIRE(result.mesh.vertices[0].tangent[0] == Catch::Approx(1.0f));
    REQUIRE(result.mesh.vertices[0].tangent[1] == Catch::Approx(0.0f));
    REQUIRE(result.mesh.vertices[0].tangent[2] == Catch::Approx(0.0f));
    REQUIRE(result.mesh.vertices[0].tangent[3] == Catch::Approx(1.0f));
    REQUIRE(result.mesh.vertices[1].texcoord[1] == 1.0f);
    REQUIRE(result.mesh.material_index == 0);
    REQUIRE(result.textures.size() == 1);
    REQUIRE(result.textures[0].mime_type == "image/png");
    REQUIRE(result.textures[0].encoded.size() == 4);
    REQUIRE(result.materials.size() == 1);
    REQUIRE(result.materials[0].material.name == "TestMaterial");
    REQUIRE(result.materials[0].material.alpha_mode == arc::render::material_alpha_mode::masked);
    REQUIRE(result.materials[0].material.alpha_cutoff == Catch::Approx(0.35f));
    REQUIRE(result.materials[0].material.base_color[2] == Catch::Approx(0.75f));
    REQUIRE(result.materials[0].material.metallic == Catch::Approx(0.2f));
    REQUIRE(result.materials[0].material.roughness == Catch::Approx(0.7f));
    REQUIRE(result.materials[0].material.double_sided);
    REQUIRE(result.materials[0].material.normal_scale == Catch::Approx(0.8f));
    REQUIRE(result.materials[0].material.occlusion_strength == Catch::Approx(0.6f));
    REQUIRE(result.materials[0].material.emissive_factor[1] == Catch::Approx(0.2f));
    REQUIRE(result.materials[0].textures.base_color == 0);
    REQUIRE(result.materials[0].textures.normal == 0);

    std::filesystem::remove(path);
}

TEST_CASE("DDS loader parses BC1 texture metadata")
{
    auto bytes = make_dds_header(4, 4, 1, 0x00000004u, 0x31545844u);
    bytes.resize(bytes.size() + 8);

    const auto result = arc::render::parse_dds_texture(bytes, "bc1.dds");

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE(result.texture.dds);
    REQUIRE(result.texture.compressed);
    REQUIRE(result.texture.format == arc::render::texture_format::bc1_rgba_unorm);
    REQUIRE(result.texture.width == 4);
    REQUIRE(result.texture.height == 4);
    REQUIRE(result.texture.mips.size() == 1);
    REQUIRE(result.texture.mips[0].size == 8);
    REQUIRE(result.texture.encoded.size() == 8);
}

TEST_CASE("DDS loader parses uncompressed RGBA8 texture metadata")
{
    auto bytes = make_dds_header(2, 2, 1, 0x00000041u, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
    bytes.resize(bytes.size() + 16);

    const auto result = arc::render::parse_dds_texture(bytes, "rgba.dds");

    INFO(result.message);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.texture.compressed);
    REQUIRE(result.texture.format == arc::render::texture_format::rgba8_unorm);
    REQUIRE(result.texture.mips.size() == 1);
    REQUIRE(result.texture.mips[0].size == 16);
}

TEST_CASE("texture loader infers material texture color space from file names")
{
    auto bytes = make_dds_header(4, 4, 1, 0x00000004u, 0x31545844u);
    bytes.resize(bytes.size() + 8);

    const auto root = std::filesystem::temp_directory_path();
    const auto base_color_path = root / "MASTER_Stone_BaseColor.dds";
    const auto normal_path = root / "MASTER_Stone_Normal.dds";
    {
        std::ofstream file(base_color_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    {
        std::ofstream file(normal_path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    const auto base_color = arc::render::load_texture_asset(base_color_path);
    const auto normal = arc::render::load_texture_asset(normal_path);

    REQUIRE(base_color.succeeded());
    REQUIRE(normal.succeeded());
    REQUIRE(base_color.texture.format == arc::render::texture_format::bc1_rgba_srgb);
    REQUIRE(normal.texture.format == arc::render::texture_format::bc1_rgba_unorm);

    std::filesystem::remove(base_color_path);
    std::filesystem::remove(normal_path);
}

TEST_CASE("texture loader prepares checked-in landscape maps for GPU upload")
{
    const auto path = std::filesystem::path(ARC_RENDER_TEST_ASSET_ROOT) / "textures" / "terrain" / "aerial_grass_rock" /
                      "aerial_grass_rock_diff_1k.jpg";
    const auto result = arc::render::load_texture_asset(path);

    INFO(result.message);
    REQUIRE(result.succeeded());
#if defined(ARC_RENDER_TEST_EXPECT_IMAGE_DECODER)
    REQUIRE(result.texture.width == 1024);
    REQUIRE(result.texture.height == 1024);
    REQUIRE(result.texture.has_pixels());
    REQUIRE(result.texture.mips.size() == 11);
    REQUIRE(result.texture.encoded.empty());
#else
    REQUIRE_FALSE(result.texture.encoded.empty());
#endif
}

TEST_CASE("DDS loader rejects invalid and truncated payloads")
{
    std::vector<std::byte> invalid(8);
    REQUIRE_FALSE(arc::render::parse_dds_texture(invalid, "bad.dds").succeeded());

    auto truncated = make_dds_header(8, 8, 1, 0x00000004u, 0x31545844u);
    truncated.resize(truncated.size() + 4);
    const auto result = arc::render::parse_dds_texture(truncated, "truncated.dds");
    REQUIRE_FALSE(result.succeeded());
}

TEST_CASE("scene asset loader wraps GLB meshes and reports scene import failures cleanly")
{
    const auto path = write_triangle_glb();
    const auto glb = arc::render::load_scene_asset(path);

    INFO(glb.message);
    REQUIRE(glb.succeeded());
    REQUIRE(glb.meshes.size() == 1);
    REQUIRE(glb.nodes.size() == 1);
    REQUIRE(glb.nodes[0].mesh_index == 0);
    REQUIRE(glb.materials.size() == 1);

    const auto fbx = arc::render::load_scene_asset(path.parent_path() / "missing.fbx");
    REQUIRE_FALSE(fbx.succeeded());
    REQUIRE_FALSE(fbx.message.empty());

    std::filesystem::remove(path);
}

#if defined(ARC_RENDER_TEST_UFBX_DATA_ROOT)
TEST_CASE("scene asset loader imports static FBX meshes with ufbx")
{
    const std::filesystem::path fixture =
        std::filesystem::path(ARC_RENDER_TEST_UFBX_DATA_ROOT) / "blender_279_nested_meshes_7400_binary.fbx";
    REQUIRE(std::filesystem::exists(fixture));

    const auto temp_root = std::filesystem::temp_directory_path() / "arc-render-fbx-import-test";
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
    std::filesystem::create_directories(temp_root, ec);

    arc::render::scene_import_options options;
    options.asset_root = temp_root;
    options.import_directory = temp_root / "imported" / "nested_meshes";

    std::vector<arc::render::scene_import_progress> progress;
    const auto result = arc::render::load_scene_asset(fixture, options,
                                                      [&](const arc::render::scene_import_progress& value)
                                                      {
                                                          progress.push_back(value);
                                                          return true;
                                                      });

    INFO(result.message);
    for (const auto& diagnostic : result.diagnostics)
        INFO(diagnostic);
    REQUIRE(result.succeeded());
    REQUIRE(result.meshes.size() >= 1);
    REQUIRE(result.nodes.size() >= 1);
    REQUIRE(result.nodes.front().mesh_index < result.meshes.size());
    REQUIRE(std::filesystem::exists(result.manifest_path));
    REQUIRE_FALSE(progress.empty());
    REQUIRE(progress.back().stage == arc::render::scene_import_stage::finalizing);

    std::filesystem::remove_all(temp_root, ec);
}

TEST_CASE("scene asset loader extracts FBX material assets and embedded textures")
{
    const std::filesystem::path fixture =
        std::filesystem::path(ARC_RENDER_TEST_UFBX_DATA_ROOT) / "blender_279_internal_textures_7400_binary.fbx";
    REQUIRE(std::filesystem::exists(fixture));

    const auto temp_root = std::filesystem::temp_directory_path() / "arc-render-fbx-texture-import-test";
    std::error_code ec;
    std::filesystem::remove_all(temp_root, ec);
    std::filesystem::create_directories(temp_root, ec);

    arc::render::scene_import_options options;
    options.asset_root = temp_root;
    options.import_directory = temp_root / "imported" / "internal_textures";

    const auto result = arc::render::load_scene_asset(fixture, options);

    INFO(result.message);
    for (const auto& diagnostic : result.diagnostics)
        INFO(diagnostic);
    REQUIRE(result.succeeded());
    REQUIRE_FALSE(result.materials.empty());
    REQUIRE_FALSE(result.textures.empty());
    REQUIRE(std::filesystem::exists(result.manifest_path));
    REQUIRE(std::filesystem::exists(result.materials.front().asset_path));
    REQUIRE_FALSE(result.textures.front().source_path.empty());
    REQUIRE(std::filesystem::exists(temp_root / result.textures.front().source_path));

    std::filesystem::remove_all(temp_root, ec);
}
#endif

TEST_CASE("primitive mesh builders create renderable geometry")
{
    const auto plane = arc::render::make_plane_mesh(2.0f);
    REQUIRE(plane.name == "Plane");
    REQUIRE(plane.vertices.size() == 4);
    REQUIRE(plane.indices == std::vector<std::uint32_t>{0, 1, 2, 0, 2, 3});
    REQUIRE(plane.vertices[0].normal[1] == Catch::Approx(1.0f));

    const auto cube = arc::render::make_cube_mesh();
    REQUIRE(cube.vertices.size() == 24);
    REQUIRE(cube.indices.size() == 36);

    const auto sphere = arc::render::make_uv_sphere_mesh(0.5f, 8, 4);
    REQUIRE(sphere.vertices.size() == 45);
    REQUIRE(sphere.indices.size() == 8 * 4 * 6);

    const auto cylinder = arc::render::make_cylinder_mesh(0.5f, 1.0f, 8);
    REQUIRE(cylinder.vertices.size() == 20);
    REQUIRE(cylinder.indices.size() == 8 * 12);

    const auto cone = arc::render::make_cone_mesh(0.5f, 1.0f, 8);
    REQUIRE(cone.name == "Cone");
    REQUIRE(cone.vertices.size() == 28);
    REQUIRE(cone.indices.size() == 8 * 6);

    const auto capsule = arc::render::make_capsule_mesh(0.5f, 1.0f, 8, 4);
    REQUIRE(capsule.name == "Capsule");
    REQUIRE(capsule.vertices.size() == 90);
    REQUIRE(capsule.indices.size() == 9 * 8 * 6);

    const auto terrain = arc::render::make_terrain_grid_mesh(8.0f, 8, 1.0f);
    REQUIRE(terrain.name == "Terrain");
    REQUIRE(terrain.vertices.size() == 81);
    REQUIRE(terrain.indices.size() == 8 * 8 * 6);
    bool has_height_variation = false;
    bool has_tilted_normal = false;
    bool has_color_variation = false;
    for (const auto& vertex : terrain.vertices)
    {
        has_height_variation = has_height_variation || std::abs(vertex.position[1]) > 0.01f;
        has_tilted_normal = has_tilted_normal || vertex.normal[1] < 0.995f;
        has_color_variation = has_color_variation ||
                              std::abs(vertex.color[0] - terrain.vertices.front().color[0]) > 0.01f ||
                              std::abs(vertex.color[1] - terrain.vertices.front().color[1]) > 0.01f ||
                              std::abs(vertex.color[2] - terrain.vertices.front().color[2]) > 0.01f;
    }
    REQUIRE(has_height_variation);
    REQUIRE(has_tilted_normal);
    REQUIRE(has_color_variation);
    const auto& center = terrain.vertices[4 * 9 + 4];
    REQUIRE(center.position[1] == Catch::Approx(arc::render::sample_terrain_height(0.0f, 0.0f, 8.0f, 1.0f)));
    REQUIRE(terrain.vertices.back().texcoord[0] - terrain.vertices.front().texcoord[0] > 1.0f);
}

TEST_CASE("Water Ocean grid uses progressive rings and a quantized camera-relative origin")
{
    const arc::render::water_ocean_grid_descriptor descriptor{
        .visible_distance = 2048.0f, .inner_grid_cells = 16u, .ring_count = 6u};
    const auto grid = arc::render::make_water_ocean_grid(descriptor);
    REQUIRE(grid.name == "Water Ocean Clipmap");
    REQUIRE_FALSE(grid.vertices.empty());
    REQUIRE(grid.indices.size() % 6u == 0u);
    REQUIRE(grid.vertices.size() < 100000u);

    float minimum_x = std::numeric_limits<float>::max();
    float maximum_x = std::numeric_limits<float>::lowest();
    float minimum_z = std::numeric_limits<float>::max();
    float maximum_z = std::numeric_limits<float>::lowest();
    bool flat = true;
    bool upward_facing = true;
    for (const auto& vertex : grid.vertices)
    {
        minimum_x = std::min(minimum_x, vertex.position[0]);
        maximum_x = std::max(maximum_x, vertex.position[0]);
        minimum_z = std::min(minimum_z, vertex.position[2]);
        maximum_z = std::max(maximum_z, vertex.position[2]);
        flat = flat && vertex.position[1] == 0.0f;
        upward_facing = upward_facing && vertex.normal[1] == 1.0f;
    }
    CHECK(flat);
    CHECK(upward_facing);
    CHECK(minimum_x == Catch::Approx(-descriptor.visible_distance));
    CHECK(maximum_x == Catch::Approx(descriptor.visible_distance));
    CHECK(minimum_z == Catch::Approx(-descriptor.visible_distance));
    CHECK(maximum_z == Catch::Approx(descriptor.visible_distance));

    const float cell_size = arc::render::water_ocean_grid_cell_size(descriptor);
    const auto first = arc::render::water_ocean_grid_origin({101.2f, 80.0f, -47.7f}, 3.5f, cell_size);
    const auto stable =
        arc::render::water_ocean_grid_origin({101.2f + cell_size * 0.25f, 2.0f, -47.7f}, 3.5f, cell_size);
    CHECK(first[0] == Catch::Approx(stable[0]));
    CHECK(first[2] == Catch::Approx(stable[2]));
    CHECK(first[1] == Catch::Approx(3.5f));
}

TEST_CASE("Water optical material maps absorption scattering and refraction into the transmission path")
{
    arc::water::water_appearance_settings appearance;
    appearance.absorption = {0.10f, 0.20f, 0.40f};
    appearance.scattering = {0.01f, 0.04f, 0.08f};
    appearance.roughness = 0.16f;
    appearance.refraction_strength = 0.12f;

    const auto material = arc::render::make_water_material(appearance, "Test Water");
    CHECK(material.name == "Test Water");
    CHECK(material.shading_model == arc::render::material_shading_model::transmission);
    CHECK(material.render_path == arc::render::material_render_path::clustered_forward);
    CHECK_FALSE(material.deferred_compatible);
    CHECK(material.alpha_mode == arc::render::material_alpha_mode::blend);
    CHECK(material.index_of_refraction == Catch::Approx(1.333f));
    CHECK(material.roughness == Catch::Approx(appearance.roughness));
    CHECK(material.transmission_factor == Catch::Approx(0.54f));
    CHECK(material.base_color[0] < material.base_color[1]);
    CHECK(material.base_color[1] < material.base_color[2]);
    CHECK(material.attenuation_color[0] > material.attenuation_color[1]);
    CHECK(material.attenuation_color[1] > material.attenuation_color[2]);
    CHECK(material.attenuation_distance == Catch::Approx(2.5f));
}

TEST_CASE("OBJ scene import triangulates polygons and supports negative indices", "[render][mesh][obj]")
{
    const auto path = std::filesystem::temp_directory_path() / "arc-render-obj-import-test.obj";
    {
        std::ofstream output(path, std::ios::trunc);
        REQUIRE(output.good());
        output << "v 0 0 0\n"
               << "v 1 0 0\n"
               << "v 1 1 0\n"
               << "v 0 1 0\n"
               << "vt 0 0\n"
               << "vt 1 0\n"
               << "vt 1 1\n"
               << "vt 0 1\n"
               << "f -4/-4 -3/-3 -2/-2 -1/-1\n";
    }

    arc::render::scene_import_options options;
    const auto imported = arc::render::load_scene_asset(path, options);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    REQUIRE(imported.succeeded());
    REQUIRE(imported.meshes.size() == 1);
    REQUIRE(imported.nodes.size() == 1);
    CHECK(imported.meshes.front().indices.size() == 6);
    CHECK(imported.meshes.front().vertices.size() == 6);
    CHECK(imported.nodes.front().mesh_index == 0);
}

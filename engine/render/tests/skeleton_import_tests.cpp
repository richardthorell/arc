#include <arc/render/mesh.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{

void append_u32(std::vector<std::byte>& bytes, std::uint32_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void append_u16(std::vector<std::byte>& bytes, std::uint16_t value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void append_f32(std::vector<std::byte>& bytes, float value)
{
    const auto* data = reinterpret_cast<const std::byte*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}

void pad4(std::vector<std::byte>& bytes, std::byte value)
{
    while ((bytes.size() % 4u) != 0u)
        bytes.push_back(value);
}

std::filesystem::path write_skinned_triangle_glb()
{
    std::vector<std::byte> bin;

    const auto position_offset = bin.size();
    for (const float value : {0.0f, 1.0f, 0.0f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f})
        append_f32(bin, value);

    const auto joints_offset = bin.size();
    for (std::size_t vertex = 0; vertex < 3; ++vertex)
    {
        append_u16(bin, vertex == 0 ? 1u : 0u);
        append_u16(bin, vertex == 0 ? 0u : 1u);
        append_u16(bin, 0u);
        append_u16(bin, 0u);
    }

    const auto weights_offset = bin.size();
    for (std::size_t vertex = 0; vertex < 3; ++vertex)
    {
        append_f32(bin, vertex == 0 ? 0.75f : 0.8f);
        append_f32(bin, vertex == 0 ? 0.25f : 0.2f);
        append_f32(bin, 0.0f);
        append_f32(bin, 0.0f);
    }

    const auto index_offset = bin.size();
    append_u16(bin, 0u);
    append_u16(bin, 1u);
    append_u16(bin, 2u);
    pad4(bin, std::byte{0});

    const auto inverse_bind_offset = bin.size();
    const float identity[16]{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const float spine_inverse[16]{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f};
    for (const float value : identity)
        append_f32(bin, value);
    for (const float value : spine_inverse)
        append_f32(bin, value);

    const std::string json =
        "{\"asset\":{\"version\":\"2.0\"},"
        "\"buffers\":[{\"byteLength\":" +
        std::to_string(bin.size()) +
        "}],"
        "\"bufferViews\":["
        "{\"buffer\":0,\"byteOffset\":" +
        std::to_string(position_offset) +
        ",\"byteLength\":36},"
        "{\"buffer\":0,\"byteOffset\":" +
        std::to_string(joints_offset) +
        ",\"byteLength\":24},"
        "{\"buffer\":0,\"byteOffset\":" +
        std::to_string(weights_offset) +
        ",\"byteLength\":48},"
        "{\"buffer\":0,\"byteOffset\":" +
        std::to_string(index_offset) +
        ",\"byteLength\":6},"
        "{\"buffer\":0,\"byteOffset\":" +
        std::to_string(inverse_bind_offset) +
        ",\"byteLength\":128}],"
        "\"accessors\":["
        "{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"},"
        "{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"VEC4\"},"
        "{\"bufferView\":2,\"componentType\":5126,\"count\":3,\"type\":\"VEC4\"},"
        "{\"bufferView\":3,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"},"
        "{\"bufferView\":4,\"componentType\":5126,\"count\":2,\"type\":\"MAT4\"}],"
        "\"nodes\":["
        "{\"name\":\"Hips\",\"children\":[1]},"
        "{\"name\":\"Spine\",\"translation\":[0,1,0]},"
        "{\"name\":\"Character\",\"mesh\":0,\"skin\":0}],"
        "\"skins\":[{\"name\":\"Rig\",\"inverseBindMatrices\":4,\"skeleton\":0,\"joints\":[0,1]}],"
        "\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0,\"JOINTS_0\":1,\"WEIGHTS_0\":2},"
        "\"indices\":3}]}]}";

    std::vector<std::byte> json_bytes(reinterpret_cast<const std::byte*>(json.data()),
                                      reinterpret_cast<const std::byte*>(json.data() + json.size()));
    pad4(json_bytes, std::byte{' '});

    std::vector<std::byte> glb;
    append_u32(glb, 0x46546C67u);
    append_u32(glb, 2u);
    append_u32(glb, static_cast<std::uint32_t>(12u + 8u + json_bytes.size() + 8u + bin.size()));
    append_u32(glb, static_cast<std::uint32_t>(json_bytes.size()));
    append_u32(glb, 0x4E4F534Au);
    glb.insert(glb.end(), json_bytes.begin(), json_bytes.end());
    append_u32(glb, static_cast<std::uint32_t>(bin.size()));
    append_u32(glb, 0x004E4942u);
    glb.insert(glb.end(), bin.begin(), bin.end());

    const auto path = std::filesystem::temp_directory_path() / "arc_basic_skeleton_import.glb";
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(glb.data()), static_cast<std::streamsize>(glb.size()));
    return path;
}

} // namespace

TEST_CASE("GLB import preserves skeleton hierarchy bind pose and mesh skin mapping", "[render][skeleton][gltf]")
{
    const auto path = write_skinned_triangle_glb();
    const auto imported = arc::render::load_gltf_mesh(path);
    std::filesystem::remove(path);

    REQUIRE(imported.succeeded());
    REQUIRE(imported.mesh.skin_vertices.size() == 3);
    REQUIRE(imported.skin_index == 0);
    REQUIRE(imported.skeletons.size() == 1);

    const auto& skeleton = imported.skeletons.front();
    REQUIRE(skeleton.name == "Rig");
    REQUIRE(skeleton.valid());
    REQUIRE(skeleton.root_joint == 0);
    REQUIRE(skeleton.joints.size() == 2);

    REQUIRE(skeleton.joints[0].name == "Hips");
    REQUIRE(skeleton.joints[0].parent == -1);
    REQUIRE(skeleton.joints[1].name == "Spine");
    REQUIRE(skeleton.joints[1].parent == 0);
    REQUIRE(skeleton.joints[1].bind_position[1] == Catch::Approx(1.0f));
    REQUIRE(skeleton.joints[1].inverse_bind_matrix(1, 3) == Catch::Approx(-1.0f));

    REQUIRE(imported.mesh.skin_vertices[0].joint_indices[0] == 1);
    REQUIRE(imported.mesh.skin_vertices[0].joint_indices[1] == 0);
    REQUIRE(imported.mesh.skin_vertices[0].joint_weights[0] == Catch::Approx(0.75f));
    REQUIRE(imported.mesh.skin_vertices[0].joint_weights[1] == Catch::Approx(0.25f));
}

TEST_CASE("scene import exposes a GLB skeleton as a model sub-resource", "[render][skeleton][gltf]")
{
    const auto path = write_skinned_triangle_glb();
    const auto imported = arc::render::load_scene_asset(path);
    std::filesystem::remove(path);

    REQUIRE(imported.succeeded());
    REQUIRE(imported.skeletons.size() == 1);
    REQUIRE(imported.nodes.size() == 1);
    REQUIRE(imported.nodes.front().skin_index == 0);
    REQUIRE(imported.skeletons.front().joints.size() == 2);
}

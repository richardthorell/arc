#include <arc/render/mesh_artifact.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>

namespace
{
template <class T> void append(std::vector<std::byte>& output, const T& value)
{
    const auto* p = reinterpret_cast<const std::byte*>(&value);
    output.insert(output.end(), p, p + sizeof(T));
}

void append_string(std::vector<std::byte>& output, std::string_view value)
{
    append(output, static_cast<std::uint64_t>(value.size()));
    const auto* first = reinterpret_cast<const std::byte*>(value.data());
    output.insert(output.end(), first, first + value.size());
}

void append_blob(std::vector<std::byte>& output, std::span<const std::byte> bytes)
{
    append(output, static_cast<std::uint64_t>(bytes.size()));
    output.insert(output.end(), bytes.begin(), bytes.end());
}

void add_mesh(std::vector<std::byte>& output, std::string_view name, std::uint32_t lods = 1)
{
    using namespace arc::render;
    std::array<mesh_vertex, 3> vertices{};
    vertices[0].position[0] = 1.0f;
    vertices[1].position[1] = 1.0f;
    vertices[2].position[2] = 1.0f;
    std::array<std::uint32_t, 3> indices{0, 1, 2};
    append_string(output, name);
    append(output, static_cast<std::uint64_t>(7));
    append(output, lods);
    for (std::uint32_t lod = 0; lod < lods; ++lod)
    {
        append(output, lod == 0 ? 1.0f : 0.5f);
        append(output, lod == 0 ? 0.0f : 0.25f);
        append_blob(output, std::as_bytes(std::span(vertices)));
        append_blob(output, std::as_bytes(std::span(indices)));
    }
}

std::vector<std::byte> artifact(bool multiple = false, std::uint32_t lods = 1)
{
    std::vector<std::byte> bytes;
    append_string(bytes, "ARC_MESH_2");
    append(bytes, multiple ? 2u : 1u);
    add_mesh(bytes, "Body", lods);
    if (multiple) add_mesh(bytes, "Wheels");
    return bytes;
}
} // namespace

TEST_CASE("V3.4c.1 decodes cooked conventional LOD0 into renderer-neutral mesh data")
{
    const auto decoded = arc::render::decode_conventional_mesh_artifact(artifact(false, 2));
    REQUIRE(decoded);
    CHECK(decoded.value().mesh.name == "Body");
    CHECK(decoded.value().mesh.material_index == 7);
    CHECK(decoded.value().mesh.vertices.size() == 3);
    const std::vector<std::uint32_t> expected{0, 1, 2};
    CHECK(decoded.value().mesh.indices == expected);
    CHECK(decoded.value().mesh.vertices[0].position[0] == 1.0f);
    CHECK(decoded.value().lod_ratio == 1.0f);
    CHECK(decoded.value().geometric_error == 0.0f);
}

TEST_CASE("V3.4c.1 requires an exact subresource for multi-mesh artifacts")
{
    const auto bytes = artifact(true);
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
    const auto selected = arc::render::decode_conventional_mesh_artifact(bytes, "Wheels");
    REQUIRE(selected);
    CHECK(selected.value().mesh.name == "Wheels");
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes, "Missing"));
}

TEST_CASE("V3.4c.1 rejects truncated payloads and trailing bytes")
{
    auto bytes = artifact();
    bytes.pop_back();
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
    bytes = artifact();
    bytes.push_back(std::byte{0});
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
}

TEST_CASE("V3.4c.1 rejects malformed magic and zero mesh count")
{
    auto bytes = artifact();
    bytes[sizeof(std::uint64_t)] = std::byte{'X'};
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
    bytes = artifact();
    const std::uint32_t zero{};
    // Length-prefixed magic occupies 8 + 10 bytes.
    std::memcpy(bytes.data() + sizeof(std::uint64_t) + 10, &zero, sizeof(zero));
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
}

TEST_CASE("V3.4c.1 rejects out-of-range triangle indices")
{
    auto bytes = artifact();
    // Final twelve bytes are the triangle index payload.
    const std::uint32_t invalid = 99;
    std::memcpy(bytes.data() + bytes.size() - sizeof(std::uint32_t), &invalid, sizeof(invalid));
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));
}

TEST_CASE("V3.4c.1 rejects corrupt unselected LODs and duplicate mesh names")
{
    auto bytes = artifact(false, 2);
    const std::uint32_t invalid = 99;
    std::memcpy(bytes.data() + bytes.size() - sizeof(std::uint32_t), &invalid, sizeof(invalid));
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes));

    bytes.clear();
    append_string(bytes, "ARC_MESH_2");
    append(bytes, static_cast<std::uint32_t>(2));
    add_mesh(bytes, "Body");
    add_mesh(bytes, "Body");
    CHECK_FALSE(arc::render::decode_conventional_mesh_artifact(bytes, "Body"));
}

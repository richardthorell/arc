#include <arc/render/cooked_material_instance.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{
template <class T> void append(std::vector<std::byte>& output, const T& value)
{
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(value));
}

void append_string(std::vector<std::byte>& output, std::string_view value)
{
    append(output, static_cast<std::uint64_t>(value.size()));
    for (const auto ch : value)
        output.push_back(static_cast<std::byte>(ch));
}

std::vector<std::byte> parent_package()
{
    std::vector<std::byte> bytes;
    append_string(bytes, "ARC_MATERIAL_4");
    append(bytes, arc::render::material_pass_contract_version);
    append(bytes, arc::render::material_abi_version);
    append(bytes, std::uint64_t{9});
    append(bytes, std::uint64_t{10});
    append(bytes, std::uint32_t{0});
    append(bytes, std::uint32_t{1});
    append(bytes, std::uint64_t{31});
    append_string(bytes, "Roughness");
    append(bytes, arc::render::shader_parameter_type::float32);
    append(bytes, std::uint32_t{0});
    append(bytes, std::uint32_t{4});
    append(bytes, true);
    append(bytes, 0.0f);
    append(bytes, 1.0f);
    append_string(bytes, R"({"version":4,"domain":"surface","graph":{"version":1,"nodes":[],"connections":[]}})");
    return bytes;
}

std::vector<std::byte> instance_package(std::string_view overrides,
                                        std::string_view parent = "00000000-0000-0009-0000-00000000000a",
                                        std::uint64_t specialization = 0)
{
    std::vector<std::byte> bytes;
    append_string(bytes, "ARC_MATERIAL_INSTANCE_1");
    append(bytes, std::uint32_t{1});
    append_string(bytes, parent);
    append(bytes, specialization);
    const auto parent_bytes = parent_package();
    append(bytes, static_cast<std::uint64_t>(parent_bytes.size()));
    bytes.insert(bytes.end(), parent_bytes.begin(), parent_bytes.end());
    std::string authored = R"({"version":1,"name":"Rough Floor","parent":{"guid":")";
    authored += parent;
    authored += R"(","pathHint":"materials/base.arcmat"},"parameterOverrides":)";
    authored += overrides;
    authored += R"(,"functionOverrides":[]})";
    append_string(bytes, authored);
    return bytes;
}
} // namespace

TEST_CASE("V3.4d.7 value-only instance decodes parent and scalar override")
{
    const auto bytes = instance_package(R"([{"parameterId":"31","value":0.75}])");
    const auto decoded = arc::render::decode_cooked_material_instance_values(bytes);
    REQUIRE(decoded);
    CHECK(decoded.value().parent.high == 9);
    CHECK(decoded.value().parent.low == 10);
    REQUIRE(decoded.value().overrides.size() == 1);
    CHECK(decoded.value().overrides.front().id == arc::render::shader_parameter_id{31});
    CHECK(std::get<float>(decoded.value().overrides.front().value) == 0.75f);

    arc::render::material_descriptor parent;
    parent.shader_package = {.high = 9, .low = 10};
    parent.runtime_program = std::make_shared<arc::render::material_runtime_program>();
    const auto applied = arc::render::apply_cooked_material_instance_values(parent, decoded.value());
    REQUIRE(applied);
    REQUIRE(applied.value().parameters.size() == 1);
    CHECK(applied.value().name == "Rough Floor");
    CHECK(applied.value().shader_package == parent.shader_package);
    CHECK(parent.parameters.empty());
}

TEST_CASE("V3.4d.7 rejects unsupported instance values and parent mismatch")
{
    const auto rejected = [](const auto& bytes)
    { CHECK_FALSE(arc::render::decode_cooked_material_instance_values(bytes)); };
    rejected(instance_package(R"([{"parameterId":"31","value":2.0}])"));
    rejected(instance_package(R"([{"parameterId":"999","value":0.5}])"));
    rejected(instance_package(R"([{"parameterId":"31","value":"not scalar"}])"));
    rejected(instance_package(R"([{"parameterId":"31","value":0.5},{"parameterId":"31","value":0.6}])"));
    rejected(instance_package("[]", "00000000-0000-0009-0000-00000000000b"));
    rejected(instance_package("[]", "00000000-0000-0009-0000-00000000000a", 9));
    auto truncated = instance_package("[]");
    truncated.pop_back();
    CHECK_FALSE(arc::render::decode_cooked_material_instance_values(truncated));

    const auto decoded = arc::render::decode_cooked_material_instance_values(instance_package("[]"));
    REQUIRE(decoded);
    arc::render::material_descriptor wrong;
    wrong.shader_package = {.high = 2, .low = 3};
    wrong.runtime_program = std::make_shared<arc::render::material_runtime_program>();
    CHECK_FALSE(arc::render::apply_cooked_material_instance_values(wrong, decoded.value()));
}

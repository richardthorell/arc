#include <arc/core/core.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace
{
struct asset_tag;
struct entity_tag;
struct slot_tag;
} // namespace

TEST_CASE("core result stores values errors and move-only payloads")
{
    auto value = arc::core::result<std::unique_ptr<int>, int>::success(std::make_unique<int>(42));
    REQUIRE(value.has_value());
    REQUIRE(*value.value() == 42);

    auto error = arc::core::result<std::unique_ptr<int>, int>::failure(7);
    REQUIRE_FALSE(error.has_value());
    REQUIRE(error.error() == 7);

    auto status = arc::core::status<int>::success();
    REQUIRE(status);
}

TEST_CASE("tagged UUIDs preserve layout and domain isolation")
{
    using asset_id = arc::core::uuid<asset_tag>;
    using entity_id = arc::core::uuid<entity_tag>;
    static_assert(sizeof(asset_id) == 16);
    static_assert(!std::is_convertible_v<asset_id, entity_id>);

    const asset_id original{0x0123456789abcdefull, 0xfedcba9876543210ull};
    const auto text = arc::core::to_string(original);
    REQUIRE(text == "0123456789abcdeffedcba9876543210");
    REQUIRE(arc::core::parse_uuid<asset_tag>(text) == original);
    REQUIRE(arc::core::parse_uuid<asset_tag>(arc::core::to_string(original, arc::core::uuid_text_format::hyphenated)) ==
            original);
    REQUIRE_FALSE(arc::core::parse_uuid<asset_tag>("0"));
    REQUIRE_FALSE(arc::core::parse_uuid<asset_tag>(std::string(32, '0')));
}

TEST_CASE("strong IDs retain invalid sentinels and type safety")
{
    using slot_id = arc::core::strong_id<slot_tag, std::uint32_t, 0>;
    static_assert(!std::is_convertible_v<slot_id, std::uint32_t>);

    REQUIRE_FALSE(slot_id{}.valid());
    REQUIRE(slot_id{9}.valid());
    REQUIRE(slot_id{9}.representation() == 9);
}

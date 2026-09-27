#include <arc/assets/dependency_inspection.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
using namespace arc::assets;

constexpr asset_guid target_guid{0x100, 0x1};
constexpr asset_guid first_owner_guid{0x200, 0x1};
constexpr asset_guid second_owner_guid{0x300, 0x1};

asset_snapshot make_asset(asset_guid guid, std::string_view path, bool read_only = false)
{
    asset_snapshot result;
    result.guid = guid;
    result.type = asset_types::material;
    result.source_path = path;
    result.read_only = read_only;
    return result;
}
} // namespace

TEST_CASE("asset usages are deterministic and deduplicated")
{
    asset_registry_snapshot registry;
    auto target = make_asset(target_guid, "Assets/Target.arcmat");
    target.reverse_dependencies = {second_owner_guid, first_owner_guid, second_owner_guid};
    registry.assets = {
        target,
        make_asset(second_owner_guid, "Assets/Zebra.arcscene"),
        make_asset(first_owner_guid, "Assets/Alpha.arcscene", true),
    };

    const auto usages = find_asset_usages(registry, target_guid);

    REQUIRE(usages.size() == 2);
    CHECK(usages[0].owner == first_owner_guid);
    CHECK(usages[0].owner_path == "Assets/Alpha.arcscene");
    CHECK(usages[0].owner_read_only);
    CHECK(usages[1].owner == second_owner_guid);
}

TEST_CASE("asset usages ignore stale reverse dependency ids")
{
    asset_registry_snapshot registry;
    auto target = make_asset(target_guid, "Assets/Target.arcmat");
    target.reverse_dependencies = {first_owner_guid};
    registry.assets = {target};

    CHECK(find_asset_usages(registry, target_guid).empty());
}

TEST_CASE("delete planning blocks referenced and read-only assets")
{
    asset_registry_snapshot registry;
    auto target = make_asset(target_guid, "Assets/Target.arcmat");
    target.reverse_dependencies = {first_owner_guid};
    registry.assets = {target, make_asset(first_owner_guid, "Assets/Owner.arcscene")};

    const auto referenced = plan_asset_delete(registry, target_guid);
    CHECK(referenced.found);
    CHECK_FALSE(referenced.safe_to_delete());
    REQUIRE(referenced.usages.size() == 1);

    registry.assets[0].reverse_dependencies.clear();
    CHECK(plan_asset_delete(registry, target_guid).safe_to_delete());

    registry.assets[0].read_only = true;
    const auto read_only = plan_asset_delete(registry, target_guid);
    CHECK(read_only.read_only);
    CHECK_FALSE(read_only.safe_to_delete());
}

TEST_CASE("delete planning reports missing assets without mutation")
{
    asset_registry_snapshot registry;
    const asset_guid missing{0x999, 0x1};

    const auto plan = plan_asset_delete(registry, missing);
    CHECK_FALSE(plan.found);
    CHECK_FALSE(plan.safe_to_delete());
    CHECK(plan.usages.empty());
}

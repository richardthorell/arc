#include <arc/flow/dependencies.h>

#include <catch2/catch_test_macros.hpp>

#include <string_view>
#include <vector>

TEST_CASE("Flow dependencies use stable asset identity and report unresolved references")
{
    using arc::flow::dependency_kind;
    using arc::flow::dependency_reference;

    std::vector<dependency_reference> references{
        {"asset-texture", dependency_kind::asset, "node-b"},
        {"asset-texture", dependency_kind::asset, "node-a"},
        {"graph-shared", dependency_kind::subgraph, "node-c"},
        {"missing-material", dependency_kind::asset, "node-d"},
        {"", dependency_kind::asset, "node-empty"},
    };

    const auto result = arc::flow::validate_dependencies(references, [](std::string_view asset_id) {
        return asset_id == "asset-texture" || asset_id == "graph-shared";
    });

    REQUIRE_FALSE(result.valid());
    REQUIRE(result.dependencies.size() == 3);
    REQUIRE(result.unresolved.size() == 1);
    CHECK(result.unresolved.front().asset_id == "missing-material");
    CHECK(result.unresolved.front().source_node_id == "node-d");
}

TEST_CASE("Flow dependency identity keeps asset and subgraph references distinct")
{
    using arc::flow::dependency_kind;
    using arc::flow::dependency_reference;

    const auto result = arc::flow::validate_dependencies(
        std::vector<dependency_reference>{{"shared-id", dependency_kind::asset, "asset-node"},
                                          {"shared-id", dependency_kind::subgraph, "subgraph-node"}},
        [](std::string_view) { return true; });

    REQUIRE(result.valid());
    REQUIRE(result.dependencies.size() == 2);
    CHECK(result.dependencies[0].kind != result.dependencies[1].kind);
}

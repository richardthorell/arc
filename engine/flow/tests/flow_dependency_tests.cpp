#include <arc/flow/dependencies.h>

#include <cstdlib>
#include <string_view>
#include <vector>

namespace
{

void require(bool condition)
{
    if (!condition)
    {
        std::abort();
    }
}

bool resolves_known_asset(std::string_view asset_id)
{
    return asset_id == "asset-texture" || asset_id == "graph-shared";
}

bool always_resolves(std::string_view)
{
    return true;
}

void test_stable_asset_identity_and_unresolved_references()
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

    const auto result = arc::flow::validate_dependencies(references, resolves_known_asset);

    require(!result.valid());
    require(result.dependencies.size() == 3);
    require(result.unresolved.size() == 1);
    require(result.unresolved.front().asset_id == "missing-material");
    require(result.unresolved.front().source_node_id == "node-d");
}

void test_asset_and_subgraph_identity_remain_distinct()
{
    using arc::flow::dependency_kind;
    using arc::flow::dependency_reference;

    const auto result = arc::flow::validate_dependencies(
        std::vector<dependency_reference>{{"shared-id", dependency_kind::asset, "asset-node"},
                                          {"shared-id", dependency_kind::subgraph, "subgraph-node"}},
        always_resolves);

    require(result.valid());
    require(result.dependencies.size() == 2);
    require(result.dependencies[0].kind != result.dependencies[1].kind);
}

} // namespace

void run_flow_dependency_tests()
{
    test_stable_asset_identity_and_unresolved_references();
    test_asset_and_subgraph_identity_remain_distinct();
}

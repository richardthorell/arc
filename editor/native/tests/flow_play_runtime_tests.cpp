#include "../src/flow_play_runtime.h"

#include <arc/scene/components.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>

TEST_CASE("Flow graph paths stay inside Content")
{
    using arc::editor::valid_flow_graph_path;
    CHECK(valid_flow_graph_path("Gameplay/Player.arcflow"));
    CHECK(valid_flow_graph_path("Player.arcflow"));
    CHECK_FALSE(valid_flow_graph_path(""));
    CHECK_FALSE(valid_flow_graph_path("../Player.arcflow"));
    CHECK_FALSE(valid_flow_graph_path("Gameplay/../Player.arcflow"));
    CHECK_FALSE(valid_flow_graph_path("/tmp/Player.arcflow"));
    CHECK_FALSE(valid_flow_graph_path("Gameplay\\Player.arcflow"));
    CHECK_FALSE(valid_flow_graph_path("Gameplay/Player.txt"));
}

TEST_CASE("Play world without Flow bindings installs cleanly")
{
    arc::framework::runtime_world world({.name = "flow-test"});
    const auto root = std::filesystem::temp_directory_path() / "arc-flow-empty-content";
    const auto result = arc::editor::install_flow_play_runtime(world, root);
    CHECK(result.succeeded);
    CHECK(result.instances == 0);
    CHECK(result.unique_programs == 0);
}

TEST_CASE("Unassigned Flow binding is inert during Play install")
{
    arc::framework::runtime_world world({.name = "flow-unassigned-test"});
    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::flow_component>(entity);

    const auto root = std::filesystem::temp_directory_path() / "arc-flow-unassigned-content";
    const auto result = arc::editor::install_flow_play_runtime(world, root);
    CHECK(result.succeeded);
    CHECK(result.instances == 0);
    CHECK(result.unique_programs == 0);
}

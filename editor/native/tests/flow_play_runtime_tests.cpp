#include "../src/flow_play_runtime.h"

#include <arc/memory/memory.h>
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
    arc::memory::memory_system memory;
    arc::framework::runtime_world world(memory, arc::framework::runtime_world_id{1},
                                        {.name = "flow-test", .install_placeholder_systems = false});
    const auto root = std::filesystem::temp_directory_path() / "arc-flow-empty-content";
    const auto result = arc::editor::install_flow_play_runtime(world, root);
    CHECK(result.succeeded);
    CHECK(result.instances == 0);
    CHECK(result.unique_programs == 0);
}

TEST_CASE("Unassigned Flow binding is inert during Play install")
{
    arc::memory::memory_system memory;
    arc::framework::runtime_world world(memory, arc::framework::runtime_world_id{1},
                                        {.name = "flow-unassigned-test", .install_placeholder_systems = false});
    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::flow_component>(entity);

    const auto root = std::filesystem::temp_directory_path() / "arc-flow-unassigned-content";
    const auto result = arc::editor::install_flow_play_runtime(world, root);
    CHECK(result.succeeded);
    CHECK(result.instances == 0);
    CHECK(result.unique_programs == 0);
}

#include "../src/flow_play_runtime.h"

#include <arc/framework/framework.h>
#include <arc/memory/memory.h>
#include <arc/scene/components.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <utility>

namespace
{

class flow_test_application final : public arc::framework::application
{
};

class temporary_flow_content
{
public:
    temporary_flow_content() : root_(make_root())
    {
        std::filesystem::create_directories(root_);
    }

    ~temporary_flow_content()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    [[nodiscard]] const std::filesystem::path& root() const noexcept
    {
        return root_;
    }

    void write(std::string_view name, std::string_view source) const
    {
        std::ofstream output(root_ / std::filesystem::path{name}, std::ios::binary);
        REQUIRE(output.good());
        output.write(source.data(), static_cast<std::streamsize>(source.size()));
        REQUIRE(output.good());
    }

private:
    [[nodiscard]] static std::filesystem::path make_root()
    {
        return std::filesystem::temp_directory_path() /
               ("arc-flow-hot-lifecycle-" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    }

    std::filesystem::path root_;
};

constexpr std::string_view lifecycle_graph = R"json({
    "version": 1,
    "assetType": "flow",
    "name": "Lifecycle",
    "graph": {
        "version": 1,
        "variables": [],
        "nodes": [
            {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
            {"id": "end", "type": "endPlay", "position": [0, 180], "values": {}},
            {"id": "self", "type": "selfEntity", "position": [0, 360], "values": {}},
            {"id": "running", "type": "stringLiteral", "position": [220, 300], "values": {"value": "running"}},
            {"id": "stopped", "type": "stringLiteral", "position": [220, 420], "values": {"value": "stopped"}},
            {"id": "set-running", "type": "setName", "position": [460, 0], "values": {}},
            {"id": "set-stopped", "type": "setName", "position": [460, 180], "values": {}}
        ],
        "connections": [
            {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set-running", "pin": "exec"}},
            {"id": "e2", "kind": "execution", "from": {"nodeId": "end", "pin": "exec"}, "to": {"nodeId": "set-stopped", "pin": "exec"}},
            {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set-running", "pin": "entity"}},
            {"id": "v2", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set-stopped", "pin": "entity"}},
            {"id": "v3", "kind": "value", "from": {"nodeId": "running", "pin": "value"}, "to": {"nodeId": "set-running", "pin": "name"}},
            {"id": "v4", "kind": "value", "from": {"nodeId": "stopped", "pin": "value"}, "to": {"nodeId": "set-stopped", "pin": "name"}}
        ],
        "viewport": {"x": 0, "y": 0, "zoom": 1}
    }
})json";

constexpr std::string_view alternate_graph = R"json({
    "version": 1,
    "assetType": "flow",
    "name": "Alternate",
    "graph": {
        "version": 1,
        "variables": [],
        "nodes": [
            {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
            {"id": "end", "type": "endPlay", "position": [0, 180], "values": {}},
            {"id": "self", "type": "selfEntity", "position": [0, 360], "values": {}},
            {"id": "running", "type": "stringLiteral", "position": [220, 300], "values": {"value": "alternate"}},
            {"id": "stopped", "type": "stringLiteral", "position": [220, 420], "values": {"value": "alternate-stopped"}},
            {"id": "set-running", "type": "setName", "position": [460, 0], "values": {}},
            {"id": "set-stopped", "type": "setName", "position": [460, 180], "values": {}}
        ],
        "connections": [
            {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set-running", "pin": "exec"}},
            {"id": "e2", "kind": "execution", "from": {"nodeId": "end", "pin": "exec"}, "to": {"nodeId": "set-stopped", "pin": "exec"}},
            {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set-running", "pin": "entity"}},
            {"id": "v2", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set-stopped", "pin": "entity"}},
            {"id": "v3", "kind": "value", "from": {"nodeId": "running", "pin": "value"}, "to": {"nodeId": "set-running", "pin": "name"}},
            {"id": "v4", "kind": "value", "from": {"nodeId": "stopped", "pin": "value"}, "to": {"nodeId": "set-stopped", "pin": "name"}}
        ],
        "viewport": {"x": 0, "y": 0, "zoom": 1}
    }
})json";

} // namespace

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

TEST_CASE("Flow bindings reconcile while Play is running")
{
    temporary_flow_content content;
    content.write("Lifecycle.arcflow", lifecycle_graph);
    content.write("Alternate.arcflow", alternate_graph);

    flow_test_application app;
    arc::framework::runtime host(app);
    auto& world = host.worlds().create({.name = "flow-hot-lifecycle", .install_placeholder_systems = false});
    const auto install = arc::editor::install_flow_play_runtime(world, content.root());
    REQUIRE(install.succeeded);
    CHECK(install.instances == 0);
    CHECK(install.unique_programs == 0);

    host.start();

    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::name_component>(entity, arc::scene::name_component{"idle"});
    arc::scene::flow_component binding{"Lifecycle.arcflow", true};
    world.entities().emplace<arc::scene::flow_component>(entity, std::move(binding));
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    world.entities().get<arc::scene::flow_component>(entity).enabled = false;
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "stopped");

    world.entities().get<arc::scene::flow_component>(entity).enabled = true;
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    world.entities().get<arc::scene::flow_component>(entity).graph_path = "Alternate.arcflow";
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "alternate");

    REQUIRE(world.entities().remove<arc::scene::flow_component>(entity));
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "alternate-stopped");

    arc::scene::flow_component rebound{"Lifecycle.arcflow", true};
    world.entities().emplace<arc::scene::flow_component>(entity, std::move(rebound));
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    world.entities().emplace<arc::scene::active_component>(entity, arc::scene::active_component{false});
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "stopped");

    world.entities().get<arc::scene::active_component>(entity).active = true;
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    REQUIRE(world.entities().destroy(entity));
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(world.state() == arc::framework::runtime_world_state::running);

    host.shutdown();
}

TEST_CASE("Flow source changes hot reload bound Play instances")
{
    temporary_flow_content content;
    content.write("Lifecycle.arcflow", lifecycle_graph);

    flow_test_application app;
    arc::framework::runtime host(app);
    auto& world = host.worlds().create({.name = "flow-source-hot-reload", .install_placeholder_systems = false});
    const auto install = arc::editor::install_flow_play_runtime(world, content.root());
    REQUIRE(install.succeeded);
    host.start();

    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::name_component>(entity, arc::scene::name_component{"idle"});
    world.entities().emplace<arc::scene::flow_component>(entity,
                                                        arc::scene::flow_component{"Lifecycle.arcflow", true});
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    content.write("Lifecycle.arcflow", alternate_graph);
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "alternate");

    world.entities().get<arc::scene::flow_component>(entity).enabled = false;
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "alternate-stopped");

    host.shutdown();
}

TEST_CASE("Flow hot reload keeps the last good generation and recovers after a compile failure")
{
    temporary_flow_content content;
    content.write("Lifecycle.arcflow", lifecycle_graph);

    flow_test_application app;
    arc::framework::runtime host(app);
    auto& world = host.worlds().create({.name = "flow-source-hot-reload-recovery", .install_placeholder_systems = false});
    const auto install = arc::editor::install_flow_play_runtime(world, content.root());
    REQUIRE(install.succeeded);
    host.start();

    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::name_component>(entity, arc::scene::name_component{"idle"});
    world.entities().emplace<arc::scene::flow_component>(entity,
                                                        arc::scene::flow_component{"Lifecycle.arcflow", true});
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    content.write("Lifecycle.arcflow", "{\"version\":1");
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(world.state() == arc::framework::runtime_world_state::running);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "running");

    content.write("Lifecycle.arcflow", alternate_graph);
    REQUIRE(host.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(world.state() == arc::framework::runtime_world_state::running);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "alternate");

    host.shutdown();
}

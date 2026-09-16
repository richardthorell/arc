#include "../src/flow_play_runtime.h"

#include <arc/framework/framework.h>
#include <arc/scene/components.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace
{

class flow_input_test_application final : public arc::framework::application
{
};

class temporary_flow_input_project
{
public:
    temporary_flow_input_project()
        : root_(std::filesystem::temp_directory_path() /
                ("arc-flow-semantic-input-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(content());
        std::filesystem::create_directories(root_ / "Config");
    }

    ~temporary_flow_input_project()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    [[nodiscard]] std::filesystem::path content() const
    {
        return root_ / "Content";
    }

    void write_flow(std::string_view source) const
    {
        std::ofstream output(content() / "Input.arcflow", std::ios::binary);
        output << source;
    }

    void write_input(std::string_view control) const
    {
        std::ofstream output(root_ / "Config" / "Input.json", std::ios::binary);
        output << R"json({"version":1,"contexts":[{"name":"Gameplay","priority":0,"actions":[{"name":"Jump","bindings":[{"device":"keyboard","control":")json"
               << control << R"json("}]}]}]})json";
    }

private:
    std::filesystem::path root_;
};

constexpr std::string_view semantic_input_graph = R"json({
  "version": 1,
  "assetType": "flow",
  "name": "Semantic Input",
  "graph": {
    "version": 1,
    "variables": [],
    "nodes": [
      {"id":"input","type":"inputAction","position":[0,0],"values":{"action":"Jump"}},
      {"id":"self","type":"selfEntity","position":[0,220],"values":{}},
      {"id":"pressed","type":"stringLiteral","position":[220,180],"values":{"value":"pressed"}},
      {"id":"released","type":"stringLiteral","position":[220,320],"values":{"value":"released"}},
      {"id":"set-pressed","type":"setName","position":[480,0],"values":{}},
      {"id":"set-released","type":"setName","position":[480,120],"values":{}}
    ],
    "connections": [
      {"id":"e1","kind":"execution","from":{"nodeId":"input","pin":"triggered"},"to":{"nodeId":"set-pressed","pin":"exec"}},
      {"id":"e2","kind":"execution","from":{"nodeId":"input","pin":"completed"},"to":{"nodeId":"set-released","pin":"exec"}},
      {"id":"v1","kind":"value","from":{"nodeId":"self","pin":"entity"},"to":{"nodeId":"set-pressed","pin":"entity"}},
      {"id":"v2","kind":"value","from":{"nodeId":"self","pin":"entity"},"to":{"nodeId":"set-released","pin":"entity"}},
      {"id":"v3","kind":"value","from":{"nodeId":"pressed","pin":"value"},"to":{"nodeId":"set-pressed","pin":"name"}},
      {"id":"v4","kind":"value","from":{"nodeId":"released","pin":"value"},"to":{"nodeId":"set-released","pin":"name"}}
    ],
    "viewport":{"x":0,"y":0,"zoom":1}
  }
})json";

} // namespace

TEST_CASE("Flow Input Action uses semantic project mappings")
{
    temporary_flow_input_project project;
    project.write_flow(semantic_input_graph);
    project.write_input("Space");

    flow_input_test_application app;
    arc::framework::runtime runtime(app);
    auto& world = runtime.worlds().create({.name = "flow-semantic-input", .install_placeholder_systems = false});
    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::name_component>(entity, arc::scene::name_component{"idle"});
    world.entities().emplace<arc::scene::flow_component>(entity, arc::scene::flow_component{"Input.arcflow", true});

    const auto install = arc::editor::install_flow_play_runtime(world, project.content());
    REQUIRE(install.succeeded);
    runtime.start();

    runtime.dispatch({.type = arc::framework::event_type::key_down, .key_code = 32});
    REQUIRE(runtime.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "pressed");

    runtime.dispatch({.type = arc::framework::event_type::key_up, .key_code = 32});
    REQUIRE(runtime.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "released");

    runtime.shutdown();
}

TEST_CASE("Flow semantic action can be remapped without changing the graph")
{
    temporary_flow_input_project project;
    project.write_flow(semantic_input_graph);
    project.write_input("Enter");

    flow_input_test_application app;
    arc::framework::runtime runtime(app);
    auto& world = runtime.worlds().create({.name = "flow-semantic-remap", .install_placeholder_systems = false});
    const auto entity = world.entities().create();
    world.entities().emplace<arc::scene::name_component>(entity, arc::scene::name_component{"idle"});
    world.entities().emplace<arc::scene::flow_component>(entity, arc::scene::flow_component{"Input.arcflow", true});

    const auto install = arc::editor::install_flow_play_runtime(world, project.content());
    REQUIRE(install.succeeded);
    runtime.start();

    runtime.dispatch({.type = arc::framework::event_type::key_down, .key_code = 32});
    REQUIRE(runtime.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "idle");

    runtime.dispatch({.type = arc::framework::event_type::key_up, .key_code = 32});
    runtime.dispatch({.type = arc::framework::event_type::key_down, .key_code = 13});
    REQUIRE(runtime.advance(1.0 / 60.0).completed_ticks == 1);
    CHECK(std::as_const(world.entities()).get<arc::scene::name_component>(entity).value == "pressed");

    runtime.shutdown();
}

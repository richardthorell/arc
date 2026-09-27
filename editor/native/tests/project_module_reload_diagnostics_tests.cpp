#include "../src/project_module_reload_diagnostics.h"

#include <arc/project/project_module.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
arc::editor::project_component_schema component(std::string id, std::string name, std::uint32_t version = 1)
{
    return {.stable_id = std::move(id), .display_name = std::move(name), .schema_version = version};
}

arc::editor::project_field_schema field(std::uint64_t id, std::string name, arc::project::game_field_kind_v1 kind)
{
    return {.stable_id = id, .display_name = std::move(name), .kind = kind};
}
} // namespace

TEST_CASE("project module reload diagnostics identify removed components")
{
    const std::vector previous{component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement")};
    const std::vector<arc::editor::project_component_schema> next;

    const auto diagnostic = arc::editor::analyze_project_module_reload(previous, next);

    CHECK(diagnostic.classification == arc::editor::module_reload_classification::native_host_restart_required);
    CHECK(diagnostic.reason == arc::editor::module_reload_reason::component_removed);
    CHECK(diagnostic.component_name == "Movement");
    CHECK(diagnostic.message.find("Movement") != std::string::npos);
}

TEST_CASE("project module reload diagnostics identify schema downgrades")
{
    const std::vector previous{component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement", 3)};
    const std::vector next{component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement", 2)};

    const auto diagnostic = arc::editor::analyze_project_module_reload(previous, next);

    CHECK(diagnostic.classification == arc::editor::module_reload_classification::native_host_restart_required);
    CHECK(diagnostic.reason == arc::editor::module_reload_reason::component_schema_downgraded);
    CHECK(diagnostic.message.find("3") != std::string::npos);
    CHECK(diagnostic.message.find("2") != std::string::npos);
}

TEST_CASE("project module reload diagnostics identify field type changes")
{
    auto previous_component = component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement");
    previous_component.fields.push_back(field(7, "Speed", arc::project::game_field_kind_v1::floating_point));
    auto next_component = component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement");
    next_component.fields.push_back(field(7, "Speed", arc::project::game_field_kind_v1::signed_integer));

    const auto diagnostic = arc::editor::analyze_project_module_reload({previous_component}, {next_component});

    CHECK(diagnostic.classification == arc::editor::module_reload_classification::play_session_restart_required);
    CHECK(diagnostic.reason == arc::editor::module_reload_reason::field_kind_changed);
    CHECK(diagnostic.field_id == 7);
    CHECK(diagnostic.field_name == "Speed");
    CHECK(diagnostic.message.find("Movement.Speed") != std::string::npos);
}

TEST_CASE("project module reload diagnostics preserve safe hot reload")
{
    auto previous_component = component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement");
    previous_component.fields.push_back(field(7, "Speed", arc::project::game_field_kind_v1::floating_point));
    auto next_component = component("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "Movement", 2);
    next_component.fields.push_back(field(7, "Speed", arc::project::game_field_kind_v1::floating_point));

    const auto diagnostic = arc::editor::analyze_project_module_reload({previous_component}, {next_component});

    CHECK(diagnostic.classification == arc::editor::module_reload_classification::safe_hot_reload);
    CHECK(diagnostic.reason == arc::editor::module_reload_reason::none);
}

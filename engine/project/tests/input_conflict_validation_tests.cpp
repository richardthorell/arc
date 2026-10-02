#include <arc/project/input_conflict_validation.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
arc::input::input_binding key_binding(arc::input::key key)
{
    return {.device = arc::input::input_device_type::keyboard, .control = arc::input::make_key_control(key)};
}
} // namespace

TEST_CASE("project input conflict validation uses shared conflict semantics")
{
    arc::project::input_config config;
    arc::project::input_context_config gameplay;
    gameplay.name = "Gameplay";
    gameplay.priority = 10;
    gameplay.actions.push_back({"Jump", {key_binding(arc::input::key::space)}});
    gameplay.actions.push_back({"Interact", {key_binding(arc::input::key::space)}});
    config.contexts.push_back(gameplay);

    const auto conflicts = arc::project::validate_input_config_conflicts(config);

    REQUIRE(conflicts.size() == 1);
    CHECK(conflicts[0].existing_action == "Jump");
    CHECK(conflicts[0].candidate_action == "Interact");
    CHECK(conflicts[0].kind == arc::input::input_conflict_kind::same_context);
    CHECK(conflicts[0].ambiguous);
    CHECK(conflicts[0].existing_binding_id == "Gameplay/Jump/0");
    CHECK(conflicts[0].candidate_binding_id == "Gameplay/Interact/0");
}

TEST_CASE("project input conflict validation preserves intentional context layering")
{
    arc::project::input_config config;
    arc::project::input_context_config gameplay;
    gameplay.name = "Gameplay";
    gameplay.priority = 10;
    gameplay.actions.push_back({"Interact", {key_binding(arc::input::key::space)}});

    arc::project::input_context_config menu;
    menu.name = "Menu";
    menu.priority = 100;
    menu.actions.push_back({"Accept", {key_binding(arc::input::key::space)}});

    config.contexts = {gameplay, menu};
    const auto conflicts = arc::project::validate_input_config_conflicts(config);

    REQUIRE(conflicts.size() == 1);
    CHECK(conflicts[0].kind == arc::input::input_conflict_kind::layered_context);
    CHECK_FALSE(conflicts[0].ambiguous);
}

TEST_CASE("disabled project contexts are excluded from conflict validation")
{
    arc::project::input_config config;
    arc::project::input_context_config gameplay;
    gameplay.name = "Gameplay";
    gameplay.actions.push_back({"Jump", {key_binding(arc::input::key::space)}});

    arc::project::input_context_config disabled;
    disabled.name = "Disabled";
    disabled.enabled = false;
    disabled.actions.push_back({"Other", {key_binding(arc::input::key::space)}});

    config.contexts = {gameplay, disabled};
    CHECK(arc::project::validate_input_config_conflicts(config).empty());
}

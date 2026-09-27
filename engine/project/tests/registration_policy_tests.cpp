#include <arc/project/registration_policy.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

TEST_CASE("project registration kinds have stable diagnostic names")
{
    using namespace arc::project;

    CHECK(registration_kind_name(game_registration_kind_v1::ecs_system) == "ecs_system");
    CHECK(registration_kind_name(game_registration_kind_v1::service) == "service");
    CHECK(registration_kind_name(game_registration_kind_v1::asset_type) == "asset_type");
    CHECK(registration_kind_name(game_registration_kind_v1::importer) == "importer");
    CHECK(registration_kind_name(game_registration_kind_v1::cook_processor) == "cook_processor");
    CHECK(registration_kind_name(game_registration_kind_v1::console_command) == "console_command");
    CHECK(registration_kind_name(game_registration_kind_v1::editor_extension) == "editor_extension");
    CHECK(registration_kind_name(game_registration_kind_v1::play_lifecycle) == "play_lifecycle");

    const auto unknown = static_cast<game_registration_kind_v1>(static_cast<std::uint8_t>(255));
    CHECK_FALSE(valid_registration_kind(unknown));
    CHECK(registration_kind_name(unknown) == "unknown");
}

TEST_CASE("editor modules may advertise every project extension category")
{
    using namespace arc::project;

    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::ecs_system));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::service));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::asset_type));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::importer));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::cook_processor));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::console_command));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::editor_extension));
    CHECK(registration_kind_allowed(game_module_kind_v1::editor, game_registration_kind_v1::play_lifecycle));
}

TEST_CASE("runtime and server modules reject editor-host extension categories")
{
    using namespace arc::project;

    for (const auto module_kind : {game_module_kind_v1::runtime, game_module_kind_v1::server})
    {
        CHECK(registration_kind_allowed(module_kind, game_registration_kind_v1::ecs_system));
        CHECK(registration_kind_allowed(module_kind, game_registration_kind_v1::service));
        CHECK(registration_kind_allowed(module_kind, game_registration_kind_v1::asset_type));
        CHECK(registration_kind_allowed(module_kind, game_registration_kind_v1::console_command));
        CHECK(registration_kind_allowed(module_kind, game_registration_kind_v1::play_lifecycle));
        CHECK_FALSE(registration_kind_allowed(module_kind, game_registration_kind_v1::importer));
        CHECK_FALSE(registration_kind_allowed(module_kind, game_registration_kind_v1::cook_processor));
        CHECK_FALSE(registration_kind_allowed(module_kind, game_registration_kind_v1::editor_extension));
    }
}

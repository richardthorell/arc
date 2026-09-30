#include <arc/input/gamepad.h>
#include <arc/project/input_rebinding.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

arc::input::input_binding key_binding(arc::input::key key)
{
    return {.device = arc::input::input_device_type::keyboard, .control = arc::input::make_key_control(key)};
}

arc::input::input_binding gamepad_binding(arc::input::gamepad_button button)
{
    return {.device = arc::input::input_device_type::gamepad,
            .control = arc::input::make_gamepad_button_control(button)};
}

arc::project::input_config basic_defaults()
{
    arc::project::input_config config{};
    arc::project::input_context_config gameplay{};
    gameplay.name = "Gameplay";
    gameplay.actions.push_back({.name = "Jump", .bindings = {key_binding(arc::input::key::space)}});
    gameplay.actions.push_back({.name = "Interact", .bindings = {key_binding(arc::input::key::e)}});
    gameplay.axes.push_back({.name = "MoveForward",
                             .bindings = {{.binding = key_binding(arc::input::key::w), .contribution = 1.0f},
                                          {.binding = key_binding(arc::input::key::s), .contribution = -1.0f}}});
    gameplay.axes2d.push_back(
        {.name = "Move",
         .bindings = {{.binding = key_binding(arc::input::key::w), .contribution = {0.0f, 1.0f}},
                      {.binding = key_binding(arc::input::key::d), .contribution = {1.0f, 0.0f}}}});
    config.contexts.push_back(std::move(gameplay));
    return config;
}

arc::project::input_config gamepad_defaults()
{
    arc::project::input_config config{};
    arc::project::input_context_config gameplay{};
    gameplay.name = "Gameplay";
    gameplay.actions.push_back({.name = "Jump", .bindings = {gamepad_binding(arc::input::gamepad_button::south)}});
    config.contexts.push_back(std::move(gameplay));
    return config;
}

arc::project::input_mapping_target jump_target()
{
    return {.context = "Gameplay", .name = "Jump", .kind = arc::project::input_mapping_kind::action};
}

class temporary_user_settings
{
public:
    temporary_user_settings()
        : root_(std::filesystem::temp_directory_path() /
                ("arc-input-overrides-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
    {
        std::filesystem::create_directories(root_);
    }

    ~temporary_user_settings()
    {
        std::error_code error;
        std::filesystem::remove_all(root_, error);
    }

    [[nodiscard]] const std::filesystem::path& root() const noexcept
    {
        return root_;
    }

private:
    std::filesystem::path root_;
};

} // namespace

TEST_CASE("runtime input rebinding replaces project defaults and resets immediately")
{
    arc::input::input_system input;
    const auto keyboard = input.connect_device(
        {.type = arc::input::input_device_type::keyboard, .name = "Keyboard", .capabilities = {.buttons = true}});

    arc::project::input_rebinding_profile profile(basic_defaults());
    REQUIRE(profile.install(input).succeeded);
    auto& player = input.player(0);

    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::space), true));
    CHECK(player.pressed("Jump"));

    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::space), false));
    REQUIRE(profile.replace_bindings(jump_target(), {{.binding = key_binding(arc::input::key::enter)}}));

    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::space), true));
    CHECK_FALSE(player.down("Jump"));
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::enter), true));
    CHECK(player.down("Jump"));

    REQUIRE(profile.reset_mapping(jump_target()));
    CHECK(player.down("Jump"));

    const auto defaults = profile.default_mapping(jump_target());
    REQUIRE(defaults);
    REQUIRE(defaults->bindings.size() == 1);
    CHECK(defaults->bindings[0].binding.control == arc::input::make_key_control(arc::input::key::space));
}

TEST_CASE("runtime input rebinding can unbind and restore one binding")
{
    arc::input::input_system input;
    const auto keyboard = input.connect_device(
        {.type = arc::input::input_device_type::keyboard, .name = "Keyboard", .capabilities = {.buttons = true}});
    arc::project::input_rebinding_profile profile(basic_defaults());
    REQUIRE(profile.install(input).succeeded);

    REQUIRE(profile.remove_binding(jump_target(), 0));
    input.begin_frame();
    REQUIRE(input.submit_button(keyboard, arc::input::make_key_control(arc::input::key::space), true));
    CHECK_FALSE(input.player(0).down("Jump"));

    REQUIRE(profile.reset_mapping(jump_target()));
    CHECK(input.player(0).down("Jump"));

    const arc::project::input_mapping_target move_target{
        .context = "Gameplay", .name = "Move", .kind = arc::project::input_mapping_kind::axis2d};
    REQUIRE(profile.replace_bindings(move_target,
                                     {{.binding = key_binding(arc::input::key::up), .contribution2d = {0.0f, 1.0f}},
                                      {.binding = key_binding(arc::input::key::d), .contribution2d = {1.0f, 0.0f}}}));
    REQUIRE(profile.reset_binding(move_target, 0));
    auto move = profile.effective_mapping(move_target);
    REQUIRE(move);
    CHECK(move->bindings[0].binding.control == arc::input::make_key_control(arc::input::key::w));

    REQUIRE(profile.add_binding(move_target,
                                {.binding = key_binding(arc::input::key::a), .contribution2d = {-1.0f, 0.0f}}));
    move = profile.effective_mapping(move_target);
    REQUIRE(move);
    REQUIRE(move->bindings.size() == 3);
    REQUIRE(profile.reset_binding(move_target, 2));
    move = profile.effective_mapping(move_target);
    REQUIRE(move);
    CHECK(move->bindings.size() == 2);
}

TEST_CASE("input user overrides persist separately and preserve valid semantic targets")
{
    temporary_user_settings temporary;
    const auto path = arc::project::input_user_overrides_path(temporary.root(), 0);

    arc::project::input_rebinding_profile profile(basic_defaults());
    REQUIRE(profile.replace_bindings(jump_target(), {{.binding = key_binding(arc::input::key::enter)}}));
    REQUIRE(arc::project::save_input_user_overrides(profile, path).succeeded);
    CHECK(std::filesystem::exists(path));

    auto updated_defaults = basic_defaults();
    updated_defaults.contexts[0].actions.push_back(
        {.name = "Pause", .bindings = {key_binding(arc::input::key::escape)}});
    arc::project::input_rebinding_profile restored(std::move(updated_defaults));
    REQUIRE(arc::project::load_input_user_overrides(restored, path).succeeded);

    const auto effective = restored.effective_mapping(jump_target());
    REQUIRE(effective);
    REQUIRE(effective->bindings.size() == 1);
    CHECK(effective->bindings[0].binding.control == arc::input::make_key_control(arc::input::key::enter));
    CHECK(
        restored
            .default_mapping({.context = "Gameplay", .name = "Pause", .kind = arc::project::input_mapping_kind::action})
            .has_value());
}

TEST_CASE("missing and stale input user overrides do not corrupt project defaults")
{
    temporary_user_settings temporary;
    arc::project::input_rebinding_profile profile(basic_defaults());
    REQUIRE(
        arc::project::load_input_user_overrides(profile, arc::project::input_user_overrides_path(temporary.root(), 0))
            .succeeded);
    CHECK(profile.user_overrides().contexts.empty());

    arc::project::input_config stale{};
    arc::project::input_context_config gameplay{};
    gameplay.name = "Gameplay";
    gameplay.actions.push_back({.name = "RemovedAction", .bindings = {key_binding(arc::input::key::enter)}});
    stale.contexts.push_back(std::move(gameplay));
    profile.set_user_overrides(std::move(stale));

    CHECK_FALSE(profile.effective_mapping(
        {.context = "Gameplay", .name = "RemovedAction", .kind = arc::project::input_mapping_kind::action}));
    const auto jump = profile.effective_mapping(jump_target());
    REQUIRE(jump);
    CHECK(jump->bindings[0].binding.control == arc::input::make_key_control(arc::input::key::space));
    CHECK(profile.user_overrides().contexts.size() == 1);
}

TEST_CASE("runtime rebinding profiles remain independent per local player")
{
    arc::input::input_system input;
    const auto first = input.connect_device(
        {.type = arc::input::input_device_type::gamepad, .name = "Pad 1", .capabilities = {.buttons = true}});
    const auto second = input.connect_device(
        {.type = arc::input::input_device_type::gamepad, .name = "Pad 2", .capabilities = {.buttons = true}});
    REQUIRE(input.assign_device(0, first));
    REQUIRE(input.assign_device(1, second));

    arc::project::input_rebinding_profile player0(gamepad_defaults(), 0);
    arc::project::input_rebinding_profile player1(gamepad_defaults(), 1);
    REQUIRE(player0.install(input).succeeded);
    REQUIRE(player1.install(input).succeeded);
    REQUIRE(player1.replace_bindings(jump_target(), {{.binding = gamepad_binding(arc::input::gamepad_button::east)}}));

    input.begin_frame();
    REQUIRE(
        input.submit_button(second, arc::input::make_gamepad_button_control(arc::input::gamepad_button::east), true));
    CHECK(input.player(1).down("Jump"));
    CHECK_FALSE(input.player(0).down("Jump"));

    REQUIRE(
        input.submit_button(first, arc::input::make_gamepad_button_control(arc::input::gamepad_button::south), true));
    CHECK(input.player(0).down("Jump"));
}

TEST_CASE("rebind capture accepts assigned hot-plugged devices and rejects disconnected or foreign devices")
{
    arc::input::input_system input;
    const auto foreign = input.connect_device(
        {.type = arc::input::input_device_type::gamepad, .name = "Foreign", .capabilities = {.buttons = true}});
    REQUIRE(input.assign_device(0, foreign));

    arc::project::input_rebind_capture capture;
    capture.begin(input, 1, {.device = arc::input::input_device_type::gamepad});
    CHECK_FALSE(
        capture.offer(foreign, arc::input::make_gamepad_button_control(arc::input::gamepad_button::south), 1.0f));
    CHECK(capture.active());

    const auto hotplugged = input.connect_device(
        {.type = arc::input::input_device_type::gamepad, .name = "Hotplugged", .capabilities = {.buttons = true}});
    REQUIRE(input.assign_device(1, hotplugged));
    const auto captured =
        capture.offer(hotplugged, arc::input::make_gamepad_button_control(arc::input::gamepad_button::west), 1.0f);
    REQUIRE(captured);
    CHECK_FALSE(capture.active());
    CHECK(captured->binding.device == arc::input::input_device_type::gamepad);
    CHECK(captured->binding.control == arc::input::make_gamepad_button_control(arc::input::gamepad_button::west));

    capture.begin(input, 1);
    REQUIRE(input.disconnect_device(hotplugged));
    CHECK_FALSE(
        capture.offer(hotplugged, arc::input::make_gamepad_button_control(arc::input::gamepad_button::north), 1.0f));
    capture.cancel();
    CHECK(capture.canceled());
    CHECK_FALSE(capture.active());
}

TEST_CASE("runtime rebinding exposes physical binding conflicts without choosing a policy")
{
    arc::project::input_rebinding_profile profile(basic_defaults());
    REQUIRE(profile.replace_bindings(
        {.context = "Gameplay", .name = "Interact", .kind = arc::project::input_mapping_kind::action},
        {{.binding = key_binding(arc::input::key::space)}}));

    const auto conflicts = profile.conflicts_for(key_binding(arc::input::key::space));
    REQUIRE(conflicts.size() == 2);
    CHECK(conflicts[0].name == "Jump");
    CHECK(conflicts[1].name == "Interact");
}

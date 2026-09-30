#include <arc/project/input_rebinding.h>

#include <catch2/catch_test_macros.hpp>

namespace
{

arc::input::input_binding key_binding(arc::input::key key)
{
    return {.device = arc::input::input_device_type::keyboard, .control = arc::input::make_key_control(key)};
}

} // namespace

TEST_CASE("rebinding treats modifier changes as user overrides")
{
    auto default_binding = key_binding(arc::input::key::w);
    default_binding.modifiers.push_back(key_binding(arc::input::key::left_shift));

    arc::project::input_context_config gameplay{.name = "Gameplay"};
    gameplay.actions.push_back({.name = "Sprint", .bindings = {default_binding}});
    arc::project::input_config defaults;
    defaults.contexts.push_back(std::move(gameplay));

    arc::project::input_rebinding_profile profile(std::move(defaults));
    const arc::project::input_mapping_target target{
        .context = "Gameplay", .name = "Sprint", .kind = arc::project::input_mapping_kind::action};

    auto replacement = profile.effective_mapping(target);
    REQUIRE(replacement);
    REQUIRE(replacement->bindings.size() == 1);
    replacement->bindings[0].binding.modifiers = {key_binding(arc::input::key::left_control)};

    REQUIRE(profile.replace_bindings(target, replacement->bindings));
    REQUIRE_FALSE(profile.user_overrides().contexts.empty());

    const auto effective = profile.effective_mapping(target);
    REQUIRE(effective);
    REQUIRE(effective->bindings[0].binding.modifiers.size() == 1);
    CHECK(effective->bindings[0].binding.modifiers[0].control ==
          arc::input::make_key_control(arc::input::key::left_control));

    REQUIRE(profile.reset_mapping(target));
    const auto reset = profile.effective_mapping(target);
    REQUIRE(reset);
    REQUIRE(reset->bindings[0].binding.modifiers.size() == 1);
    CHECK(reset->bindings[0].binding.modifiers[0].control == arc::input::make_key_control(arc::input::key::left_shift));
}

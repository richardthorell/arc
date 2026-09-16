#pragma once

#include <arc/input/input.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace arc::project
{

inline constexpr std::uint32_t input_config_version = 1;

struct input_action_config
{
    std::string name;
    std::vector<input::input_binding> bindings;
};

struct input_context_config
{
    std::string name;
    int priority{};
    bool enabled{true};
    std::vector<input_action_config> actions;
};

struct input_config
{
    std::uint32_t version{input_config_version};
    std::vector<input_context_config> contexts;
};

struct [[nodiscard]] input_config_load_result
{
    bool succeeded{};
    input_config config;
    std::string error;
};

/** Load the project semantic input mapping stored in Config/Input.json. */
[[nodiscard]] input_config_load_result load_input_config(const std::filesystem::path& path);

/** Apply one project mapping to a player in ARC's runtime-owned input system. */
[[nodiscard]] bool apply_input_config(const input_config& config, input::input_system& system,
                                      input::player_id player = 0, std::string* error = nullptr);

/** Return unique semantic action names in deterministic config order. */
[[nodiscard]] std::vector<std::string> input_action_names(const input_config& config);

} // namespace arc::project

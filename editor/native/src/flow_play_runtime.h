#pragma once

#include <arc/assets/assets.h>
#include <arc/framework/runtime_world.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace arc::editor
{

struct flow_play_install_result
{
    bool succeeded{};
    std::size_t instances{};
    std::size_t unique_programs{};
    std::string error;
};

struct flow_play_source
{
    assets::asset_reference reference;
    std::uint64_t generation{};
    std::string display_name;
    std::string source;
};

using flow_play_source_resolver =
    std::function<std::optional<flow_play_source>(const assets::asset_reference&, std::string& error)>;

/**
 * Compile and attach authored Flow bindings to an isolated Play World.
 *
 * Compiled programs are shared by asset identity; VM state remains per entity. The resolver supplies immutable source
 * generations and never exposes physical paths to the Play World. Initial bindings receive Begin Play before this
 * returns. Runtime Flow/Active component and entity lifecycle changes are then reconciled at fixed-tick phase
 * boundaries. Source generations publish new immutable programs after successful compilation; rejected edits retain
 * the last-good generation. Input Action nodes consume semantic actions resolved through ARC's input mapping system.
 */
[[nodiscard]] flow_play_install_result install_flow_play_runtime(framework::runtime_world& world,
                                                                 flow_play_source_resolver source_resolver,
                                                                 std::filesystem::path input_config_path = {});

} // namespace arc::editor

#pragma once

#include <arc/framework/runtime_world.h>

#include <cstddef>
#include <filesystem>
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

/** Returns true for a normalized Content-relative .arcflow path. */
[[nodiscard]] bool valid_flow_graph_path(std::string_view value) noexcept;

/**
 * Compile and attach authored Flow bindings to an isolated Play World.
 *
 * Compiled programs are shared by path; VM state remains per entity. Initial bindings receive Begin Play before this
 * returns. Runtime Flow/Active component and entity lifecycle changes are then reconciled at fixed-tick phase
 * boundaries so newly eligible bindings receive Begin Play and bindings that become ineligible receive End Play.
 */
[[nodiscard]] flow_play_install_result install_flow_play_runtime(framework::runtime_world& world,
                                                                 const std::filesystem::path& content_root);

} // namespace arc::editor

#pragma once

#include <arc/project/project_module.h>

#include <string_view>

namespace arc::project
{

/** @brief Return whether a registration kind is defined by the current project-module ABI. */
[[nodiscard]] bool valid_registration_kind(game_registration_kind_v1 kind) noexcept;

/**
 * @brief Return whether a registration category is valid for a module role.
 *
 * Runtime and server modules may expose runtime facilities only. Import/cook/editor
 * extensions are editor-host facilities and therefore belong to editor modules.
 */
[[nodiscard]] bool registration_kind_allowed(game_module_kind_v1 module_kind,
                                             game_registration_kind_v1 registration_kind) noexcept;

/** @brief Human-readable stable name used by diagnostics and tooling. */
[[nodiscard]] std::string_view registration_kind_name(game_registration_kind_v1 kind) noexcept;

} // namespace arc::project

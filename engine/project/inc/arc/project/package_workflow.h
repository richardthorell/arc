#pragma once

#include <arc/project/project.h>

#include <filesystem>
#include <string>

namespace arc::project
{

/** @brief High-level deploy workflow requested by editor or CLI tooling. */
enum class package_workflow_action : std::uint8_t
{
    cook,
    package,
    run
};

/** @brief Runtime role produced or launched by a package workflow. */
enum class package_workflow_role : std::uint8_t
{
    runtime,
    server
};

/** @brief Inputs used to resolve one deterministic project packaging workflow. */
struct package_workflow_request
{
    std::string profile; ///< Cook profile ID. Empty selects the first declared profile.
    package_workflow_action action{package_workflow_action::package}; ///< Requested operation.
    package_workflow_role role{package_workflow_role::runtime};       ///< Runtime or dedicated-server role.
    std::string finalizer; ///< Optional explicit platform adapter. Empty selects the target default.
};

/** @brief Fully resolved paths and build settings shared by editor and command-line frontends. */
struct package_workflow_plan
{
    std::string profile;                  ///< Selected cook profile ID.
    std::string project_guid;             ///< Persistent project identity.
    std::string engine_version;           ///< Required engine version.
    std::string configuration;            ///< Build configuration required by the profile.
    std::string target_id;                ///< Enabled target identity, independent from cook profile.
    std::string platform;                 ///< Selected operating system.
    std::string architecture;             ///< Selected CPU architecture.
    std::string renderer;                 ///< Selected renderer or none.
    std::string finalizer;                ///< Native distribution adapter identity.
    package_workflow_action action{};     ///< Requested operation.
    package_workflow_role role{};         ///< Runtime role being produced/launched.
    std::filesystem::path cook_output;    ///< Deterministic cooked-asset output root.
    std::filesystem::path package_output; ///< Deterministic packaged-artifact output root.
};

using package_workflow_result = core::result<package_workflow_plan, project_error>;

/**
 * @brief Resolve a project packaging request without executing external tools.
 *
 * Keeps editor and CLI frontends on one policy for profile/configuration validation,
 * runtime/server role validation, and artifact destinations.
 */
[[nodiscard]] package_workflow_result plan_package_workflow(const project_descriptor& descriptor,
                                                            const project_context& context,
                                                            const package_workflow_request& request = {});

} // namespace arc::project

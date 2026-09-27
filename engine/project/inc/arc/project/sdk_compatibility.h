#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace arc::project
{

/** @brief Compatibility classification between a project and an installed ARC SDK. */
enum class sdk_compatibility : std::uint8_t
{
    compatible,
    upgrade_available,
    incompatible,
    invalid_version
};

/** @brief Result of comparing a project engine version with an installed SDK version. */
struct sdk_compatibility_result
{
    sdk_compatibility status{sdk_compatibility::invalid_version};
    bool writable{};
    bool upgrade_required{};
    std::string message;
};

namespace detail
{
struct semantic_version
{
    std::uint32_t major{};
    std::uint32_t minor{};
    std::uint32_t patch{};
};

[[nodiscard]] inline std::optional<semantic_version> parse_semantic_version(std::string_view version)
{
    if (!version.empty() && version.front() == 'v')
    {
        version.remove_prefix(1);
    }

    const auto suffix = version.find_first_of("-+");
    if (suffix != std::string_view::npos)
    {
        version = version.substr(0, suffix);
    }

    std::array<std::uint32_t, 3> parts{};
    for (std::size_t index = 0; index < parts.size(); ++index)
    {
        const auto separator = version.find('.');
        const auto token = separator == std::string_view::npos ? version : version.substr(0, separator);
        if (token.empty())
        {
            return std::nullopt;
        }

        const auto* begin = token.data();
        const auto* end = token.data() + token.size();
        const auto [parsed_end, error] = std::from_chars(begin, end, parts[index]);
        if (error != std::errc{} || parsed_end != end)
        {
            return std::nullopt;
        }

        if (index + 1 < parts.size())
        {
            if (separator == std::string_view::npos)
            {
                return std::nullopt;
            }
            version.remove_prefix(separator + 1);
        }
        else if (separator != std::string_view::npos)
        {
            return std::nullopt;
        }
    }

    return semantic_version{parts[0], parts[1], parts[2]};
}
} // namespace detail

/**
 * @brief Classify whether an installed SDK can open and mutate a project safely.
 *
 * ARC treats major-version differences as incompatible. A newer SDK within the same major version may open the
 * project read-only and offer an explicit upgrade. An older SDK must not mutate a project authored by a newer SDK.
 */
[[nodiscard]] inline sdk_compatibility_result evaluate_sdk_compatibility(std::string_view project_version,
                                                                         std::string_view installed_version)
{
    const auto project = detail::parse_semantic_version(project_version);
    const auto installed = detail::parse_semantic_version(installed_version);
    if (!project || !installed)
    {
        return {.status = sdk_compatibility::invalid_version,
                .message = "Project and installed SDK versions must use semantic versioning (major.minor.patch)."};
    }

    if (project->major != installed->major)
    {
        return {.status = sdk_compatibility::incompatible,
                .message = "Project and installed SDK use incompatible major versions."};
    }

    const auto project_key = std::array{project->minor, project->patch};
    const auto installed_key = std::array{installed->minor, installed->patch};
    if (project_key == installed_key)
    {
        return {.status = sdk_compatibility::compatible,
                .writable = true,
                .message = "Project and installed SDK versions are compatible."};
    }

    if (installed_key > project_key)
    {
        return {.status = sdk_compatibility::upgrade_available,
                .upgrade_required = true,
                .message = "Project was authored with an older ARC SDK and requires an explicit upgrade before mutation."};
    }

    return {.status = sdk_compatibility::incompatible,
            .message = "Project requires a newer ARC SDK; open it with a matching or newer compatible installation."};
}

} // namespace arc::project

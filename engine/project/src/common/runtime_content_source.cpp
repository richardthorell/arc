#include <arc/project/runtime_content_source.h>

#include <arc/http/content_http.h>

#include <string>
#include <utility>

namespace arc::project
{
namespace
{

project_error invalid(std::string field, std::string message)
{
    return {.code = project_error_code::invalid_descriptor, .field = std::move(field), .message = std::move(message)};
}

runtime_content_options_result bad_option(std::string field, std::string message)
{
    return runtime_content_options_result::failure(invalid(std::move(field), std::move(message)));
}

std::optional<runtime_content_mode> parse_mode(std::string_view value)
{
    if (value == "packaged") return runtime_content_mode::packaged;
    if (value == "remote") return runtime_content_mode::remote;
    if (value == "hybrid") return runtime_content_mode::hybrid;
    return std::nullopt;
}

bool any(const runtime_content_options& options)
{
    return options.server.has_value() || options.mode.has_value();
}

} // namespace

std::string_view runtime_content_mode_name(runtime_content_mode mode) noexcept
{
    switch (mode)
    {
        case runtime_content_mode::packaged:
            return "packaged";
        case runtime_content_mode::remote:
            return "remote";
        case runtime_content_mode::hybrid:
            return "hybrid";
    }
    return "invalid";
}

std::string_view runtime_content_origin_name(runtime_content_origin origin) noexcept
{
    switch (origin)
    {
        case runtime_content_origin::distribution:
            return "distribution";
        case runtime_content_origin::project_development:
            return "project-development";
        case runtime_content_origin::trusted_launch:
            return "trusted-launch";
    }
    return "invalid";
}

runtime_content_options_result parse_android_content_metadata(std::optional<std::string_view> server,
                                                              std::optional<std::string_view> mode)
{
    runtime_content_options result;
    if (server)
    {
        if (server->empty()) return bad_option("content.server", "Content server origin must not be empty");
        result.server = std::string(*server);
    }
    if (mode)
    {
        const auto parsed = parse_mode(*mode);
        if (!parsed) return bad_option("content.mode", "Content mode must be packaged, remote or hybrid");
        result.mode = *parsed;
    }
    return runtime_content_options_result::success(std::move(result));
}

runtime_content_options_result parse_runtime_content_arguments(std::span<const std::string_view> arguments)
{
    std::optional<std::string_view> server;
    std::optional<std::string_view> mode;
    for (std::size_t index = 0; index < arguments.size(); ++index)
    {
        const auto option = arguments[index];
        const auto read = [&](std::string_view name) -> std::optional<std::string_view>
        {
            if (option == name)
            {
                if (index + 1 >= arguments.size()) return std::string_view{};
                return arguments[++index];
            }
            if (option.starts_with(name) && option.size() > name.size() && option[name.size()] == '=')
                return option.substr(name.size() + 1);
            return std::nullopt;
        };
        if (auto value = read("--content-server"))
        {
            if (server) return bad_option("content.server", "Duplicate content server argument");
            server = *value;
        }
        else if (auto mode_value = read("--content-mode"))
        {
            if (mode) return bad_option("content.mode", "Duplicate content mode argument");
            mode = *mode_value;
        }
        else if (option.starts_with("--content-"))
            return bad_option("content", "Unknown content launch argument");
    }
    return parse_android_content_metadata(server, mode);
}

runtime_content_selection_result select_runtime_content(const distribution_manifest& manifest,
                                                        const runtime_content_launch& config,
                                                        const std::filesystem::path& writable_storage_root)
{
    const bool from_launch = any(config.trusted_launch);
    const bool from_project = any(config.project_development);
    if (manifest.target.configuration == "Shipping" && (from_launch || from_project))
        return runtime_content_selection_result::failure(
            invalid("content", "Shipping runtime forbids Development content server and mode overrides"));

    runtime_content_selection selected;
    selected.project_guid = manifest.project_guid;
    selected.target_id = manifest.target.id;
    selected.cook_profile = manifest.target.cook_profile;
    selected.cache_root = writable_storage_root / "RemoteContent";
    selected.origin = from_launch    ? runtime_content_origin::trusted_launch
                      : from_project ? runtime_content_origin::project_development
                                     : runtime_content_origin::distribution;

    const auto& launch = config.trusted_launch;
    const auto& project = config.project_development;
    const auto server = launch.server ? launch.server : project.server;
    const auto explicit_mode = launch.mode ? launch.mode : project.mode;
    // A host-supplied endpoint is an explicit remote selection even if project
    // defaults say "packaged". An explicit launch mode still takes priority.
    selected.mode = explicit_mode.value_or(server ? runtime_content_mode::remote : runtime_content_mode::packaged);
    if (launch.server && !launch.mode) selected.mode = runtime_content_mode::remote;

    if (selected.mode == runtime_content_mode::packaged)
    {
        if (launch.server)
            return runtime_content_selection_result::failure(
                invalid("content.server", "Content server cannot be combined with packaged-only mode"));
        return runtime_content_selection_result::success(std::move(selected));
    }
    if (!server || server->empty())
        return runtime_content_selection_result::failure(
            invalid("content.server", "Remote/hybrid content requires a Development content-server origin"));

    // The shared C9 HTTP request policy rejects userinfo, paths, query,
    // fragment, redirects and non-loopback plaintext. Never echo rejected
    // URLs in diagnostics: malformed inputs may contain credential material.
    http::content_http_release_request probe{
        .origin = *server, .project_guid = manifest.project_guid, .target_profile = manifest.target.cook_profile};
    http::content_http_limits policy;
    policy.allow_loopback_http = true;
    const auto checked = http::validate_content_http_release_request(probe, policy);
    if (!checked.succeeded())
        return runtime_content_selection_result::failure(invalid(
            "content.server", "Invalid Development content-server origin (HTTPS or explicit loopback HTTP required)"));

    selected.server_origin = *server;
    return runtime_content_selection_result::success(std::move(selected));
}

} // namespace arc::project

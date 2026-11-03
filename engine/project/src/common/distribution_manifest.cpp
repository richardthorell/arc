#include <arc/project/distribution_manifest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <utility>

namespace arc::project
{
namespace
{
using json = nlohmann::json;

project_error failure(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}

bool contains(std::string_view value, std::initializer_list<std::string_view> options)
{
    return std::find(options.begin(), options.end(), value) != options.end();
}

bool valid_guid(std::string_view guid)
{
    if (guid.size() != 32 && guid.size() != 36) return false;
    for (std::size_t i = 0; i < guid.size(); ++i)
    {
        const bool separator = guid.size() == 36 && (i == 8 || i == 13 || i == 18 || i == 23);
        if (separator && guid[i] == '-') continue;
        if (separator || !std::isxdigit(static_cast<unsigned char>(guid[i]))) return false;
    }
    return true;
}

bool valid_sha256(std::string_view value)
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c)
                                             { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool valid_relative_path(std::string_view path)
{
    if (path.empty() || path[0] == '/' || path.find('\\') != path.npos || path.find(':') != path.npos) return false;
    std::size_t start = 0;
    while (start < path.size())
    {
        const auto end = path.find('/', start);
        const auto length = (end == path.npos ? path.size() : end) - start;
        const auto component = path.substr(start, length);
        if (component.empty() || component == "." || component == ".." || component.back() == '.' ||
            component.back() == ' ')
            return false;
        for (const unsigned char c : component)
            if (c < 32 || c == 127) return false;
        if (end == path.npos) return true;
        start = end + 1;
    }
    return false;
}

std::string portable_key(std::string_view path)
{
    std::string key(path);
    std::transform(key.begin(), key.end(), key.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return key;
}

bool valid_activity(std::string_view value)
{
    if (value.empty() || value.front() == '.' || value.back() == '.') return false;
    for (const unsigned char c : value)
        if (!std::isalnum(c) && c != '.' && c != '_' && c != '$') return false;
    return value.find('.') != value.npos;
}

bool valid_finalizer(std::string_view value)
{
    if (value.empty()) return false;
    for (const unsigned char c : value)
        if (!std::isalnum(c) && c != '-' && c != '_') return false;
    return true;
}

json target_json(const distribution_target& value)
{
    return {{"id", value.id},
            {"platform", value.platform},
            {"architecture", value.architecture},
            {"renderer", value.renderer},
            {"configuration", value.configuration},
            {"cookProfile", value.cook_profile}};
}

json file_json(const distribution_file& file)
{
    return {{"path", file.path}, {"kind", file.kind}, {"size", file.size}, {"sha256", file.sha256}};
}
} // namespace

project_status validate_distribution_manifest(const distribution_manifest& manifest)
{
    const auto invalid = [](std::string field, std::string message)
    {
        return project_status::failure(
            failure(project_error_code::invalid_descriptor, std::move(field), std::move(message)));
    };
    const auto unsafe = [](std::string field, std::string message)
    { return project_status::failure(failure(project_error_code::unsafe_path, std::move(field), std::move(message))); };
    if (manifest.format_version != distribution_format_version)
        return project_status::failure(failure(project_error_code::unsupported_version, "formatVersion",
                                               "Unsupported distribution manifest version; only v1 is accepted"));
    if (!valid_guid(manifest.project_guid))
        return invalid("projectGuid", "Project GUID must contain 32 hexadecimal digits");
    if (manifest.engine_version.empty() || manifest.sdk_version.empty())
        return invalid("engineVersion/sdkVersion", "Engine and SDK versions are required");
    const auto& target = manifest.target;
    if (target.id.empty() || target.cook_profile.empty())
        return invalid("target.cookProfile", "Target ID and cook profile must both be specified");
    if (!contains(target.platform, {"windows", "linux", "macos", "android", "ios"}) ||
        !contains(target.architecture, {"x86_64", "arm64"}) ||
        !contains(target.renderer, {"none", "vulkan", "direct3d12", "metal"}) ||
        !contains(target.configuration, {"Debug", "RelWithDebInfo", "Shipping"}))
        return invalid("target", "Unsupported platform, architecture, renderer, or configuration");
    if ((target.platform == "android" || target.platform == "ios") && target.architecture != "arm64")
        return invalid("target.architecture", "Mobile distribution requires arm64");
    if ((target.renderer == "direct3d12" && target.platform != "windows") ||
        (target.renderer == "metal" && target.platform != "macos" && target.platform != "ios") ||
        (target.platform == "android" && target.renderer != "vulkan"))
        return invalid("target.renderer", "Renderer is incompatible with target platform");
    if (!contains(manifest.role, {"client", "server"}))
        return invalid("role", "Distribution role must be client or server");
    if (manifest.role == "server" && target.renderer != "none")
        return invalid("target.renderer", "Dedicated server must use renderer 'none'");
    if (manifest.role == "client" && target.renderer == "none")
        return invalid("target.renderer", "Client requires a renderer");
    if (!valid_finalizer(manifest.finalizer))
        return invalid("finalizer", "Finalizer must be a nonempty portable identifier");
    if (manifest.entrypoint.kind == "android-activity")
    {
        if (target.platform != "android" || manifest.role != "client" || !valid_activity(manifest.entrypoint.value))
            return invalid("entrypoint", "Android activity requires an Android client and a qualified component name");
    }
    else if (manifest.entrypoint.kind == "executable")
    {
        if (!valid_relative_path(manifest.entrypoint.value))
            return unsafe("entrypoint.value", "Executable path must be package-relative");
        if (target.platform == "android" || target.platform == "ios")
            return invalid("entrypoint.kind", "Mobile distributions require a platform entrypoint");
    }
    else
        return invalid("entrypoint.kind", "Unsupported entrypoint kind");
    if (!valid_relative_path(manifest.runtime_config))
        return unsafe("runtimeConfig", "Runtime config must be a package-relative file");
    if (manifest.cook_manifests.empty())
        return invalid("content.cookManifests", "At least one cooked-content manifest is required");

    std::set<std::string> file_keys;
    for (const auto& file : manifest.files)
    {
        if (!valid_relative_path(file.path))
            return unsafe("files.path", "Staged file path must be normalized and package-relative: " + file.path);
        if (!file_keys.insert(portable_key(file.path)).second)
            return invalid("files.path", "Duplicate staged destination (case-insensitive): " + file.path);
        if (!contains(file.kind, {"executable", "library", "configuration", "content", "metadata", "symbols"}))
            return invalid("files.kind", "Unknown staged file kind for " + file.path);
        if (!valid_sha256(file.sha256))
            return invalid("files.sha256", "Staged file requires a lower-case SHA-256: " + file.path);
    }
    const auto has_file = [&](std::string_view path, std::string_view kind)
    {
        return std::any_of(manifest.files.begin(), manifest.files.end(),
                           [&](const auto& file) { return file.path == path && file.kind == kind; });
    };
    if (!has_file(manifest.runtime_config, "configuration"))
        return invalid("runtimeConfig", "Runtime configuration is missing from staged file inventory");
    if (manifest.entrypoint.kind == "executable" && !has_file(manifest.entrypoint.value, "executable"))
        return invalid("entrypoint.value", "Executable is missing from staged file inventory");

    std::set<std::string> cook_keys;
    for (const auto& path : manifest.cook_manifests)
    {
        if (!valid_relative_path(path))
            return unsafe("content.cookManifests", "Cook manifest path must be package-relative: " + path);
        if (!cook_keys.insert(portable_key(path)).second)
            return invalid("content.cookManifests", "Duplicate cook manifest reference: " + path);
        if (!path.ends_with(".arccookmanifest") || !has_file(path, "content"))
            return invalid("content.cookManifests", "Cook manifest is not staged as content: " + path);
    }
    std::set<std::string> scenes;
    for (const auto& scene : manifest.startup_scenes)
        if (!valid_guid(scene) || !scenes.insert(scene).second)
            return invalid("startupScenes", "Startup scenes require unique GUIDs");
    if (manifest.role == "client" && manifest.startup_scenes.empty())
        return invalid("startupScenes", "Client distribution requires at least one startup scene");
    return project_status::success();
}

distribution_json_result serialize_distribution_manifest(const distribution_manifest& manifest)
{
    if (auto validated = validate_distribution_manifest(manifest); !validated)
        return distribution_json_result::failure(validated.error());

    auto files = manifest.files;
    std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) { return lhs.path < rhs.path; });
    auto cook_manifests = manifest.cook_manifests;
    std::sort(cook_manifests.begin(), cook_manifests.end());

    json inventory = json::array();
    for (const auto& file : files)
        inventory.push_back(file_json(file));
    const json document{{"format", std::string(distribution_format)},
                        {"formatVersion", distribution_format_version},
                        {"projectGuid", manifest.project_guid},
                        {"engineVersion", manifest.engine_version},
                        {"sdkVersion", manifest.sdk_version},
                        {"target", target_json(manifest.target)},
                        {"role", manifest.role},
                        {"finalizer", manifest.finalizer},
                        {"entrypoint", {{"kind", manifest.entrypoint.kind}, {"value", manifest.entrypoint.value}}},
                        {"startupScenes", manifest.startup_scenes},
                        {"runtimeConfig", manifest.runtime_config},
                        {"content", {{"cookManifests", cook_manifests}}},
                        {"files", std::move(inventory)}};
    return distribution_json_result::success(document.dump(2) + "\n");
}

distribution_manifest_result parse_distribution_manifest(std::string_view source)
{
    const auto fail = [](project_error_code code, std::string field, std::string message)
    { return distribution_manifest_result::failure(failure(code, std::move(field), std::move(message))); };
    try
    {
        const auto document = json::parse(source.begin(), source.end());
        if (!document.is_object() || !document.contains("format") ||
            document.at("format").get<std::string>() != distribution_format)
            return fail(project_error_code::invalid_descriptor, "format", "Not an ARC distribution manifest");
        if (!document.contains("formatVersion") ||
            !document.at("formatVersion").is_number_unsigned() && !document.at("formatVersion").is_number_integer())
            return fail(project_error_code::invalid_descriptor, "formatVersion", "Missing distribution schema version");
        if (document.at("formatVersion").get<std::int64_t>() != distribution_format_version)
            return fail(project_error_code::unsupported_version, "formatVersion",
                        "Unsupported distribution schema version; v1 has no implicit migration");

        const auto& t = document.at("target");
        const auto& e = document.at("entrypoint");
        const auto& content = document.at("content");
        distribution_manifest manifest;
        manifest.project_guid = document.at("projectGuid").get<std::string>();
        manifest.engine_version = document.at("engineVersion").get<std::string>();
        manifest.sdk_version = document.at("sdkVersion").get<std::string>();
        manifest.target = {.id = t.at("id").get<std::string>(),
                           .platform = t.at("platform").get<std::string>(),
                           .architecture = t.at("architecture").get<std::string>(),
                           .renderer = t.at("renderer").get<std::string>(),
                           .configuration = t.at("configuration").get<std::string>(),
                           .cook_profile = t.at("cookProfile").get<std::string>()};
        manifest.role = document.at("role").get<std::string>();
        manifest.finalizer = document.at("finalizer").get<std::string>();
        manifest.entrypoint = {.kind = e.at("kind").get<std::string>(), .value = e.at("value").get<std::string>()};
        manifest.startup_scenes = document.at("startupScenes").get<std::vector<std::string>>();
        manifest.runtime_config = document.at("runtimeConfig").get<std::string>();
        manifest.cook_manifests = content.at("cookManifests").get<std::vector<std::string>>();
        for (const auto& entry : document.at("files"))
            manifest.files.push_back({.path = entry.at("path").get<std::string>(),
                                      .kind = entry.at("kind").get<std::string>(),
                                      .size = entry.at("size").get<std::uint64_t>(),
                                      .sha256 = entry.at("sha256").get<std::string>()});
        if (auto validated = validate_distribution_manifest(manifest); !validated)
            return distribution_manifest_result::failure(validated.error());
        return distribution_manifest_result::success(std::move(manifest));
    }
    catch (const json::exception& error)
    {
        return fail(project_error_code::invalid_json, "manifest",
                    "Invalid distribution manifest: " + std::string(error.what()));
    }
}
} // namespace arc::project

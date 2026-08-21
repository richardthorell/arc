#include <arc/project/project.h>

#include <arc/persistence/persistence.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace arc::project
{
namespace
{
using json = nlohmann::json;

project_error make_error(project_error_code code, std::filesystem::path path, std::string message,
                         std::string field = {})
{
    return {.code = code, .path = std::move(path), .field = std::move(field), .message = std::move(message)};
}

bool is_identifier(std::string_view value)
{
    if (value.empty()) return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char character)
                       { return std::isalnum(character) || character == '-' || character == '_' || character == '.'; });
}

bool is_guid(std::string_view value)
{
    std::size_t digits = 0;
    for (const char character : value)
    {
        if (character == '-') continue;
        if (!std::isxdigit(static_cast<unsigned char>(character))) return false;
        ++digits;
    }
    return digits == 32;
}

std::string new_guid()
{
    std::random_device device;
    std::mt19937_64 random(device());
    std::array<std::uint8_t, 16> bytes{};
    for (std::size_t index = 0; index < bytes.size(); index += 8)
    {
        const auto value = random();
        for (std::size_t byte = 0; byte < 8; ++byte)
            bytes[index + byte] = static_cast<std::uint8_t>(value >> (byte * 8));
    }
    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3f) | 0x80);
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(36);
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        if (index == 4 || index == 6 || index == 8 || index == 10) result.push_back('-');
        result.push_back(hex[bytes[index] >> 4]);
        result.push_back(hex[bytes[index] & 0x0f]);
    }
    return result;
}

std::filesystem::path normal_relative_path(const json& source, std::string_view key,
                                           std::filesystem::path fallback = {})
{
    const auto value = source.value(std::string(key), fallback.generic_string());
    if (value.empty()) return {};
    const std::filesystem::path path(value);
    const auto normalized = path.lexically_normal();
    if (path.is_absolute() || normalized.empty() || normalized == ".." ||
        (!normalized.empty() && *normalized.begin() == ".."))
        throw std::runtime_error(std::string(key) + " must be a project-relative path");
    return normalized;
}

bool is_within(const std::filesystem::path& root, const std::filesystem::path& candidate)
{
    std::error_code error;
    auto checked_root = root.lexically_normal();
    auto checked_candidate = candidate.lexically_normal();
    if (std::filesystem::exists(checked_root, error) && !error && std::filesystem::exists(checked_candidate, error) &&
        !error)
    {
        checked_root = std::filesystem::weakly_canonical(checked_root, error);
        if (error) return false;
        checked_candidate = std::filesystem::weakly_canonical(checked_candidate, error);
        if (error) return false;
    }
    const auto relative = checked_candidate.lexically_relative(checked_root);
    return !relative.empty() && relative != ".." && *relative.begin() != "..";
}

std::string to_string(module_kind value)
{
    switch (value)
    {
        case module_kind::editor:
            return "editor";
        case module_kind::runtime:
            return "runtime";
        case module_kind::server:
            return "server";
    }
    return "runtime";
}

module_kind parse_module_kind(std::string_view value)
{
    if (value == "editor") return module_kind::editor;
    if (value == "server") return module_kind::server;
    if (value == "runtime") return module_kind::runtime;
    throw std::runtime_error("module kind must be editor, runtime, or server");
}

std::string to_string(dependency_kind value)
{
    switch (value)
    {
        case dependency_kind::engine:
            return "engine";
        case dependency_kind::project:
            return "project";
        case dependency_kind::plugin:
            return "plugin";
    }
    return "engine";
}

dependency_kind parse_dependency_kind(std::string_view value)
{
    if (value == "engine") return dependency_kind::engine;
    if (value == "project") return dependency_kind::project;
    if (value == "plugin") return dependency_kind::plugin;
    throw std::runtime_error("dependency kind must be engine, project, or plugin");
}

std::string to_string(renderer_backend value)
{
    return value == renderer_backend::none ? "none" : "vulkan";
}

renderer_backend parse_renderer(std::string_view value)
{
    if (value == "none") return renderer_backend::none;
    if (value == "vulkan") return renderer_backend::vulkan;
    throw std::runtime_error("renderer backend must be none or vulkan");
}

json asset_reference_json(const project_asset_reference& value)
{
    return {{"guid", value.guid}, {"expectedType", value.expected_type}, {"pathHint", value.path_hint}};
}

project_asset_reference parse_asset_reference(const json& value)
{
    project_asset_reference result{.guid = value.value("guid", ""),
                                   .expected_type = value.value("expectedType", ""),
                                   .path_hint = value.value("pathHint", "")};
    if (!result.guid.empty() && !is_guid(result.guid)) throw std::runtime_error("asset reference GUID is malformed");
    if (!result.path_hint.empty()) normal_relative_path(json{{"path", result.path_hint}}, "path");
    return result;
}

project_descriptor parse_v2(const json& source)
{
    if (source.value("format", "") != project_format) throw std::runtime_error("unexpected project format");
    if (source.value("formatVersion", 0u) != project_format_version)
        throw std::runtime_error("unsupported project format version");

    project_descriptor result;
    result.guid = source.value("guid", "");
    result.name = source.value("name", "");
    result.engine_version = source.value("engineVersion", "");
    if (!is_guid(result.guid)) throw std::runtime_error("project GUID is malformed");
    if (result.name.empty()) throw std::runtime_error("project name is required");
    if (result.engine_version.empty()) throw std::runtime_error("engine version is required");

    const auto& paths = source.value("paths", json::object());
    result.paths.source = normal_relative_path(paths, "source", "Source");
    result.paths.content = normal_relative_path(paths, "content", "Content");
    result.paths.config = normal_relative_path(paths, "config", "Config");
    result.paths.plugins = normal_relative_path(paths, "plugins", "Plugins");
    result.paths.saved = normal_relative_path(paths, "saved", "Saved");
    result.paths.intermediate = normal_relative_path(paths, "intermediate", "Intermediate");
    result.paths.build = normal_relative_path(paths, "build", "Build");

    result.asset_roots.clear();
    for (const auto& root : source.value("assetRoots", json::array()))
        result.asset_roots.push_back(normal_relative_path(json{{"root", root}}, "root"));
    if (result.asset_roots.empty()) result.asset_roots.push_back(result.paths.content);

    for (const auto& entry : source.value("modules", json::array()))
    {
        project_module_descriptor module{.id = entry.value("id", ""),
                                         .kind = parse_module_kind(entry.value("kind", "runtime")),
                                         .target = entry.value("target", ""),
                                         .source_root = normal_relative_path(entry, "sourceRoot"),
                                         .enabled = entry.value("enabled", true)};
        for (const auto& dependency : entry.value("dependencies", json::array()))
            module.dependencies.push_back({.kind = parse_dependency_kind(dependency.value("kind", "engine")),
                                           .id = dependency.value("id", ""),
                                           .version = dependency.value("version", "")});
        result.modules.push_back(std::move(module));
    }

    for (const auto& entry : source.value("plugins", json::array()))
        result.plugins.push_back({.id = entry.value("id", ""),
                                  .version = entry.value("version", ""),
                                  .origin = entry.value("origin", "engine"),
                                  .required = entry.value("required", true),
                                  .enabled = entry.value("enabled", true),
                                  .path = normal_relative_path(entry, "path")});

    if (source.contains("defaultScene") && !source["defaultScene"].is_null())
        result.default_scene = parse_asset_reference(source["defaultScene"]);
    for (const auto& entry : source.value("startupScenes", json::array()))
        result.startup_scenes.push_back(parse_asset_reference(entry));
    for (const auto& entry : source.value("targetPlatforms", json::array()))
        result.target_platforms.push_back({.id = entry.value("id", ""), .enabled = entry.value("enabled", true)});

    const auto& toolchain = source.value("toolchain", json::object());
    result.toolchain.compiler = toolchain.value("compiler", "auto");
    result.toolchain.minimum_compiler_version = toolchain.value("minimumVersion", "");
    result.toolchain.generator = toolchain.value("generator", "auto");
    result.toolchain.architecture = toolchain.value("architecture", "x86_64");
    result.toolchain.cpp_standard = toolchain.value("cppStandard", 20u);
    result.build_configurations =
        source.value("buildConfigurations", std::vector<std::string>{"Debug", "RelWithDebInfo", "Shipping"});

    const auto& renderer = source.value("renderer", json::object());
    result.renderer.backend = parse_renderer(renderer.value("backend", "vulkan"));
    result.renderer.api = renderer.value("api", result.renderer.backend == renderer_backend::none ? "" : "1.2");
    result.renderer.quality = renderer.value("quality", "standard");
    result.renderer.anti_aliasing = renderer.value("antiAliasing", "auto");

    for (const auto& entry : source.value("cookProfiles", json::array()))
        result.cook_profiles.push_back({.id = entry.value("id", ""),
                                        .platform = entry.value("platform", ""),
                                        .architecture = entry.value("architecture", "x86_64"),
                                        .renderer = entry.value("renderer", "vulkan"),
                                        .api = entry.value("api", "1.2"),
                                        .texture_family = entry.value("textureFamily", "bc"),
                                        .configuration = entry.value("configuration", "Shipping")});

    const auto& package = source.value("package", json::object());
    result.package.application_name = package.value("applicationName", result.name);
    result.package.company_name = package.value("companyName", "");
    result.package.output = normal_relative_path(package, "output", "Build/Packages");
    result.package.region_chunks = package.value("regionChunks", true);

    const auto& settings = source.value("settings", json::object());
    result.settings.editor = normal_relative_path(settings, "editor", "Config/Editor.json");
    result.settings.renderer = normal_relative_path(settings, "renderer", "Config/Renderer.json");
    result.settings.input = normal_relative_path(settings, "input", "Config/Input.json");
    return result;
}

json descriptor_json(const project_descriptor& value)
{
    json modules = json::array();
    for (const auto& module : value.modules)
    {
        json dependencies = json::array();
        for (const auto& dependency : module.dependencies)
            dependencies.push_back(
                {{"kind", to_string(dependency.kind)}, {"id", dependency.id}, {"version", dependency.version}});
        modules.push_back({{"id", module.id},
                           {"kind", to_string(module.kind)},
                           {"target", module.target},
                           {"sourceRoot", module.source_root.generic_string()},
                           {"enabled", module.enabled},
                           {"dependencies", std::move(dependencies)}});
    }
    json plugins = json::array();
    for (const auto& plugin : value.plugins)
        plugins.push_back({{"id", plugin.id},
                           {"version", plugin.version},
                           {"origin", plugin.origin},
                           {"required", plugin.required},
                           {"enabled", plugin.enabled},
                           {"path", plugin.path.generic_string()}});
    json targets = json::array();
    for (const auto& target : value.target_platforms)
        targets.push_back({{"id", target.id}, {"enabled", target.enabled}});
    json cook_profiles = json::array();
    for (const auto& profile : value.cook_profiles)
        cook_profiles.push_back({{"id", profile.id},
                                 {"platform", profile.platform},
                                 {"architecture", profile.architecture},
                                 {"renderer", profile.renderer},
                                 {"api", profile.api},
                                 {"textureFamily", profile.texture_family},
                                 {"configuration", profile.configuration}});
    json startup = json::array();
    for (const auto& scene : value.startup_scenes)
        startup.push_back(asset_reference_json(scene));
    json roots = json::array();
    for (const auto& root : value.asset_roots)
        roots.push_back(root.generic_string());
    return {{"format", project_format},
            {"formatVersion", project_format_version},
            {"guid", value.guid},
            {"name", value.name},
            {"engineVersion", value.engine_version},
            {"paths",
             {{"source", value.paths.source.generic_string()},
              {"content", value.paths.content.generic_string()},
              {"config", value.paths.config.generic_string()},
              {"plugins", value.paths.plugins.generic_string()},
              {"saved", value.paths.saved.generic_string()},
              {"intermediate", value.paths.intermediate.generic_string()},
              {"build", value.paths.build.generic_string()}}},
            {"assetRoots", std::move(roots)},
            {"modules", std::move(modules)},
            {"plugins", std::move(plugins)},
            {"defaultScene", value.default_scene ? asset_reference_json(*value.default_scene) : json(nullptr)},
            {"startupScenes", std::move(startup)},
            {"targetPlatforms", std::move(targets)},
            {"toolchain",
             {{"compiler", value.toolchain.compiler},
              {"minimumVersion", value.toolchain.minimum_compiler_version},
              {"generator", value.toolchain.generator},
              {"architecture", value.toolchain.architecture},
              {"cppStandard", value.toolchain.cpp_standard}}},
            {"buildConfigurations", value.build_configurations},
            {"renderer",
             {{"backend", to_string(value.renderer.backend)},
              {"api", value.renderer.api},
              {"quality", value.renderer.quality},
              {"antiAliasing", value.renderer.anti_aliasing}}},
            {"cookProfiles", std::move(cook_profiles)},
            {"package",
             {{"applicationName", value.package.application_name},
              {"companyName", value.package.company_name},
              {"output", value.package.output.generic_string()},
              {"regionChunks", value.package.region_chunks}}},
            {"settings",
             {{"editor", value.settings.editor.generic_string()},
              {"renderer", value.settings.renderer.generic_string()},
              {"input", value.settings.input.generic_string()}}}};
}

project_status write_json_atomic(const std::filesystem::path& target, const json& value)
{
    std::error_code error;
    std::filesystem::create_directories(target.parent_path(), error);
    if (error) return project_status::failure(make_error(project_error_code::io_failed, target, error.message()));
    const auto temporary =
        target.string() + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
            return project_status::failure(make_error(project_error_code::io_failed, target, "cannot write file"));
        stream << value.dump(2) << '\n';
        stream.flush();
        if (!stream)
        {
            std::filesystem::remove(temporary, error);
            return project_status::failure(make_error(project_error_code::io_failed, target, "cannot flush file"));
        }
    }
#if defined(_WIN32)
    const std::filesystem::path temporary_path(temporary);
    const bool published = std::filesystem::exists(target)
                               ? ReplaceFileW(target.c_str(), temporary_path.c_str(), nullptr,
                                              REPLACEFILE_IGNORE_MERGE_ERRORS, nullptr, nullptr) != FALSE
                               : MoveFileExW(temporary_path.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH) != FALSE;
    if (!published)
#else
    std::filesystem::rename(temporary, target, error);
    if (error)
#endif
    {
        std::filesystem::remove(temporary, error);
        return project_status::failure(make_error(project_error_code::io_failed, target, "cannot publish file"));
    }
    return project_status::success();
}

json read_json(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file");
    return json::parse(stream);
}

std::string replace_all(std::string text, std::string_view from, std::string_view to)
{
    std::size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos)
    {
        text.replace(position, from.size(), to);
        position += to.size();
    }
    return text;
}

std::string safe_project_token(std::string_view name)
{
    std::string result;
    for (const unsigned char character : name)
        if (std::isalnum(character) || character == '_') result.push_back(static_cast<char>(character));
    if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front()))) result.insert(result.begin(), '_');
    return result;
}

std::optional<std::string> environment_value(const char* name)
{
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return std::nullopt;
    std::string result(value);
    std::free(value);
    return result;
#else
    const char* value = std::getenv(name);
    return value ? std::optional<std::string>(value) : std::nullopt;
#endif
}

std::optional<std::filesystem::path> find_on_path(std::string_view executable)
{
    const auto path_value = environment_value("PATH");
    if (!path_value) return std::nullopt;
#if defined(_WIN32)
    constexpr char separator = ';';
    constexpr std::string_view suffix = ".exe";
#else
    constexpr char separator = ':';
    constexpr std::string_view suffix = "";
#endif
    std::stringstream stream(*path_value);
    std::string directory;
    while (std::getline(stream, directory, separator))
    {
        auto candidate = std::filesystem::path(directory) / (std::string(executable) + std::string(suffix));
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) return std::filesystem::absolute(candidate);
    }
    return std::nullopt;
}

#if defined(_WIN32)
std::wstring quote_command_argument(std::string_view value)
{
    std::wstring input(value.begin(), value.end());
    std::wstring output{L"\""};
    std::size_t slashes{};
    for (const wchar_t character : input)
    {
        if (character == L'\\')
        {
            ++slashes;
            continue;
        }
        if (character == L'\"')
            output.append(slashes * 2u + 1u, L'\\');
        else
            output.append(slashes, L'\\');
        slashes = 0;
        output.push_back(character);
    }
    output.append(slashes * 2u, L'\\');
    output.push_back(L'\"');
    return output;
}
#endif

std::string probe_tool_version(const std::filesystem::path& executable, const std::vector<std::string>& arguments)
{
    std::string output;
#if defined(_WIN32)
    SECURITY_ATTRIBUTES security{.nLength = sizeof(SECURITY_ATTRIBUTES), .bInheritHandle = TRUE};
    HANDLE read_handle{};
    HANDLE write_handle{};
    if (!CreatePipe(&read_handle, &write_handle, &security, 0)) return {};
    SetHandleInformation(read_handle, HANDLE_FLAG_INHERIT, 0);
    STARTUPINFOW startup{.cb = sizeof(STARTUPINFOW),
                         .dwFlags = STARTF_USESTDHANDLES,
                         .hStdOutput = write_handle,
                         .hStdError = write_handle};
    PROCESS_INFORMATION process{};
    std::wstring command = quote_command_argument(executable.string());
    for (const auto& argument : arguments)
        command += L" " + quote_command_argument(argument);
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process))
    {
        CloseHandle(write_handle);
        std::array<char, 512> buffer{};
        DWORD read{};
        while (output.size() < 4096u &&
               ReadFile(read_handle, buffer.data(), static_cast<DWORD>(buffer.size()), &read, nullptr) && read)
            output.append(buffer.data(), read);
        WaitForSingleObject(process.hProcess, 5000);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
    }
    else
        CloseHandle(write_handle);
    CloseHandle(read_handle);
#else
    int descriptors[2];
    if (pipe(descriptors) != 0) return {};
    const pid_t child = fork();
    if (child == 0)
    {
        dup2(descriptors[1], STDOUT_FILENO);
        dup2(descriptors[1], STDERR_FILENO);
        close(descriptors[0]);
        close(descriptors[1]);
        std::vector<std::string> storage = arguments;
        std::vector<char*> argv{const_cast<char*>(executable.c_str())};
        for (auto& argument : storage)
            argv.push_back(argument.data());
        argv.push_back(nullptr);
        execv(executable.c_str(), argv.data());
        _exit(127);
    }
    close(descriptors[1]);
    std::array<char, 512> buffer{};
    ssize_t count{};
    while (output.size() < 4096u && (count = read(descriptors[0], buffer.data(), buffer.size())) > 0)
        output.append(buffer.data(), static_cast<std::size_t>(count));
    close(descriptors[0]);
    int status{};
    waitpid(child, &status, 0);
#endif
    const auto line_end = output.find_first_of("\r\n");
    if (line_end != std::string::npos) output.resize(line_end);
    return output;
}
} // namespace

descriptor_result load_descriptor(const std::filesystem::path& descriptor_path)
{
    try
    {
        if (!std::filesystem::is_regular_file(descriptor_path))
            return descriptor_result::failure(
                make_error(project_error_code::not_found, descriptor_path, "project descriptor not found"));
        return descriptor_result::success(parse_v2(read_json(descriptor_path)));
    }
    catch (const json::exception& error)
    {
        return descriptor_result::failure(make_error(project_error_code::invalid_json, descriptor_path, error.what()));
    }
    catch (const std::exception& error)
    {
        return descriptor_result::failure(
            make_error(project_error_code::invalid_descriptor, descriptor_path, error.what()));
    }
}

project_status save_descriptor(const std::filesystem::path& descriptor_path, const project_descriptor& descriptor)
{
    const auto validation = validate_descriptor(descriptor_path, descriptor);
    if (!validation) return project_status::failure(validation.error());
    return write_json_atomic(descriptor_path, descriptor_json(descriptor));
}

validation_result validate_descriptor(const std::filesystem::path& descriptor_path,
                                      const project_descriptor& descriptor, const project_validation_options& options)
{
    project_validation_result result;
    if (!is_guid(descriptor.guid))
        return validation_result::failure(
            make_error(project_error_code::invalid_descriptor, descriptor_path, "project GUID is malformed", "guid"));
    if (descriptor.name.empty() || descriptor.engine_version.empty())
        return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                     "project name and engine version are required"));
    if (options.require_exact_engine && !options.engine_version.empty() &&
        descriptor.engine_version != options.engine_version)
    {
        result.writable = false;
        result.diagnostics.push_back(make_error(project_error_code::incompatible_engine, descriptor_path,
                                                "project requires ARC " + descriptor.engine_version, "engineVersion"));
        if (!options.allow_read_only) return validation_result::failure(result.diagnostics.back());
    }
    std::set<std::string> module_ids;
    std::set<std::string> targets;
    for (const auto& module : descriptor.modules)
    {
        if (!is_identifier(module.id) || module.target.empty() || module.source_root.empty() ||
            !module_ids.insert(module.id).second || !targets.insert(module.target).second)
            return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                         "module IDs and targets must be non-empty and unique",
                                                         "modules"));
    }
    std::set<std::string> plugin_ids;
    for (const auto& plugin : descriptor.plugins)
        if (!is_identifier(plugin.id) || plugin.version.empty() || !plugin_ids.insert(plugin.id).second)
            return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                         "plugin IDs must be unique and versioned", "plugins"));
    for (const auto& module : descriptor.modules)
        for (const auto& dependency : module.dependencies)
        {
            if (!is_identifier(dependency.id))
                return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                             "module dependency ID is malformed",
                                                             "modules.dependencies"));
            if (dependency.kind == dependency_kind::project && !module_ids.contains(dependency.id))
                return validation_result::failure(make_error(project_error_code::missing_module, descriptor_path,
                                                             "missing project module " + dependency.id));
            if (dependency.kind == dependency_kind::plugin && !plugin_ids.contains(dependency.id))
                return validation_result::failure(make_error(project_error_code::missing_plugin, descriptor_path,
                                                             "missing project plugin " + dependency.id));
        }
    std::unordered_map<std::string, std::vector<std::string>> project_edges;
    for (const auto& module : descriptor.modules)
        for (const auto& dependency : module.dependencies)
            if (dependency.kind == dependency_kind::project) project_edges[module.id].push_back(dependency.id);
    std::unordered_map<std::string, std::uint8_t> visit_state;
    const std::function<bool(const std::string&)> visit = [&](const std::string& id)
    {
        auto& state = visit_state[id];
        if (state == 1u) return false;
        if (state == 2u) return true;
        state = 1u;
        for (const auto& dependency : project_edges[id])
            if (!visit(dependency)) return false;
        state = 2u;
        return true;
    };
    for (const auto& module : descriptor.modules)
        if (!visit(module.id))
            return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                         "project module dependencies contain a cycle", "modules"));
    if (descriptor.asset_roots.empty())
        return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                     "at least one content root is required", "assetRoots"));
    if (std::none_of(descriptor.target_platforms.begin(), descriptor.target_platforms.end(),
                     [](const auto& platform) { return platform.enabled && !platform.id.empty(); }))
        return validation_result::failure(make_error(project_error_code::unsupported_platform, descriptor_path,
                                                     "at least one target platform must be enabled",
                                                     "targetPlatforms"));
    const std::set<std::string_view> supported_platforms{"windows-x64-vulkan", "windows-x64-headless",
                                                         "linux-x64-headless"};
    for (const auto& platform : descriptor.target_platforms)
        if (platform.enabled && !supported_platforms.contains(platform.id))
            return validation_result::failure(make_error(project_error_code::unsupported_platform, descriptor_path,
                                                         "unsupported target platform: " + platform.id,
                                                         "targetPlatforms"));
    if (descriptor.renderer.backend == renderer_backend::vulkan &&
        std::none_of(descriptor.target_platforms.begin(), descriptor.target_platforms.end(),
                     [](const auto& platform) { return platform.enabled && platform.id.ends_with("-vulkan"); }))
        return validation_result::failure(make_error(project_error_code::unsupported_platform, descriptor_path,
                                                     "the Vulkan renderer requires a Vulkan target platform",
                                                     "renderer.backend"));
    for (const auto& plugin : descriptor.plugins)
        if (plugin.required && !plugin.enabled)
            return validation_result::failure(make_error(project_error_code::missing_plugin, descriptor_path,
                                                         "required plugin is disabled: " + plugin.id, "plugins"));
    if (descriptor.toolchain.cpp_standard < 20)
        return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                     "ARC projects require C++20", "toolchain.cppStandard"));
    if (descriptor.build_configurations.empty())
        return validation_result::failure(make_error(project_error_code::invalid_descriptor, descriptor_path,
                                                     "at least one build configuration is required"));
    const auto context = resolve_context(descriptor_path, descriptor);
    if (!context) return validation_result::failure(context.error());
    const auto source_root = context.value().root / descriptor.paths.source;
    for (const auto& module : descriptor.modules)
        if (!is_within(source_root, context.value().root / module.source_root))
            return validation_result::failure(make_error(project_error_code::unsafe_path, module.source_root,
                                                         "module source root escapes the project Source directory",
                                                         module.id));
    for (const auto& plugin : descriptor.plugins)
        if (!plugin.path.empty() && !is_within(context.value().plugin_root, context.value().root / plugin.path))
            return validation_result::failure(make_error(project_error_code::unsafe_path, plugin.path,
                                                         "plugin path escapes the project Plugins directory",
                                                         plugin.id));
    for (const auto& setting : {descriptor.settings.editor, descriptor.settings.renderer, descriptor.settings.input})
        if (!is_within(context.value().config_root, context.value().root / setting))
            return validation_result::failure(make_error(project_error_code::unsafe_path, setting,
                                                         "settings path escapes the project Config directory"));
    if (!is_within(context.value().build_root, context.value().root / descriptor.package.output))
        return validation_result::failure(make_error(project_error_code::unsafe_path, descriptor.package.output,
                                                     "package output escapes the project Build directory",
                                                     "package.output"));
    if (options.require_paths)
    {
        for (const auto& root : context.value().asset_roots)
            if (!std::filesystem::is_directory(root))
                return validation_result::failure(make_error(project_error_code::invalid_descriptor, root,
                                                             "asset root does not exist", "assetRoots"));
        const auto validate_scene_reference =
            [&](const project_asset_reference& reference) -> std::optional<project_error>
        {
            if (!is_guid(reference.guid) || reference.expected_type != "scene" || reference.path_hint.empty())
                return make_error(project_error_code::invalid_scene, descriptor_path,
                                  "scene references require a GUID, scene type, and path hint");
            const auto scene = context.value().root / reference.path_hint;
            const bool in_content = std::any_of(context.value().asset_roots.begin(), context.value().asset_roots.end(),
                                                [&](const auto& root) { return is_within(root, scene); });
            if (!in_content || !std::filesystem::is_regular_file(scene))
                return make_error(project_error_code::invalid_scene, scene,
                                  "scene reference is missing or outside the declared content roots");
            const auto metadata_path = std::filesystem::path(scene.string() + ".arcmeta");
            if (!std::filesystem::is_regular_file(metadata_path))
                return make_error(project_error_code::invalid_scene, metadata_path, "scene asset metadata is missing");
            try
            {
                const auto metadata = read_json(metadata_path);
                if (metadata.value("format", "") != "arc.asset-meta" || metadata.value("guid", "") != reference.guid)
                    return make_error(project_error_code::invalid_scene, metadata_path,
                                      "scene asset GUID does not match its metadata");
            }
            catch (const std::exception& error)
            {
                return make_error(project_error_code::invalid_scene, metadata_path, error.what());
            }
            return std::nullopt;
        };
        if (descriptor.default_scene)
            if (const auto error = validate_scene_reference(*descriptor.default_scene))
                return validation_result::failure(*error);
        for (const auto& scene : descriptor.startup_scenes)
            if (const auto error = validate_scene_reference(scene)) return validation_result::failure(*error);
        for (const auto& module : descriptor.modules)
        {
            const auto module_source_root = context.value().root / module.source_root;
            if (!is_within(context.value().root, module_source_root) ||
                !std::filesystem::is_directory(module_source_root))
                return validation_result::failure(make_error(project_error_code::missing_module, module_source_root,
                                                             "module source root does not exist", module.id));
        }
        for (const auto& plugin : descriptor.plugins)
        {
            if (plugin.path.empty()) continue;
            const auto plugin_path = context.value().root / plugin.path;
            if (!is_within(context.value().root, plugin_path) ||
                (plugin.required && !std::filesystem::exists(plugin_path)))
                return validation_result::failure(make_error(project_error_code::missing_plugin, plugin_path,
                                                             "plugin path is missing or unsafe", plugin.id));
        }
    }
    return validation_result::success(std::move(result));
}

context_result resolve_context(const std::filesystem::path& descriptor_path, const project_descriptor& descriptor)
{
    const auto root = std::filesystem::absolute(descriptor_path).lexically_normal().parent_path();
    project_context result{.descriptor_path = std::filesystem::absolute(descriptor_path).lexically_normal(),
                           .root = root,
                           .config_root = root / descriptor.paths.config,
                           .plugin_root = root / descriptor.paths.plugins,
                           .saved_root = root / descriptor.paths.saved,
                           .intermediate_root = root / descriptor.paths.intermediate,
                           .build_root = root / descriptor.paths.build,
                           .asset_cache_root = root / descriptor.paths.intermediate / "Cache",
                           .recovery_root = root / descriptor.paths.saved / "Recovery"};
    for (const auto& asset_root : descriptor.asset_roots)
        result.asset_roots.push_back(root / asset_root);
    const std::array paths{result.config_root, result.plugin_root,      result.saved_root,   result.intermediate_root,
                           result.build_root,  result.asset_cache_root, result.recovery_root};
    for (const auto& path : paths)
        if (!is_within(root, path))
            return context_result::failure(
                make_error(project_error_code::unsafe_path, path, "project path escapes the project root"));
    for (const auto& path : result.asset_roots)
        if (!is_within(root, path))
            return context_result::failure(
                make_error(project_error_code::unsafe_path, path, "asset root escapes the project root"));
    return context_result::success(std::move(result));
}

project_status upgrade_descriptor(const std::filesystem::path& descriptor_path, std::string_view target_engine_version)
{
    try
    {
        const auto source = read_json(descriptor_path);
        const auto version = source.value("formatVersion", 0u);
        if (version == project_format_version)
        {
            auto descriptor = parse_v2(source);
            descriptor.engine_version = target_engine_version;
            return save_descriptor(descriptor_path, descriptor);
        }
        if (version != 1u)
            return project_status::failure(make_error(project_error_code::unsupported_version, descriptor_path,
                                                      "only version 1 projects can be upgraded"));
        project_descriptor descriptor;
        descriptor.guid = source.value("guid", "");
        descriptor.name = source.value("name", "");
        descriptor.engine_version = std::string(target_engine_version);
        descriptor.asset_roots.clear();
        for (const auto& root : source.value("assetRoots", json::array({"assets"})))
            descriptor.asset_roots.push_back(normal_relative_path(json{{"root", root}}, "root"));
        descriptor.paths.content = descriptor.asset_roots.front();
        descriptor.paths.config = "config";
        descriptor.settings.editor =
            normal_relative_path(source.value("settings", json::object()), "editor", "config/editor.settings.json");
        descriptor.settings.renderer =
            normal_relative_path(source.value("settings", json::object()), "renderer", "config/renderer.settings.json");
        descriptor.settings.input =
            normal_relative_path(source.value("settings", json::object()), "input", "config/input.settings.json");
        for (const auto& scene : source.value("startupScenes", json::array()))
        {
            if (!scene.is_string()) continue;
            project_asset_reference reference{.expected_type = "scene", .path_hint = scene.get<std::string>()};
            const auto metadata_path = descriptor_path.parent_path() / (reference.path_hint + ".arcmeta");
            if (std::filesystem::is_regular_file(metadata_path))
                reference.guid = read_json(metadata_path).value("guid", "");
            descriptor.startup_scenes.push_back(reference);
        }
        if (!descriptor.startup_scenes.empty()) descriptor.default_scene = descriptor.startup_scenes.front();
        for (const auto& extension : source.value("extensions", json::array()))
        {
            if (!extension.is_string()) continue;
            const auto extension_path = extension.get<std::string>();
            descriptor.plugins.push_back(
                {.id = "legacy." + safe_project_token(std::filesystem::path(extension_path).stem().string()),
                 .version = "legacy",
                 .origin = "project",
                 .required = false,
                 .enabled = true,
                 .path = extension_path});
        }
        const auto token = safe_project_token(descriptor.name);
        const auto runtime_source = std::filesystem::path("Source") / (token + "Runtime");
        const auto editor_source = std::filesystem::path("Source") / (token + "Editor");
        if (std::filesystem::is_directory(descriptor_path.parent_path() / runtime_source))
            descriptor.modules.push_back({.id = token + ".runtime",
                                          .kind = module_kind::runtime,
                                          .target = token + "Runtime",
                                          .source_root = runtime_source,
                                          .enabled = true,
                                          .dependencies = {{.kind = dependency_kind::engine,
                                                            .id = "ARC.Runtime",
                                                            .version = std::string(target_engine_version)}}});
        if (std::filesystem::is_directory(descriptor_path.parent_path() / editor_source))
        {
            std::vector<module_dependency> editor_dependencies;
            if (!descriptor.modules.empty())
                editor_dependencies.push_back({.kind = dependency_kind::project, .id = token + ".runtime"});
            editor_dependencies.push_back({.kind = dependency_kind::engine,
                                           .id = "ARC.EditorModuleSDK",
                                           .version = std::string(target_engine_version)});
            descriptor.modules.push_back({.id = token + ".editor",
                                          .kind = module_kind::editor,
                                          .target = token + "Editor",
                                          .source_root = editor_source,
                                          .enabled = true,
                                          .dependencies = std::move(editor_dependencies)});
        }
#if defined(_WIN32)
        descriptor.target_platforms = {{.id = "windows-x64-vulkan"}};
#else
        descriptor.target_platforms = {{.id = "linux-x64-headless"}};
        descriptor.renderer = {.backend = renderer_backend::none, .api = {}, .quality = "standard"};
#endif
        descriptor.cook_profiles = {{.id = "windows-x64-vulkan", .platform = "windows"},
                                    {.id = "linux-x64-headless", .platform = "linux", .renderer = "none", .api = ""}};
        const auto legacy_cook = descriptor_path.parent_path() / "arc.cook.json";
        if (std::filesystem::is_regular_file(legacy_cook))
        {
            const auto cook = read_json(legacy_cook);
            descriptor.cook_profiles.clear();
            const auto profiles = cook.value("profiles", json::object());
            for (auto iterator = profiles.begin(); iterator != profiles.end(); ++iterator)
                descriptor.cook_profiles.push_back(
                    {.id = iterator.key(),
                     .platform = iterator.value().value("platform", ""),
                     .architecture = iterator.value().value("architecture", "x86_64"),
                     .renderer = iterator.value().value("renderer", "vulkan"),
                     .api = iterator.value().value("api", "1.2"),
                     .texture_family = iterator.value().value("textureFamily", "bc"),
                     .configuration = iterator.value().value("configuration", "Shipping")});
        }
        const auto backup = descriptor_path.string() + ".v1.bak";
        std::filesystem::copy_file(descriptor_path, backup, std::filesystem::copy_options::overwrite_existing);
        const auto saved = save_descriptor(descriptor_path, descriptor);
        if (!saved)
            std::filesystem::copy_file(backup, descriptor_path, std::filesystem::copy_options::overwrite_existing);
        return saved;
    }
    catch (const std::exception& error)
    {
        return project_status::failure(
            make_error(project_error_code::invalid_descriptor, descriptor_path, error.what()));
    }
}

templates_result discover_templates(const std::filesystem::path& templates_root)
{
    std::vector<project_template_snapshot> result;
    try
    {
        if (!std::filesystem::is_directory(templates_root))
            return templates_result::failure(
                make_error(project_error_code::not_found, templates_root, "template root does not exist"));
        for (const auto& entry : std::filesystem::directory_iterator(templates_root))
        {
            const auto manifest_path = entry.path() / "template.arc-template.json";
            if (!entry.is_directory() || !std::filesystem::is_regular_file(manifest_path)) continue;
            const auto manifest = read_json(manifest_path);
            if (manifest.value("format", "") != template_format ||
                manifest.value("formatVersion", 0u) != template_format_version)
                continue;
            result.push_back({.id = manifest.value("id", ""),
                              .name = manifest.value("name", ""),
                              .description = manifest.value("description", ""),
                              .engine_version = manifest.value("engineVersion", ""),
                              .root = entry.path()});
        }
        std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
        return templates_result::success(std::move(result));
    }
    catch (const std::exception& error)
    {
        return templates_result::failure(
            make_error(project_error_code::invalid_descriptor, templates_root, error.what()));
    }
}

project_status create_project(const create_project_request& request)
{
    std::error_code error;
    const auto destination = std::filesystem::absolute(request.destination).lexically_normal();
    const auto template_root = request.templates_root / request.template_id;
    const bool invalid_name =
        request.name.empty() || request.name == "." || request.name == ".." ||
        request.name.find_first_of("/\\:<>|?*") != std::string::npos ||
        std::any_of(request.name.begin(), request.name.end(), [](unsigned char character) { return character < 32u; });
    if (invalid_name || request.template_id.empty())
        return project_status::failure(make_error(project_error_code::invalid_descriptor, destination,
                                                  "project name is invalid or the template ID is missing"));
    if (!std::filesystem::is_regular_file(template_root / "template.arc-template.json"))
        return project_status::failure(
            make_error(project_error_code::template_not_found, template_root, "project template not found"));
    if (std::filesystem::exists(destination) && !std::filesystem::is_empty(destination))
        return project_status::failure(
            make_error(project_error_code::destination_not_empty, destination, "project destination must be empty"));
    const auto staging = destination.parent_path() / (destination.filename().string() + ".arc-staging-" + new_guid());
    try
    {
        std::filesystem::create_directories(staging);
        const auto token = safe_project_token(request.name);
        const auto guid = new_guid();
        const auto scene_guid = new_guid();
        const auto scene_asset_guid = new_guid();
        const auto camera_guid = new_guid();
        const auto light_guid = new_guid();
        const auto environment_guid = new_guid();
        const auto floor_guid = new_guid();
        const auto copy_template_root = [&](const std::filesystem::path& source_root)
        {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(source_root))
            {
                const auto relative = entry.path().lexically_relative(source_root);
                if (relative == "template.arc-template.json") continue;
                auto relative_text = replace_all(relative.generic_string(), "__PROJECT__", token);
                const auto output = staging / std::filesystem::path(relative_text);
                if (!is_within(staging, output)) throw std::runtime_error("template output escapes destination");
                if (entry.is_directory())
                {
                    std::filesystem::create_directories(output);
                    continue;
                }
                if (!entry.is_regular_file() || entry.is_symlink())
                    throw std::runtime_error("template contains unsupported entry");
                std::ifstream input(entry.path(), std::ios::binary);
                std::string content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
                content = replace_all(std::move(content), "{{PROJECT_NAME}}", request.name);
                content = replace_all(std::move(content), "{{PROJECT_TOKEN}}", token);
                content = replace_all(std::move(content), "{{PROJECT_GUID}}", guid);
                content = replace_all(std::move(content), "{{SCENE_GUID}}", scene_guid);
                content = replace_all(std::move(content), "{{SCENE_ASSET_GUID}}", scene_asset_guid);
                content = replace_all(std::move(content), "{{CAMERA_GUID}}", camera_guid);
                content = replace_all(std::move(content), "{{LIGHT_GUID}}", light_guid);
                content = replace_all(std::move(content), "{{ENVIRONMENT_GUID}}", environment_guid);
                content = replace_all(std::move(content), "{{FLOOR_GUID}}", floor_guid);
                content = replace_all(std::move(content), "{{ENGINE_VERSION}}", request.engine_version);
                if (output.extension() == ".arcscene" || output.extension() == ".arcprefab")
                {
                    auto sealed = persistence::seal_json_document(content);
                    if (!sealed.succeeded())
                        throw std::runtime_error("template document could not be sealed: " + sealed.error);
                    content = std::move(sealed.text);
                }
                std::filesystem::create_directories(output.parent_path());
                std::ofstream stream(output, std::ios::binary | std::ios::trunc);
                stream.write(content.data(), static_cast<std::streamsize>(content.size()));
                if (!stream) throw std::runtime_error("failed to write template output");
            }
        };
        const auto template_manifest = read_json(template_root / "template.arc-template.json");
        const auto base = template_manifest.value("base", "");
        if (!base.empty())
        {
            if (!is_identifier(base) || base == request.template_id)
                throw std::runtime_error("template base is invalid");
            const auto base_root = request.templates_root / base;
            if (!is_within(request.templates_root, base_root) ||
                !std::filesystem::is_regular_file(base_root / "template.arc-template.json"))
                throw std::runtime_error("template base is missing");
            copy_template_root(base_root);
        }
        copy_template_root(template_root);
        for (const auto* directory : {"Source", "Content", "Config", "Plugins", "Saved", "Intermediate", "Build"})
            std::filesystem::create_directories(staging / directory);
        const auto descriptor_path = staging / (token + ".arcproject");
        const auto descriptor = load_descriptor(descriptor_path);
        if (!descriptor) throw std::runtime_error(descriptor.error().message);
        const auto validation = validate_descriptor(descriptor_path, descriptor.value());
        if (!validation) throw std::runtime_error(validation.error().message);
        if (std::filesystem::exists(destination)) std::filesystem::remove(destination);
        std::filesystem::rename(staging, destination);
        return project_status::success();
    }
    catch (const std::exception& exception)
    {
        std::filesystem::remove_all(staging, error);
        return project_status::failure(make_error(project_error_code::io_failed, destination, exception.what()));
    }
}

std::filesystem::path default_installation_registry_path()
{
#if defined(_WIN32)
    const auto root = environment_value("LOCALAPPDATA");
    return std::filesystem::path(root.value_or(".")) / "ARC" / "installations.v1.json";
#else
    const auto root = environment_value("XDG_DATA_HOME");
    if (root) return std::filesystem::path(*root) / "arc" / "installations.v1.json";
    const auto home = environment_value("HOME");
    return std::filesystem::path(home.value_or(".")) / ".local" / "share" / "arc" / "installations.v1.json";
#endif
}

installation_result load_installation_manifest(const std::filesystem::path& manifest_path)
{
    try
    {
        const auto source = read_json(manifest_path);
        if (source.value("format", "") != installation_format ||
            source.value("formatVersion", 0u) != installation_format_version)
            throw std::runtime_error("unsupported installation manifest");
        engine_installation_manifest result;
        result.installation_id = source.value("installationId", "");
        result.engine_version = source.value("engineVersion", "");
        result.manifest_path = std::filesystem::absolute(manifest_path).lexically_normal();
        result.root = result.manifest_path.parent_path();
        const auto editor = normal_relative_path(source, "editor");
        result.editor = editor.empty() ? std::filesystem::path{} : result.root / editor;
        result.sdk = result.root / normal_relative_path(source, "sdk", ".");
        const auto cooker = normal_relative_path(source, "cooker");
        result.cooker = cooker.empty() ? std::filesystem::path{} : result.root / cooker;
        const auto project_tool = normal_relative_path(source, "projectTool");
        result.project_tool = project_tool.empty() ? std::filesystem::path{} : result.root / project_tool;
        result.platforms = source.value("supportedPlatforms", std::vector<std::string>{});
        result.configurations = source.value("configurations", std::vector<std::string>{});
        const auto& requirements = source.value("toolchain", json::object());
        result.toolchain.compiler = requirements.value("compiler", "auto");
        result.toolchain.minimum_compiler_version = requirements.value("minimumVersion", "");
        result.toolchain.generator = requirements.value("generator", "auto");
        result.toolchain.architecture = requirements.value("architecture", "x86_64");
        result.toolchain.cpp_standard = requirements.value("cppStandard", 20u);
        for (const auto& plugin : source.value("plugins", json::array()))
            result.plugins.push_back({.id = plugin.value("id", ""),
                                      .version = plugin.value("version", ""),
                                      .platforms = plugin.value("platforms", std::vector<std::string>{})});
        for (const auto& item : source.value("templates", json::array()))
            result.templates.push_back({.id = item.value("id", ""),
                                        .name = item.value("name", ""),
                                        .description = item.value("description", ""),
                                        .engine_version = result.engine_version,
                                        .root = result.root / normal_relative_path(item, "path")});
        if (!is_identifier(result.installation_id) || result.engine_version.empty())
            throw std::runtime_error("installation identity and engine version are required");
        if (result.platforms.empty() || result.configurations.empty() || result.sdk.empty() ||
            result.project_tool.empty())
            throw std::runtime_error("installation platforms, configurations, SDK, and project tool are required");
        std::set<std::string> template_ids;
        for (const auto& item : result.templates)
            if (!is_identifier(item.id) || item.name.empty() || !template_ids.insert(item.id).second ||
                !is_within(result.root, item.root))
                throw std::runtime_error("installation templates must have unique IDs and contained paths");
        return installation_result::success(std::move(result));
    }
    catch (const std::exception& error)
    {
        return installation_result::failure(
            make_error(project_error_code::invalid_descriptor, manifest_path, error.what()));
    }
}

project_status register_installation(const std::filesystem::path& registry_path,
                                     const std::filesystem::path& manifest_path)
{
    const auto installation = load_installation_manifest(manifest_path);
    if (!installation) return project_status::failure(installation.error());
    json registry{{"format", "arc-installation-registry"}, {"formatVersion", 1}, {"installations", json::array()}};
    try
    {
        if (std::filesystem::is_regular_file(registry_path)) registry = read_json(registry_path);
        auto& entries = registry["installations"];
        entries.erase(
            std::remove_if(entries.begin(), entries.end(), [&](const json& value)
                           { return value.value("installationId", "") == installation.value().installation_id; }),
            entries.end());
        entries.push_back({{"installationId", installation.value().installation_id},
                           {"manifest", std::filesystem::absolute(manifest_path).generic_string()}});
        return write_json_atomic(registry_path, registry);
    }
    catch (const std::exception& error)
    {
        return project_status::failure(make_error(project_error_code::io_failed, registry_path, error.what()));
    }
}

project_status unregister_installation(const std::filesystem::path& registry_path, std::string_view installation_id)
{
    try
    {
        if (!std::filesystem::is_regular_file(registry_path)) return project_status::success();
        auto registry = read_json(registry_path);
        auto& entries = registry["installations"];
        entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const json& value)
                                     { return value.value("installationId", "") == installation_id; }),
                      entries.end());
        return write_json_atomic(registry_path, registry);
    }
    catch (const std::exception& error)
    {
        return project_status::failure(make_error(project_error_code::io_failed, registry_path, error.what()));
    }
}

installations_result discover_installations(const std::filesystem::path& registry_path)
{
    std::vector<engine_installation_manifest> result;
    try
    {
        if (!std::filesystem::is_regular_file(registry_path)) return installations_result::success({});
        const auto registry = read_json(registry_path);
        for (const auto& entry : registry.value("installations", json::array()))
        {
            const auto installation = load_installation_manifest(entry.value("manifest", ""));
            if (installation) result.push_back(installation.value());
        }
        std::sort(result.begin(), result.end(),
                  [](const auto& left, const auto& right)
                  {
                      return std::tie(left.engine_version, left.installation_id) <
                             std::tie(right.engine_version, right.installation_id);
                  });
        return installations_result::success(std::move(result));
    }
    catch (const std::exception& error)
    {
        return installations_result::failure(make_error(project_error_code::invalid_json, registry_path, error.what()));
    }
}

installations_result repair_installations(const std::filesystem::path& registry_path,
                                          const std::vector<std::filesystem::path>& search_roots)
{
    std::vector<std::filesystem::path> candidates;
    try
    {
        if (std::filesystem::is_regular_file(registry_path))
        {
            try
            {
                const auto registry = read_json(registry_path);
                for (const auto& entry : registry.value("installations", json::array()))
                    if (entry.contains("manifest") && entry["manifest"].is_string())
                        candidates.emplace_back(entry["manifest"].get<std::string>());
            }
            catch (const std::exception&)
            {
                // A corrupt registry is rebuilt entirely from explicit search roots.
            }
        }
        for (const auto& root : search_roots)
        {
            std::error_code iterator_error;
            for (std::filesystem::recursive_directory_iterator
                     iterator(root, std::filesystem::directory_options::skip_permission_denied, iterator_error),
                 end;
                 iterator != end; iterator.increment(iterator_error))
            {
                if (iterator_error)
                {
                    iterator_error.clear();
                    continue;
                }
                if (iterator->is_regular_file() && iterator->path().filename() == "arc-installation.json")
                    candidates.push_back(iterator->path());
            }
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        std::vector<engine_installation_manifest> repaired;
        std::set<std::string> identities;
        for (const auto& candidate : candidates)
        {
            const auto installation = load_installation_manifest(candidate);
            if (installation && identities.insert(installation.value().installation_id).second)
                repaired.push_back(installation.value());
        }
        json registry{{"format", "arc-installation-registry"}, {"formatVersion", 1}, {"installations", json::array()}};
        for (const auto& installation : repaired)
            registry["installations"].push_back({{"installationId", installation.installation_id},
                                                 {"manifest", installation.manifest_path.generic_string()}});
        const auto written = write_json_atomic(registry_path, registry);
        if (!written) return installations_result::failure(written.error());
        std::sort(repaired.begin(), repaired.end(),
                  [](const auto& left, const auto& right)
                  {
                      return std::tie(left.engine_version, left.installation_id) <
                             std::tie(right.engine_version, right.installation_id);
                  });
        return installations_result::success(std::move(repaired));
    }
    catch (const std::exception& error)
    {
        return installations_result::failure(make_error(project_error_code::io_failed, registry_path, error.what()));
    }
}

tools_result detect_toolchains()
{
    std::vector<tool_snapshot> result;
    const auto add = [&](std::string id, std::string executable)
    {
        const auto found = find_on_path(executable);
        result.push_back(
            {.id = std::move(id),
             .executable = found.value_or(std::filesystem::path{}),
             .version = found ? probe_tool_version(*found, executable == "cl" ? std::vector<std::string>{}
                                                                              : std::vector<std::string>{"--version"})
                              : "",
             .available = found.has_value()});
    };
    add("cmake", "cmake");
    add("ninja", "ninja");
    add("clang", "clang++");
    add("gcc", "g++");
    add("msvc", "cl");
#if defined(_WIN32)
    const auto program_files = environment_value("ProgramFiles(x86)");
    if (program_files)
    {
        const auto vswhere =
            std::filesystem::path(*program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe";
        result.push_back({.id = "visual-studio",
                          .executable = vswhere,
                          .version = std::filesystem::is_regular_file(vswhere)
                                         ? probe_tool_version(vswhere, {"-latest", "-property", "installationVersion"})
                                         : "",
                          .available = std::filesystem::is_regular_file(vswhere)});
    }
#endif
    const auto vulkan_sdk = environment_value("VULKAN_SDK");
    const auto vulkan_path = vulkan_sdk ? std::filesystem::path(*vulkan_sdk) : std::filesystem::path{};
    result.push_back({.id = "vulkan-sdk",
                      .executable = vulkan_path,
                      .version = vulkan_sdk ? vulkan_path.filename().string() : "",
                      .available = vulkan_sdk.has_value()});
    return tools_result::success(std::move(result));
}

} // namespace arc::project

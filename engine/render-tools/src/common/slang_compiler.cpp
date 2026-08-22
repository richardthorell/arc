#include <arc/render_tools/render_tools.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <sstream>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace arc::render::tools
{
namespace
{

struct process_result
{
    int exit_code{-1};
    std::string output;
};

std::string read_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    std::ostringstream text;
    text << input.rdbuf();
    return text.str();
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto size = input.tellg();
    if (size <= 0) return {};
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!input) return {};
    return bytes;
}

#if defined(_WIN32)
std::wstring quote_windows_argument(std::wstring_view argument)
{
    if (argument.find_first_of(L" \t\"") == std::wstring_view::npos) return std::wstring(argument);
    std::wstring result{L'\"'};
    std::size_t slashes{};
    for (const wchar_t character : argument)
    {
        if (character == L'\\')
        {
            ++slashes;
            continue;
        }
        if (character == L'\"')
        {
            result.append(slashes * 2 + 1, L'\\');
            result.push_back(L'\"');
            slashes = 0;
            continue;
        }
        result.append(slashes, L'\\');
        slashes = 0;
        result.push_back(character);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'\"');
    return result;
}
#endif

process_result run_process(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
                           const std::filesystem::path& output_path)
{
#if defined(_WIN32)
    const auto executable_wide = executable.wstring();
    std::wstring command_line = quote_windows_argument(executable_wide);
    for (const auto& argument : arguments)
    {
        command_line.push_back(L' ');
        command_line += quote_windows_argument(std::filesystem::path(argument).wstring());
    }

    SECURITY_ATTRIBUTES security{.nLength = sizeof(SECURITY_ATTRIBUTES), .bInheritHandle = TRUE};
    const HANDLE output = CreateFileW(output_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security, CREATE_ALWAYS,
                                      FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (output == INVALID_HANDLE_VALUE) return {};

    STARTUPINFOW startup{.cb = sizeof(STARTUPINFOW)};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = output;
    startup.hStdError = output;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION process{};
    const BOOL created = CreateProcessW(executable_wide.c_str(), command_line.data(), nullptr, nullptr, TRUE,
                                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    if (!created)
    {
        CloseHandle(output);
        return {};
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code{};
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(output);
    return {.exit_code = static_cast<int>(exit_code), .output = read_text(output_path)};
#else
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.push_back(executable.string());
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& argument : storage)
        argv.push_back(argument.data());
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) return {};
    const auto output_text = output_path.string();
    if (posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, output_text.c_str(),
                                         O_WRONLY | O_CREAT | O_TRUNC, 0600) != 0 ||
        posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO) != 0)
    {
        posix_spawn_file_actions_destroy(&actions);
        return {};
    }
    pid_t process{};
    const int spawn_error = posix_spawn(&process, executable.c_str(), &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawn_error != 0) return {};
    int status{};
    if (waitpid(process, &status, 0) < 0) return {};
    const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    return {.exit_code = exit_code, .output = read_text(output_path)};
#endif
}

std::filesystem::path find_slangc()
{
#if defined(ARC_DEFAULT_SLANGC_EXECUTABLE)
    const std::filesystem::path configured{ARC_DEFAULT_SLANGC_EXECUTABLE};
    if (std::filesystem::is_regular_file(configured)) return configured;
#endif
#if defined(_WIN32)
    const auto environment = [](const char* name) -> std::string
    {
        char* value{};
        std::size_t size{};
        if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
        std::string result{value};
        std::free(value);
        return result;
    };
#else
    const auto environment = [](const char* name) -> std::string
    {
        // Tool discovery runs during single-threaded service construction.
        const char* value = std::getenv(name); // NOLINT(concurrency-mt-unsafe)
        return value == nullptr ? std::string{} : std::string{value};
    };
#endif
    const auto configured_value = environment("ARC_SLANGC_EXECUTABLE");
    if (!configured_value.empty())
    {
        const std::filesystem::path path{configured_value};
        if (std::filesystem::is_regular_file(path)) return path;
    }
    const auto path_value = environment("PATH");
    if (path_value.empty()) return {};
#if defined(_WIN32)
    constexpr char separator = ';';
    constexpr std::string_view name = "slangc.exe";
#else
    constexpr char separator = ':';
    constexpr std::string_view name = "slangc";
#endif
    std::string_view paths{path_value};
    std::size_t cursor{};
    while (cursor <= paths.size())
    {
        const auto end = paths.find(separator, cursor);
        const auto part = paths.substr(cursor, end == std::string_view::npos ? paths.size() - cursor : end - cursor);
        const auto candidate = std::filesystem::path(part) / name;
        if (std::filesystem::is_regular_file(candidate)) return candidate;
        if (end == std::string_view::npos) break;
        cursor = end + 1;
    }
    return {};
}

std::filesystem::path make_temporary_directory()
{
    static std::atomic<std::uint64_t> sequence{};
    std::error_code error;
    const auto root = std::filesystem::temp_directory_path(error);
    if (error) return {};
    const auto directory = root / ("arc-slang-" + std::to_string(++sequence));
    std::filesystem::create_directories(directory, error);
    return error ? std::filesystem::path{} : directory;
}

bool reports_pinned_slang_version(std::string_view output)
{
    const std::string escaped = std::regex_replace(std::string(pinned_slang_version), std::regex{R"(\.)"}, R"(\.)");
    return std::regex_search(std::string(output), std::regex{"(^|[^0-9.])" + escaped + "([^0-9.]|$)"});
}

struct temporary_directory
{
    std::filesystem::path path;
    ~temporary_directory()
    {
        std::error_code error;
        if (!path.empty()) std::filesystem::remove_all(path, error);
    }
};

std::string stage_name(shader_stage stage)
{
    switch (stage)
    {
        case shader_stage::vertex:
            return "vertex";
        case shader_stage::fragment:
            return "fragment";
        case shader_stage::compute:
            return "compute";
        case shader_stage::ray_generation:
            return "raygeneration";
        case shader_stage::closest_hit:
            return "closesthit";
        case shader_stage::any_hit:
            return "anyhit";
        case shader_stage::miss:
            return "miss";
    }
    return "fragment";
}

std::vector<shader_diagnostic> parse_diagnostics(std::string_view text, const shader_compile_request& request)
{
    static const std::regex pattern{R"(^(.+?)[(:](\d+)(?:[,)]|:)(\d+)?[:)]?\s*:?\s*(warning|error|note)?\s*:?[ ]*(.*)$)",
                                    std::regex::icase};
    std::vector<shader_diagnostic> result;
    std::istringstream lines{std::string(text)};
    std::string line;
    while (std::getline(lines, line))
    {
        if (line.empty()) continue;
        std::smatch match;
        shader_diagnostic diagnostic;
        diagnostic.message = line;
        if (std::regex_match(line, match, pattern))
        {
            diagnostic.location.path = match[1].str();
            diagnostic.location.line = static_cast<std::uint32_t>(std::stoul(match[2].str()));
            if (match[3].matched && !match[3].str().empty())
                diagnostic.location.column = static_cast<std::uint32_t>(std::stoul(match[3].str()));
            const auto severity = match[4].str();
            diagnostic.severity = severity == "warning" ? shader_diagnostic_severity::warning
                                  : severity == "note"  ? shader_diagnostic_severity::information
                                                        : shader_diagnostic_severity::error;
            if (match[5].matched && !match[5].str().empty()) diagnostic.message = match[5].str();
            if (const auto node = request.generated_line_nodes.find(diagnostic.location.line);
                node != request.generated_line_nodes.end())
                diagnostic.location.graph_node_id = node->second;
        }
        result.push_back(std::move(diagnostic));
    }
    return result;
}

shader_resource_kind resource_kind(std::string_view kind)
{
    std::string normalized(kind);
    std::ranges::transform(normalized, normalized.begin(),
                           [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    if (normalized.find("sampler") != std::string::npos) return shader_resource_kind::sampler;
    if (normalized.find("texture") != std::string::npos) return shader_resource_kind::sampled_texture;
    if (normalized.find("storage") != std::string::npos || normalized.find("rw") != std::string::npos)
        return shader_resource_kind::read_write_buffer;
    if (normalized.find("structured") != std::string::npos) return shader_resource_kind::structured_buffer;
    if (normalized.find("acceleration") != std::string::npos)
        return shader_resource_kind::acceleration_structure;
    return shader_resource_kind::constant_buffer;
}

std::string reflected_kind(const nlohmann::json& value)
{
    if (value.is_string()) return value.get<std::string>();
    if (!value.is_object()) return {};
    if (value.contains("kind") && value["kind"].is_string()) return value["kind"].get<std::string>();
    if (value.contains("type")) return reflected_kind(value["type"]);
    return {};
}

shader_parameter_type reflected_parameter_type(const nlohmann::json& type)
{
    auto kind = reflected_kind(type);
    std::ranges::transform(kind, kind.begin(), [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const auto columns = type.is_object() ? type.value("columnCount", type.value("elementCount", 1u)) : 1u;
    if (kind.find("bool") != std::string::npos) return shader_parameter_type::boolean;
    if (kind.find("uint") != std::string::npos) return shader_parameter_type::uint32;
    if (kind.find("int") != std::string::npos) return shader_parameter_type::int32;
    if (kind.find("matrix") != std::string::npos) return shader_parameter_type::matrix4x4;
    if (kind.find("texturecube") != std::string::npos) return shader_parameter_type::texture_cube;
    if (kind.find("texture") != std::string::npos) return shader_parameter_type::texture_2d;
    if (kind.find("sampler") != std::string::npos) return shader_parameter_type::sampler;
    if (columns == 4 || kind.find("vector4") != std::string::npos) return shader_parameter_type::float4;
    if (columns == 3 || kind.find("vector3") != std::string::npos) return shader_parameter_type::float3;
    if (columns == 2 || kind.find("vector2") != std::string::npos) return shader_parameter_type::float2;
    return shader_parameter_type::float32;
}

std::uint32_t reflected_size(shader_parameter_type type)
{
    switch (type)
    {
        case shader_parameter_type::float2: return 8;
        case shader_parameter_type::float3: return 12;
        case shader_parameter_type::float4: return 16;
        case shader_parameter_type::matrix4x4: return 64;
        default: return 4;
    }
}

void append_reflected_fields(const nlohmann::json& fields, shader_reflection& reflection)
{
    if (!fields.is_array()) return;
    for (const auto& field : fields)
    {
        if (!field.is_object()) continue;
        shader_parameter_descriptor parameter;
        parameter.name = field.value("name", "");
        if (parameter.name.empty()) continue;
        parameter.id = make_shader_parameter_id(parameter.name);
        const auto& type = field.contains("type") ? field["type"] : field;
        parameter.type = reflected_parameter_type(type);
        parameter.offset = field.value("offset", 0u);
        if (field.contains("binding") && field["binding"].is_object())
            parameter.offset = field["binding"].value("offset", parameter.offset);
        parameter.size = field.value("size", reflected_size(parameter.type));
        reflection.parameter_block_size = std::max(reflection.parameter_block_size,
                                                   parameter.offset + parameter.size);
        reflection.parameters.push_back(std::move(parameter));
    }
}

shader_reflection parse_reflection(const std::filesystem::path& path, const shader_compile_request& request)
{
    shader_reflection reflection;
    reflection.domain = request.domain;
    const auto document = nlohmann::json::parse(read_text(path), nullptr, false);
    if (!document.is_discarded() && document.contains("entryPoints") && document["entryPoints"].is_array())
    {
        for (const auto& entry : document["entryPoints"])
        {
            shader_entry_point_descriptor descriptor;
            descriptor.name = entry.value("name", request.entry_point);
            descriptor.stage = request.stage;
            descriptor.profile = request.profile;
            descriptor.id = make_shader_entry_point_id(descriptor.name, descriptor.stage);
            if (entry.contains("threadGroupSize") && entry["threadGroupSize"].is_array() &&
                entry["threadGroupSize"].size() == 3)
                for (std::size_t index = 0; index < 3; ++index)
                    descriptor.thread_group_size[index] = entry["threadGroupSize"][index].get<std::uint32_t>();
            reflection.entry_points.push_back(std::move(descriptor));
        }
    }
    if (reflection.entry_points.empty())
        reflection.entry_points.push_back({.id = make_shader_entry_point_id(request.entry_point, request.stage),
                                           .name = request.entry_point,
                                           .stage = request.stage,
                                           .profile = request.profile});

    if (!document.is_discarded() && document.contains("parameters") && document["parameters"].is_array())
    {
        for (const auto& parameter : document["parameters"])
        {
            if (!parameter.is_object()) continue;
            shader_resource_descriptor resource;
            resource.name = parameter.value("name", "");
            const auto kind = reflected_kind(parameter);
            resource.kind = resource_kind(kind);
            if (parameter.contains("binding") && parameter["binding"].is_object())
            {
                resource.binding = parameter["binding"].value("index", parameter["binding"].value("binding", 0u));
                resource.set = parameter["binding"].value("space", 0u);
            }
            else
            {
                resource.binding = parameter.value("binding", 0u);
                resource.set = parameter.value("space", 0u);
            }
            if (!resource.name.empty()) reflection.resources.push_back(std::move(resource));

            if (parameter.contains("type") && parameter["type"].is_object())
            {
                const auto& type = parameter["type"];
                if (type.contains("fields")) append_reflected_fields(type["fields"], reflection);
                if (type.contains("elementType") && type["elementType"].is_object() &&
                    type["elementType"].contains("fields"))
                    append_reflected_fields(type["elementType"]["fields"], reflection);
            }
            if (parameter.contains("fields")) append_reflected_fields(parameter["fields"], reflection);
        }
    }
    return reflection;
}

} // namespace

slang_shader_compiler::slang_shader_compiler(slang_compiler_config config) : config_(std::move(config))
{
    if (config_.executable.empty()) config_.executable = find_slangc();
    if (config_.executable.empty() || !std::filesystem::is_regular_file(config_.executable))
    {
        fingerprint_ = "slang/unavailable";
        return;
    }
    temporary_directory temporary{make_temporary_directory()};
    if (temporary.path.empty())
    {
        fingerprint_ = "slang/unavailable";
        return;
    }
    const auto version = run_process(config_.executable, {"-version"}, temporary.path / "version.txt");
    available_ = version.exit_code == 0 &&
                 (!config_.require_pinned_version || reports_pinned_slang_version(version.output));
    fingerprint_ = available_ ? "slang/" + std::string(pinned_slang_version) : "slang/version-mismatch";
}

shader_compile_result slang_shader_compiler::compile(const shader_compile_request& request)
{
    if (!available_)
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::compiler_unavailable,
             .source_path = request.source_path,
             .message = "Slang " + std::string(pinned_slang_version) +
                        " is unavailable; configure ARC_SLANGC_EXECUTABLE with the pinned toolchain"});
    if (request.target != shader_target::spirv || request.source_path.empty() || request.entry_point.empty())
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::invalid_request,
             .source_path = request.source_path,
             .message = "the initial ARC Slang adapter requires a source, entry point, and SPIR-V target"});
    const auto source_for_validation = request.source_override.empty() ? read_text(request.source_path)
                                                                       : request.source_override;
    if (source_for_validation.find("[[vk::") != std::string::npos ||
        source_for_validation.find("__spirv_") != std::string::npos)
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .source_path = request.source_path,
             .message = "backend-specific Vulkan/SPIR-V source annotations are not allowed in ARC shaders"});

    temporary_directory temporary{make_temporary_directory()};
    if (temporary.path.empty())
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::compilation_failed,
             .source_path = request.source_path,
             .message = "could not create shader compiler staging directory"});

    auto source_path = std::filesystem::path(request.source_path);
    if (!request.source_override.empty())
    {
        source_path = temporary.path / (source_path.stem().string() + ".slang");
        std::ofstream source(source_path, std::ios::binary);
        source.write(request.source_override.data(), static_cast<std::streamsize>(request.source_override.size()));
        if (!source)
            return shader_compile_result::failure(
                {.code = shader_compile_error_code::source_unavailable,
                 .source_path = request.source_path,
                 .message = "could not stage the transient shader source"});
    }

    const auto output_path = temporary.path / "shader.spv";
    const auto reflection_path = temporary.path / "reflection.json";
    const auto diagnostics_path = temporary.path / "diagnostics.txt";
    std::vector<std::string> arguments{source_path.string(),
                                       "-entry",
                                       request.entry_point,
                                       "-stage",
                                       stage_name(request.stage),
                                       "-target",
                                       "spirv",
                                       "-profile",
                                       request.profile.empty() ? "spirv_1_5" : request.profile,
                                       "-reflection-json",
                                       reflection_path.string(),
                                       "-o",
                                       output_path.string()};
    arguments.push_back(request.optimization == shader_optimization::disabled
                            ? "-O0"
                            : request.optimization == shader_optimization::performance ? "-O3" : "-O1");
    if (request.generate_debug_information) arguments.push_back("-g");
    for (const auto& directory : request.include_directories)
    {
        arguments.push_back("-I");
        arguments.push_back(directory.string());
    }
    for (const auto& define : request.defines)
    {
        arguments.push_back("-D");
        arguments.push_back(define);
    }
    for (const auto& value : request.static_switches)
    {
        arguments.push_back("-D");
        arguments.push_back(value.name + "=" + (value.value ? "1" : "0"));
    }

    const auto process = run_process(config_.executable, arguments, diagnostics_path);
    auto diagnostics = parse_diagnostics(process.output, request);
    if (process.exit_code != 0)
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::compilation_failed,
             .source_path = request.source_path,
             .message = "Slang rejected the shader source",
             .diagnostics = std::move(diagnostics)});

    auto bytecode = read_bytes(output_path);
    if (bytecode.empty())
        return shader_compile_result::failure(
            {.code = shader_compile_error_code::compilation_failed,
             .source_path = request.source_path,
             .message = "Slang completed without producing SPIR-V",
             .diagnostics = std::move(diagnostics)});

    auto reflection = parse_reflection(reflection_path, request);
    const auto entry_id = make_shader_entry_point_id(request.entry_point, request.stage);
    for (const auto pass : request.required_passes)
        reflection.passes.push_back({.pass = pass, .entry_point = entry_id, .generated = true});
    return shader_compile_result::success({.bytecode = std::move(bytecode),
                                           .reflection = std::move(reflection),
                                           .diagnostics = std::move(diagnostics),
                                           .compiler_fingerprint = fingerprint_});
}

std::string_view slang_shader_compiler::fingerprint() const noexcept
{
    return fingerprint_;
}

bool slang_shader_compiler::available() const noexcept
{
    return available_;
}

const std::filesystem::path& slang_shader_compiler::executable() const noexcept
{
    return config_.executable;
}

} // namespace arc::render::tools

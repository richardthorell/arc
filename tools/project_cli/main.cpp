#include <arc/project/project.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace
{
using json = nlohmann::json;

struct arguments
{
    std::vector<std::string> values;

    bool has(std::string_view value) const
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    }

    std::optional<std::string> option(std::string_view name) const
    {
        const auto found = std::find(values.begin(), values.end(), name);
        if (found == values.end() || std::next(found) == values.end()) return std::nullopt;
        return *std::next(found);
    }
};

void print_json(bool enabled, const json& value)
{
    if (enabled) std::cout << value.dump() << '\n';
}

int fail(bool json_output, std::string_view message, int code = 1)
{
    if (json_output)
        print_json(true, {{"succeeded", false}, {"error", message}});
    else
        std::cerr << "arc-project: " << message << '\n';
    return code;
}

std::filesystem::path descriptor_from(const arguments& args)
{
    if (const auto value = args.option("--project"))
    {
        auto path = std::filesystem::absolute(*value).lexically_normal();
        if (std::filesystem::is_regular_file(path)) return path;
        if (std::filesystem::is_directory(path))
        {
            std::vector<std::filesystem::path> descriptors;
            for (const auto& entry : std::filesystem::directory_iterator(path))
                if (entry.is_regular_file() && entry.path().extension() == ".arcproject")
                    descriptors.push_back(entry.path());
            if (descriptors.size() == 1) return descriptors.front();
        }
    }
    return {};
}

#if defined(_WIN32)
std::wstring quote_windows(std::string_view value)
{
    std::wstring wide(value.begin(), value.end());
    std::wstring result = L"\"";
    std::size_t slashes = 0;
    for (const wchar_t character : wide)
    {
        if (character == L'\\') { ++slashes; continue; }
        if (character == L'\"') result.append(slashes * 2 + 1, L'\\');
        else result.append(slashes, L'\\');
        slashes = 0;
        result.push_back(character);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'\"');
    return result;
}
#endif

int run_process(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
                const std::filesystem::path& working_directory)
{
#if defined(_WIN32)
    std::wstring command = quote_windows(executable.string());
    for (const auto& argument : arguments) command += L" " + quote_windows(argument);
    STARTUPINFOW startup{.cb = sizeof(STARTUPINFOW)};
    PROCESS_INFORMATION process{};
    std::wstring directory = working_directory.wstring();
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr,
                        directory.empty() ? nullptr : directory.c_str(), &startup, &process))
        return -1;
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
#else
    const pid_t child = fork();
    if (child < 0) return -1;
    if (child == 0)
    {
        if (!working_directory.empty()) chdir(working_directory.c_str());
        std::vector<char*> argv;
        std::string executable_text = executable.string();
        argv.push_back(executable_text.data());
        std::vector<std::string> storage = arguments;
        for (auto& argument : storage) argv.push_back(argument.data());
        argv.push_back(nullptr);
        execvp(executable_text.c_str(), argv.data());
        _exit(127);
    }
    int status = 1;
    if (waitpid(child, &status, 0) < 0) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

std::filesystem::path registry_path(const arguments& args)
{
    return args.option("--registry").value_or(arc::project::default_installation_registry_path().string());
}

int handle_engine(const arguments& args, bool json_output)
{
    if (args.values.size() < 2) return fail(json_output, "engine requires list, register, unregister, verify, or repair", 2);
    const auto action = args.values[1];
    if (action == "list" || action == "verify" || action == "repair")
    {
        const auto result = action == "repair"
                                ? arc::project::repair_installations(
                                      registry_path(args),
                                      args.option("--search")
                                          ? std::vector<std::filesystem::path>{*args.option("--search")}
                                          : std::vector<std::filesystem::path>{})
                                : arc::project::discover_installations(registry_path(args));
        if (!result) return fail(json_output, result.error().message);
        if (action == "verify")
            for (const auto& installation : result.value())
            {
                if (!std::filesystem::is_directory(installation.sdk) ||
                    !std::filesystem::is_regular_file(installation.sdk / "lib" / "cmake" / "ARC" /
                                                      "ARCConfig.cmake") ||
                    !std::filesystem::is_regular_file(installation.project_tool) ||
                    (!installation.cooker.empty() && !std::filesystem::is_regular_file(installation.cooker)) ||
                    (!installation.editor.empty() && !std::filesystem::is_regular_file(installation.editor)) ||
                    std::any_of(installation.templates.begin(), installation.templates.end(),
                                [](const auto& item) { return !std::filesystem::is_directory(item.root); }))
                    return fail(json_output, "registered ARC installation is incomplete: " +
                                                     installation.manifest_path.string());
            }
        json values = json::array();
        for (const auto& installation : result.value())
            values.push_back({{"installationId", installation.installation_id},
                              {"engineVersion", installation.engine_version},
                              {"manifest", installation.manifest_path.generic_string()},
                              {"root", installation.root.generic_string()},
                              {"editor", installation.editor.generic_string()},
                              {"sdk", installation.sdk.generic_string()},
                              {"projectTool", installation.project_tool.generic_string()}});
        print_json(json_output, {{"succeeded", true}, {"installations", values}});
        if (!json_output)
            for (const auto& value : values)
                std::cout << value["engineVersion"].get<std::string>() << "  "
                          << value["manifest"].get<std::string>() << '\n';
        return 0;
    }
    if (action == "register")
    {
        const auto manifest = args.option("--manifest");
        if (!manifest) return fail(json_output, "engine register requires --manifest", 2);
        const auto result = arc::project::register_installation(registry_path(args), *manifest);
        if (!result) return fail(json_output, result.error().message);
    }
    else if (action == "unregister")
    {
        const auto id = args.option("--id");
        if (!id) return fail(json_output, "engine unregister requires --id", 2);
        const auto result = arc::project::unregister_installation(registry_path(args), *id);
        if (!result) return fail(json_output, result.error().message);
    }
    else
        return fail(json_output, "unknown engine command", 2);
    print_json(json_output, {{"succeeded", true}});
    return 0;
}

int configure_or_build(const arguments& args, bool build, bool json_output)
{
    const auto descriptor_path = descriptor_from(args);
    if (descriptor_path.empty()) return fail(json_output, "command requires a project descriptor", 2);
    const auto descriptor = arc::project::load_descriptor(descriptor_path);
    if (!descriptor) return fail(json_output, descriptor.error().message);
    const auto context = arc::project::resolve_context(descriptor_path, descriptor.value());
    if (!context) return fail(json_output, context.error().message);
    const auto cmake = args.option("--cmake").value_or("cmake");
    const auto build_directory = context.value().build_root / args.option("--build-dir").value_or("default");
    std::vector<std::string> command;
    if (build)
    {
        command = {"--build", build_directory.string(), "--config", args.option("--config").value_or("RelWithDebInfo")};
        if (const auto target = args.option("--target"))
        {
            command.push_back("--target");
            command.push_back(*target);
        }
    }
    else
    {
        command = {"-S", context.value().root.string(), "-B", build_directory.string()};
        if (const auto sdk = args.option("--sdk")) command.push_back("-DCMAKE_PREFIX_PATH=" + *sdk);
        if (const auto generator = args.option("--generator")) { command.push_back("-G"); command.push_back(*generator); }
    }
    const int code = run_process(cmake, command, context.value().root);
    if (code != 0) return fail(json_output, "CMake command failed", code < 0 ? 1 : code);
    if (build)
    {
        const auto configuration = args.option("--config").value_or("RelWithDebInfo");
        const auto module_manifest = build_directory / ("arc-modules-" + configuration + ".json");
        if (std::filesystem::is_regular_file(module_manifest))
        {
            const auto editor_state = context.value().saved_root / "Editor";
            std::filesystem::create_directories(editor_state);
            std::ofstream(editor_state / "active-build.json", std::ios::trunc)
                << json{{"format", "arc-active-build"}, {"formatVersion", 1},
                        {"configuration", configuration},
                        {"moduleManifest", std::filesystem::relative(module_manifest, context.value().root).generic_string()}}
                       .dump(2)
                << '\n';
        }
    }
    print_json(json_output, {{"succeeded", true}, {"buildDirectory", build_directory.generic_string()}});
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    arguments args;
    for (int index = 1; index < argc; ++index) args.values.emplace_back(argv[index]);
    const bool json_output = args.has("--json");
    if (args.values.empty())
        return fail(json_output, "expected create, validate, upgrade, engine, toolchains, configure, build, or ide", 2);
    const auto command = args.values.front();
    if (command == "create")
    {
        const auto name = args.option("--name");
        const auto destination = args.option("--destination");
        const auto template_id = args.option("--template");
        const auto templates = args.option("--templates");
        const auto engine = args.option("--engine");
        if (!name || !destination || !template_id || !templates || !engine)
            return fail(json_output, "create requires --name, --destination, --template, --templates, and --engine", 2);
        const auto result = arc::project::create_project({.name = *name, .destination = *destination,
                                                          .template_id = *template_id, .templates_root = *templates,
                                                          .engine_version = *engine});
        if (!result) return fail(json_output, result.error().message);
        print_json(json_output, {{"succeeded", true}, {"destination", std::filesystem::absolute(*destination).generic_string()}});
        return 0;
    }
    if (command == "validate")
    {
        const auto descriptor_path = descriptor_from(args);
        if (descriptor_path.empty()) return fail(json_output, "validate requires --project", 2);
        const auto descriptor = arc::project::load_descriptor(descriptor_path);
        if (!descriptor) return fail(json_output, descriptor.error().message);
        const auto validation = arc::project::validate_descriptor(descriptor_path, descriptor.value(),
            {.engine_version = args.option("--engine").value_or(""),
             .require_exact_engine = args.option("--engine").has_value(),
             .require_paths = args.has("--require-paths"), .allow_read_only = args.has("--read-only")});
        if (!validation) return fail(json_output, validation.error().message);
        print_json(json_output, {{"succeeded", true}, {"writable", validation.value().writable},
                                 {"name", descriptor.value().name}, {"engineVersion", descriptor.value().engine_version}});
        if (!json_output) std::cout << descriptor.value().name << " is valid\n";
        return 0;
    }
    if (command == "upgrade")
    {
        const auto descriptor_path = descriptor_from(args);
        const auto engine = args.option("--engine");
        if (descriptor_path.empty() || !engine) return fail(json_output, "upgrade requires --project and --engine", 2);
        const auto result = arc::project::upgrade_descriptor(descriptor_path, *engine);
        if (!result) return fail(json_output, result.error().message);
        print_json(json_output, {{"succeeded", true}});
        return 0;
    }
    if (command == "engine") return handle_engine(args, json_output);
    if (command == "toolchains")
    {
        const auto result = arc::project::detect_toolchains();
        if (!result) return fail(json_output, result.error().message);
        json tools = json::array();
        for (const auto& tool : result.value())
            tools.push_back({{"id", tool.id}, {"available", tool.available},
                             {"executable", tool.executable.generic_string()}, {"version", tool.version}});
        print_json(json_output, {{"succeeded", true}, {"tools", tools}});
        if (!json_output)
            for (const auto& tool : result.value()) std::cout << tool.id << ": " << (tool.available ? "available" : "missing") << '\n';
        return 0;
    }
    if (command == "configure") return configure_or_build(args, false, json_output);
    if (command == "build") return configure_or_build(args, true, json_output);
    if (command == "ide")
    {
        if (args.values.size() < 2) return fail(json_output, "ide requires generate or launch", 2);
        const auto descriptor_path = descriptor_from(args);
        const auto ide = args.option("--ide");
        if (descriptor_path.empty() || !ide) return fail(json_output, "ide requires --project and --ide", 2);
        const auto root = descriptor_path.parent_path();
        if (args.values[1] == "generate" && *ide == "vscode")
        {
            std::ofstream(root / (descriptor_path.stem().string() + ".code-workspace"))
                << json{{"folders", json::array({{{"path", "."}}})}, {"settings", {{"cmake.useCMakePresets", "always"}}}}.dump(2) << '\n';
        }
        else if (args.values[1] == "generate" && *ide == "visual-studio")
        {
            const auto directory = root / "Build" / "VisualStudio";
            const int code = run_process(args.option("--cmake").value_or("cmake"),
                                         {"-S", root.string(), "-B", directory.string(), "-G",
                                          "Visual Studio 17 2022", "-A", "x64"}, root);
            if (code != 0) return fail(json_output, "Visual Studio generation failed");
        }
        else if (args.values[1] == "generate" && *ide == "clion")
        {
            if (!std::filesystem::is_regular_file(root / "CMakePresets.json"))
                return fail(json_output, "CLion requires the generated CMakePresets.json");
        }
        else if (args.values[1] == "launch")
        {
            const std::string executable = *ide == "visual-studio" ? "devenv" : *ide == "vscode" ? "code" : "clion";
            std::filesystem::path launch_target = root;
            if (*ide == "visual-studio")
            {
                const auto solution_root = root / "Build" / "VisualStudio";
                if (!std::filesystem::is_directory(solution_root))
                    return fail(json_output, "Generate the Visual Studio solution before launching it");
                for (const auto& entry : std::filesystem::directory_iterator(solution_root))
                    if (entry.path().extension() == ".sln") { launch_target = entry.path(); break; }
            }
            const int code = run_process(executable, {launch_target.string()}, root);
            if (code != 0) return fail(json_output, "IDE launch failed");
        }
        else return fail(json_output, "unsupported IDE operation", 2);
        print_json(json_output, {{"succeeded", true}});
        return 0;
    }
    return fail(json_output, "unknown command", 2);
}

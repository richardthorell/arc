#include <arc/project/project.h>
#include "android_content_command.h"
#include "package_command.h"
#include "workflow_command.h"

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
        if (character == L'\\')
        {
            ++slashes;
            continue;
        }
        if (character == L'\"')
            result.append(slashes * 2 + 1, L'\\');
        else
            result.append(slashes, L'\\');
        slashes = 0;
        result.push_back(character);
    }
    result.append(slashes * 2, L'\\');
    result.push_back(L'\"');
    return result;
}
#endif

void relay_process_output(const char* data, std::size_t size, std::string* captured, bool diagnostic_stream)
{
    if (captured)
    {
        captured->append(data, size);
        constexpr std::size_t max_capture = 8192;
        if (captured->size() > max_capture) captured->erase(0, captured->size() - max_capture);
    }
    auto& stream = diagnostic_stream ? std::cerr : std::cout;
    stream.write(data, static_cast<std::streamsize>(size));
    stream.flush();
}

int run_process(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
                const std::filesystem::path& working_directory, std::string* captured = nullptr,
                bool diagnostic_stream = false)
{
#if defined(_WIN32)
    std::wstring command = quote_windows(executable.string());
    for (const auto& argument : arguments)
        command += L" " + quote_windows(argument);

    SECURITY_ATTRIBUTES security{.nLength = sizeof(SECURITY_ATTRIBUTES), .bInheritHandle = TRUE};
    HANDLE output_read = nullptr;
    HANDLE output_write = nullptr;
    if (!CreatePipe(&output_read, &output_write, &security, 0)) return -1;
    if (!SetHandleInformation(output_read, HANDLE_FLAG_INHERIT, 0))
    {
        CloseHandle(output_write);
        CloseHandle(output_read);
        return -1;
    }

    STARTUPINFOW startup{.cb = sizeof(STARTUPINFOW)};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = output_write;
    startup.hStdError = output_write;
    PROCESS_INFORMATION process{};
    std::wstring directory = working_directory.wstring();
    const BOOL started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, 0, nullptr,
                                        directory.empty() ? nullptr : directory.c_str(), &startup, &process);
    CloseHandle(output_write);
    if (!started)
    {
        CloseHandle(output_read);
        return -1;
    }

    char buffer[4096];
    DWORD count = 0;
    while (ReadFile(output_read, buffer, sizeof(buffer), &count, nullptr) && count != 0)
        relay_process_output(buffer, count, captured, diagnostic_stream);
    CloseHandle(output_read);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
#else
    int output_pipe[2];
    if (pipe(output_pipe) != 0) return -1;
    const pid_t child = fork();
    if (child < 0)
    {
        close(output_pipe[0]);
        close(output_pipe[1]);
        return -1;
    }
    if (child == 0)
    {
        close(output_pipe[0]);
        if (dup2(output_pipe[1], STDOUT_FILENO) < 0 || dup2(output_pipe[1], STDERR_FILENO) < 0) _exit(127);
        close(output_pipe[1]);
        if (!working_directory.empty() && chdir(working_directory.c_str()) != 0) _exit(127);
        std::vector<char*> argv;
        std::string executable_text = executable.string();
        argv.push_back(executable_text.data());
        std::vector<std::string> storage = arguments;
        for (auto& argument : storage)
            argv.push_back(argument.data());
        argv.push_back(nullptr);
        execvp(executable_text.c_str(), argv.data());
        _exit(127);
    }
    close(output_pipe[1]);
    char buffer[4096];
    ssize_t count = 0;
    while ((count = read(output_pipe[0], buffer, sizeof(buffer))) > 0)
        relay_process_output(buffer, static_cast<std::size_t>(count), captured, diagnostic_stream);
    close(output_pipe[0]);
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
    if (args.values.size() < 2)
        return fail(json_output, "engine requires list, register, unregister, verify, or repair", 2);
    const auto action = args.values[1];
    if (action == "list" || action == "verify" || action == "repair")
    {
        const auto result =
            action == "repair"
                ? arc::project::repair_installations(registry_path(args),
                                                     args.option("--search")
                                                         ? std::vector<std::filesystem::path>{*args.option("--search")}
                                                         : std::vector<std::filesystem::path>{})
                : arc::project::discover_installations(registry_path(args));
        if (!result) return fail(json_output, result.error().message);
        if (action == "verify")
            for (const auto& installation : result.value())
            {
                if (!std::filesystem::is_directory(installation.sdk) ||
                    !std::filesystem::is_regular_file(installation.sdk / "lib" / "cmake" / "ARC" / "ARCConfig.cmake") ||
                    !std::filesystem::is_regular_file(installation.project_tool) ||
                    (!installation.cooker.empty() && !std::filesystem::is_regular_file(installation.cooker)) ||
                    (!installation.editor.empty() && !std::filesystem::is_regular_file(installation.editor)) ||
                    std::any_of(installation.templates.begin(), installation.templates.end(),
                                [](const auto& item) { return !std::filesystem::is_directory(item.root); }))
                    return fail(json_output,
                                "registered ARC installation is incomplete: " + installation.manifest_path.string());
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
                std::cout << value["engineVersion"].get<std::string>() << "  " << value["manifest"].get<std::string>()
                          << '\n';
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
    const auto configuration = args.option("--config").value_or("RelWithDebInfo");
    const auto build_directory = context.value().build_root / args.option("--build-dir").value_or("default");
    const bool configured = std::filesystem::is_regular_file(build_directory / "CMakeCache.txt");

    // A clean on a never-configured project is already satisfied. This also
    // makes the editor's Rebuild action work for a freshly created project.
    if (build && !configured && args.option("--target") == "clean")
    {
        json response;
        response["succeeded"] = true;
        response["buildDirectory"] = build_directory.generic_string();
        response["nothingToClean"] = true;
        print_json(json_output, response);
        if (!json_output) std::cout << "No configured project build to clean\n";
        return 0;
    }

    const auto configure = [&]() -> int
    {
        std::vector<std::string> command{"-S", context.value().root.string(), "-B", build_directory.string(),
                                         "-DCMAKE_BUILD_TYPE=" + configuration};
        auto sdk = args.option("--sdk");
        if (!sdk)
        {
            // The source-editor launcher supplies one exact local SDK staging root.
            // Never guess a checkout, pick a newest build, or register it globally.
            std::filesystem::path development;
#if defined(_WIN32)
            const auto length = GetEnvironmentVariableW(L"ARC_DEVELOPMENT_SDK", nullptr, 0);
            if (length > 0)
            {
                std::wstring value(length, L'\0');
                const auto written = GetEnvironmentVariableW(L"ARC_DEVELOPMENT_SDK", value.data(), length);
                if (written > 0 && written < length)
                {
                    value.resize(written);
                    development = value;
                }
            }
#else
            if (const char* value = std::getenv("ARC_DEVELOPMENT_SDK")) development = value;
#endif
            if (!development.empty())
            {
                const auto& root = development;
                if (configuration.find_first_of("/\\:") != std::string::npos || configuration == "." ||
                    configuration == "..")
                    return fail(json_output, "Invalid development SDK configuration");
                std::ifstream manifest(root / configuration / "arc-development-sdk.json");
                const auto metadata = json::parse(manifest, nullptr, false);
                if (!metadata.is_object() || metadata.value("version", json{}) != 1 ||
                    metadata.value("engineVersion", json{}) != descriptor.value().engine_version ||
                    metadata.value("configuration", json{}) != configuration || !metadata.contains("buildDirectory") ||
                    !metadata["buildDirectory"].is_string())
                    return fail(json_output, "The development ARC SDK does not support this engine version or "
                                             "configuration. Restart run_editor.py with a matching build, or select "
                                             "an installed SDK using --sdk.");
                const auto engine_build = metadata["buildDirectory"].get<std::string>();
                std::string output;
                const int code = run_process(
                    cmake, {"--build", engine_build, "--config", configuration, "--target", "arc-development-sdk"},
                    context.value().root, &output, json_output);
                if (code != 0)
                    return fail(json_output,
                                "Preparing the matching development ARC SDK failed. Review the "
                                "engine build output above. If editor tool sources changed, restart "
                                "through run_editor.py before retrying.",
                                code < 0 ? 1 : code);
                sdk = (root / configuration).string();
            }
        }
        if (!sdk)
        {
            // Select only an unambiguous installation for the exact engine
            // version. Never silently use another engine version or SDK.
            const auto installations =
                arc::project::discover_installations(arc::project::default_installation_registry_path());
            if (installations)
            {
                std::vector<std::filesystem::path> matches;
                for (const auto& installed : installations.value())
                    if (installed.engine_version == descriptor.value().engine_version &&
                        std::filesystem::is_regular_file(installed.sdk / "lib" / "cmake" / "ARC" / "ARCConfig.cmake"))
                        if (std::find(matches.begin(), matches.end(), installed.sdk) == matches.end())
                            matches.push_back(installed.sdk);
                if (matches.size() == 1) sdk = matches.front().string();
                if (matches.size() > 1)
                    return fail(json_output, "Multiple matching ARC SDK installations exist for " +
                                                 descriptor.value().engine_version +
                                                 ". Select the intended installation using --sdk <ARC SDK root>.");
            }
        }
        if (sdk)
        {
            command.push_back("-DCMAKE_PREFIX_PATH=" + *sdk);
            // Explicit SDK selection must override an older cached ARC_DIR.
            command.push_back("-DARC_DIR=" + (std::filesystem::path(*sdk) / "lib" / "cmake" / "ARC").generic_string());
        }
        if (const auto generator = args.option("--generator"))
        {
            command.push_back("-G");
            command.push_back(*generator);
        }

        std::string output;
        const int code = run_process(cmake, command, context.value().root, &output, json_output);
        if (code != 0)
        {
            const std::string hint =
                "Configure CMake and the matching ARC SDK, or pass --sdk <installed ARC SDK root>.";
            std::string description = "CMake configuration failed for ";
            description += build_directory.string();
            description += " (exit " + std::to_string(code) + "). ";
            description += hint;
            return fail(json_output, description, code < 0 ? 1 : code);
        }
        return 0;
    };

    // A cache is written even when configuration/generation fails; it does
    // not prove that a native build system exists. Reconfigure before each
    // build so retries recover and explicit SDK/configuration changes apply.
    // CMake still reuses its cache and incremental compilation outputs.
    {
        const int code = configure();
        if (code != 0) return code;
    }
    if (build)
    {
        std::vector<std::string> command{"--build", build_directory.string(), "--config", configuration};
        if (const auto target = args.option("--target"))
        {
            command.push_back("--target");
            command.push_back(*target);
        }
        std::string output;
        const int code = run_process(cmake, command, context.value().root, &output, json_output);
        if (code != 0)
        {
            std::string description = "CMake build failed in ";
            description += build_directory.string();
            description += " for " + configuration;
            description += " (exit " + std::to_string(code) + "). Review the CMake output above.";
            return fail(json_output, description, code < 0 ? 1 : code);
        }
        const auto module_manifest = build_directory / ("arc-modules-" + configuration + ".json");
        if (std::filesystem::is_regular_file(module_manifest))
        {
            const auto editor_state = context.value().saved_root / "Editor";
            std::filesystem::create_directories(editor_state);
            std::ofstream(editor_state / "active-build.json", std::ios::trunc)
                << json{{"format", "arc-active-build"},
                        {"formatVersion", 1},
                        {"configuration", configuration},
                        {"moduleManifest",
                         std::filesystem::relative(module_manifest, context.value().root).generic_string()}}
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
    for (int index = 1; index < argc; ++index)
        args.values.emplace_back(argv[index]);
    const bool json_output = args.has("--json");
    if (args.values.empty())
        return fail(json_output,
                    "expected create, validate, upgrade, settings, engine, toolchains, configure, build, package, "
                    "workflow, or ide",
                    2);
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
        const auto result = arc::project::create_project({.name = *name,
                                                          .destination = *destination,
                                                          .template_id = *template_id,
                                                          .templates_root = *templates,
                                                          .engine_version = *engine});
        if (!result) return fail(json_output, result.error().message);
        print_json(json_output,
                   {{"succeeded", true}, {"destination", std::filesystem::absolute(*destination).generic_string()}});
        return 0;
    }
    if (command == "validate")
    {
        const auto descriptor_path = descriptor_from(args);
        if (descriptor_path.empty()) return fail(json_output, "validate requires --project", 2);
        const auto descriptor = arc::project::load_descriptor(descriptor_path);
        if (!descriptor) return fail(json_output, descriptor.error().message);
        const auto validation =
            arc::project::validate_descriptor(descriptor_path, descriptor.value(),
                                              {.engine_version = args.option("--engine").value_or(""),
                                               .require_exact_engine = args.option("--engine").has_value(),
                                               .require_paths = args.has("--require-paths"),
                                               .allow_read_only = args.has("--read-only")});
        if (!validation) return fail(json_output, validation.error().message);
        print_json(json_output, {{"succeeded", true},
                                 {"writable", validation.value().writable},
                                 {"name", descriptor.value().name},
                                 {"engineVersion", descriptor.value().engine_version}});
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
            tools.push_back({{"id", tool.id},
                             {"available", tool.available},
                             {"executable", tool.executable.generic_string()},
                             {"version", tool.version}});
        print_json(json_output, {{"succeeded", true}, {"tools", tools}});
        if (!json_output)
            for (const auto& tool : result.value())
                std::cout << tool.id << ": " << (tool.available ? "available" : "missing") << '\n';
        return 0;
    }
    if (command == "settings")
    {
        if (args.values.size() > 1 && args.values[1] == "describe")
        {
            print_json(true, {{"version", 1},
                              {"targets", arc::project::supported_project_targets()},
                              {"textureOutputs", {"bc", "astc", "etc2", "portable"}},
                              {"textureQualities", {"balanced", "quality", "size"}},
                              {"androidMinimumSdk", 26},
                              {"androidCompileSdk", 35}});
            return 0;
        }
        if (args.values.size() <= 1 || args.values[1] != "export-android")
            return fail(json_output, "settings expects describe or export-android", 2);
        const auto file = descriptor_from(args);
        if (file.empty()) return fail(json_output, "settings export-android requires --project", 2);
        const auto descriptor = arc::project::load_descriptor(file);
        if (!descriptor) return fail(json_output, descriptor.error().message);
        const auto validated = arc::project::validate_descriptor(file, descriptor.value());
        if (!validated) return fail(json_output, validated.error().message);
        if (!descriptor.value().android || descriptor.value().android->application_id.empty())
            return fail(json_output, "Configure the Android application ID in Project Settings before packaging");
        const auto& android = *descriptor.value().android;
        print_json(true, {{"format", "arc.android-project-settings"},
                          {"version", 1},
                          {"projectGuid", descriptor.value().guid},
                          {"applicationId", android.application_id},
                          {"displayName", descriptor.value().package.application_name},
                          {"versionName", android.version_name},
                          {"versionCode", android.version_code},
                          {"minimumSdk", android.minimum_sdk},
                          {"targetSdk", android.target_sdk}});
        return 0;
    }
    if ((command == "build" && args.option("--platform") == "android") ||
        (command == "workflow" && args.values.size() > 1 &&
         (args.values[1] == "status" || args.values[1] == "logs" || args.values[1] == "cancel")))
    {
        const auto descriptor_path = descriptor_from(args);
        const auto sdk_root = args.option("--sdk-root");
        if (descriptor_path.empty() || !sdk_root)
            return fail(json_output, "Android workflow requires --project and --sdk-root", 2);
        const auto sdk = std::filesystem::absolute(*sdk_root).lexically_normal();
        const auto script = sdk / "share" / "arc" / "android" / "android_build_workflow.py";
        if (!std::filesystem::is_regular_file(script))
            return fail(json_output, "Installed ARC SDK lacks the A5 Android build executor");
#if defined(_WIN32)
        const auto python = args.option("--python").value_or("python");
#else
        const auto python = args.option("--python").value_or("python3");
#endif
        std::vector<std::string> invoke;
        invoke.push_back(script.string());
        invoke.push_back(command == "build" ? "execute" : args.values[1]);
        invoke.push_back("--project");
        invoke.push_back(descriptor_path.string());
        invoke.push_back("--sdk-root");
        invoke.push_back(sdk.string());
        invoke.push_back("--project-tool");
        const auto executable = std::filesystem::absolute(argv[0]).lexically_normal();
        invoke.push_back(std::filesystem::is_regular_file(executable) ? executable.string() : argv[0]);
        for (const auto* key :
             {"--profile", "--target", "--cooker", "--output", "--runtime-config", "--gradle", "--run-id"})
        {
            if (const auto value = args.option(key))
            {
                invoke.push_back(key);
                invoke.push_back(*value);
            }
        }
        if (json_output) invoke.push_back("--json");
        return run_process(python, invoke, std::filesystem::current_path());
    }
    if (command == "configure") return configure_or_build(args, false, json_output);
    if (command == "build") return configure_or_build(args, true, json_output);
    if (command == "package" && args.values.size() > 1 && args.values[1] == "android-apk")
    {
        const auto descriptor_path = descriptor_from(args);
        const auto sdk_root = args.option("--sdk-root");
        const auto assets_root = args.option("--assets");
        const auto output_root = args.option("--output");
        if (descriptor_path.empty() || !sdk_root || !assets_root || !output_root)
            return fail(json_output, "android-apk requires --project, --sdk-root, --assets and --output", 2);
        const auto sdk = std::filesystem::absolute(*sdk_root).lexically_normal();
        const auto script = sdk / "share" / "arc" / "android" / "android_apk_finalizer.py";
        if (!std::filesystem::is_regular_file(script))
            return fail(json_output, "Installed ARC SDK lacks the A4 Android APK finalizer");
#if defined(_WIN32)
        const auto python = args.option("--python").value_or("python");
#else
        const auto python = args.option("--python").value_or("python3");
#endif
        std::vector<std::string> invoke;
        invoke.push_back(script.string());
        invoke.push_back("--project-root");
        invoke.push_back(descriptor_path.parent_path().string());
        invoke.push_back("--sdk-root");
        invoke.push_back(sdk.string());
        invoke.push_back("--assets");
        invoke.push_back(std::filesystem::absolute(*assets_root).lexically_normal().string());
        invoke.push_back("--output");
        invoke.push_back(std::filesystem::absolute(*output_root).lexically_normal().string());
        if (const auto settings = args.option("--settings"))
        {
            invoke.push_back("--settings");
            invoke.push_back(std::filesystem::absolute(*settings).lexically_normal().string());
        }
        else
        {
            invoke.push_back("--project-tool");
            invoke.push_back(argv[0]);
        }
        if (const auto gradle = args.option("--gradle"))
        {
            invoke.push_back("--gradle");
            invoke.push_back(*gradle);
        }
        if (const auto workspace = args.option("--workspace-root"))
        {
            invoke.push_back("--workspace-root");
            invoke.push_back(std::filesystem::absolute(*workspace).lexically_normal().string());
        }
        if (json_output) invoke.push_back("--json");
        return run_process(python, invoke, std::filesystem::current_path());
    }
    if (command == "package" && args.values.size() > 1 && args.values[1] == "android-content")
        return handle_android_content_command(args.values, json_output);
    if (command == "package") return handle_package_command(args.values, json_output);
    if (command == "workflow") return handle_workflow_command(args.values);
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
                << json{{"folders", json::array({{{"path", "."}}})},
                        {"settings", {{"cmake.useCMakePresets", "always"}}}}
                       .dump(2)
                << '\n';
        }
        else if (args.values[1] == "generate" && *ide == "visual-studio")
        {
            const auto directory = root / "Build" / "VisualStudio";
            const int code = run_process(
                args.option("--cmake").value_or("cmake"),
                {"-S", root.string(), "-B", directory.string(), "-G", "Visual Studio 17 2022", "-A", "x64"}, root);
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
                    if (entry.path().extension() == ".sln")
                    {
                        launch_target = entry.path();
                        break;
                    }
            }
            const int code = run_process(executable, {launch_target.string()}, root);
            if (code != 0) return fail(json_output, "IDE launch failed");
        }
        else
            return fail(json_output, "unsupported IDE operation", 2);
        print_json(json_output, {{"succeeded", true}});
        return 0;
    }
    return fail(json_output, "unknown command", 2);
}

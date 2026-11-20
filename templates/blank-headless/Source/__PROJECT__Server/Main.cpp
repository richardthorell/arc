#include <arc/project/headless_server.h>

#include <charconv>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#if defined(_WIN32)
#include <windows.h>
#endif

#ifndef ARC_SERVER_BUILD_CONFIGURATION
#error "CMake must define ARC_SERVER_BUILD_CONFIGURATION for packaged server compatibility checks"
#endif

namespace {{PROJECT_TOKEN}}
{
void register_server(arc::project::packaged_game_registrations& registrations);
}

namespace
{

volatile std::sig_atomic_t stop_signal{};

void on_stop_signal(int)
{
    stop_signal = 1;
}

#if defined(_WIN32)
// Windows console control events are not reliably translated into C runtime
// SIGINT/SIGTERM. Handle Ctrl+C and Ctrl+Break explicitly; never perform
// engine teardown or I/O from the console-handler thread.
BOOL WINAPI on_console_control(DWORD event)
{
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT)
    {
        stop_signal = 1;
        return TRUE;
    }
    return FALSE;
}
#endif

bool parse_ticks(std::string_view text, std::uint64_t& value)
{
    const char* const begin = text.data();
    const auto [end, error] = std::from_chars(begin, begin + text.size(), value);
    return error == std::errc{} && end == begin + text.size();
}

std::filesystem::path current_executable()
{
#if defined(_WIN32)
    std::wstring buffer(512, L'\0');
    for (;;)
    {
        const auto length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0) return {};
        if (length < buffer.size())
        {
            buffer.resize(length);
            return std::filesystem::path(buffer);
        }
        buffer.resize(buffer.size() * 2);
        if (buffer.size() > 32768) return {};
    }
#elif defined(__linux__)
    std::error_code error;
    const auto path = std::filesystem::read_symlink("/proc/self/exe", error);
    return error ? std::filesystem::path{} : path;
#else
    return {};
#endif
}

arc::framework::platform_family native_family()
{
#if defined(_WIN32)
    return arc::framework::platform_family::windows;
#elif defined(__linux__)
    return arc::framework::platform_family::linux_os;
#else
    return arc::framework::platform_family::unknown;
#endif
}

std::string native_architecture()
{
#if defined(_M_X64) || defined(__x86_64__)
    return "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#else
    return {};
#endif
}

} // namespace

int main(int argc, char** argv)
{
    namespace fs = std::filesystem;
    namespace project = arc::project;

    std::uint64_t maximum_ticks{};
    bool sleep_to_clock = true;
    bool json_report = false;
    fs::path distribution_root;
    fs::path data_root;
    fs::path stop_file;

    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--version")
        {
            // Compiled-in identity; does not depend on an installed SDK or
            // on the current working directory.
            std::cout << "ARC server engine {{ENGINE_VERSION}} ("
                      << ARC_SERVER_BUILD_CONFIGURATION << ")\n";
            return 0;
        }
        else if (argument == "--report-json")
            json_report = true;
        else if (argument == "--no-sleep")
            sleep_to_clock = false;
        else if (argument == "--ticks" && index + 1 < argc)
        {
            if (!parse_ticks(argv[++index], maximum_ticks))
            {
                std::cerr << "--ticks requires an unsigned integer\n";
                return 2;
            }
        }
        else if (argument == "--distribution-root" && index + 1 < argc)
            distribution_root = fs::path(argv[++index]);
        else if (argument == "--data-root" && index + 1 < argc)
            data_root = fs::path(argv[++index]);
        else if (argument == "--stop-file" && index + 1 < argc)
            stop_file = fs::path(argv[++index]);
        else
        {
            std::cerr << "Unknown or incomplete server option: " << argument << '\n';
            return 2;
        }
    }

    if ((!distribution_root.empty() && !distribution_root.is_absolute()) ||
        (!data_root.empty() && !data_root.is_absolute()) ||
        (!stop_file.empty() && (!stop_file.is_absolute() || stop_file != stop_file.lexically_normal())))
    {
        std::cerr << "Distribution, data and stop-file paths must be absolute and normalized\n";
        return 2;
    }

    const fs::path executable = current_executable();
    if (distribution_root.empty() && (!executable.is_absolute() || executable.empty()))
    {
        std::cerr << "Cannot locate native server executable; specify --distribution-root\n";
        return 2;
    }

    if (data_root.empty())
    {
#if defined(_WIN32)
        // Read the native wide environment to preserve Unicode paths and avoid
        // MSVC's deprecated narrow getenv interface under /WX.
        const DWORD required = GetEnvironmentVariableW(L"ARC_SERVER_DATA_ROOT", nullptr, 0);
        if (required > 1)
        {
            std::wstring base(required, L'\0');
            const DWORD length = GetEnvironmentVariableW(L"ARC_SERVER_DATA_ROOT", base.data(), required);
            if (length > 0 && length < required)
            {
                base.resize(length);
                data_root = fs::path(base);
            }
        }
#else
        const char* base = std::getenv("ARC_SERVER_DATA_ROOT");
        if (base && *base) data_root = fs::path(base);
#endif
    }
    if (data_root.empty())
    {
        std::error_code error;
        data_root = fs::temp_directory_path(error) / "{{PROJECT_TOKEN}}-server";
        if (error)
        {
            std::cerr << "Cannot resolve a writable data directory\n";
            return 2;
        }
    }
    if (!data_root.is_absolute())
    {
        std::cerr << "ARC_SERVER_DATA_ROOT must be an absolute path\n";
        return 2;
    }

    std::error_code storage_error;
    fs::create_directories(data_root, storage_error);
    if (storage_error)
    {
        std::cerr << "Cannot initialize server writable data directory: " << storage_error.message() << '\n';
        return 2;
    }

    if (native_family() == arc::framework::platform_family::unknown || native_architecture().empty())
    {
        std::cerr << "Unsupported packaged server platform or architecture\n";
        return 2;
    }

#if defined(_WIN32)
    if (!SetConsoleCtrlHandler(on_console_control, TRUE))
    {
        std::cerr << "Cannot register Windows console stop handler\n";
        return 2;
    }
    std::signal(SIGTERM, on_stop_signal);
#else
    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);
#endif

    arc::jobs::job_system jobs{{.worker_count = 1, .run_inline = false, .io_worker_count = 2,
                                .enable_render_thread = false}};
    arc::io::async_file_service native_files{jobs};
    arc::io::virtual_file_system files{jobs};

    project::runtime_boot_request boot;
    boot.role = project::runtime_launch_role::server;
    boot.host.distribution.kind = distribution_root.empty()
                                      ? project::runtime_distribution_location_kind::executable
                                      : project::runtime_distribution_location_kind::directory;
    boot.host.distribution.native_path = distribution_root.empty() ? executable : distribution_root;
    boot.host.writable_storage_root = std::move(data_root);
    boot.host.capabilities = {.family = native_family(),
                              .form_factor = arc::framework::device_form_factor::server,
                              .logical_processor_count = std::thread::hardware_concurrency(),
                              .persistent_local_storage = true};
    boot.expected = {.engine_version = "{{ENGINE_VERSION}}",
                     .sdk_version = "{{ENGINE_VERSION}}",
                     .architecture = native_architecture(),
                     .renderer = "none",
                     .configuration = ARC_SERVER_BUILD_CONFIGURATION,
                     .project_guid = "{{PROJECT_GUID}}"};

    project::packaged_runtime_request request{.boot = std::move(boot),
                                               .files = files,
                                               .native_files = &native_files};
    {{PROJECT_TOKEN}}::register_server(request.registrations);

    bool external_stop_requested = false;
    auto result = project::run_packaged_headless_server({.runtime = std::move(request),
                                                         .maximum_ticks = maximum_ticks,
                                                         .sleep_to_clock = sleep_to_clock,
                                                         .stop_requested = [&]
                                                         {
                                                             if (stop_signal != 0) return true;
                                                             if (stop_file.empty()) return false;
                                                             std::error_code error;
                                                             external_stop_requested = fs::exists(stop_file, error) && !error;
                                                             return external_stop_requested;
                                                         }});
    if (!result)
    {
        std::cerr << result.error().field << ": " << result.error().message << '\n';
        return 1;
    }
    const char* reason = stop_signal != 0 ? "signal"
                          : external_stop_requested ? "external-stop"
                          : maximum_ticks != 0 && result.value() >= maximum_ticks ? "tick-limit"
                                                                                  : "runtime-stop";
    if (json_report)
    {
        // Machine-readable stdout for external supervisors and CI. Values are
        // fixed engine/configuration literals and numeric simulation counters.
        std::cout << "{\"event\":\"server.stopped\",\"role\":\"server\","
                  << "\"engineVersion\":\"{{ENGINE_VERSION}}\","
                  << "\"configuration\":\"" << ARC_SERVER_BUILD_CONFIGURATION << "\","
                  << "\"reason\":\"" << reason << "\","
                  << "\"requestedTicks\":" << maximum_ticks << ","
                  << "\"completedTicks\":" << result.value() << "}\n";
    }
    else
        std::cout << "Server stopped after " << result.value() << " simulation ticks\n";
    return 0;
}

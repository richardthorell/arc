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
    fs::path distribution_root;
    fs::path data_root;

    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        if (argument == "--no-sleep")
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
        else
        {
            std::cerr << "Unknown or incomplete server option: " << argument << '\n';
            return 2;
        }
    }

    if ((!distribution_root.empty() && !distribution_root.is_absolute()) ||
        (!data_root.empty() && !data_root.is_absolute()))
    {
        std::cerr << "Distribution and data roots must be absolute paths\n";
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
        const char* base = std::getenv("ARC_SERVER_DATA_ROOT");
        if (base && *base) data_root = fs::path(base);
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

    std::signal(SIGINT, on_stop_signal);
    std::signal(SIGTERM, on_stop_signal);

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

    auto result = project::run_packaged_headless_server({.runtime = std::move(request),
                                                         .maximum_ticks = maximum_ticks,
                                                         .sleep_to_clock = sleep_to_clock,
                                                         .stop_requested = [] { return stop_signal != 0; }});
    if (!result)
    {
        std::cerr << result.error().field << ": " << result.error().message << '\n';
        return 1;
    }
    std::cout << "Server stopped after " << result.value() << " simulation ticks\n";
    return 0;
}

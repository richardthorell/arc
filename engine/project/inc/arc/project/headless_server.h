#pragma once

/** @file
 * @brief Drive a validated D0-packaged server through D1.5 without a window or renderer.
 */

#include <arc/project/runtime_session.h>

#include <cstdint>
#include <functional>

namespace arc::project
{

struct packaged_headless_server_options
{
    packaged_runtime_request runtime;
    std::uint64_t maximum_ticks{}; ///< Zero runs until a stop request or a runtime failure.
    bool sleep_to_clock{true};
    /** Borrowed callback polled on the host thread; e.g. a SIGINT/SIGTERM flag. */
    std::function<bool()> stop_requested;
};

using packaged_headless_server_result = core::result<std::uint64_t, project_error>;

/**
 * Run a staged dedicated server. All distribution validation, cooked mounts,
 * registration, simulation and teardown go through the same D1.5 session used
 * by packaged clients. The host supplies the physical executable/distribution
 * location, compiled-in target expectations, and any project registrations.
 *
 * The server loop uses fixed simulation deltas and optional wall-clock pacing,
 * never creates a native window, and handles a stop request at a safe outer
 * frame boundary. Any startup, system, or shutdown error returns a structured
 * project_error; owned mounts are released even on failure.
 *
 * The caller owns the jobs/VFS/native file services for the full call duration.
 */
[[nodiscard]] packaged_headless_server_result run_packaged_headless_server(packaged_headless_server_options options);

} // namespace arc::project

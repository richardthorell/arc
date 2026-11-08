#pragma once

/** @file
 * @brief Shared, host-neutral packaged game/server startup and shutdown session (D1.5).
 */

#include <arc/project/runtime_application.h>

#include <memory>

namespace arc::project
{

/** The host owns the VFS, async file service, native process and surface. */
struct packaged_runtime_request
{
    runtime_boot_request boot;
    io::virtual_file_system& files;
    io::async_file_service* native_files{}; ///< Native staged packages; unused for mounted-VFS packages.
    io::virtual_path native_mount;
    io::virtual_path artifact_mount;
    packaged_game_registrations registrations;
    framework::application_config config;
};

/** Lifecycle state of a successfully launched session; failures never produce sessions. */
enum class packaged_runtime_state : std::uint8_t
{
    running,
    suspended,      ///< Host pause or unavailable client surface; no simulation steps occur.
    stop_requested, ///< Host requested termination, but shutdown/unmount has not yet run.
    stopped         ///< All runtime/module and cooked catalog resources have been released.
};

class packaged_runtime_session;
using packaged_runtime_result = core::result<std::unique_ptr<packaged_runtime_session>, project_error>;
using packaged_frame_result = core::result<framework::frame_time, project_error>;

/**
 * Owns the D1.2 context, D1.3 catalog, D1.4 application and framework runtime.
 *
 * Destruction and shutdown release resources in reverse startup order:
 * stop worlds/modules/services -> destroy runtime -> destroy app and scene
 * buffers -> unmount cooked catalog -> unmount owned distribution mount.
 * The host's VFS/file services and any pre-existing package mounts survive.
 *
 * Call lifecycle methods serially on the host's runtime thread. The host must
 * keep its borrowed file services alive until this session is destroyed.
 */
class packaged_runtime_session final
{
public:
    ~packaged_runtime_session() noexcept;

    packaged_runtime_session(const packaged_runtime_session&) = delete;
    packaged_runtime_session& operator=(const packaged_runtime_session&) = delete;
    packaged_runtime_session(packaged_runtime_session&&) = delete;
    packaged_runtime_session& operator=(packaged_runtime_session&&) = delete;

    [[nodiscard]] packaged_runtime_state state() const noexcept;
    [[nodiscard]] const runtime_boot_context& boot() const noexcept;

    /** Borrowed runtime, valid only while the session is not stopped. */
    [[nodiscard]] framework::runtime* runtime() noexcept;
    [[nodiscard]] const framework::runtime* runtime() const noexcept;

    /** Borrowed mounted catalog, valid only while the session is not stopped. */
    [[nodiscard]] runtime_catalog_session* catalog() noexcept;

    /** Advance an explicit host-measured delta. No work is run while suspended. */
    [[nodiscard]] packaged_frame_result advance(double delta_seconds);
    /** Forward ordinary platform-neutral events, including close_requested. */
    [[nodiscard]] project_status dispatch(const framework::event& event);

    /** Process foreground/background pause, independent of headless debug time controls. */
    [[nodiscard]] project_status pause();
    [[nodiscard]] project_status resume();

    /** Client surface lifecycle. Does not allocate or destroy OS/GPU surfaces. */
    [[nodiscard]] project_status surface_lost();
    /** Restore a client surface; optionally dispatch its nonzero pixel extent. */
    [[nodiscard]] project_status surface_ready(std::uint32_t width, std::uint32_t height);

    /** Stop the host loop without destroying resources inside an event callback. */
    void request_stop() noexcept;
    /** Idempotent. Returns the first shutdown exception as a structured error. */
    [[nodiscard]] project_status shutdown() noexcept;

private:
    friend packaged_runtime_result start_packaged_runtime(packaged_runtime_request request);
    explicit packaged_runtime_session(runtime_boot_context boot);

    [[nodiscard]] project_status check_active() const;
    void update_pause_state();

    runtime_boot_context boot_;
    std::unique_ptr<runtime_catalog_session> catalog_;
    std::unique_ptr<packaged_runtime_application> application_;
    std::unique_ptr<framework::runtime> runtime_;
    framework::frame_time last_frame_{};
    bool host_paused_{};
    bool surface_unavailable_{};
    bool stopped_{};
};

 /**
  * One authoritative startup path: discover/validate -> mount -> preflight
  * scenes/registrations -> start framework. Failure destroys partial state and
  * never returns a half-initialized session. Hosts invoke request_stop/shutdown
  * during OS termination, rather than directly manipulating the catalog.
  */
[[nodiscard]] packaged_runtime_result start_packaged_runtime(packaged_runtime_request request);

} // namespace arc::project

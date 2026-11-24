#include <arc/project/runtime_session.h>
#include <arc/assets/remote_content_discovery.h>

#include <exception>
#include <string>
#include <string_view>
#include <utility>

namespace arc::project
{
namespace
{

project_error failure(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}

project_error exception_failure(std::string_view stage, const std::exception& error)
{
    return failure(project_error_code::process_failed, std::string(stage),
                   "Packaged runtime " + std::string(stage) + " failed: " + error.what());
}

project_error unknown_failure(std::string_view stage)
{
    return failure(project_error_code::process_failed, std::string(stage),
                   "Packaged runtime " + std::string(stage) + " failed with an unknown exception");
}

} // namespace

packaged_runtime_session::packaged_runtime_session(runtime_boot_context boot) : boot_(std::move(boot)) {}

packaged_runtime_session::~packaged_runtime_session() noexcept
{
    static_cast<void>(shutdown());
}

packaged_runtime_state packaged_runtime_session::state() const noexcept
{
    if (stopped_ || !runtime_) return packaged_runtime_state::stopped;
    if (!runtime_->running()) return packaged_runtime_state::stop_requested;
    if (host_paused_ || surface_unavailable_) return packaged_runtime_state::suspended;
    return packaged_runtime_state::running;
}

const runtime_boot_context& packaged_runtime_session::boot() const noexcept
{
    return boot_;
}

framework::runtime* packaged_runtime_session::runtime() noexcept
{
    return runtime_.get();
}

const framework::runtime* packaged_runtime_session::runtime() const noexcept
{
    return runtime_.get();
}

runtime_catalog_session* packaged_runtime_session::catalog() noexcept
{
    return catalog_.get();
}

std::uint64_t packaged_runtime_session::active_remote_generation() const noexcept
{
    return active_remote_generation_;
}

project_status packaged_runtime_session::check_active() const
{
    if (stopped_ || !runtime_)
        return project_status::failure(
            failure(project_error_code::invalid_descriptor, "runtime", "Packaged runtime session is stopped"));
    if (!runtime_->running())
        return project_status::failure(
            failure(project_error_code::invalid_descriptor, "runtime", "Packaged runtime termination was requested"));
    return project_status::success();
}

void packaged_runtime_session::update_pause_state()
{
    if (host_paused_ || surface_unavailable_)
        runtime_->pause();
    else
        runtime_->resume();
}

packaged_frame_result packaged_runtime_session::advance(double delta_seconds)
{
    auto active = check_active();
    if (!active) return packaged_frame_result::failure(std::move(active).error());
    if (host_paused_ || surface_unavailable_)
    {
        auto skipped = last_frame_;
        skipped.delta_seconds = 0.0;
        skipped.completed_ticks = 0;
        return packaged_frame_result::success(skipped);
    }

    try
    {
        last_frame_ = runtime_->advance(delta_seconds);
        for (const auto id : runtime_->worlds().ordered_worlds())
        {
            const auto* world = runtime_->worlds().find(id);
            if (!world || world->state() != framework::runtime_world_state::faulted) continue;
            auto error = failure(project_error_code::process_failed, "worlds",
                                 "Packaged runtime world '" + std::string(world->name()) +
                                     "' faulted: " + world->fault_message());
            static_cast<void>(shutdown());
            return packaged_frame_result::failure(std::move(error));
        }
        return packaged_frame_result::success(last_frame_);
    }
    catch (const packaged_runtime_startup_error& error)
    {
        auto detail = error.detail();
        static_cast<void>(shutdown());
        return packaged_frame_result::failure(std::move(detail));
    }
    catch (const std::exception& error)
    {
        auto detail = exception_failure("advance", error);
        static_cast<void>(shutdown());
        return packaged_frame_result::failure(std::move(detail));
    }
    catch (...)
    {
        auto detail = unknown_failure("advance");
        static_cast<void>(shutdown());
        return packaged_frame_result::failure(std::move(detail));
    }
}

project_status packaged_runtime_session::dispatch(const framework::event& event)
{
    auto active = check_active();
    if (!active) return active;
    try
    {
        runtime_->dispatch(event);
        return project_status::success();
    }
    catch (const packaged_runtime_startup_error& error)
    {
        auto detail = error.detail();
        static_cast<void>(shutdown());
        return project_status::failure(std::move(detail));
    }
    catch (const std::exception& error)
    {
        auto detail = exception_failure("dispatch", error);
        static_cast<void>(shutdown());
        return project_status::failure(std::move(detail));
    }
    catch (...)
    {
        auto detail = unknown_failure("dispatch");
        static_cast<void>(shutdown());
        return project_status::failure(std::move(detail));
    }
}

project_status packaged_runtime_session::pause()
{
    auto active = check_active();
    if (!active) return active;
    if (!host_paused_)
    {
        host_paused_ = true;
        update_pause_state();
    }
    return project_status::success();
}

project_status packaged_runtime_session::resume()
{
    auto active = check_active();
    if (!active) return active;
    if (host_paused_)
    {
        host_paused_ = false;
        update_pause_state();
    }
    return project_status::success();
}

project_status packaged_runtime_session::surface_lost()
{
    auto active = check_active();
    if (!active) return active;
    if (boot_.launch.role == runtime_launch_role::server)
        return project_status::failure(failure(project_error_code::unsupported_platform, "surface",
                                               "Dedicated headless server has no native rendering surface"));
    if (surface_unavailable_) return project_status::success();
    surface_unavailable_ = true;
    update_pause_state();
    return dispatch({.type = framework::event_type::focus_lost});
}

project_status packaged_runtime_session::surface_ready(std::uint32_t width, std::uint32_t height)
{
    auto active = check_active();
    if (!active) return active;
    if (boot_.launch.role == runtime_launch_role::server)
        return project_status::failure(failure(project_error_code::unsupported_platform, "surface",
                                               "Dedicated headless server has no native rendering surface"));
    if ((width == 0) != (height == 0))
        return project_status::failure(
            failure(project_error_code::invalid_descriptor, "surface", "Both surface dimensions must be nonzero"));
    if (width != 0)
    {
        auto resized = dispatch({.type = framework::event_type::resized, .width = width, .height = height});
        if (!resized) return resized;
    }
    if (surface_unavailable_)
    {
        auto focused = dispatch({.type = framework::event_type::focus_gained});
        if (!focused) return focused;
        surface_unavailable_ = false;
        update_pause_state();
    }
    return project_status::success();
}

void packaged_runtime_session::request_stop() noexcept
{
    if (runtime_) runtime_->request_stop();
}

project_status packaged_runtime_session::shutdown() noexcept
{
    if (stopped_) return project_status::success();
    stopped_ = true;
    project_status outcome = project_status::success();

    if (runtime_)
    {
        try
        {
            runtime_->shutdown();
        }
        catch (const std::exception& error)
        {
            outcome = project_status::failure(exception_failure("shutdown", error));
        }
        catch (...)
        {
            outcome = project_status::failure(unknown_failure("shutdown"));
        }
    }
    // Runtime must die before the application and its registered callbacks;
    // the catalog must outlive both so scene/module shutdown can access assets.
    runtime_.reset();
    application_.reset();
    catalog_.reset();
    remote_cache_.reset();
    active_remote_generation_ = 0;
    return outcome;
}

packaged_runtime_result start_packaged_runtime(packaged_runtime_request request)
{
    auto boot = discover_runtime_boot(std::move(request.boot));
    if (!boot) return packaged_runtime_result::failure(std::move(boot).error());

    // Remote-only startup is permitted ONLY with a trusted host-owned pairing.
    // The ordinary packaged and hybrid paths remain unchanged; no remote
    // endpoint may silently replace a Shipping startup world.
    const bool remote = boot.value().content.mode == runtime_content_mode::remote;
    if (remote != request.content_pairing.has_value() ||
        (remote && (!request.content_jobs || !request.native_files ||
                    boot.value().launch.manifest.target.configuration == "Shipping")))
        return packaged_runtime_result::failure(
            failure(project_error_code::invalid_descriptor, "content.mode",
                    "Remote-only startup requires authenticated Development content pairing and IO workers"));

    auto session = std::unique_ptr<packaged_runtime_session>(new packaged_runtime_session(std::move(boot).value()));
    try
    {
        auto catalog = bootstrap_runtime_catalog({.boot = session->boot_,
                                                  .files = request.files,
                                                  .native_files = request.native_files,
                                                  .native_mount = std::move(request.native_mount),
                                                  .artifact_mount = std::move(request.artifact_mount)});
        if (!catalog) return packaged_runtime_result::failure(std::move(catalog).error());
        session->catalog_ = std::move(catalog).value();

        std::uint64_t verified_generation{};
        if (remote)
        {
            // This happens BEFORE prepare_packaged_application reads the
            // startup scene and BEFORE framework::runtime::start creates ECS
            // worlds. Mounting after startup would display the stale packaged
            // world and would not satisfy L1.3 unsaved-scene fidelity.
            const auto& pairing = *request.content_pairing;
            const auto& manifest = session->boot_.launch.manifest;
            if (manifest.startup_scenes.size() != 1 ||
                manifest.startup_scenes.front() != pairing.scene_asset_guid ||
                pairing.base_build_id.empty() || pairing.descriptor_sha256.size() != 64 ||
                pairing.generation == 0)
                return packaged_runtime_result::failure(failure(
                    project_error_code::invalid_scene, "startupScenes",
                    "Development remote startup requires one matching registered cooked scene and release identity"));

            http::content_http_limits limits;
            limits.allow_loopback_http = true;
            limits.maximum_object_bytes = 2ull * 1024ull * 1024ull * 1024ull;
            limits.development_bearer_token = pairing.token;
            assets::remote_content_expectations expected;
            expected.project_guid = manifest.project_guid;
            expected.engine_version = manifest.engine_version;
            expected.base_build_id = pairing.base_build_id;
            expected.target = {.platform = manifest.target.platform,
                               .architecture = manifest.target.architecture,
                               .renderer = manifest.target.renderer,
                               .cook_profile = manifest.target.cook_profile,
                               .texture_family = pairing.texture_family};
            const auto origin = session->boot_.content.server_origin;
            auto discovered = assets::discover_remote_release(*request.content_jobs, origin, expected, limits);
            if (!discovered.handle().wait_result().succeeded())
                return packaged_runtime_result::failure(failure(
                    project_error_code::io_failed, "content.release", "Remote release discovery did not finish"));
            auto descriptor = discovered.get();
            if (!descriptor.succeeded() || descriptor.descriptor_sha256 != pairing.descriptor_sha256 ||
                descriptor.release->generation != pairing.generation)
                return packaged_runtime_result::failure(failure(
                    project_error_code::invalid_descriptor, "content.release",
                    "Remote cooked scene descriptor is unavailable or differs from the trusted publication"));

            // The catalog's provider borrows this CAS. Keep it alive until
            // shutdown releases all runtime worlds, application and mounts.
            session->remote_cache_ =
                std::make_unique<assets::derived_data_cache>(assets::derived_data_cache_config{
                    .root = session->boot_.content.cache_root});
            auto transfer = assets::acquire_remote_generation(
                *request.content_jobs, *request.native_files, *session->remote_cache_, *descriptor.release,
                {.origin = origin, .http = limits});
            if (!transfer.handle().wait_result().succeeded() || !transfer.get().succeeded())
                return packaged_runtime_result::failure(failure(
                    project_error_code::io_failed, "content.objects",
                    "The complete SHA-verified remote cooked generation could not be acquired"));
            const auto mounted = session->catalog_->catalog().mount_remote_generation(
                *descriptor.release, *session->remote_cache_, *request.native_files, 100);
            if (!mounted)
                return packaged_runtime_result::failure(failure(
                    project_error_code::invalid_scene, "content.catalog",
                    "Verified remote cooked generation could not be mounted"));
            // A valid release that does not supply the actual startup scene
            // must not silently fall back to the saved packaged scene.
            const auto scene_guid = assets::parse_asset_guid(pairing.scene_asset_guid);
            std::optional<assets::asset_database_record> active_scene;
            if (scene_guid)
                active_scene = session->catalog_->catalog().database().query(
                    *scene_guid, assets::asset_types::scene);
            if (!active_scene || active_scene->active_provider.find(":remote:") == std::string::npos)
                return packaged_runtime_result::failure(failure(
                    project_error_code::invalid_scene, "content.startupScene",
                    "Verified remote generation does not own the active cooked startup scene"));
            verified_generation = descriptor.release->generation;
        }

        auto application = prepare_packaged_application({.boot = session->boot_,
                                                         .catalog = *session->catalog_,
                                                         .registrations = std::move(request.registrations),
                                                         .config = std::move(request.config)});
        if (!application) return packaged_runtime_result::failure(std::move(application).error());
        session->application_ = std::move(application).value();

        session->runtime_ = std::make_unique<framework::runtime>(*session->application_);
        session->runtime_->set_platform_capabilities(session->boot_.launch.host.capabilities);
        session->runtime_->start();
        // Only a successful runtime start after the exact remote provider was
        // mounted qualifies as L1.3 native scene activation.
        session->active_remote_generation_ = verified_generation;
        return packaged_runtime_result::success(std::move(session));
    }
    catch (const packaged_runtime_startup_error& error)
    {
        return packaged_runtime_result::failure(error.detail());
    }
    catch (const std::exception& error)
    {
        return packaged_runtime_result::failure(exception_failure("startup", error));
    }
    catch (...)
    {
        return packaged_runtime_result::failure(unknown_failure("startup"));
    }
}

} // namespace arc::project

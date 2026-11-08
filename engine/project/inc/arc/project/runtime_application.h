#pragma once

/** @file
 * @brief Packaged game/server registrations and explicit startup-world assembly.
 */

#include <arc/framework/runtime.h>
#include <arc/project/runtime_catalog.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace arc::project
{

/** A verified, bounded cooked scene payload, held until world registration. */
struct packaged_scene_payload
{
    assets::asset_guid guid{};
    std::string artifact_name;
    io::file_buffer bytes;
};

/**
 * Shared runtime hooks supplied by the game/module integration layer.
 * These are in-process callbacks, not an editor workspace or a native loader.
 * D1.6/D1.7 hosts supply the module implementation and game-specific scene
 * materializer; there is no fallback to an empty or placeholder client world.
 */
struct packaged_game_registrations
{
    std::function<void(framework::runtime_service_registry&)> services;
    std::function<void(framework::module_registry&)> modules;
    /** Required when startup scenes are present; must populate the provided world or report failure. */
    std::function<project_status(framework::runtime_world&, const packaged_scene_payload&)> materialize_scene;
    /** Optional server world assembly for dedicated servers with no startup scene. */
    std::function<project_status(framework::runtime_world&)> initialize_headless_world;
};

struct packaged_application_request
{
    const runtime_boot_context& boot;
    runtime_catalog_session& catalog;
    packaged_game_registrations registrations;
    framework::application_config config;
};

class packaged_runtime_application final : public framework::application
{
public:
    ~packaged_runtime_application() override;

    packaged_runtime_application(const packaged_runtime_application&) = delete;
    packaged_runtime_application& operator=(const packaged_runtime_application&) = delete;

    [[nodiscard]] framework::application_config configure() const override;
    void register_services(framework::runtime_service_registry& services) override;
    void register_modules(framework::module_registry& modules) override;
    void register_worlds(framework::runtime_world_manager& worlds) override;

    [[nodiscard]] const std::vector<packaged_scene_payload>& scenes() const noexcept;
    [[nodiscard]] runtime_launch_role role() const noexcept;

private:
    friend core::result<std::unique_ptr<packaged_runtime_application>, project_error>
    prepare_packaged_application(packaged_application_request request);

    packaged_runtime_application(runtime_launch_role role, framework::application_config config,
                                 packaged_game_registrations registrations,
                                 std::vector<packaged_scene_payload> scenes);

    runtime_launch_role role_{runtime_launch_role::client};
    framework::application_config config_;
    packaged_game_registrations registrations_;
    std::vector<packaged_scene_payload> scenes_;
};

using packaged_application_result = core::result<std::unique_ptr<packaged_runtime_application>, project_error>;

/**
 * Before starting framework::runtime, verify every selected startup artifact
 * is readable and SHA-256 matches the cooked database metadata. This prevents
 * partial world creation on missing/corrupt startup content.
 *
 * Creates an application whose register_worlds supplies an explicit world for
 * every ordered startup scene, or one explicit headless server world. A scene
 * materializer failure prevents the framework's implicit default world.
 * Caller owns the app and keeps the catalog session alive until shutdown.
 */
[[nodiscard]] packaged_application_result prepare_packaged_application(packaged_application_request request);

} // namespace arc::project

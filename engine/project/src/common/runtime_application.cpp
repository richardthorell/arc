#include <arc/project/runtime_application.h>

#include <arc/assets/assets.h>
#include <arc/assets/package_artifact_reader.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace arc::project
{
namespace
{

constexpr std::uint64_t maximum_scene_bytes = 128u * 1024u * 1024u;
constexpr std::uint64_t maximum_total_startup_bytes = 256u * 1024u * 1024u;

project_error failure(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}

using scene_result = core::result<packaged_scene_payload, project_error>;

scene_result read_scene(runtime_catalog_session& session, assets::asset_guid guid)
{
    const auto record = session.catalog().database().query(guid, assets::asset_types::scene);
    if (!record)
        return scene_result::failure(
            failure(project_error_code::invalid_scene, "startupScenes",
                    "Startup scene is absent from the cooked asset database: " + assets::to_string(guid)));

    for (const auto& representation : record->representations)
    {
        if (representation.provider_id != record->active_provider) continue;

        const assets::cooked_artifact_address address{
            .asset = guid, .schema = assets::artifact_schemas::scene, .name = representation.name};
        auto resolved = session.catalog().resolve(address);
        if (!resolved) continue; // Another representation of this GUID may use a non-scene schema.

        if (resolved.value().size() != representation.size || representation.size == 0 ||
            representation.size > maximum_scene_bytes)
            return scene_result::failure(
                failure(project_error_code::invalid_scene, "startupScenes",
                        "Cooked startup scene has invalid or oversized artifact range: " + assets::to_string(guid)));

        auto bytes = session.files().read_all(resolved.value()).get();
        if (!bytes)
            return scene_result::failure(failure(project_error_code::io_failed, "startupScenes",
                                                 "Could not read cooked startup scene " + assets::to_string(guid) +
                                                     ": " + bytes.error().message));
        if (bytes.value().size() != representation.size ||
            assets::hash_bytes(bytes.value()) != representation.content_hash)
            return scene_result::failure(
                failure(project_error_code::corrupt_content, "startupScenes",
                        "Cooked startup scene SHA-256 or size does not match its package manifest: " +
                            assets::to_string(guid)));
        return scene_result::success(
            {.guid = guid, .artifact_name = representation.name, .bytes = std::move(bytes).value()});
    }

    return scene_result::failure(
        failure(project_error_code::invalid_scene, "startupScenes",
                "Cooked startup scene has no resolvable scene-schema artifact in its active provider: " +
                    assets::to_string(guid)));
}

} // namespace

packaged_runtime_startup_error::packaged_runtime_startup_error(project_error error)
    : std::runtime_error(error.message), error_(std::move(error))
{
}

const project_error& packaged_runtime_startup_error::detail() const noexcept
{
    return error_;
}

packaged_runtime_application::packaged_runtime_application(runtime_launch_role role,
                                                           framework::application_config config,
                                                           packaged_game_registrations registrations,
                                                           std::vector<packaged_scene_payload> scenes)
    : role_(role), config_(std::move(config)), registrations_(std::move(registrations)), scenes_(std::move(scenes))
{
}

packaged_runtime_application::~packaged_runtime_application() = default;

framework::application_config packaged_runtime_application::configure() const
{
    return config_;
}

void packaged_runtime_application::register_services(framework::runtime_service_registry& services)
{
    if (registrations_.services) registrations_.services(services);
}

void packaged_runtime_application::register_modules(framework::module_registry& modules)
{
    if (registrations_.modules) registrations_.modules(modules);
}

void packaged_runtime_application::register_worlds(framework::runtime_world_manager& worlds)
{
    if (worlds.size() != 0)
        throw packaged_runtime_startup_error(failure(project_error_code::invalid_descriptor, "startupScenes",
                                                     "Packaged startup requires an initially empty world manager"));

    std::vector<framework::runtime_world_id> created;
    try
    {
        if (scenes_.empty())
        {
            if (role_ != runtime_launch_role::server)
                throw packaged_runtime_startup_error(
                    failure(project_error_code::invalid_scene, "startupScenes",
                            "Client startup has no cooked scenes; refusing framework's implicit empty world"));

            auto& world = worlds.create({.name = "Dedicated Server",
                                         .role = framework::runtime_world_role::server,
                                         .seed = config_.simulation.process_seed,
                                         .install_placeholder_systems = false,
                                         .presentation_enabled = false});
            created.push_back(world.id());
            if (registrations_.initialize_headless_world)
            {
                auto result = registrations_.initialize_headless_world(world);
                if (!result) throw packaged_runtime_startup_error(std::move(result).error());
            }
        }
        else
        {
            for (std::size_t index = 0; index < scenes_.size(); ++index)
            {
                const auto& scene = scenes_[index];
                auto& world =
                    worlds.create({.name = "Scene " + assets::to_string(scene.guid),
                                   .role = role_ == runtime_launch_role::server ? framework::runtime_world_role::server
                                                                                : framework::runtime_world_role::client,
                                   .seed = config_.simulation.process_seed + index,
                                   .install_placeholder_systems = false,
                                   .presentation_enabled = role_ == runtime_launch_role::client &&
                                                           config_.simulation.presentation_enabled});
                created.push_back(world.id());
                if (!registrations_.materialize_scene)
                    throw packaged_runtime_startup_error(
                        failure(project_error_code::invalid_scene, "startupScenes",
                                "Missing cooked scene materializer for " + assets::to_string(scene.guid)));

                auto result = registrations_.materialize_scene(world, scene);
                if (!result)
                {
                    auto error = std::move(result).error();
                    if (error.field.empty()) error.field = "startupScenes";
                    if (error.message.empty()) error.message = "Cooked scene materialization failed";
                    throw packaged_runtime_startup_error(std::move(error));
                }
            }
        }
    }
    catch (...)
    {
        for (auto it = created.rbegin(); it != created.rend(); ++it)
            static_cast<void>(worlds.destroy(*it));
        throw;
    }
}

const std::vector<packaged_scene_payload>& packaged_runtime_application::scenes() const noexcept
{
    return scenes_;
}

runtime_launch_role packaged_runtime_application::role() const noexcept
{
    return role_;
}

packaged_application_result prepare_packaged_application(packaged_application_request request)
{
    const auto& boot = request.boot;
    const auto& manifest = boot.launch.manifest;
    const auto& scenes = request.catalog.startup_scenes();

    auto checked = validate_runtime_launch({.manifest = manifest,
                                            .role = boot.launch.role,
                                            .host = boot.launch.host,
                                            .startup_bundles = boot.launch.startup_bundles});
    if (!checked) return packaged_application_result::failure(std::move(checked).error());

    if (manifest.startup_scenes.size() != scenes.size())
        return packaged_application_result::failure(
            failure(project_error_code::invalid_scene, "startupScenes",
                    "Cooked catalog startup scene count does not match the distribution manifest"));
    if (scenes.empty() && boot.launch.role == runtime_launch_role::client)
        return packaged_application_result::failure(
            failure(project_error_code::invalid_scene, "startupScenes", "Packaged client requires a startup scene"));
    if (!scenes.empty() && !request.registrations.materialize_scene)
        return packaged_application_result::failure(
            failure(project_error_code::missing_module, "materializeScene",
                    "Packaged runtime must register a cooked scene materializer before starting scenes"));

    std::vector<packaged_scene_payload> payloads;
    payloads.reserve(scenes.size());
    std::uint64_t total_bytes{};
    for (std::size_t index = 0; index < scenes.size(); ++index)
    {
        const auto parsed = assets::parse_asset_guid(manifest.startup_scenes[index]);
        if (!parsed || scenes[index] != *parsed)
            return packaged_application_result::failure(
                failure(project_error_code::invalid_scene, "startupScenes",
                        "Cooked catalog scene ordering or GUID does not match D0 at index " + std::to_string(index)));

        auto payload = read_scene(request.catalog, scenes[index]);
        if (!payload) return packaged_application_result::failure(std::move(payload).error());
        if (payload.value().bytes.size() > maximum_total_startup_bytes - total_bytes)
            return packaged_application_result::failure(
                failure(project_error_code::invalid_scene, "startupScenes",
                        "Total cooked startup scene bytes exceed the 256 MiB bootstrap budget"));
        total_bytes += payload.value().bytes.size();
        payloads.push_back(std::move(payload).value());
    }

    auto config = std::move(request.config);
    config.simulation.default_world_role = boot.launch.role == runtime_launch_role::server
                                               ? framework::runtime_world_role::server
                                               : framework::runtime_world_role::client;
    if (boot.launch.role == runtime_launch_role::server)
    {
        config.simulation.headless = true;
        config.simulation.presentation_enabled = false;
        config.visible = false;
        config.start_focused = false;
    }
    else
        config.simulation.headless = false;

    return packaged_application_result::success(
        std::unique_ptr<packaged_runtime_application>(new packaged_runtime_application(
            boot.launch.role, std::move(config), std::move(request.registrations), std::move(payloads))));
}

} // namespace arc::project

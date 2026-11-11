#include <arc/project/runtime_catalog.h>

#include <arc/assets/cook.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::project
{
namespace
{

project_error failure(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}

runtime_artifact_read_result artifact_failure(project_error_code code, std::string message)
{
    return runtime_artifact_read_result::failure(failure(code, "content.artifacts", std::move(message)));
}

using path_result = core::result<io::virtual_path, project_error>;

path_result virtual_child(const io::virtual_path& root, std::string_view relative)
{
    std::string combined(root.relative_path());
    if (!combined.empty()) combined += '/';
    combined += relative;
    auto parsed = io::virtual_path::from_parts(root.scheme(), root.authority(), combined);
    if (!parsed)
        return path_result::failure(failure(project_error_code::unsafe_path, "content.cookManifests",
                                            "Invalid virtual path for staged cook manifest: " + std::string(relative)));
    return path_result::success(std::move(parsed).value());
}

using cook_result = core::result<assets::cook_manifest, project_error>;

cook_result read_cook_manifest(io::virtual_file_system& files, const io::virtual_path& manifest_file)
{
    auto resolved = files.resolve(manifest_file);
    if (!resolved)
        return cook_result::failure(
            failure(project_error_code::not_found, "content.cookManifests",
                    "Staged cook manifest cannot be resolved: " + std::string(manifest_file.string())));
    auto bytes = files.read_all(resolved.value()).get();
    if (!bytes)
        return cook_result::failure(
            failure(project_error_code::io_failed, "content.cookManifests",
                    "Staged cook manifest cannot be read: " + std::string(manifest_file.string()) + " (" +
                        bytes.error().message + ")"));
    auto parsed = assets::parse_cook_manifest(bytes.value(), manifest_file.string());
    if (!parsed)
        return cook_result::failure(failure(project_error_code::invalid_descriptor, "content.cookManifests",
                                            "Staged cook manifest is invalid: " + parsed.error().message));
    return cook_result::success(std::move(parsed).value());
}

project_status verify_target(const assets::cook_target& cooked, const distribution_target& distribution,
                             std::string_view manifest_path)
{
    // Match the actual cooked platform. C6 (#994) will add richer
    // feature/quality fallback and variant selection; never accept a Windows
    // cooked representation for an Android distribution.
    const auto platform_matches =
        (distribution.platform == "windows" && cooked.platform == assets::cook_platform::windows) ||
        (distribution.platform == "linux" && cooked.platform == assets::cook_platform::linux_os) ||
        (distribution.platform == "macos" && cooked.platform == assets::cook_platform::macos) ||
        (distribution.platform == "android" && cooked.platform == assets::cook_platform::android);
    if (!platform_matches)
        return project_status::failure(
            failure(project_error_code::unsupported_platform, "target.platform",
                    "Cooked manifest " + std::string(manifest_path) +
                        " does not support the packaged target platform: " + distribution.platform));

    const bool architecture_matches =
        (distribution.architecture == "x86_64" && cooked.architecture == assets::cook_architecture::x86_64) ||
        (distribution.architecture == "arm64" && cooked.architecture == assets::cook_architecture::arm64);
    if (!architecture_matches)
        return project_status::failure(
            failure(project_error_code::incompatible_engine, "target.architecture",
                    "Cooked manifest " + std::string(manifest_path) + " has an incompatible CPU architecture"));

    const bool renderer_matches =
        (distribution.renderer == "none" && cooked.renderer == assets::cook_renderer::none) ||
        (distribution.renderer == "vulkan" && cooked.renderer == assets::cook_renderer::vulkan) ||
        (distribution.renderer == "direct3d12" && cooked.renderer == assets::cook_renderer::direct3d12) ||
        (distribution.renderer == "metal" && cooked.renderer == assets::cook_renderer::metal);
    if (!renderer_matches)
        return project_status::failure(
            failure(project_error_code::incompatible_engine, "target.renderer",
                    "Cooked manifest " + std::string(manifest_path) + " has an incompatible rendering backend"));

    const auto configuration = distribution.configuration == "Shipping" ? assets::cook_configuration::shipping
                                                                        : assets::cook_configuration::development;
    if (cooked.configuration != configuration)
        return project_status::failure(
            failure(project_error_code::incompatible_engine, "target.configuration",
                    "Cooked manifest " + std::string(manifest_path) + " has an incompatible build configuration"));

    return project_status::success();
}

bool same_provider_namespace(const io::virtual_path& lhs, const io::virtual_path& rhs)
{
    return lhs.scheme() == rhs.scheme() && lhs.authority() == rhs.authority();
}

} // namespace

runtime_catalog_session::runtime_catalog_session(io::virtual_file_system& files) noexcept : files_(&files) {}

runtime_catalog_session::~runtime_catalog_session()
{
    // Artifact provider handles must be released before the native package
    // provider disappears. Externally mounted VFS sources are never touched.
    catalog_.reset();
    if (native_mount_) static_cast<void>(files_->unmount(native_mount_));
}

assets::cooked_asset_catalog& runtime_catalog_session::catalog() noexcept
{
    return *catalog_;
}

const assets::cooked_asset_catalog& runtime_catalog_session::catalog() const noexcept
{
    return *catalog_;
}

io::virtual_file_system& runtime_catalog_session::files() const noexcept
{
    return *files_;
}

const std::vector<assets::asset_guid>& runtime_catalog_session::startup_scenes() const noexcept
{
    return startup_scenes_;
}

const std::vector<assets::content_bundle_id>& runtime_catalog_session::startup_bundles() const noexcept
{
    return startup_bundles_;
}

runtime_artifact_read_result runtime_catalog_session::read_active_artifact(assets::asset_guid guid,
                                                                           assets::asset_type_id type,
                                                                           assets::artifact_schema_id schema,
                                                                           std::string_view name)
{
    if (!guid.valid() || !type.valid() || !schema.valid())
        return artifact_failure(project_error_code::invalid_descriptor, "Invalid cooked artifact identity");

    const auto record = catalog_->database().query(guid, type);
    if (!record) return artifact_failure(project_error_code::not_found, "Cooked GUID/type missing from active catalog");

    // Only the active provider is authoritative, even if an older one still
    // contains a representation with the requested schema.
    std::optional<assets::cooked_artifact_address> selected;
    for (const auto& representation : record->representations)
    {
        if (representation.provider_id != record->active_provider) continue;
        if (!name.empty() && representation.name != name) continue;

        auto path = io::virtual_path::parse(representation.locator_hint);
        if (!path) continue;
        auto address = assets::parse_cooked_artifact_virtual_path(path.value());
        if (!address || address->asset != guid || address->schema != schema || address->name != representation.name)
            continue;

        if (selected) return artifact_failure(project_error_code::invalid_descriptor, "Ambiguous artifact name");
        selected = std::move(address);
    }

    if (!selected) return artifact_failure(project_error_code::not_found, "No matching active cooked representation");

    auto resolved = catalog_->resolve(*selected);
    if (!resolved)
        return artifact_failure(project_error_code::not_found, "Cannot resolve artifact: " + resolved.error().message);

    auto bytes = files_->read_all(resolved.value()).get();
    if (!bytes) return artifact_failure(project_error_code::io_failed, "Read: " + bytes.error().message);
    return runtime_artifact_read_result::success(std::move(bytes).value());
}

runtime_catalog_result bootstrap_runtime_catalog(const runtime_catalog_request& request)
{
    const auto& boot = request.boot;
    const auto& source = boot.launch.host.distribution;
    const auto& manifest = boot.launch.manifest;
    const bool native = source.kind != runtime_distribution_location_kind::mounted_vfs;

    // Recheck D1.1 role/host assumptions at the entry boundary, including
    // direct callers that did not obtain the context from D1.2.
    auto checked = validate_runtime_launch({.manifest = manifest,
                                            .role = boot.launch.role,
                                            .host = boot.launch.host,
                                            .startup_bundles = boot.launch.startup_bundles});
    if (!checked) return runtime_catalog_result::failure(std::move(checked).error());

    auto package_root = source.virtual_root;
    if (native)
    {
        if (boot.native_distribution_root.empty() || !request.native_files ||
            !boot.native_distribution_root.is_absolute())
            return runtime_catalog_result::failure(
                failure(project_error_code::invalid_descriptor, "host.distribution",
                        "Native catalog bootstrap requires a validated absolute distribution root and file service"));
        package_root =
            request.native_mount.empty() ? io::virtual_path::parse("package://runtime").value() : request.native_mount;
    }
    else if (source.files != &request.files || !boot.native_distribution_root.empty())
        return runtime_catalog_result::failure(
            failure(project_error_code::invalid_descriptor, "host.distribution",
                    "Mounted distribution must use its original host VFS without a native root"));

    const auto artifact_root =
        request.artifact_mount.empty() ? io::virtual_path::parse("artifact://game").value() : request.artifact_mount;
    if (package_root.empty() || artifact_root.empty() || same_provider_namespace(package_root, artifact_root))
        return runtime_catalog_result::failure(
            failure(project_error_code::invalid_descriptor, "host.distribution",
                    "Cooked artifact mount must use a separate virtual provider namespace"));

    auto session = std::unique_ptr<runtime_catalog_session>(new runtime_catalog_session(request.files));
    if (native)
    {
        auto provider =
            std::make_shared<io::filesystem_file_provider>(*request.native_files, boot.native_distribution_root);
        auto mounted = request.files.mount({.root = package_root,
                                            .priority = 0,
                                            .provider = std::move(provider),
                                            .debug_name = "runtime distribution"});
        if (!mounted)
            return runtime_catalog_result::failure(
                failure(project_error_code::io_failed, "host.distribution",
                        "Cannot mount staged runtime distribution: " + mounted.error().message));
        session->native_mount_ = mounted.value();
    }

    // Preflight target compatibility and advertised bundle memberships before
    // mounting any cooked providers. Only cook manifest bytes are read here.
    std::vector<io::virtual_path> cook_paths;
    cook_paths.reserve(manifest.cook_manifests.size());
    std::set<std::string> available_bundles;
    std::set<assets::asset_guid> cooked_scenes;
    for (const auto& relative : manifest.cook_manifests)
    {
        auto path = virtual_child(package_root, relative);
        if (!path) return runtime_catalog_result::failure(std::move(path).error());
        auto cooked = read_cook_manifest(request.files, path.value());
        if (!cooked) return runtime_catalog_result::failure(std::move(cooked).error());
        checked = verify_target(cooked.value().target, manifest.target, relative);
        if (!checked) return runtime_catalog_result::failure(std::move(checked).error());
        available_bundles.insert(cooked.value().bundle_ids.begin(), cooked.value().bundle_ids.end());
        for (const auto& artifact : cooked.value().artifacts)
            if (artifact.type == assets::asset_types::scene && artifact.schema == assets::artifact_schemas::scene)
                cooked_scenes.insert(artifact.asset);
        cook_paths.push_back(std::move(path).value());
    }
    for (const auto& bundle : boot.launch.startup_bundles)
        if (!available_bundles.contains(bundle))
            return runtime_catalog_result::failure(
                failure(project_error_code::not_found, "startupBundles",
                        "Required startup Content Bundle '" + bundle + "' is absent from the staged cooked manifests"));

    session->catalog_ = std::make_unique<assets::cooked_asset_catalog>(request.files, artifact_root);
    // D0 cookManifest ordering is stable and determines deterministic overlay
    // priorities; a later manifest wins for the same cooked artifact address.
    for (std::size_t index = 0; index < cook_paths.size(); ++index)
    {
        auto mounted = session->catalog_->mount_package(cook_paths[index], static_cast<std::int32_t>(index));
        if (!mounted)
            return runtime_catalog_result::failure(
                failure(project_error_code::invalid_descriptor, "content.cookManifests",
                        "Cannot mount cooked manifest '" + std::string(cook_paths[index].string()) +
                            "': " + mounted.error().message));
    }

    // Resolve GUID/type/schema at the metadata layer; actual scene deserialization
    // and world registration belong to D1.4. A server may have no startup scenes.
    for (const auto& guid : manifest.startup_scenes)
    {
        auto parsed = assets::parse_asset_guid(guid);
        if (!parsed)
            return runtime_catalog_result::failure(
                failure(project_error_code::invalid_scene, "startupScenes", "Startup scene GUID is invalid: " + guid));
        const auto record = session->catalog_->database().query(*parsed, assets::asset_types::scene);
        if (!record || !cooked_scenes.contains(*parsed))
            return runtime_catalog_result::failure(
                failure(project_error_code::invalid_scene, "startupScenes",
                        "No compatible cooked scene representation for startup GUID: " + guid));
        session->startup_scenes_.push_back(*parsed);
    }
    session->startup_bundles_ = boot.launch.startup_bundles;

    return runtime_catalog_result::success(std::move(session));
}

} // namespace arc::project

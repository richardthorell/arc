#include <arc/project/runtime_boot.h>

#include <arc/assets/assets.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace arc::project
{
namespace
{

constexpr std::uint64_t max_manifest_bytes = 16u * 1024u * 1024u;
constexpr std::size_t max_executable_ancestor_levels = 8;

project_error fail(project_error_code code, std::string field, std::filesystem::path path, std::string message)
{
    return {.code = code, .path = std::move(path), .field = std::move(field), .message = std::move(message)};
}

template <class T>
core::result<T, project_error> failure(project_error_code code, std::string field, std::filesystem::path path,
                                       std::string message)
{
    return core::result<T, project_error>::failure(fail(code, std::move(field), std::move(path), std::move(message)));
}

bool inside_root(const std::filesystem::path& root, const std::filesystem::path& path)
{
    auto parent = root.begin();
    auto child = path.begin();
    while (parent != root.end() && child != path.end())
    {
        if (*parent != *child) return false;
        ++parent;
        ++child;
    }
    return parent == root.end();
}

using native_path_result = core::result<std::filesystem::path, project_error>;

native_path_result contained_file(const std::filesystem::path& root, std::string_view relative, std::string_view field)
{
    const auto physical = root / std::filesystem::path(relative);
    std::error_code ec;
    const auto canonical = std::filesystem::canonical(physical, ec);
    if (ec)
    {
        const auto code =
            ec == std::errc::no_such_file_or_directory ? project_error_code::not_found : project_error_code::io_failed;
        return failure<std::filesystem::path>(code, std::string(field), physical,
                                              "Required staged file is unavailable: " + ec.message());
    }
    if (!inside_root(root, canonical))
        return failure<std::filesystem::path>(project_error_code::unsafe_path, std::string(field), physical,
                                              "Staged file resolves outside the distribution root");
    if (!std::filesystem::is_regular_file(canonical, ec) || ec)
        return failure<std::filesystem::path>(project_error_code::io_failed, std::string(field), physical,
                                              "Staged entry is not a readable regular file");
    return native_path_result::success(canonical);
}

using text_result = core::result<std::string, project_error>;

text_result read_native_manifest(const std::filesystem::path& root)
{
    auto path = contained_file(root, runtime_distribution_manifest_name, "manifest");
    if (!path) return text_result::failure(std::move(path).error());

    std::error_code ec;
    const auto size = std::filesystem::file_size(path.value(), ec);
    if (ec)
        return failure<std::string>(project_error_code::io_failed, "manifest", path.value(),
                                    "Could not determine distribution manifest size: " + ec.message());
    if (size > max_manifest_bytes)
        return failure<std::string>(project_error_code::invalid_descriptor, "manifest", path.value(),
                                    "Distribution manifest exceeds the 16 MiB boot limit");

    std::ifstream input(path.value(), std::ios::binary);
    if (!input)
        return failure<std::string>(project_error_code::io_failed, "manifest", path.value(),
                                    "Cannot open distribution manifest");
    std::string text(static_cast<std::size_t>(size), '\0');
    if (!text.empty()) input.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        return failure<std::string>(project_error_code::io_failed, "manifest", path.value(),
                                    "Distribution manifest changed or could not be read");
    return text_result::success(std::move(text));
}

using virtual_path_result = core::result<io::virtual_path, project_error>;

virtual_path_result virtual_child(const runtime_distribution_location& location, std::string_view relative)
{
    std::string combined(location.virtual_root.relative_path());
    if (!combined.empty()) combined += '/';
    combined += relative;
    auto path =
        io::virtual_path::from_parts(location.virtual_root.scheme(), location.virtual_root.authority(), combined);
    if (!path)
        return failure<io::virtual_path>(project_error_code::unsafe_path, "host.distribution", {},
                                         "Invalid staged virtual path: " + std::string(relative));
    return virtual_path_result::success(std::move(path).value());
}

using bytes_result = core::result<io::file_buffer, project_error>;

bytes_result read_virtual_file(const runtime_distribution_location& location, std::string_view relative,
                               std::string_view field, std::uint64_t max_bytes)
{
    auto path = virtual_child(location, relative);
    if (!path) return bytes_result::failure(std::move(path).error());

    auto resolved = location.files->resolve(path.value());
    if (!resolved)
    {
        const auto code = resolved.error().code == io::file_error_code::not_found ||
                                  resolved.error().code == io::file_error_code::tombstoned
                              ? project_error_code::not_found
                              : project_error_code::io_failed;
        return failure<io::file_buffer>(code, std::string(field), {},
                                        "Staged virtual file '" + std::string(path.value().string()) +
                                            "' is unavailable: " + resolved.error().message);
    }
    if (resolved.value().size() > max_bytes)
        return failure<io::file_buffer>(project_error_code::invalid_descriptor, std::string(field), {},
                                        "Staged virtual file '" + std::string(path.value().string()) +
                                            "' exceeds the configured boot read limit");
    auto read = location.files->read_all(resolved.value()).get();
    if (!read)
        return failure<io::file_buffer>(project_error_code::io_failed, std::string(field), {},
                                        "Could not read staged virtual file '" + std::string(path.value().string()) +
                                            "': " + read.error().message);
    if (read.value().size() > max_bytes)
        return failure<io::file_buffer>(project_error_code::invalid_descriptor, std::string(field), {},
                                        "Staged virtual file grew beyond the boot read limit");
    return bytes_result::success(std::move(read).value());
}

using manifest_result = core::result<distribution_manifest, project_error>;

manifest_result parse_manifest_document(std::string_view text, std::filesystem::path location)
{
    auto parsed = parse_distribution_manifest(text);
    if (!parsed)
    {
        auto error = std::move(parsed).error();
        error.path = std::move(location);
        return manifest_result::failure(std::move(error));
    }
    return manifest_result::success(std::move(parsed).value());
}

struct native_discovery
{
    distribution_manifest manifest;
    std::filesystem::path root;
};

using native_discovery_result = core::result<native_discovery, project_error>;

native_discovery_result read_native_at_root(const std::filesystem::path& candidate)
{
    std::error_code ec;
    const auto root = std::filesystem::canonical(candidate, ec);
    if (ec || !std::filesystem::is_directory(root))
        return failure<native_discovery>(project_error_code::not_found, "host.distribution", candidate,
                                         "Distribution root is not an accessible directory");

    auto contents = read_native_manifest(root);
    if (!contents) return native_discovery_result::failure(std::move(contents).error());
    auto parsed = parse_manifest_document(contents.value(), root / runtime_distribution_manifest_name);
    if (!parsed) return native_discovery_result::failure(std::move(parsed).error());
    return native_discovery_result::success({.manifest = std::move(parsed).value(), .root = root});
}

native_discovery_result discover_native(const runtime_distribution_location& location)
{
    if (location.kind == runtime_distribution_location_kind::directory)
        return read_native_at_root(location.native_path);

    std::error_code ec;
    const auto executable = std::filesystem::canonical(location.native_path, ec);
    if (ec || !std::filesystem::is_regular_file(executable))
        return failure<native_discovery>(project_error_code::not_found, "host.distribution", location.native_path,
                                         "Host executable is missing or not a regular file");

    // Start beside the executable, never at current_path(). A package may use
    // bin/ or a bundle's Contents/MacOS/ layout; avoid an unbounded root scan.
    auto candidate = location.native_path.parent_path();
    for (std::size_t level = 0; level < max_executable_ancestor_levels && !candidate.empty(); ++level)
    {
        ec.clear();
        const bool present = std::filesystem::exists(candidate / runtime_distribution_manifest_name, ec);
        if (ec)
            return failure<native_discovery>(project_error_code::io_failed, "manifest",
                                             candidate / runtime_distribution_manifest_name,
                                             "Cannot inspect distribution manifest location: " + ec.message());
        if (present)
        {
            auto found = read_native_at_root(candidate);
            if (!found) return found; // Never mask an invalid nearer manifest by searching higher up.
            const auto& entry = found.value().manifest.entrypoint;
            if (entry.kind != "executable")
                return failure<native_discovery>(
                    project_error_code::invalid_descriptor, "entrypoint.kind",
                    candidate / runtime_distribution_manifest_name,
                    "Native executable host requires a distribution executable entrypoint");
            auto staged_executable = contained_file(found.value().root, entry.value, "entrypoint.value");
            if (!staged_executable) return native_discovery_result::failure(std::move(staged_executable).error());
            if (staged_executable.value() != executable)
                return failure<native_discovery>(project_error_code::invalid_descriptor, "entrypoint.value",
                                                 location.native_path,
                                                 "Host executable does not match the distribution entrypoint");
            return found;
        }
        if (candidate == candidate.root_path()) break;
        candidate = candidate.parent_path();
    }
    return failure<native_discovery>(project_error_code::not_found, "manifest",
                                     location.native_path.parent_path() / runtime_distribution_manifest_name,
                                     "No arc-distribution.json found beside or within eight parents of executable");
}

project_status check_expectations(const distribution_manifest& manifest, const runtime_boot_expectations& expected)
{
    const auto mismatch = [](std::string field, std::string required, std::string staged)
    {
        return project_status::failure(
            fail(project_error_code::incompatible_engine, field, {},
                 "Packaged " + field + " '" + staged + "' does not match host expectation '" + required + "'"));
    };
    if (manifest.engine_version != expected.engine_version)
        return mismatch("engineVersion", expected.engine_version, manifest.engine_version);
    if (manifest.sdk_version != expected.sdk_version)
        return mismatch("sdkVersion", expected.sdk_version, manifest.sdk_version);
    if (manifest.target.architecture != expected.architecture)
        return mismatch("target.architecture", expected.architecture, manifest.target.architecture);
    if (manifest.target.renderer != expected.renderer)
        return mismatch("target.renderer", expected.renderer, manifest.target.renderer);
    if (manifest.target.configuration != expected.configuration)
        return mismatch("target.configuration", expected.configuration, manifest.target.configuration);
    if (!expected.project_guid.empty() && manifest.project_guid != expected.project_guid)
        return mismatch("projectGuid", expected.project_guid, manifest.project_guid);
    if (!expected.target_id.empty() && manifest.target.id != expected.target_id)
        return mismatch("target.id", expected.target_id, manifest.target.id);
    if (!expected.cook_profile.empty() && manifest.target.cook_profile != expected.cook_profile)
        return mismatch("target.cookProfile", expected.cook_profile, manifest.target.cook_profile);
    return project_status::success();
}

project_status validate_request(const runtime_boot_request& request)
{
    const auto& expected = request.expected;
    if (expected.engine_version.empty() || expected.sdk_version.empty() || expected.architecture.empty() ||
        expected.renderer.empty() || expected.configuration.empty())
        return project_status::failure(
            fail(project_error_code::invalid_descriptor, "expected", {},
                 "Host must provide engine/SDK versions, architecture, renderer and build configuration"));
    if (request.integrity != runtime_boot_integrity::startup && request.integrity != runtime_boot_integrity::full)
        return project_status::failure(fail(project_error_code::invalid_descriptor, "integrity", {},
                                            "Unknown distribution boot integrity policy"));

    // Reject ambiguous/relative native locations before touching disk. D1.1
    // validates the complete host descriptor again after reading D0.
    const auto& source = request.host.distribution;
    if (source.kind == runtime_distribution_location_kind::executable ||
        source.kind == runtime_distribution_location_kind::directory)
    {
        if (source.native_path.empty() || !source.native_path.is_absolute() ||
            source.native_path != source.native_path.lexically_normal() || source.files || !source.virtual_root.empty())
            return project_status::failure(fail(project_error_code::unsafe_path, "host.distribution",
                                                source.native_path, "Host must supply one absolute native location"));
    }
    else if (source.kind == runtime_distribution_location_kind::mounted_vfs)
    {
        if (!source.files || source.virtual_root.empty() || !source.native_path.empty())
            return project_status::failure(fail(project_error_code::invalid_descriptor, "host.distribution", {},
                                                "Mounted distribution requires a VFS and virtual root"));
    }
    else
        return project_status::failure(fail(project_error_code::invalid_descriptor, "host.distribution", {},
                                            "Unknown distribution location kind"));
    return project_status::success();
}

bool boot_required(const distribution_manifest& manifest, std::string_view path)
{
    return path == manifest.runtime_config || std::find(manifest.cook_manifests.begin(), manifest.cook_manifests.end(),
                                                        path) != manifest.cook_manifests.end();
}

project_status verify_native_inventory(const distribution_manifest& manifest, const std::filesystem::path& root,
                                       runtime_boot_integrity integrity)
{
    for (const auto& file : manifest.files)
    {
        auto physical = contained_file(root, file.path, "files.path");
        if (!physical) return project_status::failure(std::move(physical).error());
        std::error_code ec;
        const auto size = std::filesystem::file_size(physical.value(), ec);
        if (ec)
            return project_status::failure(fail(project_error_code::io_failed, "files.path", physical.value(),
                                                "Could not stat staged file: " + ec.message()));
        if (size != file.size)
            return project_status::failure(fail(project_error_code::corrupt_content, "files.size", physical.value(),
                                                "Staged file size disagrees with distribution manifest"));
        if (integrity != runtime_boot_integrity::full && !boot_required(manifest, file.path)) continue;
        auto hash = assets::hash_file(physical.value());
        if (!hash)
            return project_status::failure(fail(project_error_code::io_failed, "files.sha256", physical.value(),
                                                "Could not hash staged file: " + hash.error().message));
        if (assets::to_string(hash.value()) != file.sha256)
            return project_status::failure(fail(project_error_code::corrupt_content, "files.sha256", physical.value(),
                                                "Staged file SHA-256 differs from distribution manifest"));
    }
    return project_status::success();
}

project_status verify_virtual_inventory(const distribution_manifest& manifest,
                                        const runtime_distribution_location& location, runtime_boot_integrity integrity)
{
    for (const auto& file : manifest.files)
    {
        // Android hosts may expose packaged config/content through a VFS while
        // the OS loader owns native libraries. A full check requires every file
        // to be visible in the supplied logical distribution provider.
        if (integrity != runtime_boot_integrity::full && !boot_required(manifest, file.path)) continue;
        auto data = read_virtual_file(location, file.path, "files.path", file.size);
        if (!data)
        {
            // A size overrun should be reported as corruption, not as a manifest cap.
            auto error = std::move(data).error();
            if (error.code == project_error_code::invalid_descriptor)
            {
                error.code = project_error_code::corrupt_content;
                error.field = "files.size";
            }
            return project_status::failure(std::move(error));
        }
        if (data.value().size() != file.size)
            return project_status::failure(
                fail(project_error_code::corrupt_content, "files.size", {},
                     "Staged virtual file size differs from distribution manifest: " + file.path));
        if (assets::to_string(assets::hash_bytes(data.value())) != file.sha256)
            return project_status::failure(
                fail(project_error_code::corrupt_content, "files.sha256", {},
                     "Staged virtual file SHA-256 differs from distribution manifest: " + file.path));
    }
    return project_status::success();
}

} // namespace

runtime_boot_result discover_runtime_boot(runtime_boot_request request)
{
    auto checked = validate_request(request);
    if (!checked) return runtime_boot_result::failure(std::move(checked).error());

    distribution_manifest manifest;
    std::filesystem::path native_root;
    if (request.host.distribution.kind == runtime_distribution_location_kind::mounted_vfs)
    {
        auto bytes = read_virtual_file(request.host.distribution, runtime_distribution_manifest_name, "manifest",
                                       max_manifest_bytes);
        if (!bytes) return runtime_boot_result::failure(std::move(bytes).error());
        const auto text = std::string_view(reinterpret_cast<const char*>(bytes.value().data()), bytes.value().size());
        auto parsed = parse_manifest_document(text, runtime_distribution_manifest_name);
        if (!parsed) return runtime_boot_result::failure(std::move(parsed).error());
        manifest = std::move(parsed).value();
    }
    else
    {
        auto discovered = discover_native(request.host.distribution);
        if (!discovered) return runtime_boot_result::failure(std::move(discovered).error());
        manifest = std::move(discovered.value().manifest);
        native_root = std::move(discovered.value().root);
    }

    auto prepared = prepare_runtime_launch({.manifest = std::move(manifest),
                                            .role = request.role,
                                            .host = std::move(request.host),
                                            .startup_bundles = std::move(request.startup_bundles)});
    if (!prepared) return runtime_boot_result::failure(std::move(prepared).error());

    checked = check_expectations(prepared.value().manifest, request.expected);
    if (!checked) return runtime_boot_result::failure(std::move(checked).error());

    if (prepared.value().host.distribution.kind == runtime_distribution_location_kind::mounted_vfs)
        checked =
            verify_virtual_inventory(prepared.value().manifest, prepared.value().host.distribution, request.integrity);
    else
        checked = verify_native_inventory(prepared.value().manifest, native_root, request.integrity);
    if (!checked) return runtime_boot_result::failure(std::move(checked).error());

    return runtime_boot_result::success(
        {.launch = std::move(prepared).value(), .native_distribution_root = std::move(native_root)});
}

} // namespace arc::project

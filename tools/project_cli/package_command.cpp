#include "package_command.h"

#include <arc/assets/cook.h>
#include <arc/project/distribution_assembler.h>
#include <arc/project/distribution_zip.h>
#include <arc/project/package_workflow.h>
#include <arc/project/project.h>
#include <arc/project/windows_distribution_finalizer.h>
#include <arc/project/windows_dependencies.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;
namespace assets = arc::assets;
using json = nlohmann::json;

std::optional<std::string> option(const std::vector<std::string>& args, std::string_view name)
{
    const auto found = std::find(args.begin(), args.end(), name);
    if (found == args.end() || std::next(found) == args.end()) return std::nullopt;
    return *std::next(found);
}

std::vector<std::string> options(const std::vector<std::string>& args, std::string_view name)
{
    std::vector<std::string> result;
    for (auto i = args.begin(); i != args.end(); ++i)
        if (*i == name && i + 1 != args.end()) result.push_back(*(++i));
    return result;
}

int fail(bool json_output, std::string_view field, std::string_view message)
{
    if (json_output)
        std::cout << json{{"succeeded", false}, {"field", field}, {"error", message}}.dump() << '\n';
    else
        std::cerr << "arc-project package: " << field << ": " << message << '\n';
    return 1;
}

fs::path source_path(const fs::path& root, std::string_view value)
{
    fs::path source(value);
    return (source.is_absolute() ? source : root / source).lexically_normal();
}

} // namespace

int handle_package_command(const std::vector<std::string>& args, bool json_output)
{
    // The source-selection arguments are intentionally explicit until build/link
    // metadata can produce a complete native dependency closure. There is no
    // implicit directory scan of the SDK or editor in this command.
    const auto project_arg = option(args, "--project");
    const auto binary_arg = option(args, "--executable");
    const auto config_arg = option(args, "--runtime-config");
    const auto cook_arg = option(args, "--cook-manifest");
    if (!project_arg || !binary_arg || !config_arg || !cook_arg)
        return fail(json_output, "arguments", "requires --project, --executable, --runtime-config and --cook-manifest");

    const auto descriptor_path = fs::absolute(*project_arg).lexically_normal();
    auto descriptor = project::load_descriptor(descriptor_path);
    if (!descriptor) return fail(json_output, "project", descriptor.error().message);
    auto context = project::resolve_context(descriptor_path, descriptor.value());
    if (!context) return fail(json_output, "project", context.error().message);

    const bool server = std::find(args.begin(), args.end(), "--server") != args.end();
    const auto workflow = project::plan_package_workflow(
        descriptor.value(), context.value(),
        {.profile = option(args, "--profile").value_or(""),
         .action = project::package_workflow_action::package,
         .role = server ? project::package_workflow_role::server : project::package_workflow_role::runtime});
    if (!workflow) return fail(json_output, workflow.error().field, workflow.error().message);
    if (workflow.value().platform != "windows" || workflow.value().configuration != "Shipping")
        return fail(json_output, "target", "D2.3 currently supports Windows Shipping distributions only");

    const auto cook_path = source_path(context.value().root, *cook_arg);
    const auto cooked = assets::load_cook_manifest(cook_path);
    if (!cooked) return fail(json_output, "cookManifest", cooked.error().message);
    const auto& target = cooked.value().target;
    const bool matches =
        target.platform == assets::cook_platform::windows &&
        target.architecture == (workflow.value().architecture == "arm64" ? assets::cook_architecture::arm64
                                                                         : assets::cook_architecture::x86_64) &&
        target.configuration == assets::cook_configuration::shipping &&
        ((workflow.value().renderer == "none" && target.renderer == assets::cook_renderer::none) ||
         (workflow.value().renderer == "vulkan" && target.renderer == assets::cook_renderer::vulkan) ||
         (workflow.value().renderer == "direct3d12" && target.renderer == assets::cook_renderer::direct3d12));
    if (!matches)
        return fail(json_output, "cookManifest.target", "Cooked target is incompatible with selected workflow");

    project::distribution_manifest manifest;
    manifest.project_guid = workflow.value().project_guid;
    manifest.engine_version = workflow.value().engine_version;
    manifest.sdk_version = option(args, "--sdk-version").value_or(workflow.value().engine_version);
    manifest.target = {.id = workflow.value().target_id,
                       .platform = workflow.value().platform,
                       .architecture = workflow.value().architecture,
                       .renderer = workflow.value().renderer,
                       .configuration = workflow.value().configuration,
                       .cook_profile = workflow.value().profile};
    manifest.role = server ? "server" : "client";
    manifest.finalizer = workflow.value().finalizer;
    const auto binary = source_path(context.value().root, *binary_arg);
    if (binary.filename().empty()) return fail(json_output, "executable", "Executable filename is empty");
    manifest.entrypoint = {.kind = "executable", .value = "bin/" + binary.filename().generic_string()};
    manifest.runtime_config = "Config/runtime.json";
    const auto cook_destination = "Content/" + cook_path.filename().generic_string();
    manifest.cook_manifests = {cook_destination};

    if (!server && descriptor.value().startup_scenes.empty())
        return fail(json_output, "startupScenes", "Client project must declare startup scenes");

    std::set<assets::asset_guid> cooked_scenes;
    for (const auto& artifact : cooked.value().artifacts)
        if (artifact.type == assets::asset_types::scene) cooked_scenes.insert(artifact.asset);

    for (const auto& scene : descriptor.value().startup_scenes)
    {
        const auto parsed = assets::parse_asset_guid(scene.guid);
        if (!parsed || !cooked_scenes.contains(*parsed))
            return fail(json_output, "startupScenes",
                        "Startup scene is missing from the selected cooked package: " + scene.guid);
        manifest.startup_scenes.push_back(scene.guid);
    }

    std::vector<project::distribution_assembly_file> files;
    files.push_back({.source = binary, .destination = manifest.entrypoint.value, .kind = "executable"});
    files.push_back({.source = source_path(context.value().root, *config_arg),
                     .destination = manifest.runtime_config,
                     .kind = "configuration"});
    files.push_back({.source = cook_path, .destination = cook_destination, .kind = "content"});

    std::set<std::string> chunks;
    for (const auto& artifact : cooked.value().artifacts)
    {
        const auto& chunk = artifact.chunk;
        if (chunk.empty() || chunk != fs::path(chunk).filename().generic_string() ||
            !fs::path(chunk).extension().generic_string().ends_with(".arcpak"))
            return fail(json_output, "cookManifest.artifacts", "Cook manifest contains an unsafe chunk reference");
        if (chunks.insert(chunk).second)
            files.push_back(
                {.source = cook_path.parent_path() / chunk, .destination = "Content/" + chunk, .kind = "content"});
    }

    // Discover recursive PE imports. --dll-dir supplies deterministic search
    // roots; --library remains available for explicitly loaded runtime DLLs
    // that do not appear in an import table.
    std::vector<fs::path> search_directories;
    std::vector<fs::path> explicit_libraries;
    std::vector<fs::path> plugins;
    for (const auto& value : options(args, "--dll-dir"))
        search_directories.push_back(source_path(context.value().root, value));
    for (const auto& value : options(args, "--library"))
    {
        auto path = source_path(context.value().root, value);
        search_directories.push_back(path.parent_path());
        explicit_libraries.push_back(std::move(path));
    }
    for (const auto& value : options(args, "--plugin"))
        plugins.push_back(source_path(context.value().root, value));
    auto closure = project::resolve_windows_dependencies(binary, plugins, search_directories);
    if (!closure) return fail(json_output, closure.error().field, closure.error().message);

    std::vector<project::distribution_native_dependency> dependencies = std::move(closure.value().dependencies);
    std::set<std::string> selected_names;
    for (const auto& item : closure.value().files)
    {
        selected_names.insert(item.destination);
        files.push_back(item);
    }
    for (const auto& path : explicit_libraries)
    {
        const auto destination = "bin/" + path.filename().generic_string();
        if (!selected_names.insert(destination).second) continue;
        files.push_back({.source = path, .destination = destination, .kind = "library"});
        dependencies.push_back({.destination = destination, .kind = "library"});
    }

    const auto output = option(args, "--output") ? source_path(context.value().root, *option(args, "--output"))
                                                 : workflow.value().package_output;
    project::windows_directory_finalizer finalizer;
    auto assembled = project::assemble_distribution({.workflow = workflow.value(),
                                                     .manifest = std::move(manifest),
                                                     .output_root = output,
                                                     .files = std::move(files),
                                                     .dependencies = std::move(dependencies)},
                                                    finalizer);
    if (!assembled) return fail(json_output, assembled.error().field, assembled.error().message);

    // ZIP is an optional secondary export of the verified directory. The
    // directory remains available if the ZIP operation encounters an error.
    const auto zip_arg = option(args, "--zip");
    std::optional<fs::path> zip_output;
    if (zip_arg)
    {
        zip_output = source_path(context.value().root, *zip_arg);
        const auto status = project::write_distribution_zip(output, *zip_output);
        if (!status) return fail(json_output, status.error().field, status.error().message);
    }

    if (json_output)
        std::cout << json{{"succeeded", true},
                          {"output", output.generic_string()},
                          {"files", assembled.value().files.size()},
                          {"manifest", (output / "arc-distribution.json").generic_string()},
                          {"zip", zip_output ? zip_output->generic_string() : ""}}
                         .dump()
                  << '\n';
    else
        std::cout << "Packaged ARC Windows distribution: " << output << '\n';
    if (!json_output && zip_output) std::cout << "ZIP archive: " << *zip_output << '\n';
    return 0;
}

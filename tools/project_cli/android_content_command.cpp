#include "android_content_command.h"

#include <arc/assets/cook.h>
#include <arc/project/distribution_assembler.h>
#include <arc/project/distribution_finalizer.h>
#include <arc/project/package_workflow.h>
#include <arc/project/project.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;
namespace assets = arc::assets;
using json = nlohmann::json;

std::optional<std::string> argument(const std::vector<std::string>& args, std::string_view name)
{
    const auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end() || std::next(it) == args.end()) return std::nullopt;
    return *std::next(it);
}

int fail(bool json_output, std::string_view field, std::string_view message)
{
    if (json_output)
        std::cout << json{{"succeeded", false}, {"field", field}, {"error", message}}.dump() << '\n';
    else
        std::cerr << "arc-project package android-content: " << field << ": " << message << '\n';
    return 1;
}

project::project_status error(project::project_error_code code, std::string field, std::string message)
{
    return project::project_status::failure({.code = code, .field = std::move(field), .message = std::move(message)});
}

class android_content_stage_finalizer final : public project::distribution_finalizer
{
public:
    [[nodiscard]] std::string_view id() const noexcept override
    {
        return "android-gradle";
    }

    [[nodiscard]] project::project_status validate(const project::distribution_finalize_request& request) const override
    {
        const auto& manifest = request.manifest;
        if (manifest.target.platform != "android" || manifest.target.architecture != "arm64" ||
            manifest.target.renderer != "vulkan" || manifest.role != "client" ||
            manifest.entrypoint.kind != "android-activity" || manifest.entrypoint.value != "android.app.NativeActivity")
            return error(project::project_error_code::unsupported_platform, "target",
                         "Android content staging requires an ARM64 Vulkan NativeActivity client");

        if (!request.dependencies.empty())
            return error(project::project_error_code::invalid_descriptor, "dependencies",
                         "A3 stages assets only; Android JNI library dependency closure belongs to A4");

        std::error_code ec;
        const auto root = fs::canonical(request.staging_root, ec);
        if (ec || !fs::is_directory(root))
            return error(project::project_error_code::not_found, "stagingRoot", "Staging directory is missing");

        // The shared assembler already enforces safe relative paths, input file
        // types and file hash inventory. Recheck the staged bytes before publish.
        std::set<std::string> inventory;
        for (const auto& item : manifest.files)
        {
            if (item.kind != "content" && item.kind != "configuration")
                return error(project::project_error_code::invalid_descriptor, "files.kind",
                             "A3 APK assets cannot contain native binaries, plugins or symbols");
            if (!item.path.starts_with("Content/") && !item.path.starts_with("Config/"))
                return error(project::project_error_code::unsafe_path, "files.path",
                             "A3 APK assets must be under Content/ or Config/");
            const auto file = root / fs::path(item.path);
            const auto status = fs::symlink_status(file, ec);
            if (ec || !fs::is_regular_file(status) || fs::is_symlink(status))
                return error(project::project_error_code::not_found, "files.path",
                             "Missing or unsafe staged asset: " + item.path);
            if (fs::file_size(file, ec) != item.size || ec)
                return error(project::project_error_code::corrupt_content, "files.size",
                             "Staged asset size changed: " + item.path);
            const auto digest = assets::hash_file(file);
            if (!digest || assets::to_string(digest.value()) != item.sha256)
                return error(project::project_error_code::corrupt_content, "files.sha256",
                             "Staged asset hash changed: " + item.path);
            inventory.insert(item.path);
        }
        for (const auto& entry : fs::recursive_directory_iterator(root, ec))
        {
            if (ec) return error(project::project_error_code::io_failed, "stagingRoot", ec.message());
            const auto status = entry.symlink_status(ec);
            if (ec || fs::is_symlink(status))
                return error(project::project_error_code::unsafe_path, "stagingRoot",
                             "Staged assets may not contain symbolic links");
            if (fs::is_directory(status)) continue;
            const auto relative = fs::relative(entry.path(), root, ec).generic_string();
            if (ec || !fs::is_regular_file(status) || !inventory.contains(relative))
                return error(project::project_error_code::invalid_descriptor, "files",
                             "Uninventoried or non-regular staged asset: " + relative);
        }
        return project::project_status::success();
    }

    [[nodiscard]] project::project_status finalize(const project::distribution_finalize_request& request) override
    {
        if (const auto checked = validate(request); !checked) return checked;
        const auto serialized = project::serialize_distribution_manifest(request.manifest);
        if (!serialized) return project::project_status::failure(serialized.error());

        const auto pending = fs::path(request.output_root.string() + ".pending");
        std::error_code ec;
        fs::create_directories(pending, ec);
        if (ec) return error(project::project_error_code::io_failed, "outputRoot", ec.message());

        const auto rollback = [&]
        {
            std::error_code ignored;
            fs::remove_all(pending, ignored);
        };
        for (const auto& entry : request.manifest.files)
        {
            const auto destination = pending / entry.path;
            fs::create_directories(destination.parent_path(), ec);
            if (!ec) fs::copy_file(request.staging_root / entry.path, destination, fs::copy_options::none, ec);
            if (ec)
            {
                rollback();
                return error(project::project_error_code::io_failed, "files",
                             "Failed to publish Android asset: " + entry.path + " (" + ec.message() + ")");
            }
            const auto hash = assets::hash_file(destination);
            if (!hash || assets::to_string(hash.value()) != entry.sha256)
            {
                rollback();
                return error(project::project_error_code::corrupt_content, "files.sha256",
                             "Copied Android asset failed SHA-256 verification: " + entry.path);
            }
        }

        {
            std::ofstream output(pending / "arc-distribution.json", std::ios::binary | std::ios::trunc);
            output << serialized.value();
            output.flush();
            if (!output.good())
            {
                rollback();
                return error(project::project_error_code::io_failed, "manifest",
                             "Failed to write Android D0 distribution manifest");
            }
        }
        fs::rename(pending, request.output_root, ec);
        if (ec)
        {
            rollback();
            return error(project::project_error_code::io_failed, "outputRoot",
                         "Failed to atomically publish Android content stage: " + ec.message());
        }
        return project::project_status::success();
    }
};

fs::path project_path(const fs::path& root, const std::string& value)
{
    const fs::path input(value);
    return fs::absolute(input.is_absolute() ? input : root / input).lexically_normal();
}

int run(const std::vector<std::string>& args, bool json_output)
{
    const auto descriptor_arg = argument(args, "--project");
    const auto profile_arg = argument(args, "--profile");
    const auto target_arg = argument(args, "--target");
    const auto cook_arg = argument(args, "--cook-manifest");
    const auto config_arg = argument(args, "--runtime-config");
    const auto output_arg = argument(args, "--output");
    if (!descriptor_arg || !profile_arg || !target_arg || !cook_arg || !config_arg || !output_arg)
        return fail(json_output, "arguments",
                    "requires --project, --profile, --target, --cook-manifest, --runtime-config and --output");

    const auto descriptor_path = fs::absolute(*descriptor_arg).lexically_normal();
    const auto descriptor = project::load_descriptor(descriptor_path);
    if (!descriptor) return fail(json_output, "project", descriptor.error().message);
    const auto context = project::resolve_context(descriptor_path, descriptor.value());
    if (!context) return fail(json_output, "project", context.error().message);

    project::package_workflow_request selection;
    selection.profile = *profile_arg;
    selection.action = project::package_workflow_action::package;
    selection.role = project::package_workflow_role::runtime;
    selection.target_id = *target_arg;
    const auto workflow = project::plan_package_workflow(descriptor.value(), context.value(), selection);
    if (!workflow) return fail(json_output, workflow.error().field, workflow.error().message);
    if (workflow.value().platform != "android" || workflow.value().architecture != "arm64" ||
        workflow.value().renderer != "vulkan")
        return fail(json_output, "target", "Android content staging requires an ARM64 Vulkan project target");

    const auto cook_file = project_path(context.value().root, *cook_arg);
    const auto cooked = assets::load_cook_manifest(cook_file);
    if (!cooked) return fail(json_output, "cookManifest", cooked.error().message);
    const auto& target = cooked.value().target;
    auto expected_configuration = assets::cook_configuration::development;
    if (workflow.value().configuration == "Shipping") expected_configuration = assets::cook_configuration::shipping;
    if (target.platform != assets::cook_platform::android || target.architecture != assets::cook_architecture::arm64 ||
        target.renderer != assets::cook_renderer::vulkan || target.name != workflow.value().profile ||
        target.configuration != expected_configuration)
        return fail(json_output, "cookManifest.target",
                    "Cooked ARM64/Vulkan target, profile or configuration differs from Android workflow");

    const auto* startup_scene = project::resolve_startup_scene(descriptor.value());
    if (!startup_scene)
        return fail(json_output, "startupScenes", "Android client must declare at least one startup scene");

    // Verify the package is structurally mountable before selecting its chunks.
    assets::asset_package_mount package;
    if (const auto mounted = package.mount(cook_file); !mounted)
        return fail(json_output, "cookManifest", "Cooked package cannot mount: " + mounted.error().message);

    project::distribution_manifest manifest;
    manifest.project_guid = workflow.value().project_guid;
    manifest.engine_version = workflow.value().engine_version;
    manifest.sdk_version = argument(args, "--sdk-version").value_or(workflow.value().engine_version);
    manifest.target = {.id = workflow.value().target_id,
                       .platform = "android",
                       .architecture = "arm64",
                       .renderer = "vulkan",
                       .configuration = workflow.value().configuration,
                       .cook_profile = workflow.value().profile};
    manifest.role = "client";
    manifest.finalizer = "android-gradle";
    manifest.entrypoint = {.kind = "android-activity", .value = "android.app.NativeActivity"};
    manifest.runtime_config = "Config/runtime.json";
    manifest.cook_manifests = {"Content/" + cook_file.filename().generic_string()};

    std::set<assets::asset_guid> cooked_scenes;
    for (const auto& artifact : cooked.value().artifacts)
        if (artifact.type == assets::asset_types::scene && artifact.schema == assets::artifact_schemas::scene)
            cooked_scenes.insert(artifact.asset);

    if (startup_scene)
    {
        const auto& scene = *startup_scene;
        const auto guid = assets::parse_asset_guid(scene.guid);
        if (!guid || !cooked_scenes.contains(*guid))
            return fail(json_output, "startupScenes",
                        "Startup scene is absent from Android cooked package: " + scene.guid);
        const auto readable = package.read(*guid, assets::artifact_schemas::scene);
        if (!readable)
            return fail(json_output, "startupScenes",
                        "Startup scene cannot be read from cooked package: " + scene.guid);
        manifest.startup_scenes.push_back(scene.guid);
    }

    const auto config_file = project_path(context.value().root, *config_arg);
    {
        std::ifstream input(config_file);
        if (!input) return fail(json_output, "runtimeConfig", "Runtime configuration is unreadable");
        auto parsed = json::parse(input, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object())
            return fail(json_output, "runtimeConfig", "Runtime configuration must be a JSON object");
    }

    std::vector<project::distribution_assembly_file> selected{
        {.source = config_file, .destination = manifest.runtime_config, .kind = "configuration"},
        {.source = cook_file, .destination = manifest.cook_manifests.front(), .kind = "content"},
    };
    std::set<std::string> chunks;
    for (const auto& artifact : cooked.value().artifacts)
    {
        const fs::path chunk(artifact.chunk);
        if (artifact.chunk.empty() || chunk.filename().generic_string() != artifact.chunk ||
            chunk.extension() != ".arcpak")
            return fail(json_output, "cookManifest.artifacts", "Unsafe or missing cooked package chunk name");
        if (chunks.insert(artifact.chunk).second)
            selected.push_back({.source = cook_file.parent_path() / chunk,
                                .destination = "Content/" + artifact.chunk,
                                .kind = "content"});
    }
    if (chunks.empty()) return fail(json_output, "cookManifest.artifacts", "Cooked package has no chunks");

    const auto output = project_path(context.value().root, *output_arg);
    android_content_stage_finalizer finalizer;
    auto staged = project::assemble_distribution({.workflow = workflow.value(),
                                                  .manifest = std::move(manifest),
                                                  .output_root = output,
                                                  .files = std::move(selected)},
                                                 finalizer);
    if (!staged) return fail(json_output, staged.error().field, staged.error().message);

    if (json_output)
    {
        json report;
        report["succeeded"] = true;
        report["output"] = output.generic_string();
        report["manifest"] = (output / "arc-distribution.json").generic_string();
        report["target"] = workflow.value().target_id;
        report["profile"] = workflow.value().profile;
        report["files"] = staged.value().files.size();
        std::cout << report.dump() << '\n';
    }
    else
        std::cout << "Staged Android cooked distribution assets: " << output << '\n';
    return 0;
}
} // namespace

int handle_android_content_command(const std::vector<std::string>& args, bool json_output)
{
    try
    {
        return run(args, json_output);
    }
    catch (const std::exception& exception)
    {
        return fail(json_output, "staging", exception.what());
    }
}

#include <arc/project/distribution_assembler.h>
#include <arc/project/windows_distribution_finalizer.h>

#include <arc/assets/assets.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;

struct fixture
{
    fs::path root = fs::temp_directory_path() /
                    ("arc-d2-assembler-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    fs::path output = root / "Published";
    project::package_workflow_plan workflow;
    project::distribution_manifest manifest;
    std::vector<project::distribution_assembly_file> files;

    fixture()
    {
        workflow.project_guid = "11111111-1111-1111-1111-111111111111";
        workflow.engine_version = "0.1.0";
        workflow.profile = "windows-x64-vulkan";
        workflow.target_id = workflow.profile;
        workflow.platform = "windows";
        workflow.architecture = "x86_64";
        workflow.renderer = "vulkan";
        workflow.configuration = "Shipping";
        workflow.finalizer = "windows-directory";
        workflow.role = project::package_workflow_role::runtime;

        manifest.project_guid = workflow.project_guid;
        manifest.engine_version = workflow.engine_version;
        manifest.sdk_version = "0.1.0";
        manifest.target = {.id = workflow.target_id,
                           .platform = workflow.platform,
                           .architecture = workflow.architecture,
                           .renderer = workflow.renderer,
                           .configuration = workflow.configuration,
                           .cook_profile = workflow.profile};
        manifest.role = "client";
        manifest.finalizer = workflow.finalizer;
        manifest.entrypoint = {.kind = "executable", .value = "bin/Game.exe"};
        manifest.runtime_config = "Config/runtime.json";
        manifest.cook_manifests = {"Content/game.arccookmanifest"};
        manifest.startup_scenes = {"22222222-2222-2222-2222-222222222222"};

        add("Build/Game.exe", "bin/Game.exe", "executable", "game");
        add("Build/renderer.dll", "bin/renderer.dll", "library", "renderer");
        add("Cook/game.arccookmanifest", "Content/game.arccookmanifest", "content", "manifest");
        add("Cook/game.arcpack", "Content/game.arcpack", "content", "cooked bytes");
        add("Config/runtime.json", "Config/runtime.json", "configuration", "{}");
    }

    ~fixture()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    void add(std::string name, std::string destination, std::string kind, std::string value)
    {
        const auto source = root / "Sources" / name;
        fs::create_directories(source.parent_path());
        std::ofstream stream(source, std::ios::binary | std::ios::trunc);
        stream << value;
        stream.close();
        files.push_back({.source = source, .destination = std::move(destination), .kind = std::move(kind)});
    }

    project::distribution_assembly_request request() const
    {
        return {.workflow = workflow,
                .manifest = manifest,
                .output_root = output,
                .files = files,
                .dependencies = {{.destination = "bin/renderer.dll", .kind = "library"}}};
    }

    bool cleaned() const
    {
        return !fs::exists(fs::path(output.string() + ".assemble")) &&
               !fs::exists(fs::path(output.string() + ".pending"));
    }
};
} // namespace

TEST_CASE("D2.2 shared assembler produces a hashed D0 inventory and publishes through Windows finalizer")
{
    fixture f;
    project::windows_directory_finalizer adapter;
    const auto result = project::assemble_distribution(f.request(), adapter);
    REQUIRE(result);
    REQUIRE(result.value().files.size() == f.files.size());
    REQUIRE(fs::is_regular_file(f.output / "arc-distribution.json"));
    CHECK(fs::is_regular_file(f.output / "Content/game.arcpack"));
    CHECK(fs::is_regular_file(f.output / "bin/renderer.dll"));
    CHECK(f.cleaned());
    for (const auto& entry : result.value().files)
    {
        const auto digest = arc::assets::hash_file(f.output / entry.path);
        REQUIRE(digest);
        CHECK(arc::assets::to_string(digest.value()) == entry.sha256);
        CHECK(fs::file_size(f.output / entry.path) == entry.size);
    }
}

TEST_CASE("D2.2 shared assembler rejects path traversal and case-insensitive collisions without staging")
{
    fixture f;
    project::windows_directory_finalizer adapter;
    auto request = f.request();
    request.files.back().destination = "../outside";
    auto result = project::assemble_distribution(request, adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "files.path");
    CHECK(f.cleaned());

    auto collision_request = f.request();
    collision_request.files.back().destination = "BIN/game.exe";
    result = project::assemble_distribution(collision_request, adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "files.path");
    CHECK(f.cleaned());
}

TEST_CASE("D2.2 shared assembler fails closed on missing binaries and cleans failed finalization")
{
    fixture f;
    project::windows_directory_finalizer adapter;
    auto request = f.request();
    fs::remove(f.files.front().source);
    auto result = project::assemble_distribution(request, adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "files.source");
    CHECK(f.cleaned());
    CHECK_FALSE(fs::exists(f.output));

    // Even if all remaining sources are valid, a required D0 entrypoint
    // missing from the selected inputs must reject finalization.
    auto missing_entry_request = f.request();
    missing_entry_request.files.erase(missing_entry_request.files.begin());
    result = project::assemble_distribution(missing_entry_request, adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "entrypoint.value");
    CHECK(f.cleaned());
    CHECK_FALSE(fs::exists(f.output));
}

TEST_CASE("D2.2 shared assembler refuses output overwrite and malformed cooked metadata")
{
    fixture f;
    project::windows_directory_finalizer adapter;
    REQUIRE(project::assemble_distribution(f.request(), adapter));
    auto result = project::assemble_distribution(f.request(), adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "outputRoot");
    CHECK(f.cleaned());

    fs::remove_all(f.output);
    auto request = f.request();
    request.manifest.cook_manifests = {"Content/absent.arccookmanifest"};
    result = project::assemble_distribution(request, adapter);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "content.cookManifests");
    CHECK(f.cleaned());
    CHECK_FALSE(fs::exists(f.output));
}

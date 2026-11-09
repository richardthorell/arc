#include <arc/project/windows_distribution_finalizer.h>

#include <arc/assets/assets.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;

struct directory_fixture
{
    fs::path root = fs::temp_directory_path() /
                    ("arc-d2-windows-finalizer-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    fs::path staging = root / "stage";
    fs::path output = root / "release";
    project::package_workflow_plan workflow;
    project::distribution_manifest manifest;

    directory_fixture()
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
        add_file("bin/Game.exe", "executable", "windows binary placeholder");
        add_file("Config/runtime.json", "configuration", "{}");
        add_file("Content/game.arccookmanifest", "content", "{}");
        add_file("Content/startup.arcpack", "content", "binary cooked payload");
        add_file("bin/physics.dll", "library", "DLL closure placeholder");
    }

    ~directory_fixture()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    void add_file(const std::string& relative, const std::string& kind, std::string_view bytes)
    {
        const auto path = staging / relative;
        fs::create_directories(path.parent_path());
        std::ofstream stream(path, std::ios::binary);
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        stream.close();
        const auto hash = arc::assets::hash_file(path);
        REQUIRE(hash);
        manifest.files.push_back({.path = relative,
                                  .kind = kind,
                                  .size = fs::file_size(path),
                                  .sha256 = arc::assets::to_string(hash.value())});
    }

    project::distribution_finalize_request request() const
    {
        return {.workflow = workflow,
                .manifest = manifest,
                .staging_root = staging,
                .output_root = output,
                .dependencies = {{.destination = "bin/physics.dll", .kind = "library"}}};
    }
};
} // namespace

TEST_CASE("D2.1 Windows directory finalizer publishes a verified relocatable Shipping payload")
{
    directory_fixture fixture;
    project::windows_directory_finalizer finalizer;
    REQUIRE(project::run_distribution_finalizer(fixture.request(), finalizer));
    REQUIRE(fs::is_regular_file(fixture.output / "arc-distribution.json"));
    REQUIRE(fs::is_regular_file(fixture.output / "bin/Game.exe"));
    REQUIRE(fs::is_regular_file(fixture.output / "bin/physics.dll"));
    REQUIRE(fs::is_regular_file(fixture.output / "Content/startup.arcpack"));
    CHECK_FALSE(fs::exists(fs::path(fixture.output.string() + ".pending")));
    const auto relocated = fixture.root / "relocated";
    fs::rename(fixture.output, relocated);
    CHECK(fs::is_regular_file(relocated / "Config/runtime.json"));
    CHECK(fs::is_regular_file(relocated / "arc-distribution.json"));
}

TEST_CASE("D2.1 Windows finalizer rejects corrupt or missing dependency payload without publishing")
{
    directory_fixture fixture;
    project::windows_directory_finalizer finalizer;
    {
        std::ofstream stream(fixture.staging / "bin/physics.dll", std::ios::app);
        stream << "changed";
    }
    auto outcome = project::run_distribution_finalizer(fixture.request(), finalizer);
    REQUIRE_FALSE(outcome);
    CHECK_FALSE(fs::exists(fixture.output));

    fs::remove(fixture.staging / "bin/physics.dll");
    outcome = project::run_distribution_finalizer(fixture.request(), finalizer);
    REQUIRE_FALSE(outcome);
    CHECK_FALSE(fs::exists(fixture.output));
}

TEST_CASE("D2.1 Windows finalizer rejects undeclared staged files and refuses output overwrite")
{
    directory_fixture fixture;
    project::windows_directory_finalizer finalizer;
    fs::create_directories(fixture.staging / "Source");
    {
        std::ofstream stream(fixture.staging / "Source/Game.cpp");
        stream << "authoring leak";
    }
    auto outcome = project::run_distribution_finalizer(fixture.request(), finalizer);
    REQUIRE_FALSE(outcome);
    CHECK(outcome.error().field == "files");
    fs::remove_all(fixture.staging / "Source");
    REQUIRE(project::run_distribution_finalizer(fixture.request(), finalizer));
    outcome = project::run_distribution_finalizer(fixture.request(), finalizer);
    REQUIRE_FALSE(outcome);
    CHECK(outcome.error().field == "outputRoot");
}

TEST_CASE("D2.1 Windows finalizer does not accept unsupported platform or non-shipping configuration")
{
    directory_fixture fixture;
    project::windows_directory_finalizer finalizer;
    fixture.manifest.target.configuration = "Debug";
    fixture.workflow.configuration = "Debug";
    auto outcome = project::run_distribution_finalizer(fixture.request(), finalizer);
    REQUIRE_FALSE(outcome);
    CHECK(outcome.error().field == "target");
    CHECK_FALSE(fs::exists(fixture.output));
}

#include <arc/project/distribution_finalizer.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
const std::string digest(64, 'a');

arc::project::project_descriptor descriptor()
{
    arc::project::project_descriptor project;
    project.guid = "11111111-1111-1111-1111-111111111111";
    project.engine_version = "0.1.0";
    project.target_platforms = {{.id = "windows-x64-vulkan"}};
    project.cook_profiles = {{.id = "windows-x64-vulkan",
                              .platform = "windows",
                              .architecture = "x86_64",
                              .renderer = "vulkan",
                              .configuration = "Shipping"}};
    project.modules = {{.id = "runtime", .kind = arc::project::module_kind::runtime, .target = "Game"}};
    return project;
}

arc::project::package_workflow_plan plan()
{
    arc::project::project_context context;
    context.root = std::filesystem::temp_directory_path() / "arc-project-finalizer";
    context.build_root = context.root / "Build";
    const auto result = arc::project::plan_package_workflow(descriptor(), context, {.profile = "windows-x64-vulkan"});
    REQUIRE(result);
    return result.value();
}

arc::project::distribution_manifest manifest()
{
    arc::project::distribution_manifest m;
    m.project_guid = "11111111-1111-1111-1111-111111111111";
    m.engine_version = "0.1.0";
    m.sdk_version = "0.1.0";
    m.target = {.id = "windows-x64-vulkan",
                .platform = "windows",
                .architecture = "x86_64",
                .renderer = "vulkan",
                .configuration = "Shipping",
                .cook_profile = "windows-x64-vulkan"};
    m.role = "client";
    m.finalizer = "windows-directory";
    m.entrypoint = {.kind = "executable", .value = "bin/Game.exe"};
    m.startup_scenes = {"22222222-2222-2222-2222-222222222222"};
    m.runtime_config = "config/runtime.json";
    m.cook_manifests = {"content/game.arccookmanifest"};
    m.files = {{.path = "bin/Game.exe", .kind = "executable", .size = 12, .sha256 = digest},
               {.path = "config/runtime.json", .kind = "configuration", .size = 8, .sha256 = digest},
               {.path = "content/game.arccookmanifest", .kind = "content", .size = 16, .sha256 = digest}};
    return m;
}

class recording_finalizer final : public arc::project::distribution_finalizer
{
public:
    std::string selected_id{"windows-directory"};
    mutable bool validated{};
    bool called{};
    bool reject{};
    std::string_view id() const noexcept override
    {
        return selected_id;
    }
    arc::project::project_status validate(const arc::project::distribution_finalize_request&) const override
    {
        validated = true;
        if (reject)
            return arc::project::project_status::failure({.code = arc::project::project_error_code::invalid_descriptor,
                                                          .field = "adapter",
                                                          .message = "Native adapter rejected package"});
        return arc::project::project_status::success();
    }
    arc::project::project_status finalize(const arc::project::distribution_finalize_request&) override
    {
        called = true;
        return arc::project::project_status::success();
    }
};
} // namespace

TEST_CASE("distribution dispatch validates then invokes selected platform finalizer")
{
    const auto workflow = plan();
    const auto dist = manifest();
    const auto root = std::filesystem::temp_directory_path();
    recording_finalizer finalizer;
    const arc::project::distribution_finalize_request request{
        .workflow = workflow, .manifest = dist, .staging_root = root / "arc-stage", .output_root = root / "arc-dist"};
    const auto result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE(result);
    CHECK(finalizer.validated);
    CHECK(finalizer.called);
}

TEST_CASE("distribution dispatcher rejects mismatched role target or adapter without side effects")
{
    const auto workflow = plan();
    auto dist = manifest();
    const auto root = std::filesystem::temp_directory_path();
    recording_finalizer finalizer;
    const arc::project::distribution_finalize_request request{
        .workflow = workflow, .manifest = dist, .staging_root = root / "arc-stage", .output_root = root / "arc-dist"};
    dist.role = "server";
    auto result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "target.renderer");
    CHECK_FALSE(finalizer.called);
    dist.role = "client";
    dist.target.architecture = "arm64";
    result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "target");
    dist.target.architecture = "x86_64";
    finalizer.selected_id = "android-gradle";
    result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "finalizer");
    CHECK_FALSE(finalizer.validated);
    CHECK_FALSE(finalizer.called);
}

TEST_CASE("distribution dispatcher propagates finalizer validation failures")
{
    const auto workflow = plan();
    const auto dist = manifest();
    const auto root = std::filesystem::temp_directory_path();
    recording_finalizer finalizer;
    finalizer.reject = true;
    const arc::project::distribution_finalize_request request{
        .workflow = workflow, .manifest = dist, .staging_root = root / "arc-stage", .output_root = root / "arc-dist"};
    const auto result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "adapter");
    CHECK(finalizer.validated);
    CHECK_FALSE(finalizer.called);
}

TEST_CASE("distribution dispatcher rejects missing or aliased staging roots")
{
    const auto workflow = plan();
    const auto dist = manifest();
    const auto root = std::filesystem::temp_directory_path();
    recording_finalizer finalizer;
    arc::project::distribution_finalize_request request{
        .workflow = workflow, .manifest = dist, .staging_root = root / "arc-stage", .output_root = root / "arc-stage"};
    auto result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "paths");
    request.output_root = root / "arc-dist";
    request.staging_root.clear();
    result = arc::project::run_distribution_finalizer(request, finalizer);
    REQUIRE_FALSE(result);
    CHECK_FALSE(finalizer.called);
}

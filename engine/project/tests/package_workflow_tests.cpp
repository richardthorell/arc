#include <arc/project/package_workflow.h>

#include <catch2/catch_test_macros.hpp>

namespace
{
arc::project::project_descriptor descriptor()
{
    arc::project::project_descriptor result;
    result.build_configurations = {"Debug", "Shipping"};
    result.cook_profiles = {
        {.id = "windows-client", .platform = "windows", .renderer = "vulkan", .configuration = "Shipping"},
        {.id = "windows-server", .platform = "windows", .renderer = "none", .configuration = "Shipping"}};
    result.target_platforms = {{.id = "windows-x64-vulkan"}, {.id = "windows-x64-headless"}};
    result.modules = {{.id = "game", .kind = arc::project::module_kind::runtime, .target = "game"},
                      {.id = "server", .kind = arc::project::module_kind::server, .target = "server"}};
    result.package.output = "Build/Packages";
    return result;
}

arc::project::project_context context()
{
    arc::project::project_context result;
    result.root = std::filesystem::path("/project");
    result.build_root = result.root / "Build";
    return result;
}
} // namespace

TEST_CASE("unified workflows use explicit target and profile identity")
{
    using namespace arc::project;
    const auto missing = plan_project_workflow(descriptor(), context(), {});
    REQUIRE_FALSE(missing);
    const auto mismatch = plan_project_workflow(descriptor(), context(),
                                                {.profile = "windows-client", .target_id = "android-arm64-vulkan"});
    REQUIRE_FALSE(mismatch);
    CHECK(mismatch.error().field == "target");
    const auto valid = plan_project_workflow(descriptor(), context(),
                                             {.profile = "windows-client", .target_id = "windows-x64-vulkan"});
    REQUIRE(valid);
    CHECK(valid.value().output.profile == "windows-client");
    CHECK(valid.value().output.configuration == "Shipping");
}

TEST_CASE("unified workflows keep standalone actions separate")
{
    using namespace arc::project;
    using stage = project_workflow_stage;
    const auto check = [](project_workflow_action action, const std::vector<stage>& expected)
    {
        const auto plan =
            plan_project_workflow(descriptor(), context(),
                                  {.action = action, .profile = "windows-client", .target_id = "windows-x64-vulkan"});
        REQUIRE(plan);
        CHECK(plan.value().stages == expected);
    };
    check(project_workflow_action::configure, {stage::validate, stage::configure});
    check(project_workflow_action::build_code, {stage::validate, stage::configure, stage::compile});
    check(project_workflow_action::cook, {stage::validate, stage::cook});
    check(project_workflow_action::build,
          {stage::validate, stage::configure, stage::compile, stage::cook, stage::package});

    auto content_only = descriptor();
    content_only.modules.clear();
    REQUIRE(plan_project_workflow(
        content_only, context(),
        {.action = project_workflow_action::cook, .profile = "windows-client", .target_id = "windows-x64-vulkan"}));
}

TEST_CASE("unified Play snapshots before preparing output and requires development configuration")
{
    using namespace arc::project;
    using stage = project_workflow_stage;
    const project_workflow_request request{.action = project_workflow_action::play,
                                           .profile = "windows-client",
                                           .target_id = "windows-x64-vulkan",
                                           .destination = project_workflow_destination::local,
                                           .session_id = "test-1"};
    REQUIRE_FALSE(plan_project_workflow(descriptor(), context(), request));
    auto project = descriptor();
    project.cook_profiles.front().configuration = "Debug";
    const auto plan = plan_project_workflow(project, context(), request);
    REQUIRE(plan);
    CHECK(plan.value().output.package_output == context().build_root / "PlaySessions/session-test-1/Package");
    CHECK(plan.value().stages == std::vector<stage>{stage::validate, stage::snapshot, stage::configure, stage::compile,
                                                    stage::cook, stage::package, stage::launch, stage::connect});
    const auto run = plan_project_workflow(descriptor(), context(),
                                           {.action = project_workflow_action::run_built,
                                            .profile = "windows-client",
                                            .target_id = "windows-x64-vulkan",
                                            .destination = project_workflow_destination::local});
    REQUIRE(run);
    CHECK(run.value().stages == std::vector<stage>{stage::validate, stage::launch});
}

TEST_CASE("unified workflows reject implicit destinations and build deployment")
{
    using namespace arc::project;
    REQUIRE_FALSE(plan_project_workflow(
        descriptor(), context(),
        {.action = project_workflow_action::play, .profile = "windows-client", .target_id = "windows-x64-vulkan"}));
    REQUIRE_FALSE(plan_project_workflow(descriptor(), context(),
                                        {.profile = "windows-client",
                                         .target_id = "windows-x64-vulkan",
                                         .destination = project_workflow_destination::local}));
    REQUIRE_FALSE(plan_project_workflow(descriptor(), context(),
                                        {.action = project_workflow_action::run_built,
                                         .profile = "windows-client",
                                         .target_id = "windows-x64-vulkan",
                                         .destination = project_workflow_destination::device}));
}

TEST_CASE("unified Android Play deploys only after packaging")
{
    using namespace arc::project;
    auto project = descriptor();
    project.target_platforms = {{.id = "android-arm64-vulkan"}};
    project.cook_profiles = {{.id = "android-dev",
                              .platform = "android",
                              .architecture = "arm64",
                              .renderer = "vulkan",
                              .configuration = "Debug"}};
    const auto plan = plan_project_workflow(project, context(),
                                            {.action = project_workflow_action::play,
                                             .profile = "android-dev",
                                             .target_id = "android-arm64-vulkan",
                                             .destination = project_workflow_destination::device,
                                             .session_id = "test-android"});
    REQUIRE(plan);
    using stage = project_workflow_stage;
    CHECK(plan.value().stages == std::vector<stage>{stage::validate, stage::snapshot, stage::configure, stage::compile,
                                                    stage::cook, stage::package, stage::deploy, stage::launch,
                                                    stage::connect});
}

TEST_CASE("unified Play session paths reject unsafe identifiers")
{
    using namespace arc::project;
    auto project = descriptor();
    project.cook_profiles.front().configuration = "Debug";
    for (const auto* id : {"", "../escape", "a/b", "a\\b", "C:drive", "trailing."})
    {
        const auto plan = plan_project_workflow(project, context(),
                                                {.action = project_workflow_action::play,
                                                 .profile = "windows-client",
                                                 .target_id = "windows-x64-vulkan",
                                                 .destination = project_workflow_destination::local,
                                                 .session_id = id});
        REQUIRE_FALSE(plan);
        CHECK(plan.error().field == "sessionId");
    }
}

TEST_CASE("package workflow resolves deterministic runtime destinations")
{
    const auto plan = arc::project::plan_package_workflow(
        descriptor(), context(),
        {.profile = "windows-client", .action = arc::project::package_workflow_action::package});

    REQUIRE(plan);
    CHECK(plan.value().configuration == "Shipping");
    CHECK(plan.value().target_id == "windows-x64-vulkan");
    CHECK(plan.value().finalizer == "windows-directory");
    CHECK(plan.value().platform == "windows");
    CHECK(plan.value().cook_output == std::filesystem::path("/project/Build/Cooked/windows-client"));
    CHECK(plan.value().package_output == std::filesystem::path("/project/Build/Packages/windows-client/Runtime"));
}

TEST_CASE("package workflow resolves dedicated server role separately")
{
    const auto plan = arc::project::plan_package_workflow(descriptor(), context(),
                                                          {.profile = "windows-server",
                                                           .action = arc::project::package_workflow_action::run,
                                                           .role = arc::project::package_workflow_role::server});

    REQUIRE(plan);
    CHECK(plan.value().package_output == std::filesystem::path("/project/Build/Packages/windows-server/Server"));
}

TEST_CASE("package workflow rejects undeclared profiles")
{
    const auto plan = arc::project::plan_package_workflow(descriptor(), context(), {.profile = "missing"});

    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == arc::project::project_error_code::invalid_descriptor);
    CHECK(plan.error().field == "cookProfiles");
}

TEST_CASE("package workflow requires role modules for package and run")
{
    auto project = descriptor();
    project.modules.erase(project.modules.begin());
    const auto plan = arc::project::plan_package_workflow(project, context(), {.profile = "windows-client"});

    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == arc::project::project_error_code::missing_module);
}

TEST_CASE("dedicated server workflow requires a headless cook profile")
{
    const auto plan = arc::project::plan_package_workflow(
        descriptor(), context(), {.profile = "windows-client", .role = arc::project::package_workflow_role::server});

    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == arc::project::project_error_code::invalid_descriptor);
    CHECK(plan.error().field == "cookProfiles");
}

TEST_CASE("package workflow rejects a disabled target even if a cook profile exists")
{
    auto project = descriptor();
    project.target_platforms.front().enabled = false;
    const auto plan = arc::project::plan_package_workflow(project, context(), {.profile = "windows-client"});
    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == arc::project::project_error_code::unsupported_platform);
    CHECK(plan.error().field == "targetPlatforms");
}

TEST_CASE("package workflow rejects incompatible cook target and platform finalizer")
{
    auto project = descriptor();
    project.cook_profiles.front().renderer = "direct3d12";
    auto plan = arc::project::plan_package_workflow(project, context(), {.profile = "windows-client"});
    REQUIRE_FALSE(plan);
    CHECK(plan.error().code == arc::project::project_error_code::unsupported_platform);

    plan = arc::project::plan_package_workflow(descriptor(), context(),
                                               {.profile = "windows-client", .finalizer = "android-gradle"});
    REQUIRE_FALSE(plan);
    CHECK(plan.error().field == "finalizer");
}

TEST_CASE("package workflow resolves the same Android target metadata without packaging it")
{
    auto project = descriptor();
    project.target_platforms.push_back({.id = "android-arm64-vulkan"});
    project.cook_profiles.push_back({.id = "android-mobile",
                                     .platform = "android",
                                     .architecture = "arm64",
                                     .renderer = "vulkan",
                                     .configuration = "Shipping"});
    const auto plan = arc::project::plan_package_workflow(project, context(), {.profile = "android-mobile"});
    REQUIRE(plan);
    CHECK(plan.value().target_id == "android-arm64-vulkan");
    CHECK(plan.value().profile == "android-mobile");
    CHECK(plan.value().finalizer == "android-gradle");
}

TEST_CASE("package workflow rejects a headless client and unsupported configuration")
{
    auto client = arc::project::plan_package_workflow(descriptor(), context(), {.profile = "windows-server"});
    REQUIRE_FALSE(client);
    CHECK(client.error().field == "cookProfiles");

    auto project = descriptor();
    project.cook_profiles.front().configuration = "Experimental";
    const auto invalid = arc::project::plan_package_workflow(project, context(), {.profile = "windows-client"});
    REQUIRE_FALSE(invalid);
    CHECK(invalid.error().field == "buildConfigurations");
}

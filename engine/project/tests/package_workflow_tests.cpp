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

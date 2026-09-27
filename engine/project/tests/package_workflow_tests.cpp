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

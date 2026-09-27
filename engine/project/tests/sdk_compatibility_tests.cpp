#include <arc/project/sdk_compatibility.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("SDK compatibility accepts an exact project version")
{
    const auto result = arc::project::evaluate_sdk_compatibility("0.8.3", "0.8.3");
    CHECK(result.status == arc::project::sdk_compatibility::compatible);
    CHECK(result.writable);
    CHECK_FALSE(result.upgrade_required);
}

TEST_CASE("SDK compatibility requires explicit upgrade for an older project")
{
    const auto result = arc::project::evaluate_sdk_compatibility("0.8.2", "0.9.0");
    CHECK(result.status == arc::project::sdk_compatibility::upgrade_available);
    CHECK_FALSE(result.writable);
    CHECK(result.upgrade_required);
}

TEST_CASE("SDK compatibility rejects an SDK older than the project")
{
    const auto result = arc::project::evaluate_sdk_compatibility("0.9.1", "0.9.0");
    CHECK(result.status == arc::project::sdk_compatibility::incompatible);
    CHECK_FALSE(result.writable);
    CHECK_FALSE(result.upgrade_required);
}

TEST_CASE("SDK compatibility rejects major-version changes")
{
    const auto result = arc::project::evaluate_sdk_compatibility("1.0.0", "2.0.0");
    CHECK(result.status == arc::project::sdk_compatibility::incompatible);
    CHECK_FALSE(result.writable);
}

TEST_CASE("SDK compatibility accepts common semantic-version decorations")
{
    const auto result = arc::project::evaluate_sdk_compatibility("v0.9.0-dev", "0.9.0+local");
    CHECK(result.status == arc::project::sdk_compatibility::compatible);
    CHECK(result.writable);
}

TEST_CASE("SDK compatibility reports malformed versions")
{
    CHECK(arc::project::evaluate_sdk_compatibility("0.9", "0.9.0").status ==
          arc::project::sdk_compatibility::invalid_version);
    CHECK(arc::project::evaluate_sdk_compatibility("latest", "0.9.0").status ==
          arc::project::sdk_compatibility::invalid_version);
}

#include <arc/render/vulkan/vulkan_backend.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Vulkan loader availability can be queried")
{
    (void)arc::render::vulkan::vulkan_loader_available();
    SUCCEED("Vulkan loader query completed");
}

TEST_CASE("Vulkan backend creation either succeeds or reports a reason")
{
    const auto result = arc::render::vulkan::create_vulkan_backend();
    if (result.succeeded())
    {
        REQUIRE(result.backend->type() == arc::render::render_backend_type::vulkan);
        REQUIRE(result.backend->capabilities().api_major >= 1);
    }
    else
    {
        REQUIRE_FALSE(result.message.empty());
    }
}

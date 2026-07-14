#include <arc/render/vulkan/vulkan_backend.h>
#include <arc/render/renderer.h>

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
        const auto& capabilities = result.backend->capabilities();
        REQUIRE(capabilities.api_major >= 1);
        REQUIRE((capabilities.api_major > 1 || capabilities.api_minor >= 2));
        REQUIRE_FALSE(capabilities.adapter_name.empty());
        REQUIRE(capabilities.graphics_queue);
        REQUIRE(capabilities.compute_queue);
        REQUIRE(capabilities.dynamic_rendering);
        REQUIRE(capabilities.max_color_attachments >= 5);
    }
    else
    {
        REQUIRE_FALSE(result.message.empty());
    }
}

TEST_CASE("Vulkan compatibility override leaves optional device paths disabled")
{
    arc::render::vulkan::vulkan_backend_config config{};
    config.force_disable_optional_features = true;
    auto result = arc::render::vulkan::create_vulkan_backend(config);
    if (result.succeeded())
    {
        arc::render::renderer renderer({ .force_disable_optional_features = true });
        renderer.set_backend(std::move(result.backend));
        REQUIRE_FALSE(renderer.resolved_config().features.synchronization2);
        REQUIRE_FALSE(renderer.resolved_config().features.timeline_semaphores);
        REQUIRE_FALSE(renderer.resolved_config().features.descriptor_indexing);
    }
    else
    {
        REQUIRE_FALSE(result.message.empty());
    }
}

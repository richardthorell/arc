#include <arc/render/renderer.h>
#include <arc/render/renderer_profile.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

namespace
{
constexpr std::uint64_t mebibyte = 1024ull * 1024ull;
constexpr std::uint64_t gibibyte = 1024ull * mebibyte;

arc::render::render_capabilities capable_discrete_adapter()
{
    arc::render::render_capabilities result;
    result.discrete_gpu = true;
    result.dedicated_video_memory = 16ull * gibibyte;
    result.memory_budget = 14ull * gibibyte;
    result.compute_shaders = true;
    result.storage_buffers = true;
    result.storage_images = true;
    result.descriptor_indexing = true;
    result.virtual_geometry_compute = true;
    result.virtual_geometry_indexed = true;
    return result;
}
} // namespace

TEST_CASE("renderer profile document resolves deterministic capability-based layers")
{
    using namespace arc::render;
    constexpr auto source = R"json(
    {
      "format": "arc-renderer-profile",
      "formatVersion": 1,
      "deviceProfiles": [
        {
          "id": "desktop-discrete",
          "priority": 20,
          "match": {
            "gpuClass": "discrete",
            "minimumGpuMemoryMiB": 8192,
            "requires": ["computeShaders", "virtualGeometryIndexed"]
          },
          "settings": {
            "quality": "high",
            "virtualGeometry": {
              "projectedError": 0.7,
              "gpuCacheMiB": 768,
              "requestLimit": 3072,
              "computeCrossoverPixels": 2.0,
              "hardwareCrossoverPixels": 9.0
            }
          }
        },
        {
          "id": "fallback",
          "priority": 1,
          "match": {},
          "settings": { "quality": "low" }
        }
      ],
      "overrides": {
        "virtualGeometry": { "requestLimit": 1024 },
        "textureStreaming": { "gpuBudgetMiB": 1536, "requestLimit": 512 },
        "terrain": { "geometryErrorScale": 0.75 }
      }
    })json";

    const auto parsed = parse_renderer_profile_document(source);
    REQUIRE(parsed);

    renderer_profile_overrides runtime;
    runtime.virtual_geometry_request_limit = 256;
    const arc::framework::platform_capabilities platform{.form_factor = arc::framework::device_form_factor::desktop,
                                                         .logical_processor_count = 16,
                                                         .system_memory_bytes = 32ull * gibibyte};
    const auto resolved = resolve_renderer_profile(parsed.document, runtime, capable_discrete_adapter(), platform);

    CHECK(resolved.device_profile_id == "desktop-discrete");
    CHECK(resolved.quality == render_quality_tier::high);
    CHECK(resolved.profile.geometry_error_threshold == 0.7f);
    CHECK(resolved.profile.virtual_geometry_gpu_budget_bytes == 768ull * mebibyte);
    CHECK(resolved.profile.virtual_geometry_request_limit == 256u);
    CHECK(resolved.profile.texture_gpu_budget_bytes == 1536ull * mebibyte);
    CHECK(resolved.profile.texture_request_limit == 512u);
    CHECK(resolved.profile.terrain_geometry_error_scale == 0.75f);
}

TEST_CASE("renderer profile selection is stable across declaration order")
{
    using namespace arc::render;
    const auto parsed = parse_renderer_profile_document(R"json(
    {
      "deviceProfiles": [
        { "id": "z-profile", "priority": 4, "match": { "gpuClass": "discrete" }, "settings": {} },
        { "id": "a-profile", "priority": 4, "match": { "gpuClass": "discrete" }, "settings": {} }
      ]
    })json");
    REQUIRE(parsed);
    const auto resolved = resolve_renderer_profile(parsed.document, {}, capable_discrete_adapter(), {});
    CHECK(resolved.device_profile_id == "a-profile");
}

TEST_CASE("mobile renderer profiles depend on form factor rather than platform name")
{
    using namespace arc::render;
    const auto parsed = parse_renderer_profile_document(R"json(
    {
      "deviceProfiles": [{
        "id": "handheld",
        "priority": 10,
        "match": { "formFactor": "handheld", "gpuClass": "integrated" },
        "settings": { "quality": "low", "tiers": { "cpu": "constrained" } }
      }]
    })json");
    REQUIRE(parsed);
    render_capabilities adapter;
    adapter.integrated_gpu = true;
    adapter.shared_system_memory = 4ull * gibibyte;
    arc::framework::platform_capabilities windows{.family = arc::framework::platform_family::windows,
                                                  .form_factor = arc::framework::device_form_factor::handheld,
                                                  .logical_processor_count = 8,
                                                  .system_memory_bytes = 8ull * gibibyte};
    auto android = windows;
    android.family = arc::framework::platform_family::android;

    const auto windows_profile = resolve_renderer_profile(parsed.document, {}, adapter, windows);
    const auto android_profile = resolve_renderer_profile(parsed.document, {}, adapter, android);
    CHECK(windows_profile.device_profile_id == "handheld");
    CHECK(android_profile.device_profile_id == windows_profile.device_profile_id);
    CHECK(android_profile.profile.geometry_error_threshold == windows_profile.profile.geometry_error_threshold);
}

TEST_CASE("renderer profile parser accepts current flat project settings")
{
    using namespace arc::render;
    const auto parsed = parse_renderer_profile_document(
        R"json({ "renderer.qualityTier": "high", "renderer.antiAliasing": "fxaa" })json");
    REQUIRE(parsed);
    REQUIRE(parsed.document.project_overrides.quality);
    REQUIRE(parsed.document.project_overrides.anti_aliasing);
    CHECK(*parsed.document.project_overrides.quality == render_quality_tier::high);
    CHECK(*parsed.document.project_overrides.anti_aliasing == anti_aliasing_method::fxaa);
}

TEST_CASE("renderer profile parser rejects unknown capability predicates")
{
    const auto parsed = arc::render::parse_renderer_profile_document(R"json(
    {
      "deviceProfiles": [{
        "id": "invalid",
        "match": { "requires": ["madeUpFeature"] },
        "settings": {}
      }]
    })json");
    REQUIRE_FALSE(parsed);
    REQUIRE(parsed.error);
    CHECK_THAT(parsed.error->field, Catch::Matchers::ContainsSubstring("requires"));
}

TEST_CASE("resolved renderer config exposes profile-controlled streaming budgets")
{
    using namespace arc::render;
    renderer_config config;
    config.quality = render_quality_tier::ultra;
    config.profile_overrides.virtual_geometry_gpu_budget_bytes = 640ull * mebibyte;
    config.profile_overrides.virtual_geometry_cpu_budget_bytes = 320ull * mebibyte;
    config.profile_overrides.virtual_geometry_request_limit = 777u;
    config.profile_overrides.texture_gpu_budget_bytes = 1200ull * mebibyte;
    config.profile_overrides.texture_cpu_budget_bytes = 300ull * mebibyte;
    config.profile_overrides.texture_upload_budget_per_frame = 48ull * mebibyte;
    config.profile_overrides.texture_request_limit = 333u;
    config.profile_overrides.virtual_texture_cache_budget_bytes = 400ull * mebibyte;

    const arc::framework::platform_capabilities platform{.logical_processor_count = 12,
                                                         .system_memory_bytes = 32ull * gibibyte};
    const auto resolved = resolve_render_config(config, capable_discrete_adapter(), platform);
    CHECK(resolved.quality == render_quality_tier::ultra);
    CHECK(resolved.virtual_geometry_gpu_budget_bytes == 640ull * mebibyte);
    CHECK(resolved.virtual_geometry_cpu_budget_bytes == 320ull * mebibyte);
    CHECK(resolved.virtual_geometry_request_limit == 777u);
    CHECK(resolved.texture_gpu_budget_bytes == 1200ull * mebibyte);
    CHECK(resolved.texture_cpu_budget_bytes == 300ull * mebibyte);
    CHECK(resolved.texture_upload_budget_per_frame == 48ull * mebibyte);
    CHECK(resolved.texture_request_limit == 333u);
    CHECK(resolved.virtual_texture_cache_budget_bytes == 400ull * mebibyte);
}

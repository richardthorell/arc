#pragma once

#include <arc/framework/capabilities.h>
#include <arc/render/render_backend.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

inline constexpr std::string_view renderer_profile_format = "arc-renderer-profile";
inline constexpr std::uint32_t renderer_profile_format_version = 1;

/** @brief GPU topology condition used by data-driven device profiles. */
enum class renderer_gpu_class : std::uint8_t
{
    any,
    integrated,
    discrete
};

/**
 * @brief Optional settings layered over an implemented render quality tier.
 *
 * Empty members inherit from the preceding layer. This lets target, project,
 * and runtime policy override individual values without cloning renderer code.
 */
struct renderer_profile_overrides
{
    std::optional<render_quality_tier> quality;
    std::optional<render_path> path;
    std::optional<anti_aliasing_method> anti_aliasing;
    std::optional<render_scalability_tier> cpu_tier;
    std::optional<render_scalability_tier> gpu_tier;
    std::optional<render_scalability_tier> memory_tier;
    std::optional<bool> dynamic_resolution;
    std::optional<float> target_frame_time_ms;
    std::optional<float> minimum_render_scale;
    std::optional<float> maximum_render_scale;
    std::optional<float> geometry_error_threshold;
    std::optional<float> minimum_geometry_error_threshold;
    std::optional<float> maximum_geometry_error_threshold;
    std::optional<std::uint32_t> directional_shadow_cascades;
    std::optional<std::uint32_t> directional_shadow_resolution;
    std::optional<std::uint32_t> local_shadow_atlas_resolution;
    std::optional<float> minimum_shadow_resolution_scale;
    std::optional<float> maximum_shadow_resolution_scale;
    std::optional<float> minimum_volumetric_resolution_scale;
    std::optional<float> maximum_volumetric_resolution_scale;
    std::optional<std::uint64_t> virtual_geometry_gpu_budget_bytes;
    std::optional<std::uint64_t> virtual_geometry_cpu_budget_bytes;
    std::optional<std::uint32_t> virtual_geometry_request_limit;
    std::optional<float> virtual_geometry_compute_crossover_pixels;
    std::optional<float> virtual_geometry_hardware_crossover_pixels;
    std::optional<std::uint64_t> texture_gpu_budget_bytes;
    std::optional<std::uint64_t> texture_cpu_budget_bytes;
    std::optional<std::uint64_t> texture_upload_budget_per_frame;
    std::optional<std::uint32_t> texture_request_limit;
    std::optional<std::uint64_t> virtual_texture_cache_budget_bytes;
    std::optional<float> terrain_geometry_error_scale;
    std::optional<float> post_process_quality;
};

/** @brief Capability predicates for one target/device profile. */
struct renderer_device_profile_match
{
    renderer_gpu_class gpu_class{renderer_gpu_class::any};
    std::optional<framework::device_form_factor> form_factor;
    std::uint32_t minimum_logical_processors{};
    std::uint32_t maximum_logical_processors{};
    std::uint64_t minimum_system_memory_bytes{};
    std::uint64_t maximum_system_memory_bytes{};
    std::uint64_t minimum_gpu_memory_bytes{};
    std::uint64_t maximum_gpu_memory_bytes{};
    /** Backend-neutral feature names; unknown names are rejected while parsing. */
    std::vector<std::string> required_features;
};

/** @brief Named target/device layer selected from capability facts. */
struct renderer_device_profile
{
    std::string id;
    std::int32_t priority{};
    renderer_device_profile_match match;
    renderer_profile_overrides settings;
};

/** @brief Parsed project policy from Config/Renderer.json. */
struct renderer_profile_document
{
    std::string preferred_profile_id;
    std::vector<renderer_device_profile> device_profiles;
    renderer_profile_overrides project_overrides;
};

/** @brief Actionable parse/load failure for renderer profile policy. */
struct renderer_profile_error
{
    std::filesystem::path path;
    std::string field;
    std::string message;
};

/** @brief Result of parsing or loading a renderer profile document. */
struct renderer_profile_document_result
{
    renderer_profile_document document;
    std::optional<renderer_profile_error> error;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return !error.has_value();
    }
};

/** @brief Layered profile state selected before executable feature resolution. */
struct renderer_profile_resolution
{
    std::string device_profile_id{"engine-default"};
    render_quality_tier requested_quality{render_quality_tier::auto_select};
    render_quality_tier quality{render_quality_tier::medium};
    render_quality_profile profile{};
    render_path path{render_path::auto_select};
    anti_aliasing_method anti_aliasing{anti_aliasing_method::auto_select};
    render_scalability_tier cpu_tier{render_scalability_tier::balanced};
    render_scalability_tier gpu_tier{render_scalability_tier::balanced};
    render_scalability_tier memory_tier{render_scalability_tier::balanced};
    bool dynamic_resolution{true};
    std::vector<std::string> diagnostics;
};

/** @brief Parse a renderer profile document without performing filesystem I/O. */
[[nodiscard]] renderer_profile_document_result parse_renderer_profile_document(std::string_view source);

/** @brief Load and parse project renderer policy. A missing file resolves to an empty document. */
[[nodiscard]] renderer_profile_document_result load_renderer_profile_document(const std::filesystem::path& path);

/**
 * @brief Resolve engine defaults, target/device policy, project settings, then runtime overrides.
 */
[[nodiscard]] renderer_profile_resolution resolve_renderer_profile(const renderer_profile_document& document,
                                                                   const renderer_profile_overrides& runtime_overrides,
                                                                   const render_capabilities& capabilities,
                                                                   const framework::platform_capabilities& platform,
                                                                   std::string_view requested_profile_id = {});

} // namespace arc::render

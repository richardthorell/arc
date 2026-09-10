#include <arc/scene/terrain_streaming_prediction.h>

#include <arc/render/renderer.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <utility>

namespace arc::scene
{
namespace
{

constexpr float spatial_epsilon = 1.0e-6f;

float vector_length(const math::vector3f& value) noexcept
{
    return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

math::vector3f add_scaled(const math::vector3f& value, const math::vector3f& direction, float scale) noexcept
{
    return {value[0] + direction[0] * scale, value[1] + direction[1] * scale, value[2] + direction[2] * scale};
}

math::vector3f normalized_or(const math::vector3f& value, const math::vector3f& fallback) noexcept
{
    const auto length = vector_length(value);
    if (length <= spatial_epsilon) return fallback;
    return {value[0] / length, value[1] / length, value[2] / length};
}

float dot(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

float distance(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return vector_length({lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]});
}

struct page_spatial_data
{
    math::vector3f center{};
    float radius{};
    float geometric_error{};
};

bool page_spatial_bounds(const render::virtual_mesh_data& geometry, std::uint32_t page_index,
                         page_spatial_data& output) noexcept
{
    if (page_index >= geometry.pages.size()) return false;
    const auto& page = geometry.pages[page_index];
    if (page.cluster_count == 0u || page.first_cluster >= geometry.clusters.size()) return false;

    const auto end = std::min<std::uint64_t>(static_cast<std::uint64_t>(page.first_cluster) + page.cluster_count,
                                             geometry.clusters.size());
    math::vector3f minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
    math::vector3f maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest()};
    float maximum_error{};
    for (std::uint64_t cluster_index = page.first_cluster; cluster_index < end; ++cluster_index)
    {
        const auto& cluster = geometry.clusters[static_cast<std::size_t>(cluster_index)];
        for (std::size_t axis = 0; axis < 3u; ++axis)
        {
            minimum[axis] = std::min(minimum[axis], cluster.bounds_min[axis]);
            maximum[axis] = std::max(maximum[axis], cluster.bounds_max[axis]);
        }
        maximum_error = std::max(maximum_error, cluster.geometric_error);
    }

    output.center = {(minimum[0] + maximum[0]) * 0.5f, (minimum[1] + maximum[1]) * 0.5f,
                     (minimum[2] + maximum[2]) * 0.5f};
    output.radius = distance(minimum, maximum) * 0.5f;
    output.geometric_error = maximum_error;
    return true;
}

float distance_to_page(const math::vector3f& point, const page_spatial_data& page) noexcept
{
    return std::max(0.0f, distance(point, page.center) - page.radius);
}

} // namespace

terrain_streaming_predictor::terrain_streaming_predictor(terrain_streaming_prediction_config config)
{
    configure(config);
}

void terrain_streaming_predictor::configure(terrain_streaming_prediction_config config) noexcept
{
    if (!std::isfinite(config.prediction_horizon_seconds)) config.prediction_horizon_seconds = 1.25f;
    if (!std::isfinite(config.prefetch_distance)) config.prefetch_distance = 768.0f;
    if (!std::isfinite(config.hysteresis_distance)) config.hysteresis_distance = 128.0f;
    if (!std::isfinite(config.forward_bias)) config.forward_bias = 0.75f;
    if (!std::isfinite(config.streaming_importance)) config.streaming_importance = 1.0f;

    config.prediction_horizon_seconds = std::clamp(config.prediction_horizon_seconds, 0.0f, 5.0f);
    config.prefetch_distance = std::max(0.0f, config.prefetch_distance);
    config.hysteresis_distance = std::max(0.0f, config.hysteresis_distance);
    config.forward_bias = std::clamp(config.forward_bias, 0.0f, 4.0f);
    config.streaming_importance = std::clamp(config.streaming_importance, 0.01f, 16.0f);
    config_ = config;
}

void terrain_streaming_predictor::reset() noexcept
{
    active_pages_.clear();
}

terrain_streaming_prediction_result terrain_streaming_predictor::update(const terrain_render_proxy& proxy,
                                                                        render::renderer& renderer,
                                                                        const terrain_streaming_prediction_view& view)
{
    terrain_streaming_prediction_result result;
    if (view.camera_cut) active_pages_.clear();

    const auto speed = vector_length(view.camera_velocity);
    const auto horizon = view.camera_cut ? 0.0f : config_.prediction_horizon_seconds;
    result.predicted_camera_position = add_scaled(view.camera_position, view.camera_velocity, horizon);
    if (config_.maximum_prefetch_pages == 0u || config_.prefetch_distance <= 0.0f) return result;

    const auto fallback_forward = speed > spatial_epsilon ? normalized_or(view.camera_velocity, {0.0f, 0.0f, -1.0f})
                                                          : math::vector3f{0.0f, 0.0f, -1.0f};
    const auto forward = normalized_or(view.camera_forward, fallback_forward);
    const auto travel_distance = speed * horizon;

    struct candidate
    {
        active_page key;
        float rank{};
        float distance{};
        float proximity{};
        float approach{};
        float heading{};
        float geometric_error{};
        bool hysteresis{};
    };
    std::vector<candidate> candidates;

    auto& residency = renderer.virtual_geometry_residency();
    for (const auto& region : proxy.regions)
    {
        if (!region.geometry.virtualized.valid()) continue;
        const auto resource = region.geometry.virtualized;
        const auto generation = renderer.virtual_mesh_content_generation(resource);
        const auto* geometry = renderer.virtual_mesh_data_for(resource);
        if (!geometry || generation == 0u) continue;

        for (std::uint32_t page_index = 0; page_index < geometry->pages.size(); ++page_index)
        {
            const auto& descriptor = geometry->pages[page_index];
            if (descriptor.root || residency.resident(resource, generation, page_index)) continue;
            ++result.considered_pages;

            page_spatial_data page;
            if (!page_spatial_bounds(*geometry, page_index, page)) continue;

            const active_page key{.resource = resource, .resource_generation = generation, .page_index = page_index};
            const bool was_active = std::find(active_pages_.begin(), active_pages_.end(), key) != active_pages_.end();
            const auto current_distance = distance_to_page(view.camera_position, page);
            const auto predicted_distance = distance_to_page(result.predicted_camera_position, page);
            const auto nearest_distance = std::min(current_distance, predicted_distance);
            const auto selection_distance =
                config_.prefetch_distance + (was_active ? config_.hysteresis_distance : 0.0f);
            if (nearest_distance > selection_distance) continue;

            const math::vector3f to_page{page.center[0] - view.camera_position[0],
                                         page.center[1] - view.camera_position[1],
                                         page.center[2] - view.camera_position[2]};
            const auto page_direction = normalized_or(to_page, forward);
            const auto heading = std::max(0.0f, dot(forward, page_direction));
            const auto approach_distance = std::max(0.0f, current_distance - predicted_distance);
            const auto approach =
                travel_distance > spatial_epsilon ? std::clamp(approach_distance / travel_distance, 0.0f, 1.0f) : 0.0f;
            const auto proximity = selection_distance > spatial_epsilon
                                       ? std::clamp(1.0f - nearest_distance / selection_distance, 0.0f, 1.0f)
                                       : 1.0f;
            const auto error_weight = std::clamp(page.geometric_error / std::max(nearest_distance, 1.0f), 0.0f, 2.0f);
            const bool retained_by_hysteresis = was_active && nearest_distance > config_.prefetch_distance;
            const auto rank = proximity * 4.0f + approach * 3.0f + heading * config_.forward_bias + error_weight * 0.5f;
            candidates.push_back({.key = key,
                                  .rank = rank,
                                  .distance = nearest_distance,
                                  .proximity = proximity,
                                  .approach = approach,
                                  .heading = heading,
                                  .geometric_error = page.geometric_error,
                                  .hysteresis = retained_by_hysteresis});
        }
    }

    result.candidate_pages = static_cast<std::uint32_t>(candidates.size());
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const auto& lhs, const auto& rhs)
                     {
                         if (lhs.rank != rhs.rank) return lhs.rank > rhs.rank;
                         if (lhs.distance != rhs.distance) return lhs.distance < rhs.distance;
                         if (lhs.key.resource.index != rhs.key.resource.index)
                             return lhs.key.resource.index < rhs.key.resource.index;
                         return lhs.key.page_index < rhs.key.page_index;
                     });
    if (candidates.size() > config_.maximum_prefetch_pages) candidates.resize(config_.maximum_prefetch_pages);

    std::vector<render::virtual_geometry_page_request> requests;
    requests.reserve(candidates.size());
    std::vector<active_page> next_active;
    next_active.reserve(candidates.size());
    for (const auto& candidate : candidates)
    {
        const auto importance = config_.streaming_importance;
        requests.push_back(
            {.resource = candidate.key.resource,
             .resource_generation = candidate.key.resource_generation,
             .page_index = candidate.key.page_index,
             .projected_error = std::max(candidate.geometric_error, 0.25f) * (1.0f + candidate.approach) * importance,
             .screen_coverage = std::clamp(candidate.proximity + candidate.heading * 0.25f, 0.0f, 1.0f) * importance,
             .distance = candidate.distance,
             .visible_child = false,
             .shadow_view = false});
        next_active.push_back(candidate.key);
        if (candidate.hysteresis) ++result.hysteresis_pages;
    }

    if (!requests.empty()) residency.request(requests);
    result.requested_pages = static_cast<std::uint32_t>(requests.size());
    active_pages_ = std::move(next_active);
    return result;
}

} // namespace arc::scene

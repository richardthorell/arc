#pragma once

#include <arc/scene/terrain.h>

#include <cstdint>
#include <vector>

namespace arc::render
{
class renderer;
}

namespace arc::scene
{

/** @brief Tunable M2.4 policy for proactive terrain virtual-geometry requests. */
struct terrain_streaming_prediction_config
{
    float prediction_horizon_seconds{1.25f};
    float prefetch_distance{768.0f};
    float hysteresis_distance{128.0f};
    float forward_bias{0.75f};
    float streaming_importance{1.0f};
    std::uint32_t maximum_prefetch_pages{64u};
};

/**
 * @brief Camera state expressed in the same local space as the terrain render proxy.
 *
 * World-space callers should transform camera position, velocity, and forward into terrain-local space before
 * invoking the predictor. Keeping the policy local-space makes it independent of scene transform ownership.
 */
struct terrain_streaming_prediction_view
{
    math::vector3f camera_position{};
    math::vector3f camera_velocity{};
    math::vector3f camera_forward{0.0f, 0.0f, -1.0f};
    bool camera_cut{};
};

/** @brief Diagnostics from one predictive terrain streaming update. */
struct [[nodiscard]] terrain_streaming_prediction_result
{
    math::vector3f predicted_camera_position{};
    std::uint32_t considered_pages{};
    std::uint32_t candidate_pages{};
    std::uint32_t requested_pages{};
    std::uint32_t hysteresis_pages{};
};

/**
 * @brief Terrain-specific spatial policy layered over the generic virtual-geometry residency manager.
 *
 * The predictor never owns residency or IO. It only contributes bounded prefetch requests for non-root pages based on
 * current/predicted camera position, motion direction, page error and recent selection hysteresis. Normal GPU feedback
 * continues to provide visible demand and therefore remains the authoritative correctness path.
 */
class terrain_streaming_predictor
{
public:
    explicit terrain_streaming_predictor(terrain_streaming_prediction_config config = {});

    void configure(terrain_streaming_prediction_config config) noexcept;
    void reset() noexcept;

    [[nodiscard]] terrain_streaming_prediction_result
    update(const terrain_render_proxy& proxy, render::renderer& renderer,
           const terrain_streaming_prediction_view& view);

private:
    struct active_page
    {
        render::virtual_mesh_handle resource{};
        std::uint32_t resource_generation{};
        std::uint32_t page_index{};

        friend constexpr bool operator==(const active_page&, const active_page&) noexcept = default;
    };

    terrain_streaming_prediction_config config_{};
    std::vector<active_page> active_pages_;
};

} // namespace arc::scene

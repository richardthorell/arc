#pragma once

#include <arc/math/vector.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace arc::navigation
{

using navigation_layer = std::uint32_t;
inline constexpr navigation_layer all_layers = ~navigation_layer{0};

struct tile_coordinate
{
    std::int32_t x = 0;
    std::int32_t z = 0;

    [[nodiscard]] friend constexpr bool operator==(const tile_coordinate&, const tile_coordinate&) noexcept = default;
};

struct agent_definition
{
    float radius = 0.5F;
    float height = 1.8F;
    float max_step_height = 0.4F;
    float max_slope_degrees = 45.0F;
    navigation_layer layer_mask = all_layers;
};

struct navigation_bounds
{
    math::vector<float, 3> minimum{};
    math::vector<float, 3> maximum{};
};

struct path_query
{
    math::vector<float, 3> start{};
    math::vector<float, 3> goal{};
    agent_definition agent{};
    float max_path_length = std::numeric_limits<float>::infinity();
    bool allow_partial_path = false;
};

enum class path_status : std::uint8_t
{
    complete,
    partial,
    unreachable
};

struct path_result
{
    path_status status = path_status::unreachable;
    std::vector<math::vector<float, 3>> points{};
    float length = 0.0F;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_agent,
    invalid_bounds,
    invalid_query
};

[[nodiscard]] validation_error validate(const agent_definition& agent) noexcept;
[[nodiscard]] validation_error validate(const navigation_bounds& bounds) noexcept;
[[nodiscard]] validation_error validate(const path_query& query) noexcept;

} // namespace arc::navigation

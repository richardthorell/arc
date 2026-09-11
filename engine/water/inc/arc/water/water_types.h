#pragma once

#include <arc/assets/assets.h>
#include <arc/math/math.h>

#include <compare>
#include <cstdint>
#include <limits>
#include <string>

namespace arc::water
{

enum class water_body_type : std::uint8_t
{
    ocean,
    lake,
    river
};

enum class water_quality : std::uint8_t
{
    low,
    medium,
    high,
    ultra
};

struct water_body_handle
{
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t index{invalid_index};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != invalid_index;
    }

    friend constexpr auto operator<=>(const water_body_handle&, const water_body_handle&) noexcept = default;
};

/** @brief Horizontal ownership bounds. Oceans are represented by `bounded == false`. */
struct water_body_bounds
{
    float minimum_x{};
    float minimum_z{};
    float maximum_x{};
    float maximum_z{};
    bool bounded{};

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool contains(float world_x, float world_z) const noexcept;
    [[nodiscard]] float area() const noexcept;
};

struct water_simulation_settings
{
    float wind_speed{12.0f};
    math::vector2f wind_direction{1.0f, 0.0f};
    float fetch_length{50000.0f};
    float wave_amplitude{1.0f};
    float choppiness{1.0f};
    std::uint64_t seed{1};
};

struct water_foam_settings
{
    bool enabled{true};
    float threshold{0.55f};
    float decay{0.4f};
};

struct water_appearance_settings
{
    math::vector3f absorption{0.18f, 0.065f, 0.025f};
    math::vector3f scattering{0.02f, 0.075f, 0.10f};
    float roughness{0.08f};
    float refraction_strength{0.04f};
};

/** @brief Resolved authoring settings. It intentionally owns no GPU or frame-local resources. */
struct water_runtime_settings
{
    water_simulation_settings simulation;
    water_foam_settings foam;
    water_appearance_settings appearance;
    water_quality quality{water_quality::high};
};

struct water_body_descriptor
{
    water_body_type type{water_body_type::ocean};
    assets::asset_reference preset;
    water_body_bounds bounds;
    water_runtime_settings settings;
    float water_level{};
    float visible_distance{20000.0f};
    std::int32_t priority{};
    bool enabled{true};
    bool follow_camera{true};
    bool queries_enabled{true};
    std::string label;
};

struct water_surface_sample
{
    water_body_handle body;
    float height{};
    math::vector3f normal{0.0f, 1.0f, 0.0f};
    math::vector3f velocity{};
    float foam{};
};

} // namespace arc::water

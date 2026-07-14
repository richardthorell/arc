#pragma once

#include <arc/math/math.h>
#include <arc/render/render.h>

#include <chrono>
#include <cstdint>

namespace arc::editor::defaults
{

inline constexpr math::vector3f fallback_mesh_bounds_min{ -0.5f, -0.5f, -0.5f };
inline constexpr math::vector3f fallback_mesh_bounds_max{ 0.5f, 0.5f, 0.5f };
inline constexpr float fallback_mesh_radius = 1.0f;
inline constexpr float imported_mesh_fit_size = 1.8f;

inline constexpr math::vector3f default_camera_position{ 0.0f, 0.0f, 4.0f };

inline constexpr math::vector3f default_sun_rotation_degrees{ -50.0f, -35.0f, 0.0f };
inline constexpr math::vector3f default_sun_color{ 1.0f, 1.0f, 1.0f };
inline constexpr float default_sun_intensity = 1.8f;
inline constexpr std::uint32_t default_sun_shadow_resolution = 4096;
inline constexpr render::shadow_filter default_sun_shadow_filter = render::shadow_filter::pcf_3x3;
inline constexpr float default_sun_shadow_bias = 0.0008f;
inline constexpr float default_sun_shadow_normal_bias = 0.003f;

inline constexpr std::chrono::milliseconds native_viewport_frame_interval{ 16 };
inline constexpr int native_viewport_min_dimension = 1;
inline constexpr int viewport_modifier_key_down_mask = 0x8000;

} // namespace arc::editor::defaults

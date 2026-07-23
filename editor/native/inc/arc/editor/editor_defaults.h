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

inline constexpr math::vector3f default_camera_position{ 30.0f, 27.0f, 48.0f };
inline constexpr math::vector3f default_camera_focus{ 0.0f, 7.0f, -15.0f };
inline constexpr float default_camera_focus_radius = 22.0f;
inline constexpr float default_camera_orbit_x = -55.0f;
inline constexpr float default_camera_orbit_y = 35.0f;

inline constexpr float default_terrain_size = 180.0f;
inline constexpr std::uint32_t default_terrain_subdivisions = 256;
inline constexpr float default_terrain_height_scale = 28.0f;
inline constexpr float default_water_size = 30.0f;
inline constexpr math::vector3f default_water_position{ 15.0f, -1.1f, 15.0f };
inline constexpr math::vector3f default_grass_position{ -13.0f, 0.0f, 15.0f };

inline constexpr math::vector3f default_sun_rotation_degrees{ -50.0f, -35.0f, 0.0f };
inline constexpr math::vector3f default_sun_color{ 1.0f, 0.91f, 0.78f };
inline constexpr float default_sun_intensity = 3.2f;
inline constexpr std::uint32_t default_sun_shadow_resolution = 4096;
inline constexpr render::shadow_filter default_sun_shadow_filter = render::shadow_filter::pcf_3x3;
inline constexpr float default_sun_shadow_bias = 0.0008f;
inline constexpr float default_sun_shadow_normal_bias = 0.003f;

inline constexpr std::chrono::milliseconds native_viewport_frame_interval{ 16 };
inline constexpr int native_viewport_min_dimension = 1;
inline constexpr int viewport_modifier_key_down_mask = 0x8000;
inline constexpr int viewport_click_movement_threshold = 3;
inline constexpr std::uint64_t viewport_pick_fallback_frame_count = 4;

} // namespace arc::editor::defaults

#pragma once

#include <arc/math/quaternion.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <string>
#include <vector>

namespace arc::animation
{

using joint_index = std::uint32_t;
inline constexpr joint_index invalid_joint_index = ~joint_index{0};

struct joint_transform
{
    math::vector<float, 3> translation{};
    math::quaternion<float> rotation{};
    math::vector<float, 3> scale{1.0F, 1.0F, 1.0F};
};

struct skeleton_joint
{
    std::string name;
    joint_index parent = invalid_joint_index;
    joint_transform bind_pose{};
};

struct skeleton_definition
{
    std::vector<skeleton_joint> joints;
};

template <typename T>
struct keyframe
{
    float time_seconds = 0.0F;
    T value{};
};

struct joint_track
{
    joint_index joint = invalid_joint_index;
    std::vector<keyframe<math::vector<float, 3>>> translations;
    std::vector<keyframe<math::quaternion<float>>> rotations;
    std::vector<keyframe<math::vector<float, 3>>> scales;
};

struct animation_clip_definition
{
    std::string name;
    float duration_seconds = 0.0F;
    bool looping = false;
    std::vector<joint_track> tracks;
};

enum class validation_error : std::uint8_t
{
    none,
    empty_skeleton,
    invalid_joint_name,
    invalid_parent,
    invalid_bind_pose,
    invalid_duration,
    invalid_track_joint,
    duplicate_track,
    invalid_keyframe_time,
    unordered_keyframes,
    invalid_keyframe_value
};

[[nodiscard]] validation_error validate(const skeleton_definition& skeleton) noexcept;
[[nodiscard]] validation_error validate(const animation_clip_definition& clip,
                                        const skeleton_definition& skeleton) noexcept;

} // namespace arc::animation

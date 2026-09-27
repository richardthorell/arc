#include <arc/animation/animation.h>

#include <cmath>
#include <unordered_set>

namespace arc::animation
{
namespace
{

template <typename T, std::size_t N> bool finite_vector(const math::vector<T, N>& value) noexcept
{
    for (std::size_t i = 0; i < N; ++i)
        if (!std::isfinite(value[i])) return false;
    return true;
}

bool finite_quaternion(const math::quaternion<float>& value) noexcept
{
    for (std::size_t i = 0; i < 4; ++i)
        if (!std::isfinite(value[i])) return false;
    return math::length_squared(value) > 0.0F;
}

bool valid_transform(const joint_transform& transform) noexcept
{
    return finite_vector(transform.translation) && finite_quaternion(transform.rotation) &&
           finite_vector(transform.scale) && transform.scale[0] > 0.0F && transform.scale[1] > 0.0F &&
           transform.scale[2] > 0.0F;
}

template <typename T, typename ValueValidator>
validation_error validate_keys(const std::vector<keyframe<T>>& keys, float duration,
                               ValueValidator valid_value) noexcept
{
    float previous = -1.0F;
    for (const auto& key : keys)
    {
        if (!std::isfinite(key.time_seconds) || key.time_seconds < 0.0F || key.time_seconds > duration)
            return validation_error::invalid_keyframe_time;
        if (key.time_seconds < previous) return validation_error::unordered_keyframes;
        if (!valid_value(key.value)) return validation_error::invalid_keyframe_value;
        previous = key.time_seconds;
    }
    return validation_error::none;
}

} // namespace

validation_error validate(const skeleton_definition& skeleton) noexcept
{
    if (skeleton.joints.empty()) return validation_error::empty_skeleton;

    std::unordered_set<std::string> names;
    names.reserve(skeleton.joints.size());
    for (joint_index index = 0; index < skeleton.joints.size(); ++index)
    {
        const auto& joint = skeleton.joints[index];
        if (joint.name.empty() || !names.insert(joint.name).second) return validation_error::invalid_joint_name;
        if (joint.parent != invalid_joint_index && joint.parent >= index) return validation_error::invalid_parent;
        if (!valid_transform(joint.bind_pose)) return validation_error::invalid_bind_pose;
    }
    return validation_error::none;
}

validation_error validate(const animation_clip_definition& clip, const skeleton_definition& skeleton) noexcept
{
    const auto skeleton_error = validate(skeleton);
    if (skeleton_error != validation_error::none) return skeleton_error;
    if (!std::isfinite(clip.duration_seconds) || clip.duration_seconds <= 0.0F)
        return validation_error::invalid_duration;

    std::unordered_set<joint_index> animated_joints;
    animated_joints.reserve(clip.tracks.size());
    for (const auto& track : clip.tracks)
    {
        if (track.joint >= skeleton.joints.size()) return validation_error::invalid_track_joint;
        if (!animated_joints.insert(track.joint).second) return validation_error::duplicate_track;

        auto error = validate_keys(track.translations, clip.duration_seconds,
                                   [](const auto& value) { return finite_vector(value); });
        if (error != validation_error::none) return error;
        error = validate_keys(track.rotations, clip.duration_seconds,
                              [](const auto& value) { return finite_quaternion(value); });
        if (error != validation_error::none) return error;
        error =
            validate_keys(track.scales, clip.duration_seconds, [](const auto& value)
                          { return finite_vector(value) && value[0] > 0.0F && value[1] > 0.0F && value[2] > 0.0F; });
        if (error != validation_error::none) return error;
    }
    return validation_error::none;
}

} // namespace arc::animation

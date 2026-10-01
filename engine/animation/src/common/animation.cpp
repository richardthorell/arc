#include <arc/animation/animation.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

bool finite_quaternion(const math::quaternionf& value) noexcept
{
    for (std::size_t i = 0; i < 4; ++i)
        if (!std::isfinite(value[i])) return false;
    return math::length_squared(value) > 0.0F;
}

bool finite_matrix(const math::matrix4f& value) noexcept
{
    for (std::size_t row = 0; row < 4; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            if (!std::isfinite(value(row, column))) return false;
    return true;
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

math::vector3f lerp_vector(const math::vector3f& lhs, const math::vector3f& rhs, float alpha) noexcept
{
    return {lhs[0] + (rhs[0] - lhs[0]) * alpha, lhs[1] + (rhs[1] - lhs[1]) * alpha, lhs[2] + (rhs[2] - lhs[2]) * alpha};
}

math::quaternionf nlerp_quaternion(const math::quaternionf& lhs, const math::quaternionf& rhs, float alpha) noexcept
{
    float dot{};
    for (std::size_t i = 0; i < 4; ++i)
        dot += lhs[i] * rhs[i];

    const float sign = dot < 0.0F ? -1.0F : 1.0F;
    math::quaternionf result{lhs[0] + (rhs[0] * sign - lhs[0]) * alpha, lhs[1] + (rhs[1] * sign - lhs[1]) * alpha,
                             lhs[2] + (rhs[2] * sign - lhs[2]) * alpha, lhs[3] + (rhs[3] * sign - lhs[3]) * alpha};
    return math::normalize(result);
}

template <typename T, typename Interpolator>
T sample_channel(const std::vector<keyframe<T>>& keys, float time_seconds, const T& fallback,
                 Interpolator interpolate) noexcept
{
    if (keys.empty()) return fallback;
    if (time_seconds <= keys.front().time_seconds) return keys.front().value;
    if (time_seconds >= keys.back().time_seconds) return keys.back().value;

    const auto upper = std::upper_bound(keys.begin(), keys.end(), time_seconds,
                                        [](float time, const keyframe<T>& key) { return time < key.time_seconds; });
    const auto lower = upper - 1;
    const float span = upper->time_seconds - lower->time_seconds;
    if (span <= std::numeric_limits<float>::epsilon()) return upper->value;

    const float alpha = (time_seconds - lower->time_seconds) / span;
    return interpolate(lower->value, upper->value, alpha);
}

math::quaternionf combine_rotation(const math::quaternionf& parent, const math::quaternionf& local) noexcept
{
    const float px = parent[0];
    const float py = parent[1];
    const float pz = parent[2];
    const float pw = parent[3];
    const float lx = local[0];
    const float ly = local[1];
    const float lz = local[2];
    const float lw = local[3];

    return math::normalize(
        math::quaternionf{pw * lx + px * lw + py * lz - pz * ly, pw * ly - px * lz + py * lw + pz * lx,
                          pw * lz + px * ly - py * lx + pz * lw, pw * lw - px * lx - py * ly - pz * lz});
}

joint_transform combine_transform(const joint_transform& parent, const joint_transform& local) noexcept
{
    math::vector3f scaled_local{parent.scale[0] * local.translation[0], parent.scale[1] * local.translation[1],
                                parent.scale[2] * local.translation[2]};
    const math::vector3f rotated_local{math::rotate(parent.rotation, scaled_local)};

    joint_transform result{};
    result.translation = {parent.translation[0] + rotated_local[0], parent.translation[1] + rotated_local[1],
                          parent.translation[2] + rotated_local[2]};
    result.rotation = combine_rotation(parent.rotation, local.rotation);
    result.scale = {parent.scale[0] * local.scale[0], parent.scale[1] * local.scale[1],
                    parent.scale[2] * local.scale[2]};
    return result;
}

void build_model_transforms(const skeleton_definition& skeleton, animation_pose& pose) noexcept
{
    pose.model_transforms.resize(skeleton.joints.size());
    for (std::size_t index = 0; index < skeleton.joints.size(); ++index)
    {
        const auto parent = skeleton.joints[index].parent;
        if (parent == invalid_joint_index)
            pose.model_transforms[index] = pose.local_transforms[index];
        else
            pose.model_transforms[index] =
                combine_transform(pose.model_transforms[parent], pose.local_transforms[index]);
    }
}

bool sample_pose_validated(const animation_clip_definition& clip, const skeleton_definition& skeleton,
                           float time_seconds, animation_pose& output) noexcept
{
    const float resolved_time = resolve_sample_time(time_seconds, clip.duration_seconds, clip.looping);

    output.local_transforms.resize(skeleton.joints.size());
    for (std::size_t index = 0; index < skeleton.joints.size(); ++index)
        output.local_transforms[index] = skeleton.joints[index].bind_pose;

    for (const auto& track : clip.tracks)
    {
        auto& transform = output.local_transforms[track.joint];
        transform.translation = sample_channel(track.translations, resolved_time, transform.translation, lerp_vector);
        transform.rotation = sample_channel(track.rotations, resolved_time, transform.rotation, nlerp_quaternion);
        transform.scale = sample_channel(track.scales, resolved_time, transform.scale, lerp_vector);
    }

    build_model_transforms(skeleton, output);
    return true;
}

struct clip_record
{
    skeleton_id skeleton{};
    animation_clip_definition definition;
};

struct instance_record
{
    skeleton_id skeleton{};
    animation_clip_id clip{};
    float time_seconds{};
    std::array<animation_pose, 2> poses;
    std::atomic<std::uint32_t> published_index{0};
};

const animation_pose* published_pose(const instance_record& instance) noexcept
{
    const auto index = instance.published_index.load(std::memory_order_acquire);
    return &instance.poses[index];
}

animation_pose& writable_pose(instance_record& instance) noexcept
{
    const auto published = instance.published_index.load(std::memory_order_relaxed);
    return instance.poses[published == 0 ? 1 : 0];
}

void publish_pose(instance_record& instance, animation_pose& pose) noexcept
{
    const auto current = instance.published_index.load(std::memory_order_relaxed);
    const auto next = current == 0 ? 1U : 0U;
    pose.generation = instance.poses[current].generation + 1;
    instance.published_index.store(next, std::memory_order_release);
}

} // namespace

struct animation_runtime::state
{
    std::unordered_map<std::uint64_t, skeleton_definition> skeletons;
    std::unordered_map<std::uint64_t, clip_record> clips;
    std::unordered_map<std::uint64_t, std::unique_ptr<instance_record>> instances;
    std::uint64_t next_skeleton_id{1};
    std::uint64_t next_clip_id{1};
    std::uint64_t next_instance_id{1};
};

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

    if (!skeleton.inverse_bind_matrices.empty() && skeleton.inverse_bind_matrices.size() != skeleton.joints.size())
        return validation_error::invalid_inverse_bind_count;
    for (const auto& matrix : skeleton.inverse_bind_matrices)
        if (!finite_matrix(matrix)) return validation_error::invalid_inverse_bind_matrix;

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

std::string_view validation_error_message(validation_error error) noexcept
{
    switch (error)
    {
        case validation_error::none:
            return "none";
        case validation_error::empty_skeleton:
            return "skeleton contains no joints";
        case validation_error::invalid_joint_name:
            return "joint names must be non-empty and unique";
        case validation_error::invalid_parent:
            return "joint parent must precede the child in deterministic hierarchy order";
        case validation_error::invalid_bind_pose:
            return "bind pose contains an invalid transform";
        case validation_error::invalid_duration:
            return "clip duration must be finite and greater than zero";
        case validation_error::invalid_track_joint:
            return "clip track references an unknown joint";
        case validation_error::duplicate_track:
            return "clip contains more than one track for the same joint";
        case validation_error::invalid_keyframe_time:
            return "keyframe time is outside the clip range or is not finite";
        case validation_error::unordered_keyframes:
            return "keyframes must be ordered by ascending time";
        case validation_error::invalid_keyframe_value:
            return "keyframe contains an invalid transform value";
        case validation_error::invalid_inverse_bind_count:
            return "inverse-bind matrices must be empty or contain exactly one entry per joint";
        case validation_error::invalid_inverse_bind_matrix:
            return "inverse-bind matrix contains a non-finite value";
    }
    return "unknown animation validation error";
}

float resolve_sample_time(float time_seconds, float duration_seconds, bool looping) noexcept
{
    if (!std::isfinite(time_seconds) || !std::isfinite(duration_seconds) || duration_seconds <= 0.0F) return 0.0F;

    if (!looping) return std::clamp(time_seconds, 0.0F, duration_seconds);

    float resolved = std::fmod(time_seconds, duration_seconds);
    if (resolved < 0.0F) resolved += duration_seconds;
    return resolved;
}

float normalized_time(float time_seconds, float duration_seconds, bool looping) noexcept
{
    if (!std::isfinite(duration_seconds) || duration_seconds <= 0.0F) return 0.0F;
    return resolve_sample_time(time_seconds, duration_seconds, looping) / duration_seconds;
}

bool initialize_bind_pose(const skeleton_definition& skeleton, animation_pose& output) noexcept
{
    if (validate(skeleton) != validation_error::none) return false;

    output.local_transforms.resize(skeleton.joints.size());
    for (std::size_t index = 0; index < skeleton.joints.size(); ++index)
        output.local_transforms[index] = skeleton.joints[index].bind_pose;
    build_model_transforms(skeleton, output);
    return true;
}

bool sample_pose(const animation_clip_definition& clip, const skeleton_definition& skeleton, float time_seconds,
                 animation_pose& output) noexcept
{
    if (validate(clip, skeleton) != validation_error::none || !std::isfinite(time_seconds)) return false;
    return sample_pose_validated(clip, skeleton, time_seconds, output);
}

animation_runtime::animation_runtime() : state_(std::make_unique<state>()) {}

animation_runtime::~animation_runtime() = default;

animation_runtime::animation_runtime(animation_runtime&&) noexcept = default;

animation_runtime& animation_runtime::operator=(animation_runtime&&) noexcept = default;

skeleton_id animation_runtime::add_skeleton(skeleton_definition skeleton)
{
    if (validate(skeleton) != validation_error::none) return {};

    const skeleton_id id{state_->next_skeleton_id++};
    state_->skeletons.emplace(id.value, std::move(skeleton));
    return id;
}

bool animation_runtime::remove_skeleton(skeleton_id skeleton) noexcept
{
    if (!skeleton) return false;

    for (const auto& entry : state_->clips)
        if (entry.second.skeleton == skeleton) return false;
    for (const auto& entry : state_->instances)
        if (entry.second->skeleton == skeleton) return false;

    return state_->skeletons.erase(skeleton.value) != 0;
}

animation_clip_id animation_runtime::add_clip(skeleton_id skeleton, animation_clip_definition clip)
{
    const auto skeleton_iterator = state_->skeletons.find(skeleton.value);
    if (skeleton_iterator == state_->skeletons.end()) return {};
    if (validate(clip, skeleton_iterator->second) != validation_error::none) return {};

    const animation_clip_id id{state_->next_clip_id++};
    state_->clips.emplace(id.value, clip_record{skeleton, std::move(clip)});
    return id;
}

bool animation_runtime::remove_clip(animation_clip_id clip) noexcept
{
    if (!clip) return false;
    for (const auto& entry : state_->instances)
        if (entry.second->clip == clip) return false;
    return state_->clips.erase(clip.value) != 0;
}

animation_instance_id animation_runtime::create_instance(skeleton_id skeleton)
{
    return create_instance(skeleton, {});
}

animation_instance_id animation_runtime::create_instance(skeleton_id skeleton, animation_clip_id clip)
{
    const auto skeleton_iterator = state_->skeletons.find(skeleton.value);
    if (skeleton_iterator == state_->skeletons.end()) return {};

    if (clip)
    {
        const auto clip_iterator = state_->clips.find(clip.value);
        if (clip_iterator == state_->clips.end() || clip_iterator->second.skeleton != skeleton) return {};
    }

    auto instance = std::make_unique<instance_record>();
    instance->skeleton = skeleton;
    instance->clip = clip;

    bool initialized{};
    if (clip)
    {
        const auto& definition = state_->clips.at(clip.value).definition;
        initialized = sample_pose_validated(definition, skeleton_iterator->second, 0.0F, instance->poses[0]);
    }
    else
    {
        initialized = initialize_bind_pose(skeleton_iterator->second, instance->poses[0]);
    }
    if (!initialized) return {};
    instance->poses[1] = instance->poses[0];

    const animation_instance_id id{state_->next_instance_id++};
    state_->instances.emplace(id.value, std::move(instance));
    return id;
}

bool animation_runtime::destroy_instance(animation_instance_id instance) noexcept
{
    if (!instance) return false;
    return state_->instances.erase(instance.value) != 0;
}

bool animation_runtime::set_clip(animation_instance_id instance, animation_clip_id clip)
{
    const auto instance_iterator = state_->instances.find(instance.value);
    if (instance_iterator == state_->instances.end()) return false;
    auto& record = *instance_iterator->second;

    if (clip)
    {
        const auto clip_iterator = state_->clips.find(clip.value);
        if (clip_iterator == state_->clips.end() || clip_iterator->second.skeleton != record.skeleton) return false;
    }

    record.clip = clip;
    record.time_seconds = 0.0F;
    return reset_instance(instance);
}

bool animation_runtime::reset_instance(animation_instance_id instance)
{
    const auto instance_iterator = state_->instances.find(instance.value);
    if (instance_iterator == state_->instances.end()) return false;
    auto& record = *instance_iterator->second;
    record.time_seconds = 0.0F;

    const auto skeleton_iterator = state_->skeletons.find(record.skeleton.value);
    if (skeleton_iterator == state_->skeletons.end()) return false;

    auto& output = writable_pose(record);
    bool success{};
    if (record.clip)
    {
        const auto clip_iterator = state_->clips.find(record.clip.value);
        if (clip_iterator == state_->clips.end()) return false;
        success = sample_pose_validated(clip_iterator->second.definition, skeleton_iterator->second, 0.0F, output);
    }
    else
    {
        success = initialize_bind_pose(skeleton_iterator->second, output);
    }

    if (!success) return false;
    publish_pose(record, output);
    return true;
}

bool animation_runtime::update_instance(animation_instance_id instance, float delta_seconds)
{
    if (!std::isfinite(delta_seconds) || delta_seconds < 0.0F) return false;

    const auto instance_iterator = state_->instances.find(instance.value);
    if (instance_iterator == state_->instances.end()) return false;
    auto& record = *instance_iterator->second;

    if (!record.clip)
    {
        if (delta_seconds == 0.0F) return true;
        return reset_instance(instance);
    }

    const auto clip_iterator = state_->clips.find(record.clip.value);
    if (clip_iterator == state_->clips.end()) return false;
    const auto& definition = clip_iterator->second.definition;
    const float next_time =
        resolve_sample_time(record.time_seconds + delta_seconds, definition.duration_seconds, definition.looping);
    return evaluate_instance(instance, next_time);
}

bool animation_runtime::evaluate_instance(animation_instance_id instance, float time_seconds)
{
    if (!std::isfinite(time_seconds)) return false;

    const auto instance_iterator = state_->instances.find(instance.value);
    if (instance_iterator == state_->instances.end()) return false;
    auto& record = *instance_iterator->second;

    const auto skeleton_iterator = state_->skeletons.find(record.skeleton.value);
    if (skeleton_iterator == state_->skeletons.end()) return false;

    auto& output = writable_pose(record);
    if (!record.clip)
    {
        if (!initialize_bind_pose(skeleton_iterator->second, output)) return false;
        record.time_seconds = 0.0F;
    }
    else
    {
        const auto clip_iterator = state_->clips.find(record.clip.value);
        if (clip_iterator == state_->clips.end()) return false;
        const auto& definition = clip_iterator->second.definition;
        const float resolved = resolve_sample_time(time_seconds, definition.duration_seconds, definition.looping);
        if (!sample_pose_validated(definition, skeleton_iterator->second, resolved, output)) return false;
        record.time_seconds = resolved;
    }

    publish_pose(record, output);
    return true;
}

const animation_pose* animation_runtime::pose(animation_instance_id instance) const noexcept
{
    const auto iterator = state_->instances.find(instance.value);
    if (iterator == state_->instances.end()) return nullptr;
    return published_pose(*iterator->second);
}

skinning_pose_view animation_runtime::skinning_pose(animation_instance_id instance) const noexcept
{
    const auto instance_iterator = state_->instances.find(instance.value);
    if (instance_iterator == state_->instances.end()) return {};

    const auto& record = *instance_iterator->second;
    const auto skeleton_iterator = state_->skeletons.find(record.skeleton.value);
    if (skeleton_iterator == state_->skeletons.end()) return {};

    const auto* published = published_pose(record);
    return {std::span<const joint_transform>{published->model_transforms},
            std::span<const math::matrix4f>{skeleton_iterator->second.inverse_bind_matrices}, published->generation};
}

float animation_runtime::time_seconds(animation_instance_id instance) const noexcept
{
    const auto iterator = state_->instances.find(instance.value);
    return iterator == state_->instances.end() ? 0.0F : iterator->second->time_seconds;
}

skeleton_id animation_runtime::skeleton(animation_instance_id instance) const noexcept
{
    const auto iterator = state_->instances.find(instance.value);
    return iterator == state_->instances.end() ? skeleton_id{} : iterator->second->skeleton;
}

animation_clip_id animation_runtime::clip(animation_instance_id instance) const noexcept
{
    const auto iterator = state_->instances.find(instance.value);
    return iterator == state_->instances.end() ? animation_clip_id{} : iterator->second->clip;
}

const skeleton_definition* animation_runtime::skeleton_resource(skeleton_id skeleton) const noexcept
{
    const auto iterator = state_->skeletons.find(skeleton.value);
    return iterator == state_->skeletons.end() ? nullptr : &iterator->second;
}

const animation_clip_definition* animation_runtime::clip_resource(animation_clip_id clip) const noexcept
{
    const auto iterator = state_->clips.find(clip.value);
    return iterator == state_->clips.end() ? nullptr : &iterator->second.definition;
}

} // namespace arc::animation

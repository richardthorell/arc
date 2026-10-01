#pragma once

#include <arc/math/matrix.h>
#include <arc/math/quaternion.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::animation
{

using joint_index = std::uint32_t;
inline constexpr joint_index invalid_joint_index = ~joint_index{0};

/** @brief Stable runtime handle for a registered skeleton resource. */
struct skeleton_id
{
    std::uint64_t value{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return value != 0;
    }

    friend bool operator==(const skeleton_id&, const skeleton_id&) = default;
};

/** @brief Stable runtime handle for a registered animation clip resource. */
struct animation_clip_id
{
    std::uint64_t value{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return value != 0;
    }

    friend bool operator==(const animation_clip_id&, const animation_clip_id&) = default;
};

/** @brief Stable runtime handle for one animation evaluation instance. */
struct animation_instance_id
{
    std::uint64_t value{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return value != 0;
    }

    friend bool operator==(const animation_instance_id&, const animation_instance_id&) = default;
};

/**
 * @brief Local or model-space joint transform represented as translation, rotation, and scale.
 *
 * Translation uses ARC scene/world units, rotation is an `(x, y, z, w)` quaternion, and scale is component-wise.
 * Animation data does not perform coordinate-system conversion; importers must convert source data into ARC
 * conventions.
 */
struct joint_transform
{
    math::vector3f translation{};
    math::quaternionf rotation{};
    math::vector3f scale{1.0F, 1.0F, 1.0F};
};

/** @brief One joint in deterministic parent-before-child skeleton order. */
struct skeleton_joint
{
    std::string name;
    joint_index parent = invalid_joint_index;
    joint_transform bind_pose{};
};

/**
 * @brief Runtime-facing skeleton definition.
 *
 * Joints must be stored parent-before-child. Inverse-bind matrices are optional during M1; when supplied they must
 * contain one matrix per joint and are passed through the renderer-facing skinning view without exposing renderer
 * resource types.
 */
struct skeleton_definition
{
    std::vector<skeleton_joint> joints;
    std::vector<math::matrix4f> inverse_bind_matrices;
};

template <typename T> struct keyframe
{
    float time_seconds = 0.0F;
    T value{};
};

/** @brief Sparse transform channels for one joint. Missing channels fall back to the skeleton bind pose. */
struct joint_track
{
    joint_index joint = invalid_joint_index;
    std::vector<keyframe<math::vector3f>> translations;
    std::vector<keyframe<math::quaternionf>> rotations;
    std::vector<keyframe<math::vector3f>> scales;
};

/**
 * @brief Runtime-facing clip definition for the M1 authored/test representation.
 *
 * Key times are expressed directly in seconds. The representation is deliberately uncompressed; later cooked clip
 * formats can implement the same sampling contract without changing gameplay, ECS, graph, or renderer-facing APIs.
 */
struct animation_clip_definition
{
    std::string name;
    float duration_seconds = 0.0F;
    bool looping = false;
    std::vector<joint_track> tracks;
};

/** @brief Complete sampled pose with both hierarchy-local and model-space transforms. */
struct animation_pose
{
    std::vector<joint_transform> local_transforms;
    std::vector<joint_transform> model_transforms;
    std::uint64_t generation{};
};

/**
 * @brief Renderer-facing skinning handoff that contains animation data only.
 *
 * The renderer may consume model transforms and inverse-bind matrices to build backend-specific skinning resources.
 * Those resources intentionally do not appear in the animation public API.
 */
struct skinning_pose_view
{
    std::span<const joint_transform> model_transforms;
    std::span<const math::matrix4f> inverse_bind_matrices;
    std::uint64_t generation{};
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
    invalid_keyframe_value,
    invalid_inverse_bind_count,
    invalid_inverse_bind_matrix
};

[[nodiscard]] validation_error validate(const skeleton_definition& skeleton) noexcept;
[[nodiscard]] validation_error validate(const animation_clip_definition& clip,
                                        const skeleton_definition& skeleton) noexcept;
[[nodiscard]] std::string_view validation_error_message(validation_error error) noexcept;

/** @brief Resolve an arbitrary time into the clip range using clamp or loop semantics. */
[[nodiscard]] float resolve_sample_time(float time_seconds, float duration_seconds, bool looping) noexcept;

/** @brief Resolve an arbitrary time and return normalized clip time in `[0, 1]`. */
[[nodiscard]] float normalized_time(float time_seconds, float duration_seconds, bool looping) noexcept;

/** @brief Fill a pose with the skeleton bind pose and derived model-space transforms. */
[[nodiscard]] bool initialize_bind_pose(const skeleton_definition& skeleton, animation_pose& output) noexcept;

/**
 * @brief Sample the M1 clip representation into a complete local/model-space pose.
 *
 * Sparse/missing channels inherit the skeleton bind pose. Translation and scale use linear interpolation. Rotations use
 * normalized shortest-path quaternion interpolation. Invalid skeleton or clip data returns false without publishing a
 * pose.
 */
[[nodiscard]] bool sample_pose(const animation_clip_definition& clip, const skeleton_definition& skeleton,
                               float time_seconds, animation_pose& output) noexcept;

/**
 * @brief Engine-owned animation resource and instance runtime.
 *
 * Resource/instance creation and destruction are simulation-thread operations. Pose publication is double buffered so a
 * reader of the currently published pose never observes the in-progress evaluation buffer. A published pose pointer
 * remains valid until the instance is destroyed, but callers must finish reading it before two subsequent publications
 * can recycle that buffer. This frame-oriented contract leaves room for later job-system evaluation without leaking job
 * primitives here.
 */
class animation_runtime final
{
public:
    animation_runtime();
    ~animation_runtime();

    animation_runtime(const animation_runtime&) = delete;
    animation_runtime& operator=(const animation_runtime&) = delete;
    animation_runtime(animation_runtime&&) noexcept;
    animation_runtime& operator=(animation_runtime&&) noexcept;

    /** @brief Validate and register a skeleton. Returns zero on invalid input. */
    [[nodiscard]] skeleton_id add_skeleton(skeleton_definition skeleton);

    /** @brief Remove an unreferenced skeleton. Fails while clips or instances still reference it. */
    bool remove_skeleton(skeleton_id skeleton) noexcept;

    /** @brief Validate and register a clip against a skeleton. Returns zero on invalid input. */
    [[nodiscard]] animation_clip_id add_clip(skeleton_id skeleton, animation_clip_definition clip);

    /** @brief Remove an unreferenced clip. Fails while an instance still uses it. */
    bool remove_clip(animation_clip_id clip) noexcept;

    /** @brief Create an instance initialized to the skeleton bind pose. */
    [[nodiscard]] animation_instance_id create_instance(skeleton_id skeleton);

    /** @brief Create an instance and immediately sample a compatible clip at time zero. */
    [[nodiscard]] animation_instance_id create_instance(skeleton_id skeleton, animation_clip_id clip);

    /** @brief Destroy an animation instance. */
    bool destroy_instance(animation_instance_id instance) noexcept;

    /** @brief Set or clear the active clip. Passing a zero clip clears playback and publishes bind pose. */
    bool set_clip(animation_instance_id instance, animation_clip_id clip);

    /** @brief Reset time to zero and republish the active clip or bind pose. */
    bool reset_instance(animation_instance_id instance);

    /** @brief Advance one instance by non-negative delta seconds and publish a new complete pose. */
    bool update_instance(animation_instance_id instance, float delta_seconds);

    /** @brief Evaluate one instance at an explicit time and publish the result. */
    bool evaluate_instance(animation_instance_id instance, float time_seconds);

    /** @brief Return the currently published pose, or nullptr for an unknown instance. */
    [[nodiscard]] const animation_pose* pose(animation_instance_id instance) const noexcept;

    /** @brief Return an animation-only renderer handoff for the currently published pose. */
    [[nodiscard]] skinning_pose_view skinning_pose(animation_instance_id instance) const noexcept;

    /** @brief Return the resolved current sample time for an instance, or zero when unknown. */
    [[nodiscard]] float time_seconds(animation_instance_id instance) const noexcept;

    /** @brief Return the skeleton resource associated with an instance. */
    [[nodiscard]] skeleton_id skeleton(animation_instance_id instance) const noexcept;

    /** @brief Return the active clip associated with an instance, or zero when no clip is active. */
    [[nodiscard]] animation_clip_id clip(animation_instance_id instance) const noexcept;

    /** @brief Inspect a registered skeleton definition. Pointer is invalidated when that resource is removed. */
    [[nodiscard]] const skeleton_definition* skeleton_resource(skeleton_id skeleton) const noexcept;

    /** @brief Inspect a registered clip definition. Pointer is invalidated when that resource is removed. */
    [[nodiscard]] const animation_clip_definition* clip_resource(animation_clip_id clip) const noexcept;

private:
    struct state;
    std::unique_ptr<state> state_;
};

} // namespace arc::animation

#include <arc/animation/animation.h>

#include <limits>

namespace
{

using namespace arc::animation;

skeleton_definition make_skeleton()
{
    skeleton_definition skeleton{};
    skeleton.joints.push_back(skeleton_joint{"root"});
    skeleton.joints.push_back(skeleton_joint{"spine", 0});
    return skeleton;
}

} // namespace

int main()
{
    skeleton_definition skeleton{};
    if (validate(skeleton) != validation_error::empty_skeleton) return 1;

    skeleton = make_skeleton();
    if (validate(skeleton) != validation_error::none) return 2;

    auto invalid_parent = skeleton;
    invalid_parent.joints[1].parent = 1;
    if (validate(invalid_parent) != validation_error::invalid_parent) return 3;

    auto duplicate_name = skeleton;
    duplicate_name.joints[1].name = "root";
    if (validate(duplicate_name) != validation_error::invalid_joint_name) return 4;

    auto invalid_inverse_binds = skeleton;
    invalid_inverse_binds.inverse_bind_matrices.resize(1);
    if (validate(invalid_inverse_binds) != validation_error::invalid_inverse_bind_count) return 5;

    animation_clip_definition clip{};
    clip.name = "idle";
    clip.duration_seconds = 1.0F;
    clip.tracks.push_back(joint_track{0});
    clip.tracks[0].translations.push_back({0.0F, {0.0F, 0.0F, 0.0F}});
    clip.tracks[0].translations.push_back({1.0F, {0.0F, 0.1F, 0.0F}});
    if (validate(clip, skeleton) != validation_error::none) return 6;

    auto bad_joint = clip;
    bad_joint.tracks[0].joint = 4;
    if (validate(bad_joint, skeleton) != validation_error::invalid_track_joint) return 7;

    auto duplicate_track = clip;
    duplicate_track.tracks.push_back(joint_track{0});
    if (validate(duplicate_track, skeleton) != validation_error::duplicate_track) return 8;

    auto unordered = clip;
    unordered.tracks[0].translations[0].time_seconds = 0.75F;
    unordered.tracks[0].translations[1].time_seconds = 0.25F;
    if (validate(unordered, skeleton) != validation_error::unordered_keyframes) return 9;

    auto non_finite = clip;
    non_finite.tracks[0].translations[0].value[0] = std::numeric_limits<float>::quiet_NaN();
    if (validate(non_finite, skeleton) != validation_error::invalid_keyframe_value) return 10;

    if (validation_error_message(validation_error::invalid_parent).empty()) return 11;
    if (resolve_sample_time(-0.5F, 2.0F, false) != 0.0F) return 12;
    if (resolve_sample_time(3.0F, 2.0F, false) != 2.0F) return 13;
    if (resolve_sample_time(2.5F, 2.0F, true) != 0.5F) return 14;
    if (resolve_sample_time(-0.5F, 2.0F, true) != 1.5F) return 15;
    if (normalized_time(0.5F, 2.0F, false) != 0.25F) return 16;

    return 0;
}

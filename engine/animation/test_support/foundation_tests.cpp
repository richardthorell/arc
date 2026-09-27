#include <arc/animation/animation.h>

#include <cassert>
#include <limits>

int main()
{
    using namespace arc::animation;

    skeleton_definition skeleton{};
    assert(validate(skeleton) == validation_error::empty_skeleton);

    skeleton.joints.push_back(skeleton_joint{"root"});
    skeleton.joints.push_back(skeleton_joint{"spine", 0});
    assert(validate(skeleton) == validation_error::none);

    auto invalid_parent = skeleton;
    invalid_parent.joints[1].parent = 1;
    assert(validate(invalid_parent) == validation_error::invalid_parent);

    auto duplicate_name = skeleton;
    duplicate_name.joints[1].name = "root";
    assert(validate(duplicate_name) == validation_error::invalid_joint_name);

    animation_clip_definition clip{};
    clip.name = "idle";
    clip.duration_seconds = 1.0F;
    clip.tracks.push_back(joint_track{0});
    clip.tracks[0].translations.push_back({0.0F, {0.0F, 0.0F, 0.0F}});
    clip.tracks[0].translations.push_back({1.0F, {0.0F, 0.1F, 0.0F}});
    assert(validate(clip, skeleton) == validation_error::none);

    auto bad_joint = clip;
    bad_joint.tracks[0].joint = 4;
    assert(validate(bad_joint, skeleton) == validation_error::invalid_track_joint);

    auto duplicate_track = clip;
    duplicate_track.tracks.push_back(joint_track{0});
    assert(validate(duplicate_track, skeleton) == validation_error::duplicate_track);

    auto unordered = clip;
    unordered.tracks[0].translations[0].time_seconds = 0.75F;
    unordered.tracks[0].translations[1].time_seconds = 0.25F;
    assert(validate(unordered, skeleton) == validation_error::unordered_keyframes);

    auto non_finite = clip;
    non_finite.tracks[0].translations[0].value[0] = std::numeric_limits<float>::quiet_NaN();
    assert(validate(non_finite, skeleton) == validation_error::invalid_keyframe_value);

    return 0;
}

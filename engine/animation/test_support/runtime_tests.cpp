#include <arc/animation/animation.h>

#include <cmath>

namespace
{

using namespace arc::animation;

bool near(float lhs, float rhs) noexcept
{
    return std::abs(lhs - rhs) < 0.0001F;
}

skeleton_definition make_skeleton()
{
    skeleton_definition skeleton{};

    skeleton_joint root{"root"};
    root.bind_pose.translation = {1.0F, 0.0F, 0.0F};
    skeleton.joints.push_back(root);

    skeleton_joint child{"child", 0};
    child.bind_pose.translation = {0.0F, 2.0F, 0.0F};
    skeleton.joints.push_back(child);
    return skeleton;
}

animation_clip_definition make_clip(bool looping)
{
    animation_clip_definition clip{};
    clip.name = "move";
    clip.duration_seconds = 2.0F;
    clip.looping = looping;

    joint_track root{};
    root.joint = 0;
    root.translations.push_back({0.0F, {1.0F, 0.0F, 0.0F}});
    root.translations.push_back({2.0F, {3.0F, 0.0F, 0.0F}});
    clip.tracks.push_back(root);
    return clip;
}

} // namespace

int main()
{
    const auto skeleton = make_skeleton();

    animation_pose bind_pose{};
    if (!initialize_bind_pose(skeleton, bind_pose)) return 1;
    if (bind_pose.local_transforms.size() != 2 || bind_pose.model_transforms.size() != 2) return 2;
    if (!near(bind_pose.model_transforms[0].translation[0], 1.0F)) return 3;
    if (!near(bind_pose.model_transforms[1].translation[0], 1.0F) ||
        !near(bind_pose.model_transforms[1].translation[1], 2.0F))
        return 4;

    const auto clip = make_clip(false);
    animation_pose sampled{};
    if (!sample_pose(clip, skeleton, 1.0F, sampled)) return 5;
    if (!near(sampled.local_transforms[0].translation[0], 2.0F)) return 6;
    if (!near(sampled.local_transforms[1].translation[1], 2.0F)) return 7;
    if (!near(sampled.model_transforms[1].translation[0], 2.0F)) return 8;

    animation_runtime runtime{};
    if (runtime.add_skeleton({})) return 9;

    const auto skeleton_id = runtime.add_skeleton(skeleton);
    if (!skeleton_id) return 10;

    const auto clip_id = runtime.add_clip(skeleton_id, make_clip(true));
    if (!clip_id) return 11;

    const auto instance_id = runtime.create_instance(skeleton_id, clip_id);
    if (!instance_id) return 12;
    if (runtime.skeleton(instance_id) != skeleton_id || runtime.clip(instance_id) != clip_id) return 13;

    const auto* initial = runtime.pose(instance_id);
    if (initial == nullptr || initial->local_transforms.size() != 2) return 14;
    const auto initial_generation = initial->generation;
    if (!near(initial->local_transforms[0].translation[0], 1.0F)) return 15;

    if (!runtime.update_instance(instance_id, 1.0F)) return 16;
    const auto* first_update = runtime.pose(instance_id);
    if (first_update == nullptr || first_update == initial) return 17;
    if (first_update->generation != initial_generation + 1) return 18;
    if (!near(first_update->local_transforms[0].translation[0], 2.0F)) return 19;
    if (!near(runtime.time_seconds(instance_id), 1.0F)) return 20;

    if (!runtime.update_instance(instance_id, 1.5F)) return 21;
    const auto* looped = runtime.pose(instance_id);
    if (looped == nullptr || !near(runtime.time_seconds(instance_id), 0.5F)) return 22;
    if (!near(looped->local_transforms[0].translation[0], 1.5F)) return 23;

    const auto skinning = runtime.skinning_pose(instance_id);
    if (skinning.model_transforms.size() != 2 || skinning.generation != looped->generation) return 24;
    if (!skinning.inverse_bind_matrices.empty()) return 25;

    if (runtime.remove_clip(clip_id)) return 26;
    if (!runtime.reset_instance(instance_id)) return 27;
    if (!near(runtime.time_seconds(instance_id), 0.0F)) return 28;

    if (!runtime.set_clip(instance_id, {})) return 29;
    if (runtime.clip(instance_id)) return 30;
    const auto* cleared = runtime.pose(instance_id);
    if (cleared == nullptr || !near(cleared->local_transforms[0].translation[0], 1.0F)) return 31;

    if (!runtime.remove_clip(clip_id)) return 32;
    if (runtime.remove_skeleton(skeleton_id)) return 33;
    if (!runtime.destroy_instance(instance_id)) return 34;
    if (!runtime.remove_skeleton(skeleton_id)) return 35;

    return 0;
}

#include <arc/project/runtime_application.h>
#include <arc/scene/runtime_scene.h>

namespace arc_packaged_sample
{

void register_runtime(arc::project::packaged_game_registrations& registrations)
{
    // Compile the production scene-loading contract in the native Windows
    // sample, independently of generated project reflection headers.
    registrations.materialize_scene =
        [](arc::framework::runtime_world& world, const arc::project::packaged_scene_payload& scene)
    {
        const auto error = arc::scene::materialize_cooked_scene(world.entities(), scene.bytes);
        if (!error.empty())
            return arc::project::project_status::failure(
                {.code = arc::project::project_error_code::invalid_scene,
                 .field = "startupScenes",
                 .message = "Startup scene " + arc::assets::to_string(scene.guid) + ": " + error});
        return arc::project::project_status::success();
    };
}

} // namespace arc_packaged_sample

#include <arc/project/runtime_application.h>
#include <arc/scene/runtime_scene.h>
#include "{{PROJECT_TOKEN}}Runtime.reflection.h"

namespace {{PROJECT_TOKEN}}
{

void register_runtime(arc::project::packaged_game_registrations& registrations)
{
    registrations.materialize_scene = [](arc::framework::runtime_world& world,
                                         const arc::project::packaged_scene_payload& scene)
    {
        const auto error = arc::scene::materialize_cooked_scene(world.entities(), scene.bytes);
        if (!error.empty())
            return arc::project::project_status::failure({
                .code = arc::project::project_error_code::invalid_scene,
                .field = "startupScenes",
                .message = "Startup scene " + arc::assets::to_string(scene.guid) + ": " + error});
        return arc::project::project_status::success();
    };
}

} // namespace {{PROJECT_TOKEN}}

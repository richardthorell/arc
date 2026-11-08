#include <arc/project/runtime_application.h>
#include "{{PROJECT_TOKEN}}Runtime.reflection.h"

namespace {{PROJECT_TOKEN}}
{
void register_runtime(arc::project::packaged_game_registrations& registrations)
{
    // Register native game services/modules and a cooked-scene materializer here.
    // Leaving materialize_scene unset fails closed when a packaged startup scene
    // is present; a silent placeholder world is never accepted.
    (void)registrations;
}
}

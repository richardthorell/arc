#include <arc/project/headless_server.h>

namespace {{PROJECT_TOKEN}}
{

void register_server(arc::project::packaged_game_registrations& registrations)
{
    // Deliberate blank server world: add game-specific services, ECS systems,
    // or world initialization here. D1.4 registers the headless world itself.
    registrations.initialize_headless_world = [](arc::framework::runtime_world&)
    { return arc::project::project_status::success(); };
}

} // namespace {{PROJECT_TOKEN}}

#include <arc/project/headless_server.h>

namespace arc_packaged_sample
{

void register_server(arc::project::packaged_game_registrations& registrations)
{
    registrations.initialize_headless_world = [](arc::framework::runtime_world&)
    { return arc::project::project_status::success(); };
}

} // namespace arc_packaged_sample

#include <arc/project/runtime_application.h>

namespace arc_packaged_sample
{

void register_runtime(arc::project::packaged_game_registrations&)
{
    // The sample is a link/compile contract only. A staged client must provide
    // real cooked startup content and a materializer before it can boot.
}

} // namespace arc_packaged_sample

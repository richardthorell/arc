#include <arc/project/project_module.h>
#include <arc/project/runtime_application.h>
#include "{{PROJECT_TOKEN}}Runtime.reflection.h"

#include <memory>

namespace {{PROJECT_TOKEN}}
{
void register_runtime(arc::project::packaged_game_registrations& registrations);
}

namespace
{
std::unique_ptr<arc::project::packaged_game_registrations> registrations;

bool start(const arc::project::game_module_host_v1* host)
{
    if (!host || host->abi_version != arc::project::game_module_abi_version)
        return false;
    registrations = std::make_unique<arc::project::packaged_game_registrations>();
    {{PROJECT_TOKEN}}::register_runtime(*registrations);
    if (host->log)
        host->log(host->user_data, "{{PROJECT_TOKEN}}.runtime", "Android project runtime module started");
    return true;
}

void stop()
{
    registrations.reset();
}

const arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "{{ENGINE_VERSION}}",
    .project_guid = "{{PROJECT_GUID}}",
    .module_id = "{{PROJECT_TOKEN}}.runtime",
    .kind = arc::project::game_module_kind_v1::runtime,
    .components = {{PROJECT_TOKEN}}::generated::components.data(),
    .component_count = {{PROJECT_TOKEN}}::generated::components.size(),
    .start = start,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}

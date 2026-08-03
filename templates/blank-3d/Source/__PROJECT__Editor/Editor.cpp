#include <arc/project/project_module.h>
#include "{{PROJECT_TOKEN}}Runtime.reflection.h"

namespace
{
bool start(const arc::project::game_module_host_v1* host)
{
    if (host->log) host->log(host->user_data, "{{PROJECT_TOKEN}}.editor", "Project editor module started");
    return true;
}
bool prepare_reload() { return true; }
void stop() {}

const arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "{{ENGINE_VERSION}}",
    .project_guid = "{{PROJECT_GUID}}",
    .module_id = "{{PROJECT_TOKEN}}.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .components = {{PROJECT_TOKEN}}::generated::components.data(),
    .component_count = std::size({{PROJECT_TOKEN}}::generated::components),
    .start = start,
    .prepare_reload = prepare_reload,
    .stop = stop,
};
}

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}

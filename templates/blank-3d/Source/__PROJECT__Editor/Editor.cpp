#include <arc/project/project_module.h>

namespace
{
bool start(const arc::project::editor_module_host_v1*) { return true; }
void stop() {}
const arc::project::editor_module_descriptor_v1 descriptor{
    arc::project::editor_module_abi_version, "{{ENGINE_VERSION}}", "{{PROJECT_GUID}}", "{{PROJECT_TOKEN}}.editor", start, stop};
}

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::editor_module_descriptor_v1* arc_query_editor_module_v1()
{
    return &descriptor;
}


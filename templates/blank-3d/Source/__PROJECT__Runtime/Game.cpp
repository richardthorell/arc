#include <arc/platform/windows/packaged_client.h>

#ifndef ARC_GAME_BUILD_CONFIGURATION
#error "CMake must define ARC_GAME_BUILD_CONFIGURATION for packaged client validation"
#endif

namespace {{PROJECT_TOKEN}}
{
void register_runtime(arc::project::packaged_game_registrations& registrations);
}

namespace arc::platform::windows
{

packaged_client_definition configure_packaged_client()
{
    packaged_client_definition client;
    client.expected = {.engine_version = "{{ENGINE_VERSION}}",
                       .sdk_version = "{{ENGINE_VERSION}}",
                       .architecture = "x86_64",
                       .renderer = "vulkan",
                       .configuration = ARC_GAME_BUILD_CONFIGURATION,
                       .project_guid = "{{PROJECT_GUID}}"};
    client.config.title = "{{PROJECT_NAME}}";
    {{PROJECT_TOKEN}}::register_runtime(client.registrations);
    return client;
}

} // namespace arc::platform::windows

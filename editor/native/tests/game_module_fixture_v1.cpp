#include <arc/project/project_module.h>

#include <iterator>

namespace
{
constexpr arc::project::game_field_descriptor_v1 fields[]{
    {0x1111111111111111ull, "old_value", "Value", "Test", "Fixture value",
     arc::project::game_field_kind_v1::floating_point,
     arc::project::game_field_flags_v1::editable | arc::project::game_field_flags_v1::serialized, "1.0", 0.0, 10.0,
     true, true},
};
constexpr arc::project::game_component_descriptor_v1 components[]{
    {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "old_component", "Fixture Component", "Test", "Fixture", 1, fields,
     std::size(fields)},
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::console_command, "fixture.echo", "Fixture Echo", nullptr},
};
bool start(const arc::project::game_module_host_v1*)
{
    return true;
}
bool prepare_reload()
{
    return true;
}
void stop() {}
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 1,
    .components = components,
    .component_count = std::size(components),
    .registrations = registrations,
    .registration_count = std::size(registrations),
    .start = start,
    .prepare_reload = prepare_reload,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}

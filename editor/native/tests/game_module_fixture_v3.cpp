#include <arc/project/project_module.h>

#include <iterator>

namespace
{
constexpr arc::project::game_field_descriptor_v1 fields[]{
    {0x1111111111111111ull, "renamed_value", "Value", "Test", "Fixture value",
     arc::project::game_field_kind_v1::signed_integer,
     arc::project::game_field_flags_v1::editable | arc::project::game_field_flags_v1::serialized,
     "1", 0.0, 10.0, true, true},
};
constexpr arc::project::game_component_descriptor_v1 components[]{
    {"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "renamed_component", "Fixture Component", "Test", "Fixture", 3,
     fields, std::size(fields)},
};
bool start(const arc::project::game_module_host_v1*) { return true; }
bool prepare_reload() { return true; }
void stop() {}
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 3,
    .components = components,
    .component_count = std::size(components),
    .start = start,
    .prepare_reload = prepare_reload,
    .stop = stop,
};
}

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}

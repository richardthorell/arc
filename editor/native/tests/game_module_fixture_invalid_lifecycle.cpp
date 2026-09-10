#include <arc/project/project_module.h>

#include <iterator>

namespace
{
bool begin_play(void*, const arc::project::game_play_context_v1*)
{
    return true;
}

bool start(const arc::project::game_module_host_v1*)
{
    return true;
}

void stop() {}

constexpr arc::project::game_play_lifecycle_descriptor_v1 invalid_lifecycle{
    .begin_play = begin_play,
};

constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::play_lifecycle, "fixture.runtime.invalid-play-lifecycle",
     "Invalid Play Lifecycle", &invalid_lifecycle},
};

constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 12,
    .registrations = registrations,
    .registration_count = std::size(registrations),
    .start = start,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}
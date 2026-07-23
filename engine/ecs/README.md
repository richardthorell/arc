# ARC ECS

`arc-ecs` is ARC's backend-neutral entity/component foundation. Include it with:

```cpp
#include <arc/ecs/ecs.h>
```

The module owns stable entity and component identities, stable-address paged
pools, prepared allocation-free queries, independent change cursors,
deterministic structural command buffers, access-aware system scheduling,
intrusive hierarchy, entity templates, prefab metadata, archive and replication
visitors, and initial world-partition contracts.

Scene-specific components remain in `arc-scene`. Existing
`arc::scene::registry`, `entity`, `entity_guid`, pools, and views are
compatibility aliases over ECS types so runtime and editor code can migrate
incrementally.

Reflected scene component IDs and editor field metadata originate in
`engine/scene/schema/components.arccomponents.json`. CMake runs
`tools/ecs_codegen.py` and fails verification when the checked-in TypeScript
metadata is stale. Stable schema IDs are persistence contracts and must not be
derived from C++ type or field names.

Prepared queries must be registered before a hot execution path:

```cpp
world.prepare_typed_query<
    query_read<transform>,
    query_optional<velocity>,
    query_exclude<disabled>>();

for (entity value : world.query<
         query_read<transform>,
         query_optional<velocity>,
         query_exclude<disabled>>())
{
    // Query creation and iteration perform no allocation after preparation.
}
```

Systems declare component access up front. Independent read-only work may
overlap; write hazards, explicit dependencies, and phase boundaries form the
execution DAG. Structural changes produced while systems run must go through
the system's `entity_command_buffer` and become visible at synchronization
points.

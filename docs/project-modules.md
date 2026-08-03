# ARC C++ project modules

ARC external projects compile separate Editor, Runtime, and Server modules. The
Editor module is loaded only after its owning project has passed descriptor and
engine-version validation. Runtime and Server modules use the same descriptor
ABI in their corresponding launch targets.

## Stable module boundary

An Editor module exports exactly one C-linkage entry point:

```cpp
extern "C" ARC_PROJECT_MODULE_EXPORT
const arc::project::game_module_descriptor_v1* arc_query_game_module_v1();
```

The descriptor identifies the exact engine version, project GUID, module ID,
role, and build generation. It advertises reflected components and stable
registrations for systems, services, asset types, importers, cook processors,
console commands, and editor extensions. ARC validates and copies all metadata
before invoking the module lifecycle. Module-owned tasks must be quiescent when
`prepare_reload` returns successfully, and `stop` must be idempotent.

Project modules may include only installed public headers under
`<arc/<module>/<header>.h>`. Vulkan objects, Electron internals, native-host
implementation headers, and private third-party types are not part of the ABI.

## Reflected components

Declare explicit identities in project source. These values—not C++ names—are
the persistence contract:

```cpp
ARC_COMPONENT("a3e3a5922f9847f6b3f27710db2db109", 1,
              "Health", "Gameplay", "Current and maximum health")
struct health_component
{
    ARC_PROPERTY("0577bbc747c147de", "Current", "Health",
                 "Current hit points", "float", "100.0", "0.0", "1000.0",
                 "editable|serialized|save_game|prefab", "", "")
    float current{100.0f};
};
```

Use `arc_generate_reflection()` on the module target. The generator emits:

- `arc::ecs::component_traits<T>` with stable component and field IDs;
- persistence, prefab, asset-reference, and optional replication metadata;
- the project-module ABI descriptor arrays;
- JSON metadata consumed by host/editor tooling.

Generation rejects missing or duplicate IDs, unsupported property kinds,
invalid defaults, and undocumented annotations. Renaming a type or field while
retaining its stable ID preserves saved data. Increase the schema version when
a change needs migration.

## Build and reload workflow

The editor Build Output panel runs the shared `arc-project` configure/build
commands and parses compiler, generator, linker, and module-load diagnostics.
External IDE builds are detected through the project build manifest. Successful
Editor builds are copied to monotonically named generations such as
`GameEditor_0002`; the original build output is never locked by the host.

Reload proceeds as follows:

1. Pause simulation and ask the old module to drain module-owned work.
2. Capture reflected project-component records by stable IDs.
3. Validate and start the staged generation.
4. Compare schemas and migrate compatible records by stable field ID.
5. Publish refreshed metadata to the Inspector and Add Component menu.
6. Unload the old generation only after the new generation starts.

Compatible additive and rename-only changes are safe hot reloads. Field wire
kind changes require a play-session restart. Removed components, schema
downgrades, identity changes, or ABI changes require a native-host restart. If
startup fails, ARC restores the last-good generation when possible.

Unknown components and unknown fields remain in scene/prefab documents when a
module is missing. They become editable again after a compatible module loads;
ARC never discards them merely because project code is unavailable.

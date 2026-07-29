# ARC public modules

ARC public headers use `<arc/module/module.h>` umbrellas. Backend-neutral
headers never expose Vulkan, editor, SQLite, or JSON implementation types.

## Core results and identifiers

`arc::core::result<T, E>` represents recoverable failure without allocation or
exceptions. `arc::core::uuid<Tag>` and `strong_id<Tag, Rep, Invalid>` prevent
unrelated identifiers from mixing while retaining their serialized layout.

```cpp
auto loaded = load_asset_metadata(path);
if (!loaded)
    report(loaded.error());
asset_source_metadata metadata = std::move(loaded).value();
```

## ECS queries

Create and prewarm typed queries on `arc::ecs::world` before entering a hot
loop. Query iteration borrows component storage; references remain valid under
non-structural writes and must not escape world destruction.

```cpp
world.prepare_typed_query<query_read<transform>, query_write<velocity>>();
for (entity value : world.query<
         query_read<transform>, query_write<velocity>>())
{
    // Writes should use tracked writers or generated field setters.
}
```

## Jobs and memory

`arc::jobs::job_system` schedules dependency-aware work and affinity lanes.
`arc::memory::memory_system` owns tagged domains, budgets, arenas, pools, and
pressure callbacks. Frame- and tick-arena allocations must not escape their
reset boundary.

```cpp
job_handle decode = jobs.submit({ .name = "Decode texture" }, decode_texture);
job_handle upload = jobs.submit(
    { .name = "Upload texture", .affinity = job_affinity::render_thread,
      .dependencies = { decode } },
    upload_texture);
upload.wait();

scoped_allocation_tag tag({ "scene.extract" });
void* temporary = frame_arena.allocate(byte_count, alignment);
```

## Assets and persistence

`arc::assets::asset_manager` resolves persistent GUID references and manages
asynchronous generations. `arc::persistence` provides reflected authoring
archives and tagged runtime archives without exposing a JSON library.

```cpp
asset_reference reference{ guid, asset_types::texture_2d, "textures/albedo.png" };
auto pending = assets.load<texture_asset>({ .reference = reference });

document_store store(registry, migrations);
document_load_result loaded = store.load_json("scenes/main.arcscene");
```

## Rendering and scene extraction

`arc::render` is backend-neutral. A backend consumes immutable frame packets
and executable render graphs. `arc::scene::render_scene()` extracts an
`arc::ecs::world`; scene components never own Vulkan resources.

```cpp
render_scene_result extracted = render_scene(world, camera, packet);
compiled_render_graph graph = renderer.graph().compile();
render_submit_result submitted = backend.submit(packet, graph);
```

## Framework lifecycle

`arc::framework::runtime` starts memory, jobs, services, worlds, modules, and
the application in dependency order, then shuts them down in reverse order.
Deterministic gameplay belongs in fixed-tick ECS systems rather than the
frame-compatible application callback.

```cpp
class game final : public application
{
    void register_worlds(runtime_world_manager& worlds) override
    {
        worlds.create({ .name = "Client", .role = runtime_world_role::client });
    }
};

game app;
runtime process(app);
process.start();
```

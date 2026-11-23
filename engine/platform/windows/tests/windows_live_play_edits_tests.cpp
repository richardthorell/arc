#include "../src/windows_live_play_edits.h"
#include "../src/windows_live_play_inspection.h"

#include <arc/framework/application.h>
#include <arc/framework/runtime.h>
#include <arc/scene/components.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
void check(bool condition, const char* description)
{
    if (!condition) throw std::runtime_error(description);
}

class live_edit_test_application final : public arc::framework::application
{
public:
    void register_worlds(arc::framework::runtime_world_manager& manager) override
    {
        auto& runtime_world =
            manager.create({.name = "Live Play Test World", .role = arc::framework::runtime_world_role::client});
        auto& world = runtime_world.entities();
        const auto entity = world.create();
        world.emplace<arc::scene::persistent_id_component>(
            entity, arc::scene::persistent_id_component{arc::ecs::entity_guid{3, 4}});
        world.emplace<arc::scene::transform_component>(entity);
        const auto other = world.create();
        world.emplace<arc::scene::persistent_id_component>(
            other, arc::scene::persistent_id_component{arc::ecs::entity_guid{5, 6}});
        world.emplace<arc::scene::transform_component>(other);
    }
};

void run()
{
    namespace live = arc::network::live_play_dev;
    live_edit_test_application app;
    arc::framework::runtime runtime(app);
    runtime.start();
    arc::platform::windows::live_play_transform_applier applier;

    live::property_edit edit;
    edit.request_id = "edit-1";
    edit.entity_guid = arc::ecs::to_string(arc::ecs::entity_guid{3, 4});
    edit.component = "arc.transform";
    edit.property = "position";
    edit.phase = "update";
    edit.schema_version = 1;
    edit.value_count = 3;
    edit.values = {2.0f, 4.0f, 6.0f, 0.0f};
    edit.authoring_revision = 1;

    arc::platform::windows::live_play_runtime_inspector inspector;
    auto initial = inspector.sample(runtime);
    check(initial && initial->complete && initial->entities.size() == 2 && initial->world_revision == 1,
          "Initial runtime hierarchy snapshot missing");
    check(!inspector.sample(runtime), "Unchanged runtime hierarchy was resent");
    check(initial->entities.front().entity.authored &&
              initial->entities.front().entity.id == arc::ecs::to_string(arc::ecs::entity_guid{3, 4}),
          "Native hierarchy lost authored entity identity");
    arc::network::live_play_dev::inspection_request inspect_request{.request_id = "inspect-1",
                                                                    .entity = initial->entities.front().entity,
                                                                    .component = "arc.transform",
                                                                    .schema_version = 1};
    auto inspected = inspector.inspect(runtime, inspect_request);
    check(inspected.result && inspected.result->properties.size() == 3,
          "Read-only Transform inspection did not return component properties");
    check(inspected.result->properties.front().numbers[0] == 0.0f,
          "Initial inspected transform does not match the live ECS");

    auto result = applier.apply(runtime, edit, false);
    check(!result.applied && result.code == "restart_required", "Unactivated baseline mutated the runtime");
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 1, "Runtime refused valid transform update");
    const auto ids = runtime.worlds().ordered_worlds();
    check(ids.size() == 1, "Fixture world registration changed");
    auto* world = runtime.worlds().find(ids.front());
    check(world != nullptr, "Fixture runtime world disappeared");
    arc::ecs::entity selected{};
    arc::ecs::entity unrelated{};
    for (const auto entity : world->entities().entities())
    {
        const auto* id = world->entities().try_get<arc::scene::persistent_id_component>(entity);
        if (id && id->value == arc::ecs::entity_guid{3, 4}) selected = entity;
        if (id && id->value == arc::ecs::entity_guid{5, 6}) unrelated = entity;
    }
    check(selected.valid() && unrelated.valid(), "Fixture persistent GUID resolution failed");
    const auto& moved = world->entities().get<arc::scene::transform_component>(selected);
    const auto& still = world->entities().get<arc::scene::transform_component>(unrelated);
    check(moved.position[0] == 2.0f && moved.position[1] == 4.0f && moved.position[2] == 6.0f,
          "Valid authored position did not reach ECS");
    check(still.position[0] == 0.0f && still.position[1] == 0.0f, "Edit mutated an unrelated entity");
    inspected = inspector.inspect(runtime, inspect_request);
    check(inspected.result && inspected.result->properties.front().numbers[0] == 2.0f &&
              inspected.result->properties.front().numbers[1] == 4.0f,
          "Runtime inspection returned stale transform after a valid edit");
    inspect_request.entity = {.authored = false, .id = "w999-x1-e999-g1"};
    check(inspector.inspect(runtime, inspect_request).error_code == "unknown_entity",
          "Unknown runtime-only entity was inspected");
    inspect_request.entity = initial->entities.front().entity;
    inspect_request.component = "arc.unsafe";
    check(inspector.inspect(runtime, inspect_request).error_code == "unsupported_operation",
          "Inspector leaked an unsupported component");
    inspect_request.component = "arc.transform";
    inspect_request.schema_version = 2;
    check(inspector.inspect(runtime, inspect_request).error_code == "schema_mismatch",
          "Inspector accepted a mismatched component schema");
    inspect_request.schema_version = 1;
    auto spawned = world->entities().create();
    world->entities().emplace<arc::scene::name_component>(spawned, arc::scene::name_component{"Runtime Spawned"});
    world->entities().emplace<arc::scene::transform_component>(spawned);
    auto added = inspector.sample(runtime);
    check(added && added->entities.size() == 3 && added->world_revision == 2,
          "Runtime spawn was not reflected in read-only hierarchy");
    const auto spawned_ref = added->entities.back().entity;
    check(!spawned_ref.authored && spawned_ref.id.find("-e") != std::string::npos,
          "Spawned entity was not given a session-scoped runtime ID");
    inspect_request.entity = spawned_ref;
    inspect_request.component = "arc.scene.name";
    inspected = inspector.inspect(runtime, inspect_request);
    check(inspected.result && inspected.result->properties.size() == 1 &&
              inspected.result->properties.front().text == "Runtime Spawned",
          "Spawned entity name was not inspectable");
    check(world->entities().destroy(spawned), "Runtime entity deletion failed");
    auto removed = inspector.sample(runtime);
    check(removed && removed->entities.size() == 2 && removed->world_revision == 3,
          "Runtime deletion was not observed");
    check(inspector.inspect(runtime, inspect_request).error_code == "unknown_entity",
          "Deleted runtime entity ID incorrectly resolved");

    edit.request_id = "edit-2";
    edit.phase = "commit";
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 2, "Final drag commit was lost");
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "stale_revision", "Duplicate commit was applied twice");
    edit.authoring_revision = 0;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "stale_revision", "Stale edit was accepted");
    edit.authoring_revision = 2;
    edit.expected_runtime_revision = 20;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "stale_revision", "Future runtime revision was accepted");
    edit.expected_runtime_revision = 2;
    edit.entity_guid = arc::ecs::to_string(arc::ecs::entity_guid{100, 101});
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "unknown_entity", "Unknown authored GUID was accepted");
    edit.entity_guid = arc::ecs::to_string(arc::ecs::entity_guid{3, 4});
    edit.component = "arc.physics";
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "schema_mismatch", "Unsupported component schema was applied");
    edit.component = "arc.transform";
    edit.property = "rotation";
    edit.value_count = 4;
    edit.values = {0.0f, 0.0f, 0.0f, 1.0f};
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 3, "Valid rotation update was rejected");

    // A real ECS world larger than the former 128-entity ceiling must be
    // captured completely; the transport will paginate this immutable sample.
    for (int i = 0; i < 260; ++i)
    {
        const auto extra = world->entities().create();
        world->entities().emplace<arc::scene::name_component>(extra,
                                                              arc::scene::name_component{"Extra " + std::to_string(i)});
    }
    const auto large = inspector.sample(runtime);
    check(large && large->complete && large->entities.size() == 262 && large->world_revision == 4,
          "Native hierarchy unexpectedly truncated a >128-entity scene");
    const auto bulk_diff = live::diff_hierarchy(*removed, *large);
    check(!bulk_diff, "Over-budget native hierarchy change should use paged snapshot instead of delta");
    const auto forced = inspector.sample(runtime, true);
    check(forced && forced->complete && forced->entities == large->entities &&
              forced->world_revision == large->world_revision + 1,
          "Native resync did not produce a fresh full hierarchy revision");
    check(!inspector.sample(runtime), "Forced resync incorrectly invalidated hierarchy change coalescing");

    applier.clear();
    runtime.shutdown();
}
} // namespace

int main()
{
    try
    {
        run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Live Play Windows ECS edit test failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}

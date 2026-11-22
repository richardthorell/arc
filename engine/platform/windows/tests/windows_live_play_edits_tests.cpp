#include "../src/windows_live_play_edits.h"

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

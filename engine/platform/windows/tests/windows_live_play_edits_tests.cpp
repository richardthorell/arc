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

// In-memory adapter verifies the shared service contract at the Windows
// command boundary without pretending that a production solver exists.
struct body_state
{
    arc::ecs::entity entity{};
    bool bound{};
    bool held{};
    bool alive{true};
    bool refuse_pose{};
    float linear_velocity{3.0f};
    float angular_velocity{4.0f};
    arc::scene::transform_component pose{};
};

class test_physics_hold final : public arc::scene::physics_transform_hold
{
public:
    explicit test_physics_hold(body_state& state) noexcept : state_(state) {}

    ~test_physics_hold() override
    {
        if (!state_.alive) return;
        state_.held = false;
        state_.linear_velocity = 0.0f;
        state_.angular_velocity = 0.0f;
    }

    [[nodiscard]] bool valid() const noexcept override
    {
        return state_.alive && state_.held;
    }

    [[nodiscard]] bool set_pose(const arc::scene::transform_component& local) noexcept override
    {
        if (!valid() || state_.refuse_pose) return false;
        state_.pose = local;
        return true;
    }

private:
    body_state& state_;
};

class test_physics_service final : public arc::scene::physics_transform_authority_service
{
public:
    explicit test_physics_service(body_state& state) noexcept : state_(state) {}

    [[nodiscard]] arc::scene::physics_transform_owner owner(arc::framework::runtime_world_id,
                                                            arc::ecs::entity entity) const noexcept override
    {
        return state_.bound && state_.alive && entity == state_.entity ? arc::scene::physics_transform_owner::simulated
                                                                       : arc::scene::physics_transform_owner::unbound;
    }

    [[nodiscard]] std::unique_ptr<arc::scene::physics_transform_hold>
    acquire(arc::framework::runtime_world_id world, arc::ecs::entity entity,
            const arc::scene::transform_component& local) noexcept override
    {
        if (owner(world, entity) != arc::scene::physics_transform_owner::simulated || state_.held || state_.refuse_pose)
            return {};
        auto lease = std::make_unique<test_physics_hold>(state_);
        state_.held = true;
        state_.pose = local;
        return lease;
    }

private:
    body_state& state_;
};

class live_edit_test_application final : public arc::framework::application
{
public:
    body_state body;

    void register_services(arc::framework::runtime_service_registry& services) override
    {
        services.emplace<test_physics_service>(body);
    }

    void register_worlds(arc::framework::runtime_world_manager& manager) override
    {
        auto& runtime_world =
            manager.create({.name = "Live Play Test World", .role = arc::framework::runtime_world_role::client});
        auto& world = runtime_world.entities();
        const auto entity = world.create();
        world.emplace<arc::scene::persistent_id_component>(
            entity, arc::scene::persistent_id_component{arc::ecs::entity_guid{3, 4}});
        world.emplace<arc::scene::transform_component>(entity);
        world.emplace<arc::scene::point_light_component>(entity);
        const auto other = world.create();
        world.emplace<arc::scene::persistent_id_component>(
            other, arc::scene::persistent_id_component{arc::ecs::entity_guid{5, 6}});
        world.emplace<arc::scene::transform_component>(other);
        world.emplace<arc::scene::point_light_component>(other);
    }
};

void run()
{
    namespace live = arc::network::live_play_dev;
    live_edit_test_application app;
    arc::framework::runtime runtime(app);
    runtime.start();
    arc::platform::windows::live_play_property_applier applier;

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
    check(result.applied && result.applied_revision == 2,
          "Identical final commit replay did not return its original idempotent ACK");
    edit.values[0] = 99.0f;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "stale_revision",
          "Conflicting duplicate commit was incorrectly acknowledged");
    edit.values[0] = 2.0f;
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

    // L1.4d: a supported authored point light changes only the corresponding
    // runtime ECS component, without rebuilding the scene or writing the
    // author's project. Exactly repeated commits are idempotently ACKed.
    edit.component = "arc.light.point";
    edit.property = "intensity";
    edit.phase = "commit";
    edit.authoring_revision = 3;
    edit.expected_runtime_revision = 3;
    edit.value_count = 1;
    edit.values = {450.0f, 0.0f, 0.0f, 0.0f};
    result = applier.apply(runtime, edit, false);
    check(!result.applied && result.code == "restart_required", "Unactivated baseline accepted a light edit");
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 4, "Authored point light intensity update failed");
    check(world->entities().get<arc::scene::point_light_component>(selected).intensity == 450.0f,
          "Authored light intensity was not applied to the selected ECS entity");
    check(world->entities().get<arc::scene::point_light_component>(unrelated).intensity == 800.0f,
          "Light edit mutated an unrelated runtime entity");
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 4, "Identical light edit retry was not idempotent");
    edit.values[0] = 900.0f;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "stale_revision", "Conflicting duplicate light edit was applied");
    edit.values[0] = 450.0f;
    edit.property = "color";
    edit.authoring_revision = 4;
    edit.expected_runtime_revision = 4;
    edit.value_count = 3;
    edit.values = {0.2f, 0.3f, 0.4f, 0.0f};
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 5, "Authored point light color update failed");
    check(world->entities().get<arc::scene::point_light_component>(selected).color[1] == 0.3f,
          "Authoring light color did not reach native ECS");
    edit.authoring_revision = 5;
    edit.expected_runtime_revision = 5;
    edit.values[0] = -1.0f;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "invalid_request", "Negative light intensity/color was accepted");
    edit.values[0] = 0.1f;
    edit.component = "arc.light.spot";
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "unsupported_operation",
          "A light edit was accepted with a mismatched component identity");
    edit.component = "arc.light.point";
    edit.property = "range";
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "unsupported_operation", "Unknown light property was applied");

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

    // L1.4e.3: a simulated body requires explicit authority, synchronizes
    // before ACK, and releases with zero velocities at its final pose.
    app.body.entity = selected;
    app.body.bound = true;
    edit.component = "arc.transform";
    edit.property = "position";
    edit.phase = "update";
    edit.value_count = 3;
    edit.values = {7.0f, 8.0f, 9.0f, 0.0f};
    edit.authoring_revision = 6;
    edit.expected_runtime_revision = 5;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "permission_denied", "Simulated body accepted an implicit edit");
    edit.phase = "begin";
    app.body.refuse_pose = true;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && !app.body.held &&
              world->entities().get<arc::scene::transform_component>(selected).position[0] == 2.0f,
          "Failed physics acquire changed the ECS pose");
    app.body.refuse_pose = false;
    result = applier.apply(runtime, edit, true);
    check(result.applied && app.body.held && app.body.pose.position[0] == 7.0f, "Physics hold did not synchronize");
    edit.phase = "commit";
    edit.authoring_revision = 7;
    edit.expected_runtime_revision = 6;
    edit.values[0] = 11.0f;
    app.body.refuse_pose = true;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && app.body.pose.position[0] == 7.0f &&
              world->entities().get<arc::scene::transform_component>(selected).position[0] == 7.0f,
          "Failed physics synchronization changed the accepted pose");
    app.body.refuse_pose = false;
    result = applier.apply(runtime, edit, true);
    check(result.applied && app.body.pose.position[0] == 11.0f, "Held body did not receive committed pose");
    edit.phase = "cancel";
    edit.authoring_revision = 8;
    edit.expected_runtime_revision = 7;
    result = applier.apply(runtime, edit, true);
    check(result.applied && !app.body.held && app.body.pose.position[0] == 11.0f && app.body.linear_velocity == 0.0f &&
              app.body.angular_velocity == 0.0f,
          "Physics release did not preserve pose and zero velocities");
    edit.phase = "begin";
    edit.authoring_revision = 9;
    edit.expected_runtime_revision = 8;
    result = applier.apply(runtime, edit, true);
    check(result.applied && app.body.held, "Physics reacquisition failed");
    applier.clear();
    check(!app.body.held, "Disconnect leaked the physics authority lease");
    app.body.bound = false;

    // Reestablish the revision baseline for the independent Flow scenario.
    edit.phase = "update";
    edit.authoring_revision = 1;
    edit.expected_runtime_revision = 0;
    for (int i = 0; i < 5; ++i)
    {
        edit.authoring_revision = static_cast<std::uint64_t>(i + 1);
        result = applier.apply(runtime, edit, true);
        check(result.applied, "Could not establish Flow test revision");
    }

    // L1.4e: gameplay-owned transforms are not implicitly editor-owned.
    // An authenticated begin acquires a bounded session-scoped hold, which
    // executes after simulation but before presentation. Cancel releases it.
    world->entities().emplace<arc::scene::flow_component>(selected);
    edit.component = "arc.transform";
    edit.property = "position";
    edit.phase = "update";
    edit.value_count = 3;
    edit.values = {7.0f, 8.0f, 9.0f, 0.0f};
    edit.authoring_revision = 6;
    edit.expected_runtime_revision = 5;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "permission_denied",
          "Gameplay-owned transform incorrectly accepted an implicit editor edit");
    edit.phase = "begin";
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 6 && applier.held_entity_count() == 1,
          "Explicit begin did not acquire runtime transform authority");
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 6 && applier.held_entity_count() == 1,
          "Duplicate begin changed an already-held transform");
    runtime.set_before_presentation_hook([&] { applier.after_simulation(runtime, true); });
    auto gameplay_pose = world->entities().get<arc::scene::transform_component>(selected);
    gameplay_pose.set_position({60.0f, 70.0f, 80.0f});
    world->entities().emplace<arc::scene::transform_component>(selected, gameplay_pose);
    runtime.advance(1.0 / 60.0);
    check(world->entities().get<arc::scene::transform_component>(selected).position[0] == 7.0f,
          "Held pose was not reasserted at the pre-presentation boundary");
    edit.phase = "commit";
    edit.authoring_revision = 7;
    edit.expected_runtime_revision = 6;
    edit.values = {11.0f, 12.0f, 13.0f, 0.0f};
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 7 && applier.held_entity_count() == 1,
          "Final authored commit should preserve an explicitly held pose");
    edit.phase = "cancel";
    edit.authoring_revision = 8;
    edit.expected_runtime_revision = 7;
    result = applier.apply(runtime, edit, true);
    check(result.applied && result.applied_revision == 8 && applier.held_entity_count() == 0,
          "Explicit cancel did not release gameplay authority");
    gameplay_pose = world->entities().get<arc::scene::transform_component>(selected);
    gameplay_pose.set_position({88.0f, 89.0f, 90.0f});
    world->entities().emplace<arc::scene::transform_component>(selected, gameplay_pose);
    runtime.advance(1.0 / 60.0);
    check(world->entities().get<arc::scene::transform_component>(selected).position[0] == 88.0f,
          "Runtime was still overridden after cancel");
    edit.phase = "begin";
    edit.authoring_revision = 9;
    edit.expected_runtime_revision = 8;
    result = applier.apply(runtime, edit, true);
    check(result.applied && applier.held_entity_count() == 1, "Second explicit hold failed");
    applier.after_simulation(runtime, false);
    check(applier.held_entity_count() == 0, "Baseline loss did not revoke runtime authority");
    runtime.set_before_presentation_hook({});

    // A fresh transport/baseline has a fresh runtime revision authority.
    edit.expected_runtime_revision = 0;
    // A new ECS entity with the same authored GUID must not inherit stale
    // authority or revision state from an earlier generation.
    result = applier.apply(runtime, edit, true);
    check(result.applied && applier.held_entity_count() == 1, "Reacquisition after reconnect failed");
    check(world->entities().destroy(selected), "Could not destroy prior authored entity");
    const auto replacement = world->entities().create();
    world->entities().emplace<arc::scene::persistent_id_component>(
        replacement, arc::scene::persistent_id_component{arc::ecs::entity_guid{3, 4}});
    world->entities().emplace<arc::scene::transform_component>(replacement);
    applier.after_simulation(runtime, true);
    check(applier.held_entity_count() == 0, "Destroyed entity left a stale transform hold");
    edit.phase = "update";
    edit.authoring_revision = 10;
    edit.expected_runtime_revision = 1;
    result = applier.apply(runtime, edit, true);
    check(!result.applied && result.code == "restart_required",
          "New ECS generation inherited stale authored-entity authority");
    check(world->entities().get<arc::scene::transform_component>(replacement).position[0] == 0.0f,
          "Stale hold mutated a replacement entity");

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

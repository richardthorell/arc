#include <arc/ecs/ecs.h>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace arc::ecs::tests
{
struct position
{
    float x{};
    float y{};
};

struct velocity
{
    float x{};
    float y{};
};

struct hidden {};
}

namespace arc::ecs
{
template <>
struct component_traits<tests::position>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.tests.position";
    static constexpr component_type_id id{ 0x1000, 0x1 };
    static constexpr std::array<component_field_descriptor, 2> fields{{
        { 1, "x", "X", reflected_field_kind::floating_point },
        { 2, "y", "Y", reflected_field_kind::floating_point }
    }};
    static constexpr component_descriptor descriptor{
        id, canonical_name, "Position", 1, sizeof(tests::position), alignof(tests::position), fields, false, false
    };
};
}

using namespace arc::ecs;
using namespace arc::ecs::tests;

TEST_CASE("ECS entities and stable paged components")
{
    world owner;
    const entity first = owner.create();
    position* address = &owner.emplace<position>(first, position{ 1.0f, 2.0f });
    for (int index = 0; index < 700; ++index)
    {
        const entity value = owner.create();
        owner.emplace<position>(value, position{ static_cast<float>(index), 0.0f });
    }
    REQUIRE(&std::as_const(owner).get<position>(first) == address);
    REQUIRE(owner.destroy(first));
    REQUIRE_FALSE(owner.alive(first));
}

TEST_CASE("Prepared queries update after structural changes")
{
    world owner;
    owner.prepare_query<position, velocity>();
    const entity first = owner.create();
    owner.emplace<position>(first);
    REQUIRE(owner.view<position, velocity>().entities().empty());
    owner.emplace<velocity>(first);

    std::size_t count{};
    owner.view<position, velocity>().each([&](entity, const position&, const velocity&) { ++count; });
    REQUIRE(count == 1);
    owner.remove<velocity>(first);
    REQUIRE(owner.view<position, velocity>().entities().empty());
}

TEST_CASE("Prepared queries support exclusions and optional declarations")
{
    world owner;
    owner.prepare_typed_query<query_read<position>, query_optional<velocity>, query_exclude<hidden>>();
    const entity visible = owner.create();
    const entity excluded = owner.create();
    owner.emplace<position>(visible);
    owner.emplace<position>(excluded);
    owner.emplace<hidden>(excluded);

    std::vector<entity> matched;
    for (const entity value : owner.query<query_read<position>, query_optional<velocity>, query_exclude<hidden>>())
        matched.push_back(value);
    REQUIRE(matched == std::vector<entity>{ visible });

    owner.remove<hidden>(excluded);
    matched.clear();
    for (const entity value : owner.query<query_read<position>, query_optional<velocity>, query_exclude<hidden>>())
        matched.push_back(value);
    REQUIRE(matched == std::vector<entity>{ visible, excluded });
}

TEST_CASE("Frozen component registries assign deterministic compact indices")
{
    component_type_registry first;
    REQUIRE(first.register_component<velocity>());
    REQUIRE(first.register_component<position>());
    REQUIRE_FALSE(first.register_component<position>());
    REQUIRE(first.freeze());
    REQUIRE_FALSE(first.register_component<hidden>());

    component_type_registry second;
    REQUIRE(second.register_component<position>());
    REQUIRE(second.register_component<velocity>());
    REQUIRE(second.freeze());

    REQUIRE(first.runtime_index(component_type<position>()) ==
        second.runtime_index(component_type<position>()));
    REQUIRE(first.runtime_index(component_type<velocity>()) ==
        second.runtime_index(component_type<velocity>()));
    REQUIRE(first.descriptor(first.runtime_index(component_type<position>())) ==
        &component_metadata<position>());
    REQUIRE(first.runtime_index(component_type<hidden>()) == invalid_runtime_component_index);
}

TEST_CASE("Prepared query iteration performs no tracked allocations")
{
    ::arc::memory_system memory;
    world owner(memory, 77u);
    owner.prepare_typed_query<query_read<position>>();
    for (int index = 0; index < 128; ++index)
        owner.emplace<position>(owner.create(), position{ static_cast<float>(index), 0.0f });

    std::size_t warm_count{};
    for (const entity value : owner.query<query_read<position>>())
        warm_count += value.valid() ? 1u : 0u;
    REQUIRE(warm_count == 128u);
    const auto before = memory.snapshot().tracked_allocation_count;
    for (int repeat = 0; repeat < 100; ++repeat)
    {
        std::size_t count{};
        for (const entity value : owner.query<query_read<position>>())
            count += value.valid() ? 1u : 0u;
        REQUIRE(count == 128u);
    }
    REQUIRE(memory.snapshot().tracked_allocation_count == before);
}

TEST_CASE("World move assignment keeps component storage resources alive")
{
    world destination;
    destination.emplace<position>(destination.create(), position{ 1.0f, 2.0f });

    world source;
    const entity retained = source.create();
    source.emplace<position>(retained, position{ 7.0f, 9.0f });
    source.prepare_typed_query<query_read<position>>();

    destination = std::move(source);
    REQUIRE(destination.live_count() == 1);
    REQUIRE(destination.get<position>(retained).x == 7.0f);
    std::size_t query_count{};
    for (const entity value : destination.query<query_read<position>>())
        query_count += value.valid() ? 1u : 0u;
    REQUIRE(query_count == 1);
}

TEST_CASE("Field revisions preserve independent change cursors")
{
    world owner;
    const entity value = owner.create();
    owner.emplace<position>(value);
    const change_cursor before{ owner.revision() };
    REQUIRE(owner.patch_field<position>(value, 1, [](position& component) { component.x = 42.0f; }));

    const auto changes = owner.changes_since<position>(before);
    const auto found = changes.begin();
    REQUIRE(found != changes.end());
    REQUIRE(((*found).fields & 1u) != 0);
    REQUIRE(std::as_const(owner).get<position>(value).x == 42.0f);
}

TEST_CASE("Dirty consumers and structural tombstones are independent")
{
    world owner;
    const entity value = owner.create();
    owner.emplace<position>(value);
    const change_cursor extraction{ owner.revision() };
    const change_cursor replication = extraction;
    REQUIRE(owner.patch_field<position>(value, 2, [](position& component) { component.y = 9.0f; }));

    const auto extracted = owner.changes_since<position>(extraction);
    const auto replicated = owner.changes_since<position>(replication);
    REQUIRE(extracted.begin() != extracted.end());
    REQUIRE(replicated.begin() != replicated.end());
    REQUIRE(((*extracted.begin()).fields & 2u) != 0u);
    REQUIRE(((*replicated.begin()).fields & 2u) != 0u);

    const auto structural_before = owner.structural_changes().size();
    REQUIRE(owner.remove<position>(value));
    REQUIRE(owner.destroy(value));
    const auto structural = owner.structural_changes();
    REQUIRE(structural.size() >= structural_before + 2u);
    REQUIRE(structural[structural.size() - 2u].kind == structural_change_kind::component_removed);
    REQUIRE(structural.back().kind == structural_change_kind::entity_destroyed);
}

TEST_CASE("Command buffers resolve deferred entities deterministically")
{
    world owner;
    entity_command_buffer later({ 1, 2, 0 });
    entity_command_buffer earlier({ 1, 1, 0 });
    const deferred_entity created = earlier.create();
    earlier.add<position>(created, position{ 7.0f, 9.0f });

    std::vector<entity_command_buffer*> buffers{ &later, &earlier };
    const command_flush_result result = entity_command_buffer::flush_ordered(owner, buffers);
    REQUIRE(result.succeeded());
    REQUIRE(result.applied == 2);
    REQUIRE(owner.live_count() == 1);
    REQUIRE(std::as_const(owner).view<position>().entities().begin() !=
        std::as_const(owner).view<position>().entities().end());
}

TEST_CASE("System scheduler orders write conflicts and permits commands")
{
    world owner;
    const entity value = owner.create();
    owner.emplace<position>(value);
    arc::job_system jobs(arc::job_system::single_threaded_config());
    system_scheduler scheduler;
    std::vector<int> order;

    REQUIRE(scheduler.add({
        .name = "move",
        .components = { writes<position>() },
        .execute = [&](system_context& context) {
            order.push_back(1);
            context.write<position>(value)->x = 5.0f;
        }
    }));
    REQUIRE(scheduler.add({
        .name = "observe",
        .components = { reads<position>() },
        .execute = [&](system_context& context) {
            order.push_back(2);
            REQUIRE(context.read<position>(value)->x == 5.0f);
            const deferred_entity made = context.commands().create();
            context.commands().add<velocity>(made, velocity{ 1.0f, 1.0f });
        }
    }));

    const system_run_result result = scheduler.run(owner, jobs, 1.0f / 60.0f);
    REQUIRE(result.succeeded());
    REQUIRE(order == std::vector<int>{ 1, 2 });
    REQUIRE(owner.live_count() == 2);
}

TEST_CASE("System scheduler honors forward dependencies and validates declared access")
{
    world owner;
    const entity value = owner.create();
    owner.emplace<position>(value);
    arc::job_system jobs(arc::job_system::single_threaded_config());
    system_scheduler ordered;
    std::vector<int> order;

    REQUIRE(ordered.add({
        .name = "consumer",
        .components = { reads<position>() },
        .after = { "producer" },
        .execute = [&](system_context& context) {
            order.push_back(2);
            REQUIRE(context.read<position>(value)->x == 11.0f);
        }
    }));
    REQUIRE(ordered.add({
        .name = "producer",
        .components = { writes<position>() },
        .execute = [&](system_context& context) {
            order.push_back(1);
            context.write<position>(value)->x = 11.0f;
        }
    }));
    REQUIRE(ordered.run(owner, jobs, 1.0f / 60.0f).succeeded());
    REQUIRE(order == std::vector<int>{ 1, 2 });

    system_scheduler invalid;
    REQUIRE(invalid.add({
        .name = "undeclared",
        .components = { reads<position>() },
        .execute = [&](system_context& context) {
            (void)context.write<position>(value);
        }
    }));
    const auto result = invalid.run(owner, jobs, 1.0f / 60.0f);
    REQUIRE_FALSE(result.succeeded());
    REQUIRE(result.errors.size() == 1);
    REQUIRE(result.errors.front().system == "undeclared");
}

TEST_CASE("System schedules freeze after validation and reject impossible phase dependencies")
{
    system_scheduler valid;
    REQUIRE(valid.add({
        .name = "input",
        .phase = system_phase::input,
        .execute = [](system_context&) {}
    }));
    REQUIRE(valid.add({
        .name = "movement",
        .phase = system_phase::movement,
        .after = { "input" },
        .execute = [](system_context&) {}
    }));
    REQUIRE(valid.freeze().empty());
    REQUIRE(valid.frozen());
    REQUIRE_FALSE(valid.add({
        .name = "late",
        .execute = [](system_context&) {}
    }));
    REQUIRE_FALSE(valid.remove("input"));

    system_scheduler invalid;
    REQUIRE(invalid.add({
        .name = "future",
        .phase = system_phase::physics,
        .execute = [](system_context&) {}
    }));
    REQUIRE(invalid.add({
        .name = "past",
        .phase = system_phase::movement,
        .after = { "future" },
        .execute = [](system_context&) {}
    }));
    const auto errors = invalid.freeze();
    REQUIRE(errors.size() == 1);
    REQUIRE_FALSE(invalid.frozen());
}

TEST_CASE("Intrusive hierarchy traverses without snapshots")
{
    world owner;
    const entity root = owner.create();
    const entity first = owner.create();
    const entity second = owner.create();
    owner.emplace<hierarchy_component>(root);
    owner.emplace<hierarchy_component>(first);
    owner.emplace<hierarchy_component>(second);
    REQUIRE(reparent(owner, first, root));
    REQUIRE(reparent(owner, second, root));

    std::vector<entity> values;
    for (const entity child : children(owner, root))
        values.push_back(child);
    REQUIRE(values == std::vector<entity>{ first, second });
    REQUIRE(is_descendant(owner, second, root));
}

TEST_CASE("Templates, prefab overrides, and regions expose stable contracts")
{
    entity_template value = entity_template_builder("Mover")
        .component(position{ 2.0f, 3.0f })
        .component(velocity{ 1.0f, 0.0f })
        .build();
    world owner;
    entity_command_buffer commands;
    value.instantiate(commands);
    REQUIRE(commands.flush(owner).succeeded());
    REQUIRE(owner.view<position, velocity>().entities().begin() !=
        owner.view<position, velocity>().entities().end());

    prefab_instance_component instance;
    prefab_override override_value{
        .key = { generate_entity_guid(), component_type<position>(), 1, prefab_override_kind::field },
        .value = { std::byte{ 1 } }
    };
    REQUIRE(set_prefab_override(instance, override_value));
    REQUIRE(has_prefab_override(instance, override_value.key));
    REQUIRE(revert_prefab_override(instance, override_value.key));

    world_partition partition;
    const world_region_id region{ generate_entity_guid() };
    REQUIRE(partition.add_region({
        .id = region,
        .name = "Main",
        .state = world_region_state::loaded,
        .always_loaded = true
    }));
    const entity region_entity = owner.create();
    REQUIRE(partition.assign(owner, region_entity, region));
}

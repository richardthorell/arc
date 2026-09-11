#include <arc/flow/flow.h>

#include <arc/project/runtime_world_api.h>

#include <array>
#include <cassert>
#include <string>

namespace
{

using namespace arc::flow;
using namespace arc::project;

struct mock_world
{
    game_entity_target_v1 create_result{};
    game_entity_target_v1 last_destroy_target{};
    game_entity_target_v1 last_name_target{};
    game_entity_target_v1 last_transform_target{};
    game_entity_target_v1 last_tag_target{};
    game_entity_target_v1 last_active_target{};
    game_entity_target_v1 last_remove_target{};
    game_core_component_v1 last_removed_component{game_core_component_v1::name};
    std::string name{"Initial"};
    std::string tag{"Default"};
    game_transform_v1 transform{};
    bool active{true};
    bool alive{true};
    bool has_component{true};
    std::uint32_t create_calls{};
    std::uint32_t destroy_calls{};
    std::uint32_t remove_calls{};
};

mock_world& state(void* user_data)
{
    return *static_cast<mock_world*>(user_data);
}

game_entity_target_v1 create_entity(void* user_data)
{
    ++state(user_data).create_calls;
    return state(user_data).create_result;
}

bool destroy_entity(void* user_data, game_entity_target_v1 entity)
{
    mock_world& world = state(user_data);
    ++world.destroy_calls;
    world.last_destroy_target = entity;
    return entity.valid();
}

bool entity_alive(void* user_data, game_entity_v1 entity)
{
    return state(user_data).alive && entity.valid();
}

bool has_core_component(void* user_data, game_entity_v1 entity, game_core_component_v1)
{
    return state(user_data).has_component && entity.valid();
}

bool remove_core_component(void* user_data, game_entity_target_v1 entity, game_core_component_v1 component)
{
    mock_world& world = state(user_data);
    ++world.remove_calls;
    world.last_remove_target = entity;
    world.last_removed_component = component;
    return entity.valid();
}

game_string_view_v1 read_name(void* user_data, game_entity_v1 entity)
{
    if (!entity.valid()) return {};
    const std::string& value = state(user_data).name;
    return {.data = value.data(), .size = value.size()};
}

bool set_name(void* user_data, game_entity_target_v1 entity, const char* value, std::size_t size)
{
    if (!entity.valid()) return false;
    mock_world& world = state(user_data);
    world.last_name_target = entity;
    world.name.assign(value, size);
    return true;
}

bool read_transform(void* user_data, game_entity_v1 entity, game_transform_v1* value)
{
    if (!entity.valid() || !value) return false;
    *value = state(user_data).transform;
    return true;
}

bool set_transform(void* user_data, game_entity_target_v1 entity, const game_transform_v1* value)
{
    if (!entity.valid() || !value) return false;
    mock_world& world = state(user_data);
    world.last_transform_target = entity;
    world.transform = *value;
    return true;
}

game_string_view_v1 read_tag(void* user_data, game_entity_v1 entity)
{
    if (!entity.valid()) return {};
    const std::string& value = state(user_data).tag;
    return {.data = value.data(), .size = value.size()};
}

bool set_tag(void* user_data, game_entity_target_v1 entity, const char* value, std::size_t size)
{
    if (!entity.valid()) return false;
    mock_world& world = state(user_data);
    world.last_tag_target = entity;
    world.tag.assign(value, size);
    return true;
}

bool read_active(void* user_data, game_entity_v1 entity, bool* value)
{
    if (!entity.valid() || !value) return false;
    *value = state(user_data).active;
    return true;
}

bool set_active(void* user_data, game_entity_target_v1 entity, bool value)
{
    if (!entity.valid()) return false;
    mock_world& world = state(user_data);
    world.last_active_target = entity;
    world.active = value;
    return true;
}

game_world_api_v1 make_api(mock_world& world)
{
    game_world_api_v1 api;
    api.user_data = &world;
    api.create_entity = create_entity;
    api.destroy_entity = destroy_entity;
    api.entity_alive = entity_alive;
    api.has_core_component = has_core_component;
    api.remove_core_component = remove_core_component;
    api.read_name = read_name;
    api.set_name = set_name;
    api.read_transform = read_transform;
    api.set_transform = set_transform;
    api.read_tag = read_tag;
    api.set_tag = set_tag;
    api.read_active = read_active;
    api.set_active = set_active;
    return api;
}

bytecode_value_slot entity_slot()
{
    return {.type = value_type::entity, .initial_value = std::monostate{}};
}

} // namespace

void run_flow_world_tests()
{
    using namespace arc::flow;
    using namespace arc::project;

    {
        mock_world world;
        game_world_api_v1 api = make_api(world);

        bytecode_program program;
        program.value_slots = {
            entity_slot(),
            {.type = value_type::string, .initial_value = std::string{"Player"}},
            {.type = value_type::vector3, .initial_value = std::array<double, 3>{1.0, 2.0, 3.0}},
            {.type = value_type::vector4, .initial_value = std::array<double, 4>{0.0, 0.0, 0.0, 1.0}},
            {.type = value_type::vector3, .initial_value = std::array<double, 3>{2.0, 2.0, 2.0}},
            {.type = value_type::string, .initial_value = std::string{"Hero"}},
            {.type = value_type::boolean, .initial_value = false},
        };
        program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
        program.instructions = {
            {.opcode = bytecode_opcode::self_entity, .operand0 = 0, .operand1 = 1},
            {.opcode = bytecode_opcode::world_set_name, .operand0 = 0, .operand1 = 1, .operand2 = 2},
            {.opcode = bytecode_opcode::world_set_transform,
             .operand0 = 0,
             .operand1 = 2,
             .operand2 = 3,
             .operand3 = 4,
             .operand4 = 3},
            {.opcode = bytecode_opcode::world_set_tag, .operand0 = 0, .operand1 = 5, .operand2 = 4},
            {.opcode = bytecode_opcode::world_set_active,
             .operand0 = 0,
             .operand1 = 6,
             .operand2 = invalid_instruction},
        };
        program.instruction_nodes = {"self", "name", "transform", "tag", "active"};

        vm_instance instance{program};
        const vm_world_context context{.api = &api, .self = {.index = 7, .generation = 3}};
        const execution_result result = instance.begin_play(context);
        assert(result.succeeded());
        assert(result.instructions_executed == 5);
        assert(world.name == "Player");
        assert(world.tag == "Hero");
        assert(!world.active);
        assert(world.transform.position.x == 1.0f && world.transform.position.y == 2.0f &&
               world.transform.position.z == 3.0f);
        assert(world.transform.scale.x == 2.0f && world.transform.scale.y == 2.0f &&
               world.transform.scale.z == 2.0f);
        assert(world.last_name_target.entity.index == 7);
        assert(!world.last_name_target.is_deferred);
    }

    {
        mock_world world;
        world.name = "ReadName";
        world.tag = "ReadTag";
        world.active = false;
        world.transform.position = {4.0f, 5.0f, 6.0f};
        world.transform.rotation = {0.0f, 0.0f, 0.5f, 0.5f};
        world.transform.scale = {3.0f, 3.0f, 3.0f};
        game_world_api_v1 api = make_api(world);

        bytecode_program program;
        program.value_slots = {
            entity_slot(),
            {.type = value_type::boolean, .initial_value = false},
            {.type = value_type::boolean, .initial_value = false},
            {.type = value_type::string, .initial_value = std::string{}},
            {.type = value_type::vector3, .initial_value = std::array<double, 3>{0.0, 0.0, 0.0}},
            {.type = value_type::vector4, .initial_value = std::array<double, 4>{0.0, 0.0, 0.0, 1.0}},
            {.type = value_type::vector3, .initial_value = std::array<double, 3>{1.0, 1.0, 1.0}},
            {.type = value_type::string, .initial_value = std::string{}},
            {.type = value_type::boolean, .initial_value = true},
        };
        program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
        program.instructions = {
            {.opcode = bytecode_opcode::self_entity, .operand0 = 0, .operand1 = 1},
            {.opcode = bytecode_opcode::world_entity_alive, .operand0 = 0, .operand1 = 1, .operand2 = 2},
            {.opcode = bytecode_opcode::world_has_core_component,
             .operand0 = 0,
             .operand1 = static_cast<std::uint32_t>(world_core_component::transform),
             .operand2 = 2,
             .operand3 = 3},
            {.opcode = bytecode_opcode::world_get_name, .operand0 = 0, .operand1 = 3, .operand2 = 4},
            {.opcode = bytecode_opcode::world_get_transform,
             .operand0 = 0,
             .operand1 = 4,
             .operand2 = 5,
             .operand3 = 6,
             .operand4 = 5},
            {.opcode = bytecode_opcode::world_get_tag, .operand0 = 0, .operand1 = 7, .operand2 = 6},
            {.opcode = bytecode_opcode::world_get_active,
             .operand0 = 0,
             .operand1 = 8,
             .operand2 = invalid_instruction},
        };

        vm_instance instance{program};
        const vm_world_context context{.api = &api, .self = {.index = 9, .generation = 1}};
        assert(instance.begin_play(context).succeeded());
        assert(std::get<bool>(*instance.value_slot(1)));
        assert(std::get<bool>(*instance.value_slot(2)));
        assert(std::get<std::string>(*instance.value_slot(3)) == "ReadName");
        assert((std::get<std::array<double, 3>>(*instance.value_slot(4)) == std::array<double, 3>{4.0, 5.0, 6.0}));
        assert((std::get<std::array<double, 4>>(*instance.value_slot(5)) ==
                std::array<double, 4>{0.0, 0.0, 0.5, 0.5}));
        assert((std::get<std::array<double, 3>>(*instance.value_slot(6)) == std::array<double, 3>{3.0, 3.0, 3.0}));
        assert(std::get<std::string>(*instance.value_slot(7)) == "ReadTag");
        assert(!std::get<bool>(*instance.value_slot(8)));
    }

    {
        mock_world world;
        world.create_result.is_deferred = true;
        world.create_result.deferred = {.buffer = 77, .ordinal = 5};
        game_world_api_v1 api = make_api(world);

        bytecode_program program;
        program.value_slots = {
            entity_slot(),
            {.type = value_type::string, .initial_value = std::string{"Spawned"}},
        };
        program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
        program.instructions = {
            {.opcode = bytecode_opcode::world_create_entity, .operand0 = 0, .operand1 = 1},
            {.opcode = bytecode_opcode::world_set_name, .operand0 = 0, .operand1 = 1, .operand2 = 2},
            {.opcode = bytecode_opcode::world_destroy_entity, .operand0 = 0, .operand1 = invalid_instruction},
        };

        vm_instance instance{program};
        assert(instance.begin_play({.api = &api}).succeeded());
        assert(world.create_calls == 1);
        assert(world.destroy_calls == 1);
        assert(world.last_name_target.is_deferred);
        assert(world.last_name_target.deferred.buffer == 77);
        assert(world.last_name_target.deferred.ordinal == 5);
        assert(world.last_destroy_target.is_deferred);
        const flow_entity& created = std::get<flow_entity>(*instance.value_slot(0));
        assert(created.deferred);
        assert(created.deferred_buffer == 77);
        assert(created.deferred_ordinal == 5);
    }

    {
        bytecode_program program;
        program.value_slots.push_back(entity_slot());
        program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
        program.instructions.push_back(
            {.opcode = bytecode_opcode::world_create_entity, .operand0 = 0, .operand1 = invalid_instruction});
        program.instruction_nodes.push_back("spawn");

        vm_instance instance{program};
        const execution_result result = instance.begin_play();
        assert(result.status == execution_status::world_unavailable);
        assert(result.stopped_instruction == 0);
        assert(result.node_id == "spawn");
        assert(!instance.active());
    }

    {
        bytecode_program invalid;
        invalid.value_slots.push_back({.type = value_type::float32, .initial_value = 0.0});
        invalid.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
        invalid.instructions.push_back(
            {.opcode = bytecode_opcode::world_create_entity, .operand0 = 0, .operand1 = invalid_instruction});

        vm_instance instance{invalid};
        assert(!instance.valid());
    }
}

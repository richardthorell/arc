#include <arc/flow/flow.h>

#include <arc/project/runtime_world_api.h>

#include <algorithm>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

namespace arc::flow
{
namespace
{

bool value_matches_type(value_type type, const flow_value& value)
{
    switch (type)
    {
        case value_type::boolean:
            return std::holds_alternative<bool>(value);
        case value_type::integer:
            return std::holds_alternative<std::int64_t>(value);
        case value_type::float32:
            return std::holds_alternative<double>(value);
        case value_type::vector2:
            return std::holds_alternative<std::array<double, 2>>(value);
        case value_type::vector3:
            return std::holds_alternative<std::array<double, 3>>(value);
        case value_type::vector4:
            return std::holds_alternative<std::array<double, 4>>(value);
        case value_type::string:
        case value_type::name:
            return std::holds_alternative<std::string>(value);
        case value_type::entity:
            return std::holds_alternative<std::monostate>(value) || std::holds_alternative<flow_entity>(value);
        case value_type::component:
            return std::holds_alternative<std::monostate>(value);
    }
    return false;
}

bool valid_instruction_target(const bytecode_program& program, std::uint32_t instruction)
{
    return instruction == invalid_instruction || instruction < program.instructions.size();
}

bool slot_has_type(const bytecode_program& program, std::uint32_t slot, value_type type)
{
    return slot < program.value_slots.size() && program.value_slots[slot].type == type;
}

bool valid_world_component(std::uint32_t value)
{
    return value <= static_cast<std::uint32_t>(world_core_component::active);
}

bool binding_matches_slot(entry_value_kind source, const bytecode_value_slot& slot)
{
    switch (source)
    {
        case entry_value_kind::delta_seconds:
        case entry_value_kind::input_action_value:
            return slot.type == value_type::float32;
    }
    return false;
}

bool validate_program(const bytecode_program& program)
{
    if (program.version != flow_bytecode_version) return false;
    if (!program.instruction_nodes.empty() && program.instruction_nodes.size() != program.instructions.size())
        return false;

    std::unordered_set<std::string> variable_ids;
    for (const variable& item : program.variables)
    {
        if (item.id.empty() || !variable_ids.insert(item.id).second ||
            !value_matches_type(item.type, item.default_value))
            return false;
    }

    for (const bytecode_value_slot& slot : program.value_slots)
        if (!value_matches_type(slot.type, slot.initial_value)) return false;

    for (const bytecode_entry_point& entry : program.entry_points)
    {
        if (!valid_instruction_target(program, entry.instruction)) return false;
        if ((entry.kind == entry_point_kind::input_action_triggered ||
             entry.kind == entry_point_kind::input_action_completed) &&
            entry.action.empty())
            return false;

        for (const value_binding& binding : entry.value_bindings)
        {
            if (binding.slot >= program.value_slots.size()) return false;
            if (!binding_matches_slot(binding.source, program.value_slots[binding.slot])) return false;
        }
    }

    for (const bytecode_instruction& instruction : program.instructions)
    {
        switch (instruction.opcode)
        {
            case bytecode_opcode::branch:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean)) return false;
                if (!valid_instruction_target(program, instruction.operand1) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::self_entity:
            case bytecode_opcode::world_create_entity:
            case bytecode_opcode::world_destroy_entity:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !valid_instruction_target(program, instruction.operand1))
                    return false;
                break;
            case bytecode_opcode::world_entity_alive:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !slot_has_type(program, instruction.operand1, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::world_has_core_component:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !valid_world_component(instruction.operand1) ||
                    !slot_has_type(program, instruction.operand2, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::world_remove_core_component:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !valid_world_component(instruction.operand1) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::world_get_name:
            case bytecode_opcode::world_set_name:
            case bytecode_opcode::world_get_tag:
            case bytecode_opcode::world_set_tag:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !slot_has_type(program, instruction.operand1, value_type::string) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::world_get_transform:
            case bytecode_opcode::world_set_transform:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !slot_has_type(program, instruction.operand1, value_type::vector3) ||
                    !slot_has_type(program, instruction.operand2, value_type::vector4) ||
                    !slot_has_type(program, instruction.operand3, value_type::vector3) ||
                    !valid_instruction_target(program, instruction.operand4))
                    return false;
                break;
            case bytecode_opcode::world_get_active:
            case bytecode_opcode::world_set_active:
                if (!slot_has_type(program, instruction.operand0, value_type::entity) ||
                    !slot_has_type(program, instruction.operand1, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
        }
    }

    return true;
}

std::string instruction_node(const bytecode_program& program, std::uint32_t instruction)
{
    if (instruction >= program.instruction_nodes.size()) return {};
    return program.instruction_nodes[instruction];
}

execution_result status_result(execution_status status)
{
    execution_result result;
    result.status = status;
    return result;
}

void stop_execution(execution_result& result, execution_status status, const bytecode_program& program,
                    std::uint32_t instruction)
{
    result.status = status;
    result.stopped_instruction = instruction;
    result.node_id = instruction_node(program, instruction);
}

project::game_entity_target_v1 to_game_target(const flow_entity& entity)
{
    project::game_entity_target_v1 result;
    result.is_deferred = entity.deferred;
    if (entity.deferred)
    {
        result.deferred.buffer = entity.deferred_buffer;
        result.deferred.ordinal = entity.deferred_ordinal;
    }
    else
    {
        result.entity.index = entity.index;
        result.entity.generation = entity.generation;
    }
    return result;
}

std::optional<project::game_entity_v1> to_game_entity(const flow_entity& entity)
{
    if (!entity.valid() || entity.deferred) return std::nullopt;
    return project::game_entity_v1{.index = entity.index, .generation = entity.generation};
}

flow_entity from_game_target(const project::game_entity_target_v1& entity)
{
    flow_entity result;
    result.deferred = entity.is_deferred;
    if (entity.is_deferred)
    {
        result.deferred_buffer = entity.deferred.buffer;
        result.deferred_ordinal = entity.deferred.ordinal;
    }
    else
    {
        result.index = entity.entity.index;
        result.generation = entity.entity.generation;
    }
    return result;
}

project::game_core_component_v1 to_game_component(world_core_component component)
{
    switch (component)
    {
        case world_core_component::name:
            return project::game_core_component_v1::name;
        case world_core_component::transform:
            return project::game_core_component_v1::transform;
        case world_core_component::tag:
            return project::game_core_component_v1::tag;
        case world_core_component::active:
            return project::game_core_component_v1::active;
    }
    return project::game_core_component_v1::transform;
}

const flow_entity* entity_value(const std::vector<flow_value>& slots, std::uint32_t slot)
{
    if (slot >= slots.size()) return nullptr;
    return std::get_if<flow_entity>(&slots[slot]);
}

bool require_world(const vm_world_context& world, execution_result& result, const bytecode_program& program,
                   std::uint32_t instruction)
{
    if (world.api) return true;
    stop_execution(result, execution_status::world_unavailable, program, instruction);
    return false;
}

execution_result execute_chain(const bytecode_program& program, std::vector<flow_value>& value_slots,
                               std::uint32_t first_instruction, std::uint32_t instruction_budget,
                               const vm_world_context& world)
{
    execution_result result;
    std::uint32_t instruction = first_instruction;

    while (instruction != invalid_instruction)
    {
        if (instruction >= program.instructions.size())
        {
            stop_execution(result, execution_status::invalid_program, program, instruction);
            return result;
        }
        if (result.instructions_executed >= instruction_budget)
        {
            stop_execution(result, execution_status::instruction_budget_exceeded, program, instruction);
            return result;
        }

        const bytecode_instruction& current = program.instructions[instruction];
        ++result.instructions_executed;

        switch (current.opcode)
        {
            case bytecode_opcode::branch:
            {
                const bool* condition = std::get_if<bool>(&value_slots[current.operand0]);
                if (!condition)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                instruction = *condition ? current.operand1 : current.operand2;
                break;
            }
            case bytecode_opcode::self_entity:
                if (!world.self.valid())
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand0] = world.self;
                instruction = current.operand1;
                break;
            case bytecode_opcode::world_create_entity:
            {
                if (!require_world(world, result, program, instruction)) return result;
                if (!world.api->create_entity)
                {
                    stop_execution(result, execution_status::world_unavailable, program, instruction);
                    return result;
                }
                const project::game_entity_target_v1 created = world.api->create_entity(world.api->user_data);
                if (!created.valid())
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand0] = from_game_target(created);
                instruction = current.operand1;
                break;
            }
            case bytecode_opcode::world_destroy_entity:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                if (!entity || !entity->valid() || !world.api->destroy_entity ||
                    !world.api->destroy_entity(world.api->user_data, to_game_target(*entity)))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                instruction = current.operand1;
                break;
            }
            case bytecode_opcode::world_entity_alive:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto immediate = entity ? to_game_entity(*entity) : std::nullopt;
                if (!immediate || !world.api->entity_alive)
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = world.api->entity_alive(world.api->user_data, *immediate);
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::world_has_core_component:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto immediate = entity ? to_game_entity(*entity) : std::nullopt;
                if (!immediate || !world.api->has_core_component)
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                const auto component = static_cast<world_core_component>(current.operand1);
                value_slots[current.operand2] =
                    world.api->has_core_component(world.api->user_data, *immediate, to_game_component(component));
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::world_remove_core_component:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                if (!entity || !entity->valid() || !world.api->remove_core_component)
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                const auto component = static_cast<world_core_component>(current.operand1);
                if (!world.api->remove_core_component(world.api->user_data, to_game_target(*entity),
                                                      to_game_component(component)))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::world_get_name:
            case bytecode_opcode::world_get_tag:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto immediate = entity ? to_game_entity(*entity) : std::nullopt;
                if (!immediate)
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                const bool name = current.opcode == bytecode_opcode::world_get_name;
                const auto reader = name ? world.api->read_name : world.api->read_tag;
                if (!reader)
                {
                    stop_execution(result, execution_status::world_unavailable, program, instruction);
                    return result;
                }
                const project::game_string_view_v1 text = reader(world.api->user_data, *immediate);
                if (!text.valid())
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = std::string{text.data, text.size};
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::world_set_name:
            case bytecode_opcode::world_set_tag:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const std::string* text = std::get_if<std::string>(&value_slots[current.operand1]);
                if (!entity || !entity->valid() || !text)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                const bool name = current.opcode == bytecode_opcode::world_set_name;
                const auto writer = name ? world.api->set_name : world.api->set_tag;
                if (!writer)
                {
                    stop_execution(result, execution_status::world_unavailable, program, instruction);
                    return result;
                }
                if (!writer(world.api->user_data, to_game_target(*entity), text->data(), text->size()))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::world_get_transform:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto immediate = entity ? to_game_entity(*entity) : std::nullopt;
                if (!immediate || !world.api->read_transform)
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                project::game_transform_v1 transform;
                if (!world.api->read_transform(world.api->user_data, *immediate, &transform))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand1] =
                    std::array<double, 3>{transform.position.x, transform.position.y, transform.position.z};
                value_slots[current.operand2] = std::array<double, 4>{transform.rotation.x, transform.rotation.y,
                                                                      transform.rotation.z, transform.rotation.w};
                value_slots[current.operand3] =
                    std::array<double, 3>{transform.scale.x, transform.scale.y, transform.scale.z};
                instruction = current.operand4;
                break;
            }
            case bytecode_opcode::world_set_transform:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto* position = std::get_if<std::array<double, 3>>(&value_slots[current.operand1]);
                const auto* rotation = std::get_if<std::array<double, 4>>(&value_slots[current.operand2]);
                const auto* scale = std::get_if<std::array<double, 3>>(&value_slots[current.operand3]);
                if (!entity || !entity->valid() || !position || !rotation || !scale || !world.api->set_transform)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                project::game_transform_v1 transform;
                transform.position = {static_cast<float>((*position)[0]), static_cast<float>((*position)[1]),
                                      static_cast<float>((*position)[2])};
                transform.rotation = {static_cast<float>((*rotation)[0]), static_cast<float>((*rotation)[1]),
                                      static_cast<float>((*rotation)[2]), static_cast<float>((*rotation)[3])};
                transform.scale = {static_cast<float>((*scale)[0]), static_cast<float>((*scale)[1]),
                                   static_cast<float>((*scale)[2])};
                if (!world.api->set_transform(world.api->user_data, to_game_target(*entity), &transform))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                instruction = current.operand4;
                break;
            }
            case bytecode_opcode::world_get_active:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const auto immediate = entity ? to_game_entity(*entity) : std::nullopt;
                bool active = false;
                if (!immediate || !world.api->read_active ||
                    !world.api->read_active(world.api->user_data, *immediate, &active))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = active;
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::world_set_active:
            {
                if (!require_world(world, result, program, instruction)) return result;
                const flow_entity* entity = entity_value(value_slots, current.operand0);
                const bool* active = std::get_if<bool>(&value_slots[current.operand1]);
                if (!entity || !entity->valid() || !active || !world.api->set_active)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (!world.api->set_active(world.api->user_data, to_game_target(*entity), *active))
                {
                    stop_execution(result, execution_status::world_operation_failed, program, instruction);
                    return result;
                }
                instruction = current.operand2;
                break;
            }
        }
    }

    return result;
}

bool entry_matches(const bytecode_entry_point& entry, entry_point_kind kind, std::string_view action)
{
    if (entry.kind != kind) return false;
    if (kind == entry_point_kind::input_action_triggered || kind == entry_point_kind::input_action_completed)
        return entry.action == action;
    return true;
}

void bind_entry_values(const bytecode_entry_point& entry, std::vector<flow_value>& value_slots, double event_value)
{
    for (const value_binding& binding : entry.value_bindings)
    {
        switch (binding.source)
        {
            case entry_value_kind::delta_seconds:
            case entry_value_kind::input_action_value:
                value_slots[binding.slot] = event_value;
                break;
        }
    }
}

execution_result execute_event(const bytecode_program& program, std::vector<flow_value>& value_slots,
                               entry_point_kind kind, std::string_view action, double event_value,
                               std::uint32_t instruction_budget, const vm_world_context& world)
{
    execution_result result;

    for (const bytecode_entry_point& entry : program.entry_points)
    {
        if (!entry_matches(entry, kind, action)) continue;

        ++result.entry_points_executed;
        bind_entry_values(entry, value_slots, event_value);

        const std::uint32_t remaining_budget = instruction_budget - result.instructions_executed;
        execution_result entry_result = execute_chain(program, value_slots, entry.instruction, remaining_budget, world);
        result.instructions_executed += entry_result.instructions_executed;

        if (!entry_result.succeeded())
        {
            result.status = entry_result.status;
            result.stopped_instruction = entry_result.stopped_instruction;
            result.node_id = std::move(entry_result.node_id);
            return result;
        }
    }

    return result;
}

} // namespace

vm_instance::vm_instance(const bytecode_program& program, vm_limits limits)
    : program_(&program), limits_(limits), valid_(validate_program(program))
{
    reset();
}

vm_instance::vm_instance(vm_instance&& other) noexcept
    : program_(std::exchange(other.program_, nullptr)), variable_values_(std::move(other.variable_values_)),
      value_slots_(std::move(other.value_slots_)), limits_(other.limits_), valid_(std::exchange(other.valid_, false)),
      active_(std::exchange(other.active_, false))
{
}

vm_instance& vm_instance::operator=(vm_instance&& other) noexcept
{
    if (this == &other) return *this;

    program_ = std::exchange(other.program_, nullptr);
    variable_values_ = std::move(other.variable_values_);
    value_slots_ = std::move(other.value_slots_);
    limits_ = other.limits_;
    valid_ = std::exchange(other.valid_, false);
    active_ = std::exchange(other.active_, false);
    return *this;
}

bool vm_instance::valid() const noexcept
{
    return valid_;
}

bool vm_instance::active() const noexcept
{
    return active_;
}

vm_limits vm_instance::limits() const noexcept
{
    return limits_;
}

void vm_instance::reset()
{
    variable_values_.clear();
    value_slots_.clear();
    active_ = false;

    if (!program_) return;

    variable_values_.reserve(program_->variables.size());
    for (const variable& item : program_->variables)
        variable_values_.push_back(item.default_value);

    value_slots_.reserve(program_->value_slots.size());
    for (const bytecode_value_slot& slot : program_->value_slots)
        value_slots_.push_back(slot.initial_value);
}

const flow_value* vm_instance::variable_value(std::string_view id) const noexcept
{
    if (!program_) return nullptr;
    for (std::size_t index = 0; index < program_->variables.size(); ++index)
        if (program_->variables[index].id == id) return &variable_values_[index];
    return nullptr;
}

bool vm_instance::set_variable_value(std::string_view id, const flow_value& value)
{
    if (!program_) return false;
    for (std::size_t index = 0; index < program_->variables.size(); ++index)
    {
        const variable& item = program_->variables[index];
        if (item.id != id) continue;
        if (!value_matches_type(item.type, value)) return false;
        variable_values_[index] = value;
        return true;
    }
    return false;
}

const flow_value* vm_instance::value_slot(std::uint32_t slot) const noexcept
{
    if (slot >= value_slots_.size()) return nullptr;
    return &value_slots_[slot];
}

execution_result vm_instance::begin_play(vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (active_) return status_result(execution_status::already_active);

    active_ = true;
    execution_result result = execute_event(*program_, value_slots_, entry_point_kind::begin_play, {}, 0.0,
                                            limits_.instruction_budget, world);
    if (!result.succeeded()) active_ = false;
    return result;
}

execution_result vm_instance::end_play(vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);

    execution_result result =
        execute_event(*program_, value_slots_, entry_point_kind::end_play, {}, 0.0, limits_.instruction_budget, world);
    active_ = false;
    return result;
}

execution_result vm_instance::tick(double delta_seconds, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::tick, {}, delta_seconds, limits_.instruction_budget,
                         world);
}

execution_result vm_instance::fixed_tick(double delta_seconds, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::fixed_tick, {}, delta_seconds,
                         limits_.instruction_budget, world);
}

execution_result vm_instance::input_action_triggered(std::string_view action, double value, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::input_action_triggered, action, value,
                         limits_.instruction_budget, world);
}

execution_result vm_instance::input_action_completed(std::string_view action, double value, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::input_action_completed, action, value,
                         limits_.instruction_budget, world);
}

} // namespace arc::flow

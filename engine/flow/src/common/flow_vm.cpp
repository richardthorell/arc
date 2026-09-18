#include <arc/flow/flow.h>

#include <arc/project/runtime_world_api.h>

#include <algorithm>
#include <cmath>
#include <limits>
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

bool slot_exists(const bytecode_program& program, std::uint32_t slot)
{
    return slot < program.value_slots.size();
}

bool same_slot_type(const bytecode_program& program, std::uint32_t left, std::uint32_t right)
{
    return slot_exists(program, left) && slot_exists(program, right) &&
           program.value_slots[left].type == program.value_slots[right].type;
}

bool is_numeric_type(value_type type)
{
    return type == value_type::integer || type == value_type::float32;
}

bool is_arithmetic_type(value_type type)
{
    return is_numeric_type(type) || type == value_type::vector2 || type == value_type::vector3 ||
           type == value_type::vector4;
}

bool is_vector_type(value_type type)
{
    return type == value_type::vector2 || type == value_type::vector3 || type == value_type::vector4;
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

bool variable_matches_slot(const bytecode_program& program, std::uint32_t variable_index, std::uint32_t slot)
{
    return variable_index < program.variables.size() && slot < program.value_slots.size() &&
           program.variables[variable_index].type == program.value_slots[slot].type;
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

    for (const switch_int_table& table : program.switch_int_tables)
    {
        std::unordered_set<std::int64_t> unique;
        for (std::size_t index = 0; index < table.values.size(); ++index)
        {
            if (!unique.insert(table.values[index]).second) return false;
            if (!valid_instruction_target(program, table.instructions[index])) return false;
        }
        if (!valid_instruction_target(program, table.instructions[4])) return false;
    }

    for (const latent_action_definition& action : program.latent_actions)
    {
        if (!slot_has_type(program, action.active_slot, value_type::boolean) ||
            !slot_has_type(program, action.remaining_slot, value_type::float32) ||
            !valid_instruction_target(program, action.completed_instruction))
            return false;

        if (action.kind == latent_action_kind::timer)
        {
            if (!slot_has_type(program, action.period_slot, value_type::float32) ||
                !slot_has_type(program, action.looping_slot, value_type::boolean) ||
                !slot_has_type(program, action.generation_slot, value_type::integer) ||
                !valid_instruction_target(program, action.tick_instruction))
                return false;
        }
    }

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
            case bytecode_opcode::sequence:
                if (!valid_instruction_target(program, instruction.operand0) ||
                    !valid_instruction_target(program, instruction.operand1) ||
                    !valid_instruction_target(program, instruction.operand2) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::switch_integer:
                if (!slot_has_type(program, instruction.operand0, value_type::integer) ||
                    instruction.operand1 >= program.switch_int_tables.size())
                    return false;
                break;
            case bytecode_opcode::do_once:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand1))
                    return false;
                break;
            case bytecode_opcode::do_once_reset:
            case bytecode_opcode::gate_open:
            case bytecode_opcode::gate_close:
            case bytecode_opcode::gate_toggle:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean)) return false;
                break;
            case bytecode_opcode::gate_enter:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand1))
                    return false;
                break;
            case bytecode_opcode::for_loop:
                if (!slot_has_type(program, instruction.operand0, value_type::integer) ||
                    !slot_has_type(program, instruction.operand1, value_type::integer) ||
                    !slot_has_type(program, instruction.operand2, value_type::integer) ||
                    !valid_instruction_target(program, instruction.operand3) ||
                    !valid_instruction_target(program, instruction.operand4))
                    return false;
                break;
            case bytecode_opcode::while_loop:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand1) ||
                    !valid_instruction_target(program, instruction.operand2) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::delay:
            case bytecode_opcode::retriggerable_delay:
                if (!slot_has_type(program, instruction.operand0, value_type::float32) ||
                    instruction.operand1 >= program.latent_actions.size())
                    return false;
                if ((instruction.opcode == bytecode_opcode::delay &&
                     program.latent_actions[instruction.operand1].kind != latent_action_kind::delay) ||
                    (instruction.opcode == bytecode_opcode::retriggerable_delay &&
                     program.latent_actions[instruction.operand1].kind != latent_action_kind::retriggerable_delay))
                    return false;
                break;
            case bytecode_opcode::timer_start:
                if (!slot_has_type(program, instruction.operand0, value_type::float32) ||
                    !slot_has_type(program, instruction.operand1, value_type::boolean) ||
                    instruction.operand2 >= program.latent_actions.size() ||
                    program.latent_actions[instruction.operand2].kind != latent_action_kind::timer ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::timer_stop:
                if (instruction.operand0 >= program.latent_actions.size() ||
                    program.latent_actions[instruction.operand0].kind != latent_action_kind::timer ||
                    !valid_instruction_target(program, instruction.operand1))
                    return false;
                break;
            case bytecode_opcode::load_variable:
            case bytecode_opcode::store_variable:
                if (!variable_matches_slot(program, instruction.operand0, instruction.operand1) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::add:
            case bytecode_opcode::subtract:
            case bytecode_opcode::multiply:
            case bytecode_opcode::divide:
                if (!same_slot_type(program, instruction.operand0, instruction.operand1) ||
                    !same_slot_type(program, instruction.operand0, instruction.operand2) ||
                    !is_arithmetic_type(program.value_slots[instruction.operand0].type) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::compare_equal:
            case bytecode_opcode::compare_not_equal:
            case bytecode_opcode::compare_less:
            case bytecode_opcode::compare_less_equal:
            case bytecode_opcode::compare_greater:
            case bytecode_opcode::compare_greater_equal:
                if (!same_slot_type(program, instruction.operand0, instruction.operand1) ||
                    !is_numeric_type(program.value_slots[instruction.operand0].type) ||
                    !slot_has_type(program, instruction.operand2, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::boolean_and:
            case bytecode_opcode::boolean_or:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !slot_has_type(program, instruction.operand1, value_type::boolean) ||
                    !slot_has_type(program, instruction.operand2, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::boolean_not:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !slot_has_type(program, instruction.operand1, value_type::boolean) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::vector_dot:
                if (!same_slot_type(program, instruction.operand0, instruction.operand1) ||
                    !slot_exists(program, instruction.operand0) ||
                    !is_vector_type(program.value_slots[instruction.operand0].type) ||
                    !slot_has_type(program, instruction.operand2, value_type::float32) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::vector_length:
                if (!slot_exists(program, instruction.operand0) ||
                    !is_vector_type(program.value_slots[instruction.operand0].type) ||
                    !slot_has_type(program, instruction.operand1, value_type::float32) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::vector_normalize:
                if (!same_slot_type(program, instruction.operand0, instruction.operand1) ||
                    !slot_exists(program, instruction.operand0) ||
                    !is_vector_type(program.value_slots[instruction.operand0].type) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::vector_scale:
                if (!slot_exists(program, instruction.operand0) ||
                    !is_vector_type(program.value_slots[instruction.operand0].type) ||
                    !slot_has_type(program, instruction.operand1, value_type::float32) ||
                    !same_slot_type(program, instruction.operand0, instruction.operand2) ||
                    !valid_instruction_target(program, instruction.operand3))
                    return false;
                break;
            case bytecode_opcode::select:
                if (!slot_has_type(program, instruction.operand0, value_type::boolean) ||
                    !same_slot_type(program, instruction.operand1, instruction.operand2) ||
                    !same_slot_type(program, instruction.operand1, instruction.operand3) ||
                    !valid_instruction_target(program, instruction.operand4))
                    return false;
                break;
            case bytecode_opcode::convert_int_to_float:
                if (!slot_has_type(program, instruction.operand0, value_type::integer) ||
                    !slot_has_type(program, instruction.operand1, value_type::float32) ||
                    !valid_instruction_target(program, instruction.operand2))
                    return false;
                break;
            case bytecode_opcode::convert_float_to_int:
                if (!slot_has_type(program, instruction.operand0, value_type::float32) ||
                    !slot_has_type(program, instruction.operand1, value_type::integer) ||
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

bool checked_add(std::int64_t left, std::int64_t right, std::int64_t& result)
{
    if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
        (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right))
        return false;
    result = left + right;
    return true;
}

bool checked_subtract(std::int64_t left, std::int64_t right, std::int64_t& result)
{
    if ((right < 0 && left > std::numeric_limits<std::int64_t>::max() + right) ||
        (right > 0 && left < std::numeric_limits<std::int64_t>::min() + right))
        return false;
    result = left - right;
    return true;
}

bool checked_multiply(std::int64_t left, std::int64_t right, std::int64_t& result)
{
    if (left == 0 || right == 0)
    {
        result = 0;
        return true;
    }
    if ((left == -1 && right == std::numeric_limits<std::int64_t>::min()) ||
        (right == -1 && left == std::numeric_limits<std::int64_t>::min()))
        return false;
    if (left > 0)
    {
        if (right > 0 && left > std::numeric_limits<std::int64_t>::max() / right) return false;
        if (right < 0 && right < std::numeric_limits<std::int64_t>::min() / left) return false;
    }
    else
    {
        if (right > 0 && left < std::numeric_limits<std::int64_t>::min() / right) return false;
        if (right < 0 && left < std::numeric_limits<std::int64_t>::max() / right) return false;
    }
    result = left * right;
    return true;
}

bool checked_divide(std::int64_t left, std::int64_t right, std::int64_t& result)
{
    if (right == 0 || (left == std::numeric_limits<std::int64_t>::min() && right == -1)) return false;
    result = left / right;
    return true;
}

template <std::size_t N>
bool execute_vector_arithmetic(bytecode_opcode opcode, const flow_value& left_value, const flow_value& right_value,
                               flow_value& output)
{
    const auto* left = std::get_if<std::array<double, N>>(&left_value);
    const auto* right = std::get_if<std::array<double, N>>(&right_value);
    if (!left || !right) return false;
    std::array<double, N> result{};
    for (std::size_t index = 0; index < N; ++index)
    {
        switch (opcode)
        {
            case bytecode_opcode::add:
                result[index] = (*left)[index] + (*right)[index];
                break;
            case bytecode_opcode::subtract:
                result[index] = (*left)[index] - (*right)[index];
                break;
            case bytecode_opcode::multiply:
                result[index] = (*left)[index] * (*right)[index];
                break;
            case bytecode_opcode::divide:
                if ((*right)[index] == 0.0) return false;
                result[index] = (*left)[index] / (*right)[index];
                break;
            default:
                return false;
        }
    }
    output = result;
    return true;
}

bool execute_arithmetic(bytecode_opcode opcode, value_type type, const flow_value& left, const flow_value& right,
                        flow_value& output)
{
    if (type == value_type::integer)
    {
        const auto* lhs = std::get_if<std::int64_t>(&left);
        const auto* rhs = std::get_if<std::int64_t>(&right);
        if (!lhs || !rhs) return false;
        std::int64_t result{};
        const bool succeeded = opcode == bytecode_opcode::add        ? checked_add(*lhs, *rhs, result)
                               : opcode == bytecode_opcode::subtract ? checked_subtract(*lhs, *rhs, result)
                               : opcode == bytecode_opcode::multiply ? checked_multiply(*lhs, *rhs, result)
                                                                     : checked_divide(*lhs, *rhs, result);
        if (!succeeded) return false;
        output = result;
        return true;
    }
    if (type == value_type::float32)
    {
        const auto* lhs = std::get_if<double>(&left);
        const auto* rhs = std::get_if<double>(&right);
        if (!lhs || !rhs || (opcode == bytecode_opcode::divide && *rhs == 0.0)) return false;
        output = opcode == bytecode_opcode::add        ? *lhs + *rhs
                 : opcode == bytecode_opcode::subtract ? *lhs - *rhs
                 : opcode == bytecode_opcode::multiply ? *lhs * *rhs
                                                       : *lhs / *rhs;
        return true;
    }
    if (type == value_type::vector2) return execute_vector_arithmetic<2>(opcode, left, right, output);
    if (type == value_type::vector3) return execute_vector_arithmetic<3>(opcode, left, right, output);
    if (type == value_type::vector4) return execute_vector_arithmetic<4>(opcode, left, right, output);
    return false;
}

bool execute_compare(bytecode_opcode opcode, value_type type, const flow_value& left, const flow_value& right,
                     bool& output)
{
    if (type == value_type::integer)
    {
        const auto* lhs = std::get_if<std::int64_t>(&left);
        const auto* rhs = std::get_if<std::int64_t>(&right);
        if (!lhs || !rhs) return false;
        output = opcode == bytecode_opcode::compare_equal        ? *lhs == *rhs
                 : opcode == bytecode_opcode::compare_not_equal  ? *lhs != *rhs
                 : opcode == bytecode_opcode::compare_less       ? *lhs < *rhs
                 : opcode == bytecode_opcode::compare_less_equal ? *lhs <= *rhs
                 : opcode == bytecode_opcode::compare_greater    ? *lhs > *rhs
                                                                 : *lhs >= *rhs;
        return true;
    }
    if (type == value_type::float32)
    {
        const auto* lhs = std::get_if<double>(&left);
        const auto* rhs = std::get_if<double>(&right);
        if (!lhs || !rhs) return false;
        output = opcode == bytecode_opcode::compare_equal        ? *lhs == *rhs
                 : opcode == bytecode_opcode::compare_not_equal  ? *lhs != *rhs
                 : opcode == bytecode_opcode::compare_less       ? *lhs < *rhs
                 : opcode == bytecode_opcode::compare_less_equal ? *lhs <= *rhs
                 : opcode == bytecode_opcode::compare_greater    ? *lhs > *rhs
                                                                 : *lhs >= *rhs;
        return true;
    }
    return false;
}

template <std::size_t N> double vector_dot(const flow_value& left_value, const flow_value& right_value, bool& valid)
{
    const auto* left = std::get_if<std::array<double, N>>(&left_value);
    const auto* right = std::get_if<std::array<double, N>>(&right_value);
    if (!left || !right)
    {
        valid = false;
        return 0.0;
    }
    double result = 0.0;
    for (std::size_t index = 0; index < N; ++index)
        result += (*left)[index] * (*right)[index];
    valid = true;
    return result;
}

template <std::size_t N> bool vector_normalize(const flow_value& input_value, flow_value& output, double* length_output)
{
    const auto* input = std::get_if<std::array<double, N>>(&input_value);
    if (!input) return false;
    double squared = 0.0;
    for (double component : *input)
        squared += component * component;
    const double length = std::sqrt(squared);
    if (length_output) *length_output = length;
    std::array<double, N> result{};
    if (length > 0.0)
        for (std::size_t index = 0; index < N; ++index)
            result[index] = (*input)[index] / length;
    output = result;
    return true;
}

template <std::size_t N> bool vector_scale(const flow_value& input_value, double scale, flow_value& output)
{
    const auto* input = std::get_if<std::array<double, N>>(&input_value);
    if (!input) return false;
    std::array<double, N> result{};
    for (std::size_t index = 0; index < N; ++index)
        result[index] = (*input)[index] * scale;
    output = result;
    return true;
}

void bump_generation(std::int64_t& generation) noexcept
{
    generation = generation == std::numeric_limits<std::int64_t>::max() ? 0 : generation + 1;
}

execution_result execute_chain(const bytecode_program& program, std::vector<flow_value>& variable_values,
                               std::vector<flow_value>& value_slots, std::uint32_t first_instruction,
                               std::uint32_t instruction_budget, const vm_world_context& world)
{
    execution_result result;
    std::uint32_t instruction = first_instruction;

    const auto run_nested = [&](std::uint32_t target) -> bool
    {
        if (target == invalid_instruction) return true;
        const std::uint32_t remaining =
            result.instructions_executed < instruction_budget ? instruction_budget - result.instructions_executed : 0;
        execution_result nested = execute_chain(program, variable_values, value_slots, target, remaining, world);
        result.instructions_executed += nested.instructions_executed;
        if (nested.succeeded()) return true;
        result.status = nested.status;
        result.stopped_instruction = nested.stopped_instruction;
        result.node_id = std::move(nested.node_id);
        return false;
    };

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
            case bytecode_opcode::sequence:
            {
                const std::array<std::uint32_t, 4> targets{current.operand0, current.operand1, current.operand2,
                                                           current.operand3};
                for (const std::uint32_t target : targets)
                    if (!run_nested(target)) return result;
                instruction = invalid_instruction;
                break;
            }
            case bytecode_opcode::switch_integer:
            {
                const auto* selection = std::get_if<std::int64_t>(&value_slots[current.operand0]);
                if (!selection)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                const switch_int_table& table = program.switch_int_tables[current.operand1];
                instruction = table.instructions[4];
                for (std::size_t case_index = 0; case_index < table.values.size(); ++case_index)
                    if (*selection == table.values[case_index])
                    {
                        instruction = table.instructions[case_index];
                        break;
                    }
                break;
            }
            case bytecode_opcode::do_once:
            {
                bool* fired = std::get_if<bool>(&value_slots[current.operand0]);
                if (!fired)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (*fired)
                    instruction = invalid_instruction;
                else
                {
                    *fired = true;
                    instruction = current.operand1;
                }
                break;
            }
            case bytecode_opcode::do_once_reset:
            {
                bool* fired = std::get_if<bool>(&value_slots[current.operand0]);
                if (!fired)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                *fired = false;
                instruction = invalid_instruction;
                break;
            }
            case bytecode_opcode::gate_enter:
            {
                const bool* open = std::get_if<bool>(&value_slots[current.operand0]);
                if (!open)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                instruction = *open ? current.operand1 : invalid_instruction;
                break;
            }
            case bytecode_opcode::gate_open:
            case bytecode_opcode::gate_close:
            case bytecode_opcode::gate_toggle:
            {
                bool* open = std::get_if<bool>(&value_slots[current.operand0]);
                if (!open)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (current.opcode == bytecode_opcode::gate_open)
                    *open = true;
                else if (current.opcode == bytecode_opcode::gate_close)
                    *open = false;
                else
                    *open = !*open;
                instruction = invalid_instruction;
                break;
            }
            case bytecode_opcode::for_loop:
            {
                const auto* first_value = std::get_if<std::int64_t>(&value_slots[current.operand0]);
                const auto* last_value = std::get_if<std::int64_t>(&value_slots[current.operand1]);
                if (!first_value || !last_value)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                const std::int64_t first = *first_value;
                const std::int64_t last = *last_value;
                if (first <= last)
                {
                    for (std::int64_t index = first;; ++index)
                    {
                        value_slots[current.operand2] = index;
                        if (!run_nested(current.operand3)) return result;
                        if (index == last) break;
                        if (result.instructions_executed >= instruction_budget)
                        {
                            stop_execution(result, execution_status::instruction_budget_exceeded, program, instruction);
                            return result;
                        }
                        ++result.instructions_executed;
                    }
                }
                instruction = current.operand4;
                break;
            }
            case bytecode_opcode::while_loop:
            {
                while (true)
                {
                    const bool* condition = std::get_if<bool>(&value_slots[current.operand0]);
                    if (!condition)
                    {
                        stop_execution(result, execution_status::type_mismatch, program, instruction);
                        return result;
                    }
                    if (!*condition)
                    {
                        instruction = current.operand2;
                        break;
                    }
                    if (!run_nested(current.operand1) || !run_nested(current.operand3)) return result;
                    if (result.instructions_executed >= instruction_budget)
                    {
                        stop_execution(result, execution_status::instruction_budget_exceeded, program, instruction);
                        return result;
                    }
                    ++result.instructions_executed;
                }
                break;
            }
            case bytecode_opcode::delay:
            case bytecode_opcode::retriggerable_delay:
            {
                const auto* duration = std::get_if<double>(&value_slots[current.operand0]);
                const latent_action_definition& action = program.latent_actions[current.operand1];
                bool* active = std::get_if<bool>(&value_slots[action.active_slot]);
                double* remaining = std::get_if<double>(&value_slots[action.remaining_slot]);
                if (!duration || !active || !remaining)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (!std::isfinite(*duration) || *duration < 0.0)
                {
                    stop_execution(result, execution_status::invalid_operation, program, instruction);
                    return result;
                }
                if (*duration == 0.0)
                {
                    *active = false;
                    *remaining = 0.0;
                    instruction = action.completed_instruction;
                    break;
                }
                if (!*active || current.opcode == bytecode_opcode::retriggerable_delay)
                {
                    *active = true;
                    *remaining = *duration;
                }
                instruction = invalid_instruction;
                break;
            }
            case bytecode_opcode::timer_start:
            {
                const auto* interval = std::get_if<double>(&value_slots[current.operand0]);
                const auto* looping_input = std::get_if<bool>(&value_slots[current.operand1]);
                const latent_action_definition& action = program.latent_actions[current.operand2];
                bool* active = std::get_if<bool>(&value_slots[action.active_slot]);
                double* remaining = std::get_if<double>(&value_slots[action.remaining_slot]);
                double* period = std::get_if<double>(&value_slots[action.period_slot]);
                bool* looping = std::get_if<bool>(&value_slots[action.looping_slot]);
                auto* generation = std::get_if<std::int64_t>(&value_slots[action.generation_slot]);
                if (!interval || !looping_input || !active || !remaining || !period || !looping || !generation)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (!std::isfinite(*interval) || *interval < 0.0 || (*looping_input && *interval <= 0.0))
                {
                    stop_execution(result, execution_status::invalid_operation, program, instruction);
                    return result;
                }
                *active = true;
                *remaining = *interval;
                *period = *interval;
                *looping = *looping_input;
                bump_generation(*generation);
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::timer_stop:
            {
                const latent_action_definition& action = program.latent_actions[current.operand0];
                bool* active = std::get_if<bool>(&value_slots[action.active_slot]);
                double* remaining = std::get_if<double>(&value_slots[action.remaining_slot]);
                double* period = std::get_if<double>(&value_slots[action.period_slot]);
                bool* looping = std::get_if<bool>(&value_slots[action.looping_slot]);
                auto* generation = std::get_if<std::int64_t>(&value_slots[action.generation_slot]);
                if (!active || !remaining || !period || !looping || !generation)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                *active = false;
                *remaining = 0.0;
                *period = 0.0;
                *looping = false;
                bump_generation(*generation);
                instruction = current.operand1;
                break;
            }
            case bytecode_opcode::load_variable:
                value_slots[current.operand1] = variable_values[current.operand0];
                instruction = current.operand2;
                break;
            case bytecode_opcode::store_variable:
                variable_values[current.operand0] = value_slots[current.operand1];
                instruction = current.operand2;
                break;
            case bytecode_opcode::add:
            case bytecode_opcode::subtract:
            case bytecode_opcode::multiply:
            case bytecode_opcode::divide:
            {
                const value_type type = program.value_slots[current.operand2].type;
                flow_value output;
                if (!execute_arithmetic(current.opcode, type, value_slots[current.operand0],
                                        value_slots[current.operand1], output))
                {
                    stop_execution(result, execution_status::invalid_operation, program, instruction);
                    return result;
                }
                value_slots[current.operand2] = std::move(output);
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::compare_equal:
            case bytecode_opcode::compare_not_equal:
            case bytecode_opcode::compare_less:
            case bytecode_opcode::compare_less_equal:
            case bytecode_opcode::compare_greater:
            case bytecode_opcode::compare_greater_equal:
            {
                bool compared = false;
                const value_type type = program.value_slots[current.operand0].type;
                bool valid = false;
                valid = execute_compare(current.opcode, type, value_slots[current.operand0],
                                        value_slots[current.operand1], compared);
                if (!valid)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand2] = compared;
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::boolean_and:
            case bytecode_opcode::boolean_or:
            {
                const bool* left = std::get_if<bool>(&value_slots[current.operand0]);
                const bool* right = std::get_if<bool>(&value_slots[current.operand1]);
                if (!left || !right)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand2] =
                    current.opcode == bytecode_opcode::boolean_and ? *left && *right : *left || *right;
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::boolean_not:
            {
                const bool* value = std::get_if<bool>(&value_slots[current.operand0]);
                if (!value)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = !*value;
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::vector_dot:
            {
                const value_type type = program.value_slots[current.operand0].type;
                bool valid = false;
                double dot = 0.0;
                if (type == value_type::vector2)
                    dot = vector_dot<2>(value_slots[current.operand0], value_slots[current.operand1], valid);
                else if (type == value_type::vector3)
                    dot = vector_dot<3>(value_slots[current.operand0], value_slots[current.operand1], valid);
                else if (type == value_type::vector4)
                    dot = vector_dot<4>(value_slots[current.operand0], value_slots[current.operand1], valid);
                if (!valid)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand2] = dot;
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::vector_length:
            case bytecode_opcode::vector_normalize:
            {
                const value_type type = program.value_slots[current.operand0].type;
                flow_value normalized;
                double length = 0.0;
                const bool valid = type == value_type::vector2
                                       ? vector_normalize<2>(value_slots[current.operand0], normalized, &length)
                                   : type == value_type::vector3
                                       ? vector_normalize<3>(value_slots[current.operand0], normalized, &length)
                                   : type == value_type::vector4
                                       ? vector_normalize<4>(value_slots[current.operand0], normalized, &length)
                                       : false;
                if (!valid)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (current.opcode == bytecode_opcode::vector_length)
                    value_slots[current.operand1] = length;
                else
                    value_slots[current.operand1] = std::move(normalized);
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::vector_scale:
            {
                const auto* scale = std::get_if<double>(&value_slots[current.operand1]);
                const value_type type = program.value_slots[current.operand0].type;
                flow_value output;
                const bool valid =
                    scale &&
                    (type == value_type::vector2   ? vector_scale<2>(value_slots[current.operand0], *scale, output)
                     : type == value_type::vector3 ? vector_scale<3>(value_slots[current.operand0], *scale, output)
                     : type == value_type::vector4 ? vector_scale<4>(value_slots[current.operand0], *scale, output)
                                                   : false);
                if (!valid)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand2] = std::move(output);
                instruction = current.operand3;
                break;
            }
            case bytecode_opcode::select:
            {
                const bool* condition = std::get_if<bool>(&value_slots[current.operand0]);
                if (!condition)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand3] = value_slots[*condition ? current.operand1 : current.operand2];
                instruction = current.operand4;
                break;
            }
            case bytecode_opcode::convert_int_to_float:
            {
                const auto* value = std::get_if<std::int64_t>(&value_slots[current.operand0]);
                if (!value)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = static_cast<double>(*value);
                instruction = current.operand2;
                break;
            }
            case bytecode_opcode::convert_float_to_int:
            {
                const auto* value = std::get_if<double>(&value_slots[current.operand0]);
                if (!value)
                {
                    stop_execution(result, execution_status::type_mismatch, program, instruction);
                    return result;
                }
                if (!std::isfinite(*value) || *value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
                    *value > static_cast<double>(std::numeric_limits<std::int64_t>::max()))
                {
                    stop_execution(result, execution_status::invalid_operation, program, instruction);
                    return result;
                }
                value_slots[current.operand1] = static_cast<std::int64_t>(*value);
                instruction = current.operand2;
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

execution_result advance_latent_actions(const bytecode_program& program, std::vector<flow_value>& variable_values,
                                        std::vector<flow_value>& value_slots, double delta_seconds,
                                        std::uint32_t instruction_budget, const vm_world_context& world)
{
    execution_result result;
    if (!std::isfinite(delta_seconds) || delta_seconds < 0.0)
        return status_result(execution_status::invalid_operation);

    std::vector<bool> active_at_start;
    active_at_start.reserve(program.latent_actions.size());
    for (const latent_action_definition& action : program.latent_actions)
    {
        const bool* active = std::get_if<bool>(&value_slots[action.active_slot]);
        if (!active) return status_result(execution_status::invalid_program);
        active_at_start.push_back(*active);
    }

    const auto run_continuation = [&](std::uint32_t target) -> bool
    {
        if (target == invalid_instruction) return true;
        const std::uint32_t remaining_budget =
            result.instructions_executed < instruction_budget ? instruction_budget - result.instructions_executed : 0;
        execution_result nested =
            execute_chain(program, variable_values, value_slots, target, remaining_budget, world);
        result.instructions_executed += nested.instructions_executed;
        if (nested.succeeded()) return true;
        result.status = nested.status;
        result.stopped_instruction = nested.stopped_instruction;
        result.node_id = std::move(nested.node_id);
        return false;
    };

    for (std::size_t action_index = 0; action_index < program.latent_actions.size(); ++action_index)
    {
        if (!active_at_start[action_index]) continue;

        const latent_action_definition& action = program.latent_actions[action_index];
        bool* active = std::get_if<bool>(&value_slots[action.active_slot]);
        double* remaining = std::get_if<double>(&value_slots[action.remaining_slot]);
        if (!active || !remaining) return status_result(execution_status::invalid_program);
        if (!*active) continue;

        *remaining -= delta_seconds;
        if (*remaining > 0.0) continue;

        if (action.kind != latent_action_kind::timer)
        {
            *active = false;
            *remaining = 0.0;
            if (!run_continuation(action.completed_instruction)) return result;
            continue;
        }

        double* period = std::get_if<double>(&value_slots[action.period_slot]);
        bool* looping = std::get_if<bool>(&value_slots[action.looping_slot]);
        auto* generation = std::get_if<std::int64_t>(&value_slots[action.generation_slot]);
        if (!period || !looping || !generation) return status_result(execution_status::invalid_program);

        while (*active && *remaining <= 0.0)
        {
            if (result.instructions_executed >= instruction_budget)
            {
                result.status = execution_status::instruction_budget_exceeded;
                return result;
            }
            ++result.instructions_executed;

            const std::int64_t generation_before = *generation;
            const bool repeat = *looping;
            if (!repeat)
            {
                *active = false;
                *remaining = 0.0;
            }

            if (!run_continuation(action.tick_instruction)) return result;

            if (!repeat)
            {
                if (!run_continuation(action.completed_instruction)) return result;
                break;
            }

            if (*generation != generation_before || !*active) break;
            if (!std::isfinite(*period) || *period <= 0.0)
                return status_result(execution_status::invalid_operation);
            *remaining += *period;
        }
    }

    return result;
}

void clear_latent_actions(const bytecode_program& program, std::vector<flow_value>& value_slots)
{
    for (const latent_action_definition& action : program.latent_actions)
    {
        value_slots[action.active_slot] = false;
        value_slots[action.remaining_slot] = 0.0;
        if (action.kind != latent_action_kind::timer) continue;
        value_slots[action.period_slot] = 0.0;
        value_slots[action.looping_slot] = false;
        value_slots[action.generation_slot] = std::int64_t{0};
    }
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

execution_result execute_event(const bytecode_program& program, std::vector<flow_value>& variable_values,
                               std::vector<flow_value>& value_slots, entry_point_kind kind, std::string_view action,
                               double event_value, std::uint32_t instruction_budget, const vm_world_context& world)
{
    execution_result result;

    for (const bytecode_entry_point& entry : program.entry_points)
    {
        if (!entry_matches(entry, kind, action)) continue;

        ++result.entry_points_executed;
        bind_entry_values(entry, value_slots, event_value);

        const std::uint32_t remaining_budget = instruction_budget - result.instructions_executed;
        execution_result entry_result =
            execute_chain(program, variable_values, value_slots, entry.instruction, remaining_budget, world);
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
    execution_result result = execute_event(*program_, variable_values_, value_slots_, entry_point_kind::begin_play, {},
                                            0.0, limits_.instruction_budget, world);
    if (!result.succeeded())
    {
        clear_latent_actions(*program_, value_slots_);
        active_ = false;
    }
    return result;
}

execution_result vm_instance::end_play(vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);

    execution_result result = execute_event(*program_, variable_values_, value_slots_, entry_point_kind::end_play, {},
                                            0.0, limits_.instruction_budget, world);
    clear_latent_actions(*program_, value_slots_);
    active_ = false;
    return result;
}

execution_result vm_instance::tick(double delta_seconds, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);

    execution_result result =
        advance_latent_actions(*program_, variable_values_, value_slots_, delta_seconds, limits_.instruction_budget, world);
    if (!result.succeeded()) return result;

    const std::uint32_t remaining_budget =
        result.instructions_executed < limits_.instruction_budget ? limits_.instruction_budget - result.instructions_executed
                                                                 : 0;
    execution_result tick_result = execute_event(*program_, variable_values_, value_slots_, entry_point_kind::tick, {},
                                                 delta_seconds, remaining_budget, world);
    result.instructions_executed += tick_result.instructions_executed;
    result.entry_points_executed += tick_result.entry_points_executed;
    if (!tick_result.succeeded())
    {
        result.status = tick_result.status;
        result.stopped_instruction = tick_result.stopped_instruction;
        result.node_id = std::move(tick_result.node_id);
    }
    return result;
}

execution_result vm_instance::fixed_tick(double delta_seconds, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, variable_values_, value_slots_, entry_point_kind::fixed_tick, {}, delta_seconds,
                         limits_.instruction_budget, world);
}

execution_result vm_instance::input_action_triggered(std::string_view action, double value, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, variable_values_, value_slots_, entry_point_kind::input_action_triggered, action,
                         value, limits_.instruction_budget, world);
}

execution_result vm_instance::input_action_completed(std::string_view action, double value, vm_world_context world)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, variable_values_, value_slots_, entry_point_kind::input_action_completed, action,
                         value, limits_.instruction_budget, world);
}

} // namespace arc::flow

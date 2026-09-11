#include <arc/flow/flow.h>

#include <algorithm>
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
        case value_type::component:
            return std::holds_alternative<std::monostate>(value);
    }
    return false;
}

bool valid_instruction_target(const bytecode_program& program, std::uint32_t instruction)
{
    return instruction == invalid_instruction || instruction < program.instructions.size();
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
        if (item.id.empty() || !variable_ids.insert(item.id).second || !value_matches_type(item.type, item.default_value))
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
                if (instruction.operand0 >= program.value_slots.size()) return false;
                if (program.value_slots[instruction.operand0].type != value_type::boolean) return false;
                if (!valid_instruction_target(program, instruction.operand1) ||
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

execution_result execute_chain(const bytecode_program& program, const std::vector<flow_value>& value_slots,
                               std::uint32_t first_instruction, std::uint32_t instruction_budget)
{
    execution_result result;
    std::uint32_t instruction = first_instruction;

    while (instruction != invalid_instruction)
    {
        if (instruction >= program.instructions.size())
        {
            result.status = execution_status::invalid_program;
            result.stopped_instruction = instruction;
            return result;
        }
        if (result.instructions_executed >= instruction_budget)
        {
            result.status = execution_status::instruction_budget_exceeded;
            result.stopped_instruction = instruction;
            result.node_id = instruction_node(program, instruction);
            return result;
        }

        const bytecode_instruction& current = program.instructions[instruction];
        ++result.instructions_executed;

        switch (current.opcode)
        {
            case bytecode_opcode::branch:
            {
                if (current.operand0 >= value_slots.size())
                {
                    result.status = execution_status::invalid_program;
                    result.stopped_instruction = instruction;
                    result.node_id = instruction_node(program, instruction);
                    return result;
                }

                const bool* condition = std::get_if<bool>(&value_slots[current.operand0]);
                if (!condition)
                {
                    result.status = execution_status::type_mismatch;
                    result.stopped_instruction = instruction;
                    result.node_id = instruction_node(program, instruction);
                    return result;
                }
                instruction = *condition ? current.operand1 : current.operand2;
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
                               std::uint32_t instruction_budget)
{
    execution_result result;

    for (const bytecode_entry_point& entry : program.entry_points)
    {
        if (!entry_matches(entry, kind, action)) continue;

        ++result.entry_points_executed;
        bind_entry_values(entry, value_slots, event_value);

        const std::uint32_t remaining_budget = instruction_budget - result.instructions_executed;
        execution_result entry_result = execute_chain(program, value_slots, entry.instruction, remaining_budget);
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
    for (const variable& item : program_->variables) variable_values_.push_back(item.default_value);

    value_slots_.reserve(program_->value_slots.size());
    for (const bytecode_value_slot& slot : program_->value_slots) value_slots_.push_back(slot.initial_value);
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

execution_result vm_instance::begin_play()
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (active_) return status_result(execution_status::already_active);

    active_ = true;
    execution_result result = execute_event(*program_, value_slots_, entry_point_kind::begin_play, {}, 0.0,
                                            limits_.instruction_budget);
    if (!result.succeeded()) active_ = false;
    return result;
}

execution_result vm_instance::end_play()
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);

    execution_result result = execute_event(*program_, value_slots_, entry_point_kind::end_play, {}, 0.0,
                                            limits_.instruction_budget);
    active_ = false;
    return result;
}

execution_result vm_instance::tick(double delta_seconds)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::tick, {}, delta_seconds,
                         limits_.instruction_budget);
}

execution_result vm_instance::fixed_tick(double delta_seconds)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::fixed_tick, {}, delta_seconds,
                         limits_.instruction_budget);
}

execution_result vm_instance::input_action_triggered(std::string_view action, double value)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::input_action_triggered, action, value,
                         limits_.instruction_budget);
}

execution_result vm_instance::input_action_completed(std::string_view action, double value)
{
    if (!valid_ || !program_) return status_result(execution_status::invalid_program);
    if (!active_) return status_result(execution_status::inactive);
    return execute_event(*program_, value_slots_, entry_point_kind::input_action_completed, action, value,
                         limits_.instruction_budget);
}

} // namespace arc::flow

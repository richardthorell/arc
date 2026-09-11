#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace arc::project
{
struct game_world_api_v1;
}

namespace arc::flow
{

inline constexpr std::uint32_t flow_ir_version = 1;
inline constexpr std::uint32_t flow_bytecode_version = 2;
inline constexpr std::uint32_t invalid_instruction = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t invalid_entity_index = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t default_instruction_budget = 4096;

enum class diagnostic_severity : std::uint8_t
{
    information,
    warning,
    error,
};

enum class value_type : std::uint8_t
{
    boolean,
    integer,
    float32,
    vector2,
    vector3,
    vector4,
    string,
    name,
    entity,
    component,
};

/** @brief Runtime entity value that can preserve an immediate or M3.5 deferred entity target. */
struct flow_entity
{
    std::uint32_t index{invalid_entity_index};
    std::uint32_t generation{};
    std::uint64_t deferred_buffer{};
    std::uint32_t deferred_ordinal{};
    bool deferred{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return deferred ? deferred_buffer != 0 : index != invalid_entity_index;
    }
};

using flow_value = std::variant<std::monostate, bool, std::int64_t, double, std::array<double, 2>,
                                std::array<double, 3>, std::array<double, 4>, std::string, flow_entity>;

struct diagnostic
{
    diagnostic_severity severity{diagnostic_severity::error};
    std::string code;
    std::string message;
    std::string node_id;
    std::string pin_id;
    std::string connection_id;
};

struct variable
{
    std::string id;
    std::string name;
    value_type type{value_type::float32};
    flow_value default_value{0.0};
    bool exposed{false};
};

enum class entry_point_kind : std::uint8_t
{
    begin_play,
    end_play,
    tick,
    fixed_tick,
    input_action_triggered,
    input_action_completed,
};

enum class entry_value_kind : std::uint8_t
{
    delta_seconds,
    input_action_value,
};

struct value_binding
{
    entry_value_kind source{entry_value_kind::delta_seconds};
    std::uint32_t slot{0};
};

struct ir_value_slot
{
    std::uint32_t index{0};
    value_type type{value_type::float32};
    flow_value initial_value{0.0};
    std::string source_node_id;
    std::string source_pin_id;
};

struct ir_entry_point
{
    entry_point_kind kind{entry_point_kind::begin_play};
    std::string node_id;
    std::string action;
    std::uint32_t instruction{invalid_instruction};
    std::vector<value_binding> value_bindings;
};

enum class ir_opcode : std::uint8_t
{
    branch,
};

struct ir_instruction
{
    ir_opcode opcode{ir_opcode::branch};
    std::string node_id;
    std::uint32_t condition_slot{0};
    std::uint32_t true_instruction{invalid_instruction};
    std::uint32_t false_instruction{invalid_instruction};
};

struct ir_program
{
    std::uint32_t version{flow_ir_version};
    std::vector<variable> variables;
    std::vector<ir_value_slot> value_slots;
    std::vector<ir_entry_point> entry_points;
    std::vector<ir_instruction> instructions;
};

struct bytecode_value_slot
{
    value_type type{value_type::float32};
    flow_value initial_value{0.0};
};

struct bytecode_entry_point
{
    entry_point_kind kind{entry_point_kind::begin_play};
    std::string action;
    std::uint32_t instruction{invalid_instruction};
    std::vector<value_binding> value_bindings;
};

/** @brief Engine-owned component identifier encoded by Flow world instructions. */
enum class world_core_component : std::uint8_t
{
    name,
    transform,
    tag,
    active,
};

enum class bytecode_opcode : std::uint8_t
{
    branch,
    self_entity,
    world_create_entity,
    world_destroy_entity,
    world_entity_alive,
    world_has_core_component,
    world_remove_core_component,
    world_get_name,
    world_set_name,
    world_get_transform,
    world_set_transform,
    world_get_tag,
    world_set_tag,
    world_get_active,
    world_set_active,
};

/**
 * @brief Compact Flow instruction.
 *
 * Operand interpretation is opcode-specific. World transform instructions use all five operands; simpler opcodes use
 * only the leading operands they need. Control-flow targets use @ref invalid_instruction as the explicit return target.
 */
struct bytecode_instruction
{
    bytecode_opcode opcode{bytecode_opcode::branch};
    std::uint32_t operand0{0};
    std::uint32_t operand1{invalid_instruction};
    std::uint32_t operand2{invalid_instruction};
    std::uint32_t operand3{invalid_instruction};
    std::uint32_t operand4{invalid_instruction};
};

struct bytecode_program
{
    std::uint32_t version{flow_bytecode_version};
    std::vector<variable> variables;
    std::vector<bytecode_value_slot> value_slots;
    std::vector<bytecode_entry_point> entry_points;
    std::vector<bytecode_instruction> instructions;

    // Instruction-index aligned source map used by editor diagnostics/debugging.
    std::vector<std::string> instruction_nodes;
};

struct [[nodiscard]] compile_result
{
    bool succeeded{false};
    std::vector<diagnostic> diagnostics;
    std::optional<ir_program> ir;
    std::optional<bytecode_program> bytecode;
};

// Compiles a version-1 .arcflow asset into typed IR and runtime-ready bytecode.
// The editable graph remains source data and is never interpreted directly.
compile_result compile_asset(std::string_view source);

enum class execution_status : std::uint8_t
{
    completed,
    inactive,
    already_active,
    invalid_program,
    instruction_budget_exceeded,
    type_mismatch,
    world_unavailable,
    world_operation_failed,
};

struct vm_limits
{
    std::uint32_t instruction_budget{default_instruction_budget};
};

/**
 * @brief Borrowed world binding for one VM dispatch.
 *
 * The M3.5 world table is callback-scoped, so Flow never stores this pointer across dispatches. @ref self identifies
 * the entity that owns the Flow instance when the host has one.
 */
struct vm_world_context
{
    const project::game_world_api_v1* api{};
    flow_entity self{};
};

struct [[nodiscard]] execution_result
{
    execution_status status{execution_status::completed};
    std::uint32_t instructions_executed{0};
    std::uint32_t entry_points_executed{0};
    std::uint32_t stopped_instruction{invalid_instruction};
    std::string node_id;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return status == execution_status::completed;
    }
};

// Per-entity Flow runtime state. The bytecode program is immutable and must outlive the VM instance.
class vm_instance
{
public:
    explicit vm_instance(const bytecode_program& program, vm_limits limits = {});
    vm_instance(const vm_instance&) = delete;
    vm_instance& operator=(const vm_instance&) = delete;
    vm_instance(vm_instance&& other) noexcept;
    vm_instance& operator=(vm_instance&& other) noexcept;
    ~vm_instance() = default;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] vm_limits limits() const noexcept;

    // Restores graph variable defaults and bytecode value slots and leaves the instance inactive.
    void reset();

    [[nodiscard]] const flow_value* variable_value(std::string_view id) const noexcept;
    [[nodiscard]] bool set_variable_value(std::string_view id, const flow_value& value);
    [[nodiscard]] const flow_value* value_slot(std::uint32_t slot) const noexcept;

    [[nodiscard]] execution_result begin_play(vm_world_context world = {});
    [[nodiscard]] execution_result end_play(vm_world_context world = {});
    [[nodiscard]] execution_result tick(double delta_seconds, vm_world_context world = {});
    [[nodiscard]] execution_result fixed_tick(double delta_seconds, vm_world_context world = {});
    [[nodiscard]] execution_result input_action_triggered(std::string_view action, double value,
                                                          vm_world_context world = {});
    [[nodiscard]] execution_result input_action_completed(std::string_view action, double value,
                                                          vm_world_context world = {});

private:
    const bytecode_program* program_{nullptr};
    std::vector<flow_value> variable_values_;
    std::vector<flow_value> value_slots_;
    vm_limits limits_{};
    bool valid_{false};
    bool active_{false};
};

} // namespace arc::flow

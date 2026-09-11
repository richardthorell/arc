#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace arc::flow
{

inline constexpr std::uint32_t flow_ir_version = 1;
inline constexpr std::uint32_t flow_bytecode_version = 1;
inline constexpr std::uint32_t invalid_instruction = std::numeric_limits<std::uint32_t>::max();

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

using flow_value = std::variant<std::monostate, bool, std::int64_t, double, std::array<double, 2>,
                                std::array<double, 3>, std::array<double, 4>, std::string>;

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

enum class bytecode_opcode : std::uint8_t
{
    branch,
};

struct bytecode_instruction
{
    bytecode_opcode opcode{bytecode_opcode::branch};
    std::uint32_t operand0{0};
    std::uint32_t operand1{invalid_instruction};
    std::uint32_t operand2{invalid_instruction};
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

struct compile_result
{
    bool succeeded{false};
    std::vector<diagnostic> diagnostics;
    std::optional<ir_program> ir;
    std::optional<bytecode_program> bytecode;
};

// Compiles a version-1 .arcflow asset into typed IR and runtime-ready bytecode.
// The editable graph remains source data and is never interpreted directly.
compile_result compile_asset(std::string_view source);

} // namespace arc::flow

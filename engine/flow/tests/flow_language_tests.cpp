#include <arc/flow/flow.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace
{

using namespace arc::flow;

bytecode_value_slot int_slot(std::int64_t value = 0)
{
    return {.type = value_type::integer, .initial_value = value};
}

bytecode_value_slot float_slot(double value = 0.0)
{
    return {.type = value_type::float32, .initial_value = value};
}

bytecode_value_slot bool_slot(bool value = false)
{
    return {.type = value_type::boolean, .initial_value = value};
}

bytecode_value_slot vector3_slot(std::array<double, 3> value = {})
{
    return {.type = value_type::vector3, .initial_value = value};
}

void test_scalar_language_bytecode()
{
    bytecode_program program;
    program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    program.value_slots = {
        int_slot(6),  int_slot(3),     int_slot(),       int_slot(),   int_slot(),      int_slot(),
        bool_slot(),  bool_slot(true), bool_slot(false), bool_slot(),  bool_slot(),     bool_slot(),
        float_slot(), int_slot(),      int_slot(10),     int_slot(20), bool_slot(true), int_slot(),
    };
    program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    program.instructions = {
        {.opcode = bytecode_opcode::add, .operand0 = 0, .operand1 = 1, .operand2 = 2, .operand3 = 1},
        {.opcode = bytecode_opcode::subtract, .operand0 = 0, .operand1 = 1, .operand2 = 3, .operand3 = 2},
        {.opcode = bytecode_opcode::multiply, .operand0 = 0, .operand1 = 1, .operand2 = 4, .operand3 = 3},
        {.opcode = bytecode_opcode::divide, .operand0 = 0, .operand1 = 1, .operand2 = 5, .operand3 = 4},
        {.opcode = bytecode_opcode::compare_greater, .operand0 = 0, .operand1 = 1, .operand2 = 6, .operand3 = 5},
        {.opcode = bytecode_opcode::boolean_and, .operand0 = 7, .operand1 = 6, .operand2 = 9, .operand3 = 6},
        {.opcode = bytecode_opcode::boolean_or, .operand0 = 8, .operand1 = 6, .operand2 = 10, .operand3 = 7},
        {.opcode = bytecode_opcode::boolean_not, .operand0 = 8, .operand1 = 11, .operand2 = 8},
        {.opcode = bytecode_opcode::convert_int_to_float, .operand0 = 0, .operand1 = 12, .operand2 = 9},
        {.opcode = bytecode_opcode::convert_float_to_int, .operand0 = 12, .operand1 = 13, .operand2 = 10},
        {.opcode = bytecode_opcode::select,
         .operand0 = 16,
         .operand1 = 14,
         .operand2 = 15,
         .operand3 = 17,
         .operand4 = 11},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 17, .operand2 = invalid_instruction},
    };

    vm_instance instance{program};
    assert(instance.valid());
    const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<std::int64_t>(*instance.value_slot(2)) == 9);
    assert(std::get<std::int64_t>(*instance.value_slot(3)) == 3);
    assert(std::get<std::int64_t>(*instance.value_slot(4)) == 18);
    assert(std::get<std::int64_t>(*instance.value_slot(5)) == 2);
    assert(std::get<bool>(*instance.value_slot(6)));
    assert(std::get<bool>(*instance.value_slot(9)));
    assert(std::get<bool>(*instance.value_slot(10)));
    assert(std::get<bool>(*instance.value_slot(11)));
    assert(std::get<double>(*instance.value_slot(12)) == 6.0);
    assert(std::get<std::int64_t>(*instance.value_slot(13)) == 6);
    assert(std::get<std::int64_t>(*instance.value_slot(17)) == 10);
    assert(std::get<std::int64_t>(*instance.variable_value("result")) == 10);
}

void test_vector_language_bytecode()
{
    bytecode_program program;
    program.value_slots = {
        vector3_slot({3.0, 4.0, 0.0}),
        vector3_slot({1.0, 2.0, 3.0}),
        float_slot(),
        float_slot(),
        vector3_slot(),
        float_slot(2.0),
        vector3_slot(),
    };
    program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    program.instructions = {
        {.opcode = bytecode_opcode::vector_dot, .operand0 = 0, .operand1 = 1, .operand2 = 2, .operand3 = 1},
        {.opcode = bytecode_opcode::vector_length, .operand0 = 0, .operand1 = 3, .operand2 = 2},
        {.opcode = bytecode_opcode::vector_normalize, .operand0 = 0, .operand1 = 4, .operand2 = 3},
        {.opcode = bytecode_opcode::vector_scale,
         .operand0 = 0,
         .operand1 = 5,
         .operand2 = 6,
         .operand3 = invalid_instruction},
    };

    vm_instance instance{program};
    const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<double>(*instance.value_slot(2)) == 11.0);
    assert(std::get<double>(*instance.value_slot(3)) == 5.0);
    [[maybe_unused]] const auto normalized = std::get<std::array<double, 3>>(*instance.value_slot(4));
    assert(std::abs(normalized[0] - 0.6) < 1e-9);
    assert(std::abs(normalized[1] - 0.8) < 1e-9);
    [[maybe_unused]] const auto scaled = std::get<std::array<double, 3>>(*instance.value_slot(6));
    assert((scaled == std::array<double, 3>{6.0, 8.0, 0.0}));
}

void test_invalid_arithmetic_is_reported()
{
    bytecode_program program;
    program.value_slots = {int_slot(4), int_slot(0), int_slot()};
    program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    program.instructions = {{.opcode = bytecode_opcode::divide,
                             .operand0 = 0,
                             .operand1 = 1,
                             .operand2 = 2,
                             .operand3 = invalid_instruction}};

    vm_instance instance{program};
    const execution_result result = instance.begin_play();
    assert(result.status == execution_status::invalid_operation);
}

void test_compiler_emits_variable_math_prelude()
{
    constexpr std::string_view source = R"json({
        "version": 1,
        "assetType": "flow",
        "name": "Language",
        "graph": {
            "version": 1,
            "variables": [
                {"id": "score", "name": "Score", "type": "int", "defaultValue": 1, "exposed": false}
            ],
            "nodes": [
                {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                {"id": "left", "type": "intLiteral", "position": [0, 120], "values": {"value": 2}},
                {"id": "right", "type": "intLiteral", "position": [0, 220], "values": {"value": 3}},
                {"id": "add", "type": "add", "position": [220, 140], "values": {"valueType": "int"}},
                {"id": "set", "type": "setVariable", "position": [460, 0], "values": {"variableId": "score", "variableType": "int"}}
            ],
            "connections": [
                {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set", "pin": "exec"}},
                {"id": "v1", "kind": "value", "from": {"nodeId": "left", "pin": "value"}, "to": {"nodeId": "add", "pin": "a"}},
                {"id": "v2", "kind": "value", "from": {"nodeId": "right", "pin": "value"}, "to": {"nodeId": "add", "pin": "b"}},
                {"id": "v3", "kind": "value", "from": {"nodeId": "add", "pin": "value"}, "to": {"nodeId": "set", "pin": "value"}}
            ],
            "viewport": {"x": 0, "y": 0, "zoom": 1}
        }
    })json";

    const compile_result compiled = compile_asset(source);
    assert(compiled.succeeded && compiled.bytecode);
    assert(compiled.bytecode->instructions.size() == 2);
    [[maybe_unused]] const std::uint32_t entry_instruction = compiled.bytecode->entry_points[0].instruction;
    assert(entry_instruction < compiled.bytecode->instructions.size());
    [[maybe_unused]] const bytecode_instruction& add_instruction = compiled.bytecode->instructions[entry_instruction];
    assert(add_instruction.opcode == bytecode_opcode::add);
    assert(add_instruction.operand3 < compiled.bytecode->instructions.size());
    [[maybe_unused]] const bytecode_instruction& store_instruction = compiled.bytecode->instructions[add_instruction.operand3];
    assert(store_instruction.opcode == bytecode_opcode::store_variable);

    vm_instance instance{*compiled.bytecode};
    const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<std::int64_t>(*instance.variable_value("score")) == 5);
}

} // namespace

void run_flow_language_tests()
{
    test_scalar_language_bytecode();
    test_vector_language_bytecode();
    test_invalid_arithmetic_is_reported();
    test_compiler_emits_variable_math_prelude();
}

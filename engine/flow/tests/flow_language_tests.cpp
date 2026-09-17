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
    [[maybe_unused]] const bytecode_instruction& store_instruction =
        compiled.bytecode->instructions[add_instruction.operand3];
    assert(store_instruction.opcode == bytecode_opcode::store_variable);

    vm_instance instance{*compiled.bytecode};
    const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<std::int64_t>(*instance.variable_value("score")) == 5);
}

void test_sequence_and_switch_bytecode()
{
    bytecode_program sequence_program;
    sequence_program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    sequence_program.value_slots = {int_slot(1), int_slot(2), int_slot(3), int_slot(4)};
    sequence_program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    sequence_program.instructions = {
        {.opcode = bytecode_opcode::sequence, .operand0 = 1, .operand1 = 2, .operand2 = 3, .operand3 = 4},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 0},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 1},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 2},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 3},
    };

    vm_instance sequence{sequence_program};
    assert(sequence.valid());
    [[maybe_unused]] const execution_result sequence_result = sequence.begin_play();
    assert(sequence_result.succeeded());
    assert(std::get<std::int64_t>(*sequence.variable_value("result")) == 4);

    bytecode_program switch_program;
    switch_program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    switch_program.value_slots = {int_slot(3), int_slot(10), int_slot(20), int_slot(30), int_slot(40), int_slot(99)};
    switch_program.switch_int_tables.push_back(
        {.values = {1, 2, 3, 4}, .instructions = {1, 2, 3, 4, 5}});
    switch_program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    switch_program.instructions = {
        {.opcode = bytecode_opcode::switch_integer, .operand0 = 0, .operand1 = 0},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 1},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 2},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 3},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 4},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 5},
    };

    vm_instance switch_instance{switch_program};
    assert(switch_instance.valid());
    [[maybe_unused]] const execution_result switch_result = switch_instance.begin_play();
    assert(switch_result.succeeded());
    assert(std::get<std::int64_t>(*switch_instance.variable_value("result")) == 30);
}

void test_do_once_and_gate_bytecode()
{
    bytecode_program do_once_program;
    do_once_program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    do_once_program.value_slots = {bool_slot(false), int_slot(1)};
    do_once_program.entry_points = {
        {.kind = entry_point_kind::begin_play, .instruction = 0},
        {.kind = entry_point_kind::tick, .instruction = 0},
        {.kind = entry_point_kind::input_action_triggered, .action = "Reset", .instruction = 2},
    };
    do_once_program.instructions = {
        {.opcode = bytecode_opcode::do_once, .operand0 = 0, .operand1 = 1},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 1},
        {.opcode = bytecode_opcode::do_once_reset, .operand0 = 0},
    };

    vm_instance do_once{do_once_program};
    assert(do_once.valid());
    assert(do_once.begin_play().succeeded());
    assert(std::get<std::int64_t>(*do_once.variable_value("result")) == 1);
    assert(do_once.set_variable_value("result", std::int64_t{0}));
    assert(do_once.tick(1.0 / 60.0).succeeded());
    assert(std::get<std::int64_t>(*do_once.variable_value("result")) == 0);
    assert(do_once.input_action_triggered("Reset", 1.0).succeeded());
    assert(do_once.tick(1.0 / 60.0).succeeded());
    assert(std::get<std::int64_t>(*do_once.variable_value("result")) == 1);

    bytecode_program gate_program;
    gate_program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    gate_program.value_slots = {bool_slot(false), int_slot(1)};
    gate_program.entry_points = {
        {.kind = entry_point_kind::begin_play},
        {.kind = entry_point_kind::input_action_triggered, .action = "Enter", .instruction = 0},
        {.kind = entry_point_kind::input_action_triggered, .action = "Open", .instruction = 1},
        {.kind = entry_point_kind::input_action_triggered, .action = "Close", .instruction = 2},
        {.kind = entry_point_kind::input_action_triggered, .action = "Toggle", .instruction = 3},
    };
    gate_program.instructions = {
        {.opcode = bytecode_opcode::gate_enter, .operand0 = 0, .operand1 = 4},
        {.opcode = bytecode_opcode::gate_open, .operand0 = 0},
        {.opcode = bytecode_opcode::gate_close, .operand0 = 0},
        {.opcode = bytecode_opcode::gate_toggle, .operand0 = 0},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 1},
    };

    vm_instance gate{gate_program};
    assert(gate.valid());
    assert(gate.begin_play().succeeded());
    assert(gate.input_action_triggered("Enter", 1.0).succeeded());
    assert(std::get<std::int64_t>(*gate.variable_value("result")) == 0);
    assert(gate.input_action_triggered("Open", 1.0).succeeded());
    assert(gate.input_action_triggered("Enter", 1.0).succeeded());
    assert(std::get<std::int64_t>(*gate.variable_value("result")) == 1);
    assert(gate.set_variable_value("result", std::int64_t{0}));
    assert(gate.input_action_triggered("Close", 1.0).succeeded());
    assert(gate.input_action_triggered("Enter", 1.0).succeeded());
    assert(std::get<std::int64_t>(*gate.variable_value("result")) == 0);
    assert(gate.input_action_triggered("Toggle", 1.0).succeeded());
    assert(gate.input_action_triggered("Enter", 1.0).succeeded());
    assert(std::get<std::int64_t>(*gate.variable_value("result")) == 1);
}

void test_loop_bytecode_and_budget()
{
    bytecode_program for_program;
    for_program.variables = {
        {.id = "result", .name = "Result", .type = value_type::integer, .default_value = std::int64_t{0}}};
    for_program.value_slots = {int_slot(1), int_slot(3), int_slot()};
    for_program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    for_program.instructions = {
        {.opcode = bytecode_opcode::for_loop,
         .operand0 = 0,
         .operand1 = 1,
         .operand2 = 2,
         .operand3 = 1,
         .operand4 = invalid_instruction},
        {.opcode = bytecode_opcode::store_variable, .operand0 = 0, .operand1 = 2},
    };

    vm_instance for_loop{for_program};
    assert(for_loop.valid());
    [[maybe_unused]] const execution_result for_result = for_loop.begin_play();
    assert(for_result.succeeded());
    assert(std::get<std::int64_t>(*for_loop.variable_value("result")) == 3);

    bytecode_program while_program;
    while_program.value_slots = {bool_slot(true)};
    while_program.entry_points.push_back({.kind = entry_point_kind::begin_play, .instruction = 0});
    while_program.instructions = {{.opcode = bytecode_opcode::while_loop,
                                   .operand0 = 0,
                                   .operand1 = invalid_instruction,
                                   .operand2 = invalid_instruction,
                                   .operand3 = invalid_instruction}};

    vm_instance while_loop{while_program, {.instruction_budget = 4}};
    assert(while_loop.valid());
    [[maybe_unused]] const execution_result while_result = while_loop.begin_play();
    assert(while_result.status == execution_status::instruction_budget_exceeded);
}

void test_compiler_executes_sequence_switch_and_for_loop()
{
    constexpr std::string_view source = R"json({
        "version": 1,
        "assetType": "flow",
        "name": "Control",
        "graph": {
            "version": 1,
            "variables": [
                {"id": "result", "name": "Result", "type": "int", "defaultValue": 0, "exposed": false}
            ],
            "nodes": [
                {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                {"id": "seq", "type": "sequence", "position": [120, 0], "values": {}},
                {"id": "selection", "type": "intLiteral", "position": [120, 180], "values": {"value": 2}},
                {"id": "switch", "type": "switchInt", "position": [280, 0], "values": {"cases": [1, 2, 3, 4]}},
                {"id": "twenty", "type": "intLiteral", "position": [280, 220], "values": {"value": 20}},
                {"id": "ten", "type": "intLiteral", "position": [280, 300], "values": {"value": 10}},
                {"id": "setTwenty", "type": "setVariable", "position": [500, 0], "values": {"variableId": "result", "variableType": "int"}},
                {"id": "setTen", "type": "setVariable", "position": [500, 120], "values": {"variableId": "result", "variableType": "int"}},
                {"id": "for", "type": "forLoop", "position": [700, 0], "values": {}},
                {"id": "first", "type": "intLiteral", "position": [700, 220], "values": {"value": 1}},
                {"id": "last", "type": "intLiteral", "position": [700, 300], "values": {"value": 3}},
                {"id": "setIndex", "type": "setVariable", "position": [900, 0], "values": {"variableId": "result", "variableType": "int"}}
            ],
            "connections": [
                {"id": "e0", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "seq", "pin": "exec"}},
                {"id": "e1", "kind": "execution", "from": {"nodeId": "seq", "pin": "then0"}, "to": {"nodeId": "switch", "pin": "exec"}},
                {"id": "e2", "kind": "execution", "from": {"nodeId": "switch", "pin": "case1"}, "to": {"nodeId": "setTwenty", "pin": "exec"}},
                {"id": "e3", "kind": "execution", "from": {"nodeId": "seq", "pin": "then1"}, "to": {"nodeId": "setTen", "pin": "exec"}},
                {"id": "e4", "kind": "execution", "from": {"nodeId": "seq", "pin": "then2"}, "to": {"nodeId": "for", "pin": "exec"}},
                {"id": "e5", "kind": "execution", "from": {"nodeId": "for", "pin": "loopBody"}, "to": {"nodeId": "setIndex", "pin": "exec"}},
                {"id": "v0", "kind": "value", "from": {"nodeId": "selection", "pin": "value"}, "to": {"nodeId": "switch", "pin": "selection"}},
                {"id": "v1", "kind": "value", "from": {"nodeId": "twenty", "pin": "value"}, "to": {"nodeId": "setTwenty", "pin": "value"}},
                {"id": "v2", "kind": "value", "from": {"nodeId": "ten", "pin": "value"}, "to": {"nodeId": "setTen", "pin": "value"}},
                {"id": "v3", "kind": "value", "from": {"nodeId": "first", "pin": "value"}, "to": {"nodeId": "for", "pin": "first"}},
                {"id": "v4", "kind": "value", "from": {"nodeId": "last", "pin": "value"}, "to": {"nodeId": "for", "pin": "last"}},
                {"id": "v5", "kind": "value", "from": {"nodeId": "for", "pin": "index"}, "to": {"nodeId": "setIndex", "pin": "value"}}
            ],
            "viewport": {"x": 0, "y": 0, "zoom": 1}
        }
    })json";

    const compile_result compiled = compile_asset(source);
    assert(compiled.succeeded && compiled.bytecode);
    assert(compiled.bytecode->switch_int_tables.size() == 1);
    vm_instance instance{*compiled.bytecode};
    [[maybe_unused]] const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<std::int64_t>(*instance.variable_value("result")) == 3);
}

void test_compiler_reevaluates_while_condition()
{
    constexpr std::string_view source = R"json({
        "version": 1,
        "assetType": "flow",
        "name": "While",
        "graph": {
            "version": 1,
            "variables": [
                {"id": "count", "name": "Count", "type": "int", "defaultValue": 3, "exposed": false}
            ],
            "nodes": [
                {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                {"id": "while", "type": "whileLoop", "position": [300, 0], "values": {}},
                {"id": "getCondition", "type": "getVariable", "position": [0, 180], "values": {"variableId": "count", "variableType": "int"}},
                {"id": "zero", "type": "intLiteral", "position": [0, 260], "values": {"value": 0}},
                {"id": "greater", "type": "compare", "position": [150, 200], "values": {"valueType": "int", "operator": "greater"}},
                {"id": "getBody", "type": "getVariable", "position": [420, 180], "values": {"variableId": "count", "variableType": "int"}},
                {"id": "one", "type": "intLiteral", "position": [420, 260], "values": {"value": 1}},
                {"id": "subtract", "type": "subtract", "position": [560, 220], "values": {"valueType": "int"}},
                {"id": "set", "type": "setVariable", "position": [720, 0], "values": {"variableId": "count", "variableType": "int"}}
            ],
            "connections": [
                {"id": "e0", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "while", "pin": "exec"}},
                {"id": "e1", "kind": "execution", "from": {"nodeId": "while", "pin": "loopBody"}, "to": {"nodeId": "set", "pin": "exec"}},
                {"id": "v0", "kind": "value", "from": {"nodeId": "getCondition", "pin": "value"}, "to": {"nodeId": "greater", "pin": "a"}},
                {"id": "v1", "kind": "value", "from": {"nodeId": "zero", "pin": "value"}, "to": {"nodeId": "greater", "pin": "b"}},
                {"id": "v2", "kind": "value", "from": {"nodeId": "greater", "pin": "result"}, "to": {"nodeId": "while", "pin": "condition"}},
                {"id": "v3", "kind": "value", "from": {"nodeId": "getBody", "pin": "value"}, "to": {"nodeId": "subtract", "pin": "a"}},
                {"id": "v4", "kind": "value", "from": {"nodeId": "one", "pin": "value"}, "to": {"nodeId": "subtract", "pin": "b"}},
                {"id": "v5", "kind": "value", "from": {"nodeId": "subtract", "pin": "value"}, "to": {"nodeId": "set", "pin": "value"}}
            ],
            "viewport": {"x": 0, "y": 0, "zoom": 1}
        }
    })json";

    const compile_result compiled = compile_asset(source);
    assert(compiled.succeeded && compiled.bytecode);
    vm_instance instance{*compiled.bytecode};
    [[maybe_unused]] const execution_result result = instance.begin_play();
    assert(result.succeeded());
    assert(std::get<std::int64_t>(*instance.variable_value("count")) == 0);
}

} // namespace

void run_flow_language_tests()
{
    test_scalar_language_bytecode();
    test_vector_language_bytecode();
    test_invalid_arithmetic_is_reported();
    test_compiler_emits_variable_math_prelude();
    test_sequence_and_switch_bytecode();
    test_do_once_and_gate_bytecode();
    test_loop_bytecode_and_budget();
    test_compiler_executes_sequence_switch_and_for_loop();
    test_compiler_reevaluates_while_condition();
}

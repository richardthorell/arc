#include <arc/flow/flow.h>

#include <algorithm>
#include <cassert>
#include <string_view>

namespace
{

bool has_code(const arc::flow::compile_result& result, std::string_view code)
{
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const arc::flow::diagnostic& item) { return item.code == code; });
}

} // namespace

int main()
{
    using namespace arc::flow;

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "EmptyBeginPlay",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}}
                ],
                "connections": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(result.succeeded);
        assert(result.ir);
        assert(result.bytecode);
        assert(result.ir->entry_points.size() == 1);
        assert(result.ir->entry_points[0].kind == entry_point_kind::begin_play);
        assert(result.ir->entry_points[0].instruction == invalid_instruction);
        assert(result.bytecode->instructions.empty());
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "Branch",
            "graph": {
                "version": 1,
                "variables": [
                    {"id": "speed", "name": "Speed", "type": "float", "defaultValue": 4.5, "exposed": true}
                ],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "branch", "type": "branch", "position": [200, 0], "values": {}}
                ],
                "connections": [
                    {
                        "id": "exec-1",
                        "kind": "execution",
                        "from": {"nodeId": "begin", "pin": "exec"},
                        "to": {"nodeId": "branch", "pin": "exec"}
                    }
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(result.succeeded);
        assert(result.ir && result.bytecode);
        assert(result.ir->variables.size() == 1);
        assert(result.ir->instructions.size() == 1);
        assert(result.ir->instructions[0].opcode == ir_opcode::branch);
        assert(result.ir->entry_points[0].instruction == 0);
        assert(result.ir->value_slots.size() == 1);
        assert(result.ir->value_slots[0].type == value_type::boolean);
        assert(std::get<bool>(result.ir->value_slots[0].initial_value) == false);
        assert(result.bytecode->instructions[0].opcode == bytecode_opcode::branch);
        assert(result.bytecode->instructions[0].operand1 == invalid_instruction);
        assert(result.bytecode->instructions[0].operand2 == invalid_instruction);
        assert(result.bytecode->instruction_nodes[0] == "branch");
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "Tick",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "tick", "type": "tick", "position": [0, 0], "values": {}}
                ],
                "connections": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(result.succeeded);
        assert(result.ir);
        assert(result.ir->value_slots.size() == 1);
        assert(result.ir->entry_points.size() == 1);
        assert(result.ir->entry_points[0].kind == entry_point_kind::tick);
        assert(result.ir->entry_points[0].value_bindings.size() == 1);
        assert(result.ir->entry_points[0].value_bindings[0].source == entry_value_kind::delta_seconds);
        assert(result.ir->entry_points[0].value_bindings[0].slot == 0);
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "TypeMismatch",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "tick", "type": "tick", "position": [0, 0], "values": {}},
                    {"id": "branch", "type": "branch", "position": [200, 0], "values": {}}
                ],
                "connections": [
                    {
                        "id": "value-1",
                        "kind": "value",
                        "from": {"nodeId": "tick", "pin": "deltaSeconds"},
                        "to": {"nodeId": "branch", "pin": "condition"}
                    }
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(!result.succeeded);
        assert(!result.ir);
        assert(!result.bytecode);
        assert(has_code(result, "FLOW_VALUE_TYPE_MISMATCH"));
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "Cycle",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "a", "type": "branch", "position": [200, 0], "values": {}},
                    {"id": "b", "type": "branch", "position": [400, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "a", "pin": "exec"}},
                    {"id": "e2", "kind": "execution", "from": {"nodeId": "a", "pin": "true"}, "to": {"nodeId": "b", "pin": "exec"}},
                    {"id": "e3", "kind": "execution", "from": {"nodeId": "b", "pin": "true"}, "to": {"nodeId": "a", "pin": "exec"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(!result.succeeded);
        assert(has_code(result, "FLOW_EXECUTION_CYCLE"));
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "Fanout",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "a", "type": "branch", "position": [200, 0], "values": {}},
                    {"id": "b", "type": "branch", "position": [200, 200], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "a", "pin": "exec"}},
                    {"id": "e2", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "b", "pin": "exec"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");
        assert(!result.succeeded);
        assert(has_code(result, "FLOW_EXECUTION_FANOUT"));
    }

    return 0;
}

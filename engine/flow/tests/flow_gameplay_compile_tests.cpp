#include <arc/flow/flow.h>

#include <algorithm>
#include <cassert>
#include <string_view>

namespace
{

[[maybe_unused]] bool has_code(const arc::flow::compile_result& result, std::string_view code)
{
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const arc::flow::diagnostic& item) { return item.code == code; });
}

std::uint32_t find_slot(const arc::flow::ir_program& program, std::string_view node, std::string_view pin)
{
    const auto iterator =
        std::find_if(program.value_slots.begin(), program.value_slots.end(), [&](const arc::flow::ir_value_slot& slot)
                     { return slot.source_node_id == node && slot.source_pin_id == pin; });
    assert(iterator != program.value_slots.end());
    return iterator->index;
}

} // namespace

void run_flow_gameplay_compile_tests()
{
    using namespace arc::flow;

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "SetName",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "self", "type": "selfEntity", "position": [0, 100], "values": {}},
                    {"id": "name-value", "type": "stringLiteral", "position": [0, 200], "values": {"value": "Player"}},
                    {"id": "set", "type": "setName", "position": [240, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set", "pin": "entity"}},
                    {"id": "v2", "kind": "value", "from": {"nodeId": "name-value", "pin": "value"}, "to": {"nodeId": "set", "pin": "name"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(result.succeeded && result.ir && result.bytecode);
        [[maybe_unused]] const std::uint32_t self_slot = find_slot(*result.ir, "self", "entity");
        [[maybe_unused]] const std::uint32_t name_slot = find_slot(*result.ir, "name-value", "value");
        assert(std::get<std::string>(result.ir->value_slots[name_slot].initial_value) == "Player");
        assert(result.bytecode->instructions.size() == 2);
        assert(result.bytecode->entry_points[0].instruction == 1);
        assert(result.bytecode->instructions[1].opcode == bytecode_opcode::self_entity);
        assert(result.bytecode->instructions[1].operand0 == self_slot);
        assert(result.bytecode->instructions[1].operand1 == 0);
        assert(result.bytecode->instructions[0].opcode == bytecode_opcode::world_set_name);
        assert(result.bytecode->instructions[0].operand0 == self_slot);
        assert(result.bytecode->instructions[0].operand1 == name_slot);
        assert(result.bytecode->instruction_nodes[0] == "set");
        assert(result.bytecode->instruction_nodes[1] == "self");
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "ActiveBranch",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "tick", "type": "tick", "position": [0, 0], "values": {}},
                    {"id": "self", "type": "selfEntity", "position": [0, 120], "values": {}},
                    {"id": "active", "type": "getActive", "position": [220, 0], "values": {}},
                    {"id": "branch", "type": "branch", "position": [440, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "tick", "pin": "exec"}, "to": {"nodeId": "active", "pin": "exec"}},
                    {"id": "e2", "kind": "execution", "from": {"nodeId": "active", "pin": "then"}, "to": {"nodeId": "branch", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "active", "pin": "entity"}},
                    {"id": "v2", "kind": "value", "from": {"nodeId": "active", "pin": "active"}, "to": {"nodeId": "branch", "pin": "condition"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(result.succeeded && result.ir && result.bytecode);
        [[maybe_unused]] const std::uint32_t active_slot = find_slot(*result.ir, "active", "active");
        assert(result.bytecode->instructions.size() == 3);
        assert(result.bytecode->instructions[0].opcode == bytecode_opcode::world_get_active ||
               result.bytecode->instructions[1].opcode == bytecode_opcode::world_get_active);
        [[maybe_unused]] const auto branch = std::find_if(
            result.bytecode->instructions.begin(), result.bytecode->instructions.end(),
            [](const bytecode_instruction& instruction) { return instruction.opcode == bytecode_opcode::branch; });
        assert(branch != result.bytecode->instructions.end());
        assert(branch->operand0 == active_slot);
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "SetTransform",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "self", "type": "selfEntity", "position": [0, 100], "values": {}},
                    {"id": "position", "type": "vector3Literal", "position": [0, 200], "values": {"value": [1, 2, 3]}},
                    {"id": "rotation", "type": "vector4Literal", "position": [0, 300], "values": {"value": [0, 0, 0, 1]}},
                    {"id": "scale", "type": "vector3Literal", "position": [0, 400], "values": {"value": [2, 2, 2]}},
                    {"id": "set", "type": "setTransform", "position": [260, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "set", "pin": "entity"}},
                    {"id": "v2", "kind": "value", "from": {"nodeId": "position", "pin": "value"}, "to": {"nodeId": "set", "pin": "position"}},
                    {"id": "v3", "kind": "value", "from": {"nodeId": "rotation", "pin": "value"}, "to": {"nodeId": "set", "pin": "rotation"}},
                    {"id": "v4", "kind": "value", "from": {"nodeId": "scale", "pin": "value"}, "to": {"nodeId": "set", "pin": "scale"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(result.succeeded && result.bytecode);
        [[maybe_unused]] const auto transform =
            std::find_if(result.bytecode->instructions.begin(), result.bytecode->instructions.end(),
                         [](const bytecode_instruction& instruction)
                         { return instruction.opcode == bytecode_opcode::world_set_transform; });
        assert(transform != result.bytecode->instructions.end());
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "MissingInput",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "set", "type": "setName", "position": [200, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "set", "pin": "exec"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(!result.succeeded);
        assert(has_code(result, "FLOW_REQUIRED_INPUT"));
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "CreateAndName",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "create", "type": "createEntity", "position": [220, 0], "values": {}},
                    {"id": "name", "type": "stringLiteral", "position": [220, 180], "values": {"value": "Spawned"}},
                    {"id": "set", "type": "setName", "position": [460, 0], "values": {}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "create", "pin": "exec"}},
                    {"id": "e2", "kind": "execution", "from": {"nodeId": "create", "pin": "then"}, "to": {"nodeId": "set", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "create", "pin": "entity"}, "to": {"nodeId": "set", "pin": "entity"}},
                    {"id": "v2", "kind": "value", "from": {"nodeId": "name", "pin": "value"}, "to": {"nodeId": "set", "pin": "name"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(result.succeeded && result.ir && result.bytecode);
        [[maybe_unused]] const std::uint32_t entity_slot = find_slot(*result.ir, "create", "entity");
        assert(result.bytecode->instructions.size() == 2);
        assert(result.bytecode->entry_points[0].instruction == 0);
        assert(result.bytecode->instructions[0].opcode == bytecode_opcode::world_create_entity);
        assert(result.bytecode->instructions[0].operand0 == entity_slot);
        assert(result.bytecode->instructions[0].operand1 == 1);
        assert(result.bytecode->instructions[1].opcode == bytecode_opcode::world_set_name);
        assert(result.bytecode->instructions[1].operand0 == entity_slot);
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "CoreComponents",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "self", "type": "selfEntity", "position": [0, 160], "values": {}},
                    {"id": "has", "type": "hasCoreComponent", "position": [240, 0], "values": {"component": "tag"}},
                    {"id": "remove", "type": "removeCoreComponent", "position": [480, 0], "values": {"component": "tag"}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "has", "pin": "exec"}},
                    {"id": "e2", "kind": "execution", "from": {"nodeId": "has", "pin": "then"}, "to": {"nodeId": "remove", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "has", "pin": "entity"}},
                    {"id": "v2", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "remove", "pin": "entity"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(result.succeeded && result.bytecode);
        [[maybe_unused]] const auto has =
            std::find_if(result.bytecode->instructions.begin(), result.bytecode->instructions.end(),
                         [](const bytecode_instruction& instruction)
                         { return instruction.opcode == bytecode_opcode::world_has_core_component; });
        [[maybe_unused]] const auto remove =
            std::find_if(result.bytecode->instructions.begin(), result.bytecode->instructions.end(),
                         [](const bytecode_instruction& instruction)
                         { return instruction.opcode == bytecode_opcode::world_remove_core_component; });
        assert(has != result.bytecode->instructions.end());
        assert(remove != result.bytecode->instructions.end());
        assert(has->operand1 == static_cast<std::uint32_t>(world_core_component::tag));
        assert(remove->operand1 == static_cast<std::uint32_t>(world_core_component::tag));
    }

    {
        const compile_result result = compile_asset(R"json({
            "version": 1,
            "assetType": "flow",
            "name": "BadComponent",
            "graph": {
                "version": 1,
                "variables": [],
                "nodes": [
                    {"id": "begin", "type": "beginPlay", "position": [0, 0], "values": {}},
                    {"id": "self", "type": "selfEntity", "position": [0, 120], "values": {}},
                    {"id": "remove", "type": "removeCoreComponent", "position": [240, 0], "values": {"component": "physics"}}
                ],
                "connections": [
                    {"id": "e1", "kind": "execution", "from": {"nodeId": "begin", "pin": "exec"}, "to": {"nodeId": "remove", "pin": "exec"}},
                    {"id": "v1", "kind": "value", "from": {"nodeId": "self", "pin": "entity"}, "to": {"nodeId": "remove", "pin": "entity"}}
                ],
                "viewport": {"x": 0, "y": 0, "zoom": 1}
            }
        })json");

        assert(!result.succeeded);
        assert(has_code(result, "FLOW_CORE_COMPONENT"));
    }
}

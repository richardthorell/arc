#include "flow_play_runtime.h"

#include "project_runtime_world_bridge.h"

#include <arc/flow/flow.h>
#include <arc/scene/components.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::editor
{
namespace
{

std::string execution_error(const flow::execution_result& result, std::string_view graph, ecs::entity entity)
{
    std::string message = "Flow graph '" + std::string(graph) + "' failed for entity " + std::to_string(entity.index) +
                          ':' + std::to_string(entity.generation) + " (status " +
                          std::to_string(static_cast<unsigned>(result.status)) + ')';
    if (!result.node_id.empty()) message += " at node '" + result.node_id + '\'';
    return message;
}

flow::flow_entity flow_entity(ecs::entity entity) noexcept
{
    return {.index = entity.index, .generation = entity.generation};
}

struct bound_flow_instance
{
    ecs::entity entity{};
    std::string graph_path;
    std::shared_ptr<const flow::bytecode_program> program;
    flow::vm_instance vm;

    bound_flow_instance(ecs::entity value, std::string path, std::shared_ptr<const flow::bytecode_program> bytecode)
        : entity(value), graph_path(std::move(path)), program(std::move(bytecode)), vm(*program)
    {
    }
};

class flow_play_session
{
public:
    explicit flow_play_session(ecs::world& world) : world_(&world) {}
    flow_play_session(const flow_play_session&) = delete;
    flow_play_session& operator=(const flow_play_session&) = delete;

    ~flow_play_session()
    {
        if (!world_) return;
        runtime_world_bridge_context bridge{.world = world_, .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        for (auto& instance : instances_)
        {
            if (!instance.vm.active()) continue;
            (void)instance.vm.end_play({.api = &api, .self = flow_entity(instance.entity)});
        }
    }

    void add(ecs::entity entity, std::string path, std::shared_ptr<const flow::bytecode_program> program)
    {
        instances_.emplace_back(entity, std::move(path), std::move(program));
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return instances_.size();
    }

    flow_play_install_result begin_play()
    {
        runtime_world_bridge_context bridge{.world = world_, .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        for (auto& instance : instances_)
        {
            const auto result = instance.vm.begin_play({.api = &api, .self = flow_entity(instance.entity)});
            if (!result.succeeded()) return {.error = execution_error(result, instance.graph_path, instance.entity)};
        }
        return {.succeeded = true, .instances = instances_.size()};
    }

    void run_fixed(ecs::system_context& context)
    {
        runtime_world_bridge_context bridge{
            .world = &context.owner(), .commands = &context.commands(), .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        const auto input = context.input();
        for (auto& instance : instances_)
        {
            if (!context.owner().alive(instance.entity)) continue;
            const flow::vm_world_context world{.api = &api, .self = flow_entity(instance.entity)};
            for (const auto& command : input.commands)
                dispatch_input(instance, command, world);
            const auto result = instance.vm.fixed_tick(context.fixed_delta_seconds(), world);
            if (!result.succeeded())
                throw std::runtime_error(execution_error(result, instance.graph_path, instance.entity));
        }
    }

    void run_tick(ecs::system_context& context)
    {
        runtime_world_bridge_context bridge{
            .world = &context.owner(), .commands = &context.commands(), .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        for (auto& instance : instances_)
        {
            if (!context.owner().alive(instance.entity)) continue;
            const auto result =
                instance.vm.tick(context.frame_delta_seconds(), {.api = &api, .self = flow_entity(instance.entity)});
            if (!result.succeeded())
                throw std::runtime_error(execution_error(result, instance.graph_path, instance.entity));
        }
    }

private:
    static std::string input_action_name(const ecs::simulation_input_command& command)
    {
        switch (command.kind)
        {
            case ecs::simulation_input_kind::key:
                return "Key." + std::to_string(command.code);
            case ecs::simulation_input_kind::mouse_button:
                return "MouseButton." + std::to_string(command.code);
            case ecs::simulation_input_kind::mouse_wheel:
                return "MouseWheel";
            case ecs::simulation_input_kind::focus:
                return "Focus";
            case ecs::simulation_input_kind::mouse_position:
                break;
        }
        return {};
    }

    static void dispatch_input(bound_flow_instance& instance, const ecs::simulation_input_command& command,
                               flow::vm_world_context world)
    {
        if (command.kind == ecs::simulation_input_kind::mouse_position) return;
        const std::string action = input_action_name(command);
        if (action.empty()) return;

        flow::execution_result result;
        bool dispatched{};
        switch (command.action)
        {
            case ecs::simulation_input_action::pressed:
                result = instance.vm.input_action_triggered(action, command.value == 0.0f ? 1.0 : command.value, world);
                dispatched = true;
                break;
            case ecs::simulation_input_action::released:
                result = instance.vm.input_action_completed(action, 0.0, world);
                dispatched = true;
                break;
            case ecs::simulation_input_action::changed:
                if (command.kind == ecs::simulation_input_kind::mouse_wheel ||
                    command.kind == ecs::simulation_input_kind::focus)
                {
                    result = command.value > 0.0f ? instance.vm.input_action_triggered(action, command.value, world)
                                                  : instance.vm.input_action_completed(action, command.value, world);
                    dispatched = true;
                }
                break;
        }
        if (dispatched && !result.succeeded())
            throw std::runtime_error(execution_error(result, instance.graph_path, instance.entity));
    }

    ecs::world* world_{};
    std::vector<bound_flow_instance> instances_;
};

std::optional<std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

} // namespace

bool valid_flow_graph_path(std::string_view value) noexcept
{
    try
    {
        if (value.empty() || value.find('\\') != std::string_view::npos) return false;
        const std::filesystem::path path{value};
        if (path.is_absolute() || path.has_root_name() || path.extension() != ".arcflow") return false;
        const auto normalized = path.lexically_normal();
        if (normalized.generic_string() != value) return false;
        return std::none_of(normalized.begin(), normalized.end(), [](const auto& part) { return part == ".."; });
    }
    catch (...)
    {
        return false;
    }
}

flow_play_install_result install_flow_play_runtime(framework::runtime_world& world,
                                                   const std::filesystem::path& content_root)
{
    std::unordered_map<std::string, std::shared_ptr<const flow::bytecode_program>> programs;
    auto session = std::make_shared<flow_play_session>(world.entities());

    for (const auto entity : world.entities().entities())
    {
        const auto* binding = std::as_const(world.entities()).try_get<scene::flow_component>(entity);
        if (!binding || !binding->enabled) continue;
        if (const auto* active = std::as_const(world.entities()).try_get<scene::active_component>(entity);
            active && !active->active)
            continue;
        if (!valid_flow_graph_path(binding->graph_path))
            return {.error = "Flow entity " + std::to_string(entity.index) + " has an invalid graph path '" +
                             binding->graph_path + "'"};

        const auto found = programs.find(binding->graph_path);
        std::shared_ptr<const flow::bytecode_program> program;
        if (found != programs.end())
        {
            program = found->second;
        }
        else
        {
            const auto source = read_text_file(content_root / std::filesystem::path{binding->graph_path});
            if (!source) return {.error = "Flow graph could not be read: " + binding->graph_path};
            auto compiled = flow::compile_asset(*source);
            if (!compiled.succeeded || !compiled.bytecode)
            {
                std::string detail = "Flow graph failed to compile: " + binding->graph_path;
                if (!compiled.diagnostics.empty()) detail += ": " + compiled.diagnostics.front().message;
                return {.error = std::move(detail)};
            }
            program = std::make_shared<const flow::bytecode_program>(std::move(*compiled.bytecode));
            programs.emplace(binding->graph_path, program);
        }
        session->add(entity, binding->graph_path, std::move(program));
    }

    if (session->size() == 0) return {.succeeded = true};

    auto begun = session->begin_play();
    if (!begun.succeeded) return begun;

    if (!world.systems().add({.name = "__arc.flow.fixed",
                              .phase = ecs::system_phase::gameplay_commands,
                              .exclusive_world_access = true,
                              .execute = [session](ecs::system_context& context) { session->run_fixed(context); }}))
        return {.error = "Flow fixed-step system could not be installed"};
    if (!world.systems().add({.name = "__arc.flow.tick",
                              .phase = ecs::system_phase::presentation_extraction,
                              .exclusive_world_access = true,
                              .execute = [session](ecs::system_context& context) { session->run_tick(context); }}))
        return {.error = "Flow frame-tick system could not be installed"};

    return {.succeeded = true, .instances = session->size(), .unique_programs = programs.size()};
}

} // namespace arc::editor

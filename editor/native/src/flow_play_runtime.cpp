#include "flow_play_runtime.h"

#include "project_runtime_world_bridge.h"

#include <arc/flow/flow.h>
#include <arc/scene/components.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace arc::editor
{
namespace
{

constexpr std::string_view flow_lifecycle_pre_system = "__arc.flow.lifecycle.pre";
constexpr std::string_view flow_fixed_system = "__arc.flow.fixed";
constexpr std::string_view flow_lifecycle_post_system = "__arc.flow.lifecycle.post";
constexpr std::string_view flow_tick_system = "__arc.flow.tick";

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

const scene::flow_component* active_flow_binding(const ecs::world& world, ecs::entity entity) noexcept
{
    if (!world.alive(entity)) return nullptr;
    const auto* binding = world.try_get<scene::flow_component>(entity);
    if (!binding || !binding->enabled || binding->graph_path.empty()) return nullptr;
    if (const auto* active = world.try_get<scene::active_component>(entity); active && !active->active) return nullptr;
    return binding;
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

std::optional<std::string> read_text_file(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

class flow_play_session
{
public:
    flow_play_session(ecs::world& world, std::filesystem::path content_root)
        : world_(&world), content_root_(std::move(content_root))
    {
    }

    flow_play_session(const flow_play_session&) = delete;
    flow_play_session& operator=(const flow_play_session&) = delete;

    ~flow_play_session()
    {
        if (!world_) return;
        for (auto& instance : instances_)
            (void)end_instance(instance, nullptr);
    }

    [[nodiscard]] flow_play_install_result initialize()
    {
        const ecs::change_revision baseline = world_->revision();
        for (const auto entity : world_->entities())
        {
            const auto* binding = active_flow_binding(std::as_const(*world_), entity);
            if (!binding) continue;
            if (const auto error = append_instance(entity, binding->graph_path); !error.empty()) return {.error = error};
        }

        for (auto& instance : instances_)
        {
            const auto result = begin_instance(instance, nullptr);
            if (!result.succeeded()) return {.error = execution_error(result, instance.graph_path, instance.entity)};
        }

        cursor_.revision = baseline;
        return {.succeeded = true, .instances = instances_.size(), .unique_programs = programs_.size()};
    }

    void reconcile(ecs::system_context& context)
    {
        const ecs::change_revision target_revision = world_->revision();
        std::unordered_set<ecs::entity, ecs::entity_hash> seen;
        std::vector<ecs::entity> touched;
        const auto add_touched = [&](ecs::entity entity)
        {
            if (seen.emplace(entity).second) touched.push_back(entity);
        };

        if (target_revision < cursor_.revision)
        {
            for (const auto& instance : instances_)
                add_touched(instance.entity);
            for (const auto entity : world_->entities())
                add_touched(entity);
        }
        else
        {
            const auto flow_type = ecs::component_type<scene::flow_component>();
            const auto active_type = ecs::component_type<scene::active_component>();
            for (const auto& change : world_->structural_changes())
            {
                if (change.revision <= cursor_.revision || change.revision > target_revision) continue;
                if (change.kind == ecs::structural_change_kind::entity_destroyed || change.component == flow_type ||
                    change.component == active_type)
                    add_touched(change.value);
            }
            for (const auto change : world_->changes_since<scene::flow_component>(cursor_))
                if (change.revision <= target_revision) add_touched(change.value);
            for (const auto change : world_->changes_since<scene::active_component>(cursor_))
                if (change.revision <= target_revision) add_touched(change.value);
        }

        std::sort(touched.begin(), touched.end(), [](ecs::entity lhs, ecs::entity rhs)
                  { return lhs.index != rhs.index ? lhs.index < rhs.index : lhs.generation < rhs.generation; });
        for (const auto entity : touched)
        {
            if (const auto error = reconcile_entity(entity, context); !error.empty()) throw std::runtime_error(error);
        }
        cursor_.revision = target_revision;
    }

    void run_fixed(ecs::system_context& context)
    {
        runtime_world_bridge_context bridge{
            .world = &context.owner(), .commands = &context.commands(), .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        const auto input = context.input();
        for (auto& instance : instances_)
        {
            if (!matches_current_binding(instance)) continue;
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
            if (!matches_current_binding(instance)) continue;
            const auto result =
                instance.vm.tick(context.frame_delta_seconds(), {.api = &api, .self = flow_entity(instance.entity)});
            if (!result.succeeded())
                throw std::runtime_error(execution_error(result, instance.graph_path, instance.entity));
        }
    }

private:
    [[nodiscard]] bool matches_current_binding(const bound_flow_instance& instance) const noexcept
    {
        const auto* binding = active_flow_binding(std::as_const(*world_), instance.entity);
        return binding && binding->graph_path == instance.graph_path;
    }

    [[nodiscard]] std::shared_ptr<const flow::bytecode_program> program_for(std::string_view graph_path,
                                                                           std::string& error)
    {
        if (!valid_flow_graph_path(graph_path))
        {
            error = "Flow graph path is invalid: " + std::string(graph_path);
            return {};
        }

        const auto found = programs_.find(std::string(graph_path));
        if (found != programs_.end()) return found->second;

        const auto source = read_text_file(content_root_ / std::filesystem::path{graph_path});
        if (!source)
        {
            error = "Flow graph could not be read: " + std::string(graph_path);
            return {};
        }

        auto compiled = flow::compile_asset(*source);
        if (!compiled.succeeded || !compiled.bytecode)
        {
            error = "Flow graph failed to compile: " + std::string(graph_path);
            if (!compiled.diagnostics.empty()) error += ": " + compiled.diagnostics.front().message;
            return {};
        }

        auto program = std::make_shared<const flow::bytecode_program>(std::move(*compiled.bytecode));
        programs_.emplace(std::string(graph_path), program);
        return program;
    }

    [[nodiscard]] std::string append_instance(ecs::entity entity, std::string_view graph_path)
    {
        std::string error;
        auto program = program_for(graph_path, error);
        if (!program) return error;
        instances_.emplace_back(entity, std::string(graph_path), std::move(program));
        return {};
    }

    [[nodiscard]] flow::execution_result begin_instance(bound_flow_instance& instance, ecs::system_context* context)
    {
        runtime_world_bridge_context bridge{.world = world_,
                                            .commands = context ? &context->commands() : nullptr,
                                            .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        return instance.vm.begin_play({.api = &api, .self = flow_entity(instance.entity)});
    }

    [[nodiscard]] flow::execution_result end_instance(bound_flow_instance& instance, ecs::system_context* context)
    {
        if (!instance.vm.active()) return {};
        runtime_world_bridge_context bridge{.world = world_,
                                            .commands = context ? &context->commands() : nullptr,
                                            .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        return instance.vm.end_play({.api = &api, .self = flow_entity(instance.entity)});
    }

    [[nodiscard]] std::string stop_instance(std::size_t index, ecs::system_context& context)
    {
        const bool entity_alive = world_->alive(instances_[index].entity);
        const auto result = end_instance(instances_[index], &context);
        if (entity_alive && !result.succeeded())
            return execution_error(result, instances_[index].graph_path, instances_[index].entity);
        instances_.erase(instances_.begin() + static_cast<std::ptrdiff_t>(index));
        return {};
    }

    [[nodiscard]] std::string reconcile_entity(ecs::entity entity, ecs::system_context& context)
    {
        auto found = std::find_if(instances_.begin(), instances_.end(),
                                  [entity](const bound_flow_instance& instance) { return instance.entity == entity; });
        const auto* binding = active_flow_binding(std::as_const(*world_), entity);
        if (found != instances_.end() && binding && found->graph_path == binding->graph_path) return {};

        if (found != instances_.end())
        {
            const std::size_t index = static_cast<std::size_t>(std::distance(instances_.begin(), found));
            if (const auto error = stop_instance(index, context); !error.empty()) return error;
        }

        binding = active_flow_binding(std::as_const(*world_), entity);
        if (!binding) return {};
        if (const auto error = append_instance(entity, binding->graph_path); !error.empty()) return error;

        auto& instance = instances_.back();
        const auto result = begin_instance(instance, &context);
        if (result.succeeded()) return {};
        if (instance.vm.active()) (void)end_instance(instance, &context);
        const std::string error = execution_error(result, instance.graph_path, instance.entity);
        instances_.pop_back();
        return error;
    }

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
    std::filesystem::path content_root_;
    ecs::change_cursor cursor_{};
    std::unordered_map<std::string, std::shared_ptr<const flow::bytecode_program>> programs_;
    std::vector<bound_flow_instance> instances_;
};

} // namespace

bool valid_flow_graph_path(std::string_view value) noexcept
{
    try
    {
        if (value.empty() || value.find('\\') != std::string_view::npos) return false;
        const std::filesystem::path path{value};
        if (path.has_root_path() || path.extension() != ".arcflow") return false;
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
    auto session = std::make_shared<flow_play_session>(world.entities(), content_root);
    auto result = session->initialize();
    if (!result.succeeded) return result;

    auto& systems = world.systems();
    if (!systems.add({.name = std::string(flow_lifecycle_pre_system),
                      .phase = ecs::system_phase::network_receive,
                      .exclusive_world_access = true,
                      .execute = [session](ecs::system_context& context) { session->reconcile(context); }}))
        return {.error = "Flow pre-fixed lifecycle system could not be installed"};

    if (!systems.add({.name = std::string(flow_fixed_system),
                      .phase = ecs::system_phase::gameplay_commands,
                      .exclusive_world_access = true,
                      .execute = [session](ecs::system_context& context) { session->run_fixed(context); }}))
    {
        systems.remove(flow_lifecycle_pre_system);
        return {.error = "Flow fixed-step system could not be installed"};
    }

    if (!systems.add({.name = std::string(flow_lifecycle_post_system),
                      .phase = ecs::system_phase::replication,
                      .exclusive_world_access = true,
                      .execute = [session](ecs::system_context& context) { session->reconcile(context); }}))
    {
        systems.remove(flow_fixed_system);
        systems.remove(flow_lifecycle_pre_system);
        return {.error = "Flow post-fixed lifecycle system could not be installed"};
    }

    if (!systems.add({.name = std::string(flow_tick_system),
                      .phase = ecs::system_phase::presentation_extraction,
                      .exclusive_world_access = true,
                      .execute = [session](ecs::system_context& context) { session->run_tick(context); }}))
    {
        systems.remove(flow_lifecycle_post_system);
        systems.remove(flow_fixed_system);
        systems.remove(flow_lifecycle_pre_system);
        return {.error = "Flow frame-tick system could not be installed"};
    }

    return result;
}

} // namespace arc::editor

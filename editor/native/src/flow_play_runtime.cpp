#include "flow_play_runtime.h"

#include "project_runtime_world_bridge.h"

#include <arc/diagnostics/log.h>
#include <arc/flow/flow.h>
#include <arc/input/input.h>
#include <arc/project/input_config.h>
#include <arc/scene/components.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
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

std::string compile_error(std::string_view graph, const flow::compile_result& result)
{
    std::string message = "Flow graph '" + std::string(graph) + "' failed to compile";
    if (result.diagnostics.empty()) return message;

    const auto& diagnostic = result.diagnostics.front();
    if (!diagnostic.code.empty()) message += " [" + diagnostic.code + ']';
    if (!diagnostic.message.empty()) message += ": " + diagnostic.message;
    if (!diagnostic.node_id.empty()) message += " at node '" + diagnostic.node_id + '\'';
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
    if (!binding || !binding->enabled || (!binding->graph.guid.valid() && binding->graph.path_hint.empty()))
        return nullptr;
    if (const auto* active = world.try_get<scene::active_component>(entity); active && !active->active) return nullptr;
    return binding;
}

std::optional<input::key> simulation_key(std::int32_t code) noexcept
{
    if (code >= 'A' && code <= 'Z')
        return static_cast<input::key>(static_cast<unsigned>(input::key::a) + static_cast<unsigned>(code - 'A'));
    if (code >= 'a' && code <= 'z')
        return static_cast<input::key>(static_cast<unsigned>(input::key::a) + static_cast<unsigned>(code - 'a'));
    if (code >= '0' && code <= '9')
        return static_cast<input::key>(static_cast<unsigned>(input::key::num0) + static_cast<unsigned>(code - '0'));

    switch (code)
    {
        case 27:
            return input::key::escape;
        case 32:
            return input::key::space;
        case 13:
            return input::key::enter;
        case 9:
            return input::key::tab;
        case 8:
            return input::key::backspace;
        case 127:
            return input::key::delete_key;
        case 0x1001:
            return input::key::left;
        case 0x1002:
            return input::key::right;
        case 0x1003:
            return input::key::up;
        case 0x1004:
            return input::key::down;
        case 0x1005:
            return input::key::home;
        case 0x1006:
            return input::key::end;
        case 0x1007:
            return input::key::page_up;
        case 0x1008:
            return input::key::page_down;
        case 0x1010:
            return input::key::left_shift;
        case 0x1011:
            return input::key::left_control;
        case 0x1012:
            return input::key::left_alt;
        default:
            return std::nullopt;
    }
}

std::optional<input::mouse_button> simulation_mouse_button(std::int32_t code) noexcept
{
    switch (code)
    {
        case 1:
            return input::mouse_button::left;
        case 2:
            return input::mouse_button::right;
        case 3:
            return input::mouse_button::middle;
        case 4:
            return input::mouse_button::x1;
        case 5:
            return input::mouse_button::x2;
        default:
            return std::nullopt;
    }
}

struct compiled_flow_artifact
{
    assets::asset_reference reference;
    std::string display_name;
    std::string source;
    std::shared_ptr<const flow::bytecode_program> program;
    std::uint64_t source_generation{};
    std::uint64_t program_generation{1};
    std::optional<std::string> rejected_source;
    std::string reload_error;
};

struct bound_flow_instance
{
    ecs::entity entity{};
    assets::asset_reference graph;
    std::string graph_name;
    std::shared_ptr<const flow::bytecode_program> program;
    std::uint64_t artifact_generation{};
    flow::vm_instance vm;

    bound_flow_instance(ecs::entity value, assets::asset_reference reference, std::string name,
                        std::shared_ptr<const flow::bytecode_program> bytecode, std::uint64_t generation)
        : entity(value), graph(std::move(reference)), graph_name(std::move(name)), program(std::move(bytecode)),
          artifact_generation(generation), vm(*program)
    {
    }
};

std::string flow_key(const assets::asset_reference& reference)
{
    return reference.guid.valid() ? assets::to_string(reference.guid) : reference.path_hint;
}

void report_reload_error(compiled_flow_artifact& artifact, std::string message)
{
    if (artifact.reload_error == message) return;
    artifact.reload_error = std::move(message);
    ::arc::diagnostics::error("Flow", artifact.reload_error);
}

class flow_play_session
{
public:
    flow_play_session(ecs::world& world, flow_play_source_resolver source_resolver,
                      std::filesystem::path input_config_path)
        : world_(&world), source_resolver_(std::move(source_resolver)), input_config_path_(std::move(input_config_path))
    {
        keyboard_ = input_.connect_device({.type = input::input_device_type::keyboard,
                                           .connectivity = input::input_connectivity_type::builtin,
                                           .backend = input::input_backend_type::native,
                                           .name = "Play Keyboard",
                                           .capabilities = {.buttons = true, .button_count = 104}});
        mouse_ = input_.connect_device(
            {.type = input::input_device_type::mouse,
             .connectivity = input::input_connectivity_type::builtin,
             .backend = input::input_backend_type::native,
             .name = "Play Mouse",
             .capabilities = {
                 .buttons = true, .axes = true, .pointer = true, .scroll = true, .button_count = 5, .axis_count = 6}});
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
        if (const auto error = initialize_input(); !error.empty()) return {.error = error};

        const ecs::change_revision baseline = world_->revision();
        for (const auto entity : world_->entities())
        {
            const auto* binding = active_flow_binding(std::as_const(*world_), entity);
            if (!binding) continue;
            if (const auto error = append_instance(entity, binding->graph); !error.empty()) return {.error = error};
        }

        for (auto& instance : instances_)
        {
            const auto result = begin_instance(instance, nullptr);
            if (!result.succeeded()) return {.error = execution_error(result, instance.graph_name, instance.entity)};
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
        reload_changed_programs(context);
        sample_input(context.input());

        runtime_world_bridge_context bridge{
            .world = &context.owner(), .commands = &context.commands(), .unrestricted_access = true};
        const auto api = make_runtime_world_api(bridge);
        const auto& player = input_.player(0);

        for (auto& instance : instances_)
        {
            if (!matches_current_binding(instance)) continue;
            const flow::vm_world_context world{.api = &api, .self = flow_entity(instance.entity)};
            for (const auto& action : input_actions_)
            {
                if (player.pressed(action)) dispatch_input(instance, action, true, world);
                if (player.released(action)) dispatch_input(instance, action, false, world);
            }
            const auto result = instance.vm.fixed_tick(context.fixed_delta_seconds(), world);
            if (!result.succeeded())
                throw std::runtime_error(execution_error(result, instance.graph_name, instance.entity));
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
                throw std::runtime_error(execution_error(result, instance.graph_name, instance.entity));
        }
    }

private:
    using shared_program = std::shared_ptr<const flow::bytecode_program>;

    [[nodiscard]] std::string initialize_input()
    {
        const bool explicit_path = !input_config_path_.empty();
        if (!explicit_path) return {};

        std::error_code exists_error;
        const bool exists = std::filesystem::exists(input_config_path_, exists_error);
        if (exists_error)
            return "Input config could not be inspected: " + input_config_path_.generic_string() + ": " +
                   exists_error.message();
        if (!exists)
        {
            if (explicit_path) return "Input config could not be read: " + input_config_path_.generic_string();
            return {};
        }

        std::error_code regular_error;
        if (!std::filesystem::is_regular_file(input_config_path_, regular_error))
        {
            if (regular_error)
                return "Input config could not be inspected: " + input_config_path_.generic_string() + ": " +
                       regular_error.message();
            return "Input config is not a regular file: " + input_config_path_.generic_string();
        }

        auto loaded = project::load_input_config(input_config_path_);
        if (!loaded.succeeded) return loaded.error;
        const auto applied = project::apply_input_config(loaded.config, input_, 0);
        if (!applied.succeeded) return applied.error;
        input_actions_ = project::input_action_names(loaded.config);
        return {};
    }

    void sample_input(const ecs::simulation_input_snapshot& snapshot)
    {
        input_.begin_frame();
        for (const auto& command : snapshot.commands)
        {
            switch (command.kind)
            {
                case ecs::simulation_input_kind::key:
                {
                    const auto key = simulation_key(command.code);
                    if (!key) break;
                    if (command.action == ecs::simulation_input_action::pressed)
                        (void)input_.submit_button(keyboard_, input::make_key_control(*key), true);
                    else if (command.action == ecs::simulation_input_action::released)
                        (void)input_.submit_button(keyboard_, input::make_key_control(*key), false);
                    break;
                }
                case ecs::simulation_input_kind::mouse_button:
                {
                    const auto button = simulation_mouse_button(command.code);
                    if (!button) break;
                    if (command.action == ecs::simulation_input_action::pressed)
                        (void)input_.submit_button(mouse_, input::make_mouse_button_control(*button), true);
                    else if (command.action == ecs::simulation_input_action::released)
                        (void)input_.submit_button(mouse_, input::make_mouse_button_control(*button), false);
                    break;
                }
                case ecs::simulation_input_kind::mouse_position:
                    (void)input_.submit_axis(mouse_, input::make_mouse_axis_control(input::mouse_axis::position_x),
                                             static_cast<float>(command.x));
                    (void)input_.submit_axis(mouse_, input::make_mouse_axis_control(input::mouse_axis::position_y),
                                             static_cast<float>(command.y));
                    break;
                case ecs::simulation_input_kind::mouse_wheel:
                    (void)input_.submit_axis(mouse_, input::make_mouse_axis_control(input::mouse_axis::wheel_y),
                                             command.value);
                    break;
                case ecs::simulation_input_kind::focus:
                    if (command.value <= 0.0f) input_.release_all();
                    break;
            }
        }
    }

    [[nodiscard]] bool matches_current_binding(const bound_flow_instance& instance) const noexcept
    {
        const auto* binding = active_flow_binding(std::as_const(*world_), instance.entity);
        return binding && binding->graph == instance.graph;
    }

    void reload_changed_programs(ecs::system_context& context)
    {
        for (auto& [graph_key, artifact] : programs_)
        {
            std::string resolve_error;
            const auto resolved = source_resolver_(artifact.reference, resolve_error);
            if (!resolved)
            {
                report_reload_error(artifact, "Flow graph '" + artifact.display_name +
                                                  "' hot reload could not resolve the source; keeping generation " +
                                                  std::to_string(artifact.program_generation) +
                                                  (resolve_error.empty() ? std::string{} : ": " + resolve_error));
                continue;
            }

            if (resolved->generation == artifact.source_generation && resolved->source == artifact.source)
            {
                artifact.rejected_source.reset();
                artifact.reload_error.clear();
                continue;
            }
            if (artifact.rejected_source && *artifact.rejected_source == resolved->source) continue;

            if (!resolved->compile_error.empty())
            {
                artifact.rejected_source = resolved->source;
                report_reload_error(artifact, resolved->compile_error + "; keeping generation " +
                                                  std::to_string(artifact.program_generation));
                continue;
            }
            auto next_program = resolved->compiled_program;
            if (!next_program)
            {
                auto compiled = flow::compile_asset(resolved->source);
                if (!compiled.succeeded || !compiled.bytecode)
                {
                    artifact.rejected_source = resolved->source;
                    report_reload_error(artifact, compile_error(artifact.display_name, compiled) +
                                                      "; keeping generation " +
                                                      std::to_string(artifact.program_generation));
                    continue;
                }
                next_program = std::make_shared<const flow::bytecode_program>(std::move(*compiled.bytecode));
            }
            const std::uint64_t next_generation = artifact.program_generation + 1;
            std::vector<std::size_t> indices;
            std::vector<bound_flow_instance> replacements;
            indices.reserve(instances_.size());
            replacements.reserve(instances_.size());

            bool replacement_valid = true;
            for (std::size_t index = 0; index < instances_.size(); ++index)
            {
                const auto& instance = instances_[index];
                if (flow_key(instance.graph) != graph_key ||
                    instance.artifact_generation != artifact.program_generation)
                    continue;

                indices.push_back(index);
                replacements.emplace_back(instance.entity, instance.graph, resolved->display_name, next_program,
                                          next_generation);
                if (!replacements.back().vm.valid())
                {
                    replacement_valid = false;
                    break;
                }
            }

            if (!replacement_valid)
            {
                artifact.rejected_source = resolved->source;
                report_reload_error(artifact, "Flow graph '" + artifact.display_name +
                                                  "' hot reload produced an invalid VM; keeping generation " +
                                                  std::to_string(artifact.program_generation));
                continue;
            }

            for (const auto index : indices)
            {
                const auto result = end_instance(instances_[index], &context);
                if (!result.succeeded())
                    throw std::runtime_error(
                        execution_error(result, instances_[index].graph_name, instances_[index].entity));
            }

            artifact.reference = resolved->reference;
            artifact.display_name = resolved->display_name;
            artifact.source = resolved->source;
            artifact.source_generation = resolved->generation;
            artifact.program = next_program;
            artifact.program_generation = next_generation;
            artifact.rejected_source.reset();
            artifact.reload_error.clear();

            for (std::size_t replacement_index = 0; replacement_index < indices.size(); ++replacement_index)
            {
                auto& instance = instances_[indices[replacement_index]];
                instance = std::move(replacements[replacement_index]);
                const auto result = begin_instance(instance, &context);
                if (!result.succeeded())
                    throw std::runtime_error(execution_error(result, instance.graph_name, instance.entity));
            }

            std::string reload_message = "Reloaded Flow graph '" + artifact.display_name + "' as generation ";
            reload_message += std::to_string(artifact.program_generation);
            reload_message += "; restarted ";
            reload_message += std::to_string(indices.size());
            reload_message += " bound instance(s) from defaults";
            ::arc::diagnostics::info("Flow", reload_message);
        }
    }

    [[nodiscard]] shared_program program_for(const assets::asset_reference& graph, std::string& display_name,
                                             std::uint64_t& generation, std::string& error)
    {
        const auto key = flow_key(graph);
        const auto found = programs_.find(key);
        if (found != programs_.end())
        {
            generation = found->second.program_generation;
            display_name = found->second.display_name;
            return found->second.program;
        }

        const auto resolved = source_resolver_(graph, error);
        if (!resolved)
        {
            if (error.empty()) error = "Flow graph could not be resolved: " + key;
            return {};
        }

        display_name = resolved->display_name.empty() ? key : resolved->display_name;
        if (!resolved->compile_error.empty())
        {
            error = resolved->compile_error;
            return {};
        }
        auto program = resolved->compiled_program;
        if (!program)
        {
            auto compiled = flow::compile_asset(resolved->source);
            if (!compiled.succeeded || !compiled.bytecode)
            {
                error = compile_error(display_name, compiled);
                return {};
            }
            program = std::make_shared<const flow::bytecode_program>(std::move(*compiled.bytecode));
        }
        compiled_flow_artifact artifact;
        artifact.reference = resolved->reference;
        artifact.display_name = display_name;
        artifact.source = resolved->source;
        artifact.source_generation = resolved->generation;
        artifact.program = program;
        auto inserted = programs_.emplace(flow_key(resolved->reference), std::move(artifact));
        generation = inserted.first->second.program_generation;
        return inserted.first->second.program;
    }

    [[nodiscard]] std::string append_instance(ecs::entity entity, const assets::asset_reference& graph)
    {
        std::string error;
        std::string display_name;
        std::uint64_t generation{};
        auto program = program_for(graph, display_name, generation, error);
        if (!program) return error;
        instances_.emplace_back(entity, graph, std::move(display_name), std::move(program), generation);
        return {};
    }

    [[nodiscard]] flow::execution_result begin_instance(bound_flow_instance& instance, ecs::system_context* context)
    {
        runtime_world_bridge_context bridge{};
        bridge.world = world_;
        bridge.commands = context ? &context->commands() : nullptr;
        bridge.unrestricted_access = true;
        const auto api = make_runtime_world_api(bridge);
        return instance.vm.begin_play({.api = &api, .self = flow_entity(instance.entity)});
    }

    [[nodiscard]] flow::execution_result end_instance(bound_flow_instance& instance, ecs::system_context* context)
    {
        if (!instance.vm.active()) return {};
        runtime_world_bridge_context bridge{};
        bridge.world = world_;
        bridge.commands = context ? &context->commands() : nullptr;
        bridge.unrestricted_access = true;
        const auto api = make_runtime_world_api(bridge);
        return instance.vm.end_play({.api = &api, .self = flow_entity(instance.entity)});
    }

    [[nodiscard]] std::string stop_instance(std::size_t index, ecs::system_context& context)
    {
        const bool entity_alive = world_->alive(instances_[index].entity);
        const auto result = end_instance(instances_[index], &context);
        if (entity_alive && !result.succeeded())
            return execution_error(result, instances_[index].graph_name, instances_[index].entity);
        instances_.erase(instances_.begin() + static_cast<std::ptrdiff_t>(index));
        return {};
    }

    [[nodiscard]] std::string reconcile_entity(ecs::entity entity, ecs::system_context& context)
    {
        auto found = std::find_if(instances_.begin(), instances_.end(),
                                  [entity](const bound_flow_instance& instance) { return instance.entity == entity; });
        const auto* binding = active_flow_binding(std::as_const(*world_), entity);
        if (found != instances_.end() && binding && found->graph == binding->graph) return {};

        if (found != instances_.end())
        {
            const std::size_t index = static_cast<std::size_t>(std::distance(instances_.begin(), found));
            if (const auto error = stop_instance(index, context); !error.empty()) return error;
        }

        binding = active_flow_binding(std::as_const(*world_), entity);
        if (!binding) return {};
        if (const auto error = append_instance(entity, binding->graph); !error.empty()) return error;

        auto& instance = instances_.back();
        const auto result = begin_instance(instance, &context);
        if (result.succeeded()) return {};
        if (instance.vm.active()) (void)end_instance(instance, &context);
        const std::string error = execution_error(result, instance.graph_name, instance.entity);
        instances_.pop_back();
        return error;
    }

    static void dispatch_input(bound_flow_instance& instance, std::string_view action, bool triggered,
                               flow::vm_world_context world)
    {
        const auto result = triggered ? instance.vm.input_action_triggered(action, 1.0, world)
                                      : instance.vm.input_action_completed(action, 0.0, world);
        if (!result.succeeded())
            throw std::runtime_error(execution_error(result, instance.graph_name, instance.entity));
    }

    ecs::world* world_{};
    flow_play_source_resolver source_resolver_;
    std::filesystem::path input_config_path_;
    input::input_system input_;
    input::input_device_id keyboard_{};
    input::input_device_id mouse_{};
    std::vector<std::string> input_actions_;
    ecs::change_cursor cursor_{};
    std::unordered_map<std::string, compiled_flow_artifact> programs_;
    std::vector<bound_flow_instance> instances_;
};

} // namespace

flow_play_install_result install_flow_play_runtime(framework::runtime_world& world,
                                                   flow_play_source_resolver source_resolver,
                                                   std::filesystem::path input_config_path)
{
    if (!source_resolver) return {.error = "Flow source resolver is unavailable"};
    auto session =
        std::make_shared<flow_play_session>(world.entities(), std::move(source_resolver), std::move(input_config_path));
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

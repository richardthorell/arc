#pragma once

#include <arc/ecs/command_buffer.h>
#include <arc/ecs/simulation.h>
#include <arc/jobs/jobs.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc::ecs
{

enum class component_access_mode : std::uint8_t
{
    read,
    write
};

struct component_access
{
    component_type_id component{};
    component_access_mode mode{ component_access_mode::read };
};

template <class T>
constexpr component_access reads() noexcept
{
    return { component_type<T>(), component_access_mode::read };
}

template <class T>
constexpr component_access writes() noexcept
{
    return { component_type<T>(), component_access_mode::write };
}

enum class system_phase : std::uint8_t
{
    input,
    network_receive,
    gameplay_commands,
    movement,
    physics,
    abilities,
    ai,
    replication,
    presentation_extraction
};

inline constexpr std::array fixed_system_phases{
    system_phase::input,
    system_phase::network_receive,
    system_phase::gameplay_commands,
    system_phase::movement,
    system_phase::physics,
    system_phase::abilities,
    system_phase::ai,
    system_phase::replication
};

struct system_descriptor;

class system_context
{
public:
    world& owner() noexcept { return *owner_; }
    const world& owner() const noexcept { return *owner_; }
    entity_command_buffer& commands() noexcept { return *commands_; }
    float delta_seconds() const noexcept { return delta_seconds_; }
    float fixed_delta_seconds() const noexcept { return execution_.fixed_delta_seconds; }
    float frame_delta_seconds() const noexcept { return execution_.frame_delta_seconds; }
    simulation_tick_id tick_id() const noexcept { return execution_.tick; }
    runtime_world_role world_role() const noexcept { return execution_.world_role; }
    std::uint64_t world_id() const noexcept { return execution_.world_id; }
    float interpolation_alpha() const noexcept { return execution_.interpolation_alpha; }
    bool presentation() const noexcept { return execution_.presentation; }
    const simulation_input_snapshot& input() const noexcept
    {
        static constexpr simulation_input_snapshot empty{};
        return execution_.input ? *execution_.input : empty;
    }
    tick_arena* tick_memory() const noexcept { return execution_.tick_memory; }
    frame_arena* frame_memory() const noexcept { return execution_.frame_memory; }
    runtime_service_provider* services() const noexcept { return execution_.services; }

    template <class Service>
    Service* service(runtime_service_id id) const noexcept
    {
        return execution_.services
            ? static_cast<Service*>(execution_.services->find_service(id))
            : nullptr;
    }

    random_stream random(random_stream_id stream, std::uint64_t stable_subject = 0) const noexcept
    {
        return make_random_stream(
            execution_.process_seed,
            execution_.world_id,
            execution_.tick,
            stream,
            stable_subject);
    }

    template <class T>
    const T* read(entity value) const
    {
        require_access(component_type<T>(), component_access_mode::read);
        return std::as_const(*owner_).template try_get<T>(value);
    }

    template <class T>
    T* write(entity value)
    {
        require_access(component_type<T>(), component_access_mode::write);
        return owner_->template try_get<T>(value);
    }

private:
    system_context(
        world& owner,
        entity_command_buffer& commands,
        const system_execution_info& execution,
        const system_descriptor& descriptor)
        : owner_(&owner)
        , commands_(&commands)
        , descriptor_(&descriptor)
        , execution_(execution)
        , delta_seconds_(execution.delta_seconds)
    {
    }

    void require_access(component_type_id component, component_access_mode requested) const;

    world* owner_{};
    entity_command_buffer* commands_{};
    const system_descriptor* descriptor_{};
    system_execution_info execution_{};
    float delta_seconds_{};
    friend class system_scheduler;
};

struct system_descriptor
{
    std::string name;
    system_phase phase{ system_phase::gameplay_commands };
    job_priority priority{ job_priority::normal };
    job_affinity affinity{ job_affinity::any_worker };
    std::vector<component_access> components;
    std::vector<std::string> before;
    std::vector<std::string> after;
    std::function<void(system_context&)> execute;
};

inline void system_context::require_access(
    component_type_id component,
    component_access_mode requested) const
{
    const auto found = std::find_if(
        descriptor_->components.begin(),
        descriptor_->components.end(),
        [component](const component_access& access) { return access.component == component; });
    const bool accepted = found != descriptor_->components.end() &&
        (requested == component_access_mode::read || found->mode == component_access_mode::write);
    if (!accepted)
        throw std::logic_error(
            "system '" + descriptor_->name + "' used undeclared " +
            (requested == component_access_mode::write ? "write" : "read") + " component access");
}

struct system_schedule_error
{
    std::string system;
    std::string message;
};

struct system_run_result
{
    std::size_t systems_executed{};
    command_flush_result commands;
    std::vector<system_schedule_error> errors;

    bool succeeded() const noexcept { return errors.empty() && commands.succeeded(); }
};

/** Dependency- and access-aware scheduler layered on ARC's work-stealing jobs. */
class system_scheduler
{
public:
    bool add(system_descriptor descriptor)
    {
        if (frozen_ || descriptor.name.empty() || !descriptor.execute ||
            index_.contains(descriptor.name))
            return false;
        index_.emplace(descriptor.name, systems_.size());
        systems_.push_back(std::move(descriptor));
        invalidate_schedules();
        return true;
    }

    bool remove(std::string_view name)
    {
        if (frozen_)
            return false;
        const auto found = index_.find(std::string(name));
        if (found == index_.end())
            return false;
        systems_.erase(systems_.begin() + static_cast<std::ptrdiff_t>(found->second));
        rebuild_index();
        invalidate_schedules();
        return true;
    }

    void clear()
    {
        if (frozen_)
            return;
        systems_.clear();
        index_.clear();
        invalidate_schedules();
    }

    std::size_t size() const noexcept { return systems_.size(); }
    bool frozen() const noexcept { return frozen_; }

    std::vector<system_schedule_error> freeze()
    {
        std::vector<system_schedule_error> errors;
        for (std::size_t phase_index = 0; phase_index < phase_schedules_.size(); ++phase_index)
        {
            compiled_phase& schedule = phase_schedules_[phase_index];
            if (!schedule.compiled)
                compile_phase(static_cast<system_phase>(phase_index), schedule);
            errors.insert(errors.end(), schedule.errors.begin(), schedule.errors.end());
        }
        if (errors.empty())
            frozen_ = true;
        return errors;
    }

    system_run_result run(
        world& owner,
        job_system& jobs,
        float delta_seconds)
    {
        system_execution_info execution{};
        execution.delta_seconds = delta_seconds;
        execution.fixed_delta_seconds = delta_seconds;
        execution.frame_delta_seconds = delta_seconds;
        system_run_result aggregate;
        for (const system_phase phase : fixed_system_phases)
        {
            system_run_result phase_result = run_phase(owner, jobs, phase, execution);
            aggregate.systems_executed += phase_result.systems_executed;
            aggregate.errors.insert(
                aggregate.errors.end(),
                std::make_move_iterator(phase_result.errors.begin()),
                std::make_move_iterator(phase_result.errors.end()));
            aggregate.commands.applied += phase_result.commands.applied;
            aggregate.commands.errors.insert(
                aggregate.commands.errors.end(),
                std::make_move_iterator(phase_result.commands.errors.begin()),
                std::make_move_iterator(phase_result.commands.errors.end()));
            if (!phase_result.succeeded())
                break;
        }
        return aggregate;
    }

    system_run_result run_phase(
        world& owner,
        job_system& jobs,
        system_phase phase,
        const system_execution_info& execution)
    {
        system_run_result result;
        compiled_phase& schedule = phase_schedules_[static_cast<std::size_t>(phase)];
        if (!schedule.compiled)
            compile_phase(phase, schedule);
        if (schedule.selected.empty())
            return result;
        result.errors = schedule.errors;
        if (!result.errors.empty())
            return result;

        for (job_handle& handle : schedule.handles)
            handle = {};
        {
            owner.begin_scheduled_execution();
            struct scheduled_execution_scope
            {
                world* owner{};
                ~scheduled_execution_scope() { owner->end_scheduled_execution(); }
            } scheduled_scope{ &owner };

            try
            {
                for (const std::size_t index : schedule.execution_order)
                {
                    std::vector<job_handle>& prerequisites = schedule.prerequisites[index];
                    prerequisites.clear();
                    for (const std::size_t dependency : schedule.dependencies[index])
                        prerequisites.push_back(schedule.handles[dependency]);

                    system_descriptor& system = systems_[index];
                    schedule.handles[index] = jobs.submit({
                        .name = system.name,
                        .priority = system.priority,
                        .affinity = system.affinity,
                        .dependency_view = prerequisites
                    }, [&owner, &system, buffer = schedule.command_buffers[index].get(), execution]() {
                        system_context context(owner, *buffer, execution, system);
                        system.execute(context);
                    });
                }

                for (const std::size_t index : schedule.selected)
                {
                    const job_wait_result wait = schedule.handles[index].wait_result();
                    if (wait.succeeded())
                    {
                        ++result.systems_executed;
                        continue;
                    }

                    std::string message =
                        wait.status == job_status::cancelled
                        ? "system was cancelled"
                        : "system execution failed";
                    if (wait.exception)
                    {
                        try
                        {
                            std::rethrow_exception(wait.exception);
                        }
                        catch (const std::exception& error)
                        {
                            message += ": ";
                            message += error.what();
                        }
                        catch (...)
                        {
                        }
                    }
                    result.errors.push_back({ systems_[index].name, std::move(message) });
                }
            }
            catch (const std::exception& error)
            {
                for (const job_handle& handle : schedule.handles)
                    if (handle.valid())
                        handle.wait_result();
                result.errors.push_back({ {}, std::string("system scheduling failed: ") + error.what() });
            }
        }

        if (!result.errors.empty())
        {
            for (const std::size_t index : schedule.selected)
                schedule.command_buffers[index]->clear();
            return result;
        }

        result.commands = entity_command_buffer::flush_ordered(owner, schedule.buffer_views);
        return result;
    }

private:
    struct compiled_phase
    {
        bool compiled{};
        std::vector<std::size_t> selected;
        std::vector<std::vector<std::size_t>> dependencies;
        std::vector<std::size_t> execution_order;
        std::vector<std::unique_ptr<entity_command_buffer>> command_buffers;
        std::vector<entity_command_buffer*> buffer_views;
        std::vector<job_handle> handles;
        std::vector<std::vector<job_handle>> prerequisites;
        std::vector<system_schedule_error> errors;
    };

    void compile_phase(system_phase phase, compiled_phase& schedule)
    {
        schedule = {};
        schedule.dependencies.resize(systems_.size());
        schedule.command_buffers.resize(systems_.size());
        schedule.handles.resize(systems_.size());
        schedule.prerequisites.resize(systems_.size());
        for (std::size_t index = 0; index < systems_.size(); ++index)
        {
            if (systems_[index].phase != phase)
                continue;
            schedule.selected.push_back(index);
            schedule.command_buffers[index] = std::make_unique<entity_command_buffer>(
                entity_command_buffer::sort_key{
                    static_cast<std::uint32_t>(phase),
                    static_cast<std::uint32_t>(index),
                    0 });
        }
        build_phase_dependencies(phase, schedule.dependencies, schedule.errors);
        if (schedule.errors.empty() && has_cycle(schedule.dependencies))
            schedule.errors.push_back({ {}, "system dependency graph contains a cycle" });
        if (schedule.errors.empty())
        {
            schedule.execution_order = topological_order(schedule.dependencies);
            schedule.execution_order.erase(
                std::remove_if(
                    schedule.execution_order.begin(),
                    schedule.execution_order.end(),
                    [&](std::size_t index) { return systems_[index].phase != phase; }),
                schedule.execution_order.end());
        }
        for (const std::size_t index : schedule.selected)
        {
            schedule.buffer_views.push_back(schedule.command_buffers[index].get());
            schedule.prerequisites[index].reserve(schedule.dependencies[index].size());
        }
        schedule.compiled = true;
    }

    void invalidate_schedules() noexcept
    {
        for (compiled_phase& schedule : phase_schedules_)
            schedule.compiled = false;
    }

    static bool conflicts(const system_descriptor& lhs, const system_descriptor& rhs)
    {
        for (const component_access& left : lhs.components)
            for (const component_access& right : rhs.components)
                if (left.component == right.component &&
                    (left.mode == component_access_mode::write || right.mode == component_access_mode::write))
                    return true;
        return false;
    }

    void build_dependencies(
        std::vector<std::vector<std::size_t>>& dependencies,
        std::vector<system_schedule_error>& errors) const
    {
        for (std::size_t index = 0; index < systems_.size(); ++index)
        {
            const system_descriptor& system = systems_[index];
            for (const std::string& name : system.after)
            {
                const auto found = index_.find(name);
                if (found == index_.end())
                    errors.push_back({ system.name, "unknown after dependency '" + name + "'" });
                else
                    dependencies[index].push_back(found->second);
            }
            for (const std::string& name : system.before)
            {
                const auto found = index_.find(name);
                if (found == index_.end())
                    errors.push_back({ system.name, "unknown before dependency '" + name + "'" });
                else
                    dependencies[found->second].push_back(index);
            }
        }

        for (std::size_t right = 0; right < systems_.size(); ++right)
        {
            for (std::size_t left = 0; left < systems_.size(); ++left)
            {
                if (left == right)
                    continue;
                if (systems_[left].phase < systems_[right].phase)
                    dependencies[right].push_back(left);
                else if (
                    left < right &&
                    systems_[left].phase == systems_[right].phase &&
                    conflicts(systems_[left], systems_[right]) &&
                    !depends_on(dependencies, left, right))
                    dependencies[right].push_back(left);
            }
            auto& values = dependencies[right];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }

    void build_phase_dependencies(
        system_phase phase,
        std::vector<std::vector<std::size_t>>& dependencies,
        std::vector<system_schedule_error>& errors) const
    {
        for (std::size_t index = 0; index < systems_.size(); ++index)
        {
            const system_descriptor& system = systems_[index];
            if (system.phase != phase)
                continue;
            for (const std::string& name : system.after)
            {
                const auto found = index_.find(name);
                if (found == index_.end())
                    errors.push_back({ system.name, "unknown after dependency '" + name + "'" });
                else if (systems_[found->second].phase == phase)
                    dependencies[index].push_back(found->second);
                else if (systems_[found->second].phase > phase)
                    errors.push_back({
                        system.name,
                        "after dependency '" + name + "' is in a later system phase"
                    });
            }
            for (const std::string& name : system.before)
            {
                const auto found = index_.find(name);
                if (found == index_.end())
                    errors.push_back({ system.name, "unknown before dependency '" + name + "'" });
                else if (systems_[found->second].phase == phase)
                    dependencies[found->second].push_back(index);
                else if (systems_[found->second].phase < phase)
                    errors.push_back({
                        system.name,
                        "before dependency '" + name + "' is in an earlier system phase"
                    });
            }
        }

        for (std::size_t right = 0; right < systems_.size(); ++right)
        {
            if (systems_[right].phase != phase)
                continue;
            for (std::size_t left = 0; left < right; ++left)
            {
                if (systems_[left].phase == phase &&
                    conflicts(systems_[left], systems_[right]) &&
                    !depends_on(dependencies, left, right))
                    dependencies[right].push_back(left);
            }
            auto& values = dependencies[right];
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
        }
    }

    static bool has_cycle(const std::vector<std::vector<std::size_t>>& dependencies)
    {
        enum class visit : std::uint8_t { none, active, complete };
        std::vector<visit> states(dependencies.size());
        const auto inspect = [&](auto&& self, std::size_t index) -> bool {
            if (states[index] == visit::active)
                return true;
            if (states[index] == visit::complete)
                return false;
            states[index] = visit::active;
            for (const std::size_t dependency : dependencies[index])
                if (self(self, dependency))
                    return true;
            states[index] = visit::complete;
            return false;
        };
        for (std::size_t index = 0; index < dependencies.size(); ++index)
            if (inspect(inspect, index))
                return true;
        return false;
    }

    static bool depends_on(
        const std::vector<std::vector<std::size_t>>& dependencies,
        std::size_t system,
        std::size_t candidate,
        std::size_t depth = 0)
    {
        if (system >= dependencies.size() || depth >= dependencies.size())
            return false;
        for (const std::size_t dependency : dependencies[system])
        {
            if (dependency == candidate || depends_on(dependencies, dependency, candidate, depth + 1))
                return true;
        }
        return false;
    }

    static std::vector<std::size_t> topological_order(
        const std::vector<std::vector<std::size_t>>& dependencies)
    {
        std::vector<std::size_t> indegree(dependencies.size());
        std::vector<std::vector<std::size_t>> dependents(dependencies.size());
        for (std::size_t index = 0; index < dependencies.size(); ++index)
        {
            indegree[index] = dependencies[index].size();
            for (const std::size_t dependency : dependencies[index])
                dependents[dependency].push_back(index);
        }
        std::vector<std::size_t> ready;
        for (std::size_t index = 0; index < indegree.size(); ++index)
            if (indegree[index] == 0)
                ready.push_back(index);
        std::vector<std::size_t> result;
        while (!ready.empty())
        {
            const std::size_t current = ready.front();
            ready.erase(ready.begin());
            result.push_back(current);
            for (const std::size_t dependent : dependents[current])
            {
                if (--indegree[dependent] == 0)
                {
                    const auto insertion = std::lower_bound(ready.begin(), ready.end(), dependent);
                    ready.insert(insertion, dependent);
                }
            }
        }
        return result;
    }

    void rebuild_index()
    {
        index_.clear();
        for (std::size_t index = 0; index < systems_.size(); ++index)
            index_.emplace(systems_[index].name, index);
    }

    std::vector<system_descriptor> systems_;
    std::unordered_map<std::string, std::size_t> index_;
    std::array<compiled_phase, static_cast<std::size_t>(system_phase::presentation_extraction) + 1>
        phase_schedules_;
    bool frozen_{};
};

} // namespace arc::ecs

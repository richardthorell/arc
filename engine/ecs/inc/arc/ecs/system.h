#pragma once

#include <arc/ecs/command_buffer.h>
#include <arc/jobs/jobs.h>

#include <algorithm>
#include <cstdint>
#include <functional>
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
    pre_simulation,
    simulation,
    post_simulation,
    transforms,
    extraction,
    presentation
};

struct system_descriptor;

class system_context
{
public:
    world& owner() noexcept { return *owner_; }
    const world& owner() const noexcept { return *owner_; }
    entity_command_buffer& commands() noexcept { return *commands_; }
    float delta_seconds() const noexcept { return delta_seconds_; }

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
        float delta_seconds,
        const system_descriptor& descriptor)
        : owner_(&owner), commands_(&commands), descriptor_(&descriptor), delta_seconds_(delta_seconds)
    {
    }

    void require_access(component_type_id component, component_access_mode requested) const;

    world* owner_{};
    entity_command_buffer* commands_{};
    const system_descriptor* descriptor_{};
    float delta_seconds_{};
    friend class system_scheduler;
};

struct system_descriptor
{
    std::string name;
    system_phase phase{ system_phase::simulation };
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
        if (descriptor.name.empty() || !descriptor.execute || index_.contains(descriptor.name))
            return false;
        index_.emplace(descriptor.name, systems_.size());
        systems_.push_back(std::move(descriptor));
        return true;
    }

    bool remove(std::string_view name)
    {
        const auto found = index_.find(std::string(name));
        if (found == index_.end())
            return false;
        systems_.erase(systems_.begin() + static_cast<std::ptrdiff_t>(found->second));
        rebuild_index();
        return true;
    }

    void clear()
    {
        systems_.clear();
        index_.clear();
    }

    std::size_t size() const noexcept { return systems_.size(); }

    system_run_result run(world& owner, job_system& jobs, float delta_seconds)
    {
        system_run_result result;
        std::vector<std::vector<std::size_t>> dependencies(systems_.size());
        build_dependencies(dependencies, result.errors);
        if (!result.errors.empty() || has_cycle(dependencies))
        {
            if (result.errors.empty())
                result.errors.push_back({ {}, "system dependency graph contains a cycle" });
            return result;
        }
        const std::vector<std::size_t> execution_order = topological_order(dependencies);

        std::vector<std::unique_ptr<entity_command_buffer>> command_buffers;
        command_buffers.resize(systems_.size());
        for (std::size_t index = 0; index < systems_.size(); ++index)
        {
            command_buffers[index] = std::make_unique<entity_command_buffer>(
                entity_command_buffer::sort_key{
                    static_cast<std::uint32_t>(systems_[index].phase),
                    static_cast<std::uint32_t>(index),
                    0 });
        }

        owner.begin_scheduled_execution();
        std::vector<job_handle> handles(systems_.size());
        for (const std::size_t index : execution_order)
        {
            std::vector<job_handle> prerequisites;
            prerequisites.reserve(dependencies[index].size());
            for (const std::size_t dependency : dependencies[index])
                prerequisites.push_back(handles[dependency]);

            system_descriptor& system = systems_[index];
            handles[index] = jobs.submit({
                .name = system.name,
                .priority = system.priority,
                .affinity = system.affinity,
                .dependencies = std::move(prerequisites)
            }, [&owner, &system, buffer = command_buffers[index].get(), delta_seconds]() {
                system_context context(owner, *buffer, delta_seconds, system);
                system.execute(context);
            });
        }

        for (std::size_t index = 0; index < handles.size(); ++index)
        {
            const job_wait_result wait = handles[index].wait_result();
            if (wait.succeeded())
                ++result.systems_executed;
            else
                result.errors.push_back({
                    systems_[index].name,
                    wait.status == job_status::cancelled ? "system was cancelled" : "system execution failed"
                });
        }
        owner.end_scheduled_execution();

        std::vector<entity_command_buffer*> buffers;
        buffers.reserve(command_buffers.size());
        for (auto& buffer : command_buffers)
            buffers.push_back(buffer.get());
        result.commands = entity_command_buffer::flush_ordered(owner, buffers);
        return result;
    }

private:
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
};

} // namespace arc::ecs

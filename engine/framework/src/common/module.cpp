#include <arc/module.h>

#include <arc/application.h>

#include <stdexcept>
#include <unordered_map>

namespace arc
{

module_context::module_context(job_system& jobs, logger& diagnostics, tracked_memory_resource& memory) noexcept
    : jobs_(&jobs)
    , diagnostics_(&diagnostics)
    , memory_(&memory)
{
}

job_system& module_context::jobs() const noexcept
{
    return *jobs_;
}

logger& module_context::diagnostics() const noexcept
{
    return *diagnostics_;
}

tracked_memory_resource& module_context::memory() const noexcept
{
    return *memory_;
}

module::~module() = default;

std::vector<std::string> module::dependencies() const
{
    return {};
}

void module::on_start(module_context&)
{
}

void module::on_update(module_context&, const frame_time&)
{
}

void module::on_event(module_context&, const event&)
{
}

void module::on_shutdown(module_context&)
{
}

void module_registry::add(std::unique_ptr<module> value)
{
    if (!value)
        throw std::invalid_argument("module_registry cannot add a null module");
    if (value->name().empty())
        throw std::invalid_argument("module names must not be empty");

    modules_.push_back(std::move(value));
}

std::size_t module_registry::size() const noexcept
{
    return modules_.size();
}

bool module_registry::empty() const noexcept
{
    return modules_.empty();
}

const std::vector<std::unique_ptr<module>>& module_registry::modules() const noexcept
{
    return modules_;
}

module_registry& module_manager::registry() noexcept
{
    return registry_;
}

const module_registry& module_manager::registry() const noexcept
{
    return registry_;
}

void module_manager::start(module_context& context)
{
    if (started_)
        return;

    resolve_order();
    for (const auto index : order_)
        registry_.modules()[index]->on_start(context);
    started_ = true;
}

void module_manager::update(module_context& context, const frame_time& time)
{
    if (!started_)
        return;

    for (const auto index : order_)
        registry_.modules()[index]->on_update(context, time);
}

void module_manager::dispatch(module_context& context, const event& value)
{
    if (!started_)
        return;

    for (const auto index : order_)
        registry_.modules()[index]->on_event(context, value);
}

void module_manager::shutdown(module_context& context)
{
    if (!started_)
        return;

    for (auto iterator = order_.rbegin(); iterator != order_.rend(); ++iterator)
        registry_.modules()[*iterator]->on_shutdown(context);
    started_ = false;
}

bool module_manager::started() const noexcept
{
    return started_;
}

std::vector<std::string_view> module_manager::start_order() const
{
    std::vector<std::string_view> result;
    result.reserve(order_.size());
    for (const auto index : order_)
        result.push_back(registry_.modules()[index]->name());
    return result;
}

void module_manager::resolve_order()
{
    if (ordered_)
        return;

    const auto& modules = registry_.modules();
    std::unordered_map<std::string, std::size_t> names;
    names.reserve(modules.size());

    for (std::size_t index = 0; index < modules.size(); ++index)
    {
        const std::string name(modules[index]->name());
        if (!names.emplace(name, index).second)
            throw std::invalid_argument("duplicate module name: " + name);
    }

    order_.clear();
    order_.reserve(modules.size());
    std::vector<int> state(modules.size(), 0);

    auto visit = [&](auto& self, std::size_t index) -> void {
        if (state[index] == 2)
            return;
        if (state[index] == 1)
            throw std::invalid_argument("module dependency cycle detected");

        state[index] = 1;
        for (const auto& dependency : modules[index]->dependencies())
        {
            const auto found = names.find(dependency);
            if (found == names.end())
                throw std::invalid_argument("unknown module dependency: " + dependency);
            self(self, found->second);
        }
        state[index] = 2;
        order_.push_back(index);
    };

    for (std::size_t index = 0; index < modules.size(); ++index)
        visit(visit, index);

    ordered_ = true;
}

} // namespace arc

#include <arc/framework/service.h>

#include <algorithm>
#include <functional>
#include <stdexcept>

namespace arc
{

runtime_service_registry::~runtime_service_registry()
{
    shutdown();
}

bool runtime_service_registry::add(std::unique_ptr<runtime_service> service)
{
    if (!service || started_ || !service->id().valid() || service->name().empty() ||
        index_.contains(service->id().value))
        return false;
    index_.emplace(service->id().value, services_.size());
    services_.push_back(std::move(service));
    order_.clear();
    return true;
}

void runtime_service_registry::resolve_order()
{
    order_.clear();
    enum class visit : std::uint8_t { none, active, complete };
    std::vector<visit> states(services_.size());

    const auto inspect = [&](auto&& self, std::size_t index) -> void {
        if (states[index] == visit::active)
            throw std::invalid_argument("runtime service dependency graph contains a cycle");
        if (states[index] == visit::complete)
            return;

        states[index] = visit::active;
        for (const runtime_service_id dependency : services_[index]->dependencies())
        {
            const auto found = index_.find(dependency.value);
            if (found == index_.end())
                throw std::invalid_argument(
                    "runtime service '" + std::string(services_[index]->name()) +
                    "' depends on an unknown service");
            self(self, found->second);
        }
        states[index] = visit::complete;
        order_.push_back(index);
    };

    for (std::size_t index = 0; index < services_.size(); ++index)
        inspect(inspect, index);
}

void runtime_service_registry::start()
{
    if (started_)
        return;
    resolve_order();
    runtime_service_context context(*this);
    started_count_ = 0;
    try
    {
        for (const std::size_t index : order_)
        {
            services_[index]->on_start(context);
            ++started_count_;
        }
        started_ = true;
    }
    catch (...)
    {
        while (started_count_ > 0)
        {
            --started_count_;
            services_[order_[started_count_]]->on_shutdown(context);
        }
        throw;
    }
}

void runtime_service_registry::shutdown() noexcept
{
    if (!started_ && started_count_ == 0)
        return;

    runtime_service_context context(*this);
    while (started_count_ > 0)
    {
        --started_count_;
        services_[order_[started_count_]]->on_shutdown(context);
    }
    started_ = false;
}

void* runtime_service_registry::find_service(runtime_service_id id) noexcept
{
    const auto found = index_.find(id.value);
    return found == index_.end() ? nullptr : services_[found->second].get();
}

const void* runtime_service_registry::find_service(runtime_service_id id) const noexcept
{
    const auto found = index_.find(id.value);
    return found == index_.end() ? nullptr : services_[found->second].get();
}

bool runtime_service_registry::capture_deterministic_state(
    std::uint64_t world,
    std::vector<runtime_service_snapshot>& snapshots,
    std::string& error) const
{
    snapshots.clear();
    for (const std::size_t index : order_)
    {
        const runtime_service& service = *services_[index];
        if (!service.has_deterministic_state())
            continue;
        runtime_service_snapshot snapshot{
            .service = service.id(),
            .version = service.snapshot_version()
        };
        if (!service.capture_deterministic_state(world, snapshot.bytes, error))
        {
            if (error.empty())
                error = "runtime service '" + std::string(service.name()) + "' rejected snapshot capture";
            snapshots.clear();
            return false;
        }
        snapshots.push_back(std::move(snapshot));
    }
    return true;
}

bool runtime_service_registry::validate_deterministic_state(
    std::uint64_t world,
    const std::vector<runtime_service_snapshot>& snapshots,
    std::string& error) const
{
    for (const std::size_t index : order_)
    {
        const runtime_service& service = *services_[index];
        if (!service.has_deterministic_state())
            continue;
        const auto captured = std::find_if(
            snapshots.begin(),
            snapshots.end(),
            [&service](const runtime_service_snapshot& snapshot) {
                return snapshot.service == service.id();
            });
        if (captured == snapshots.end())
        {
            error = "snapshot is missing deterministic state for runtime service '" +
                std::string(service.name()) + "'";
            return false;
        }
    }

    for (const runtime_service_snapshot& snapshot : snapshots)
    {
        const auto found = index_.find(snapshot.service.value);
        if (found == index_.end())
        {
            error = "snapshot references an unavailable deterministic runtime service";
            return false;
        }
        const runtime_service& service = *services_[found->second];
        if (!service.has_deterministic_state() ||
            !service.validate_deterministic_state(
                world, snapshot.version, snapshot.bytes, error))
        {
            if (error.empty())
                error = "runtime service '" + std::string(service.name()) + "' rejected snapshot restore";
            return false;
        }
    }
    return true;
}

void runtime_service_registry::restore_deterministic_state(
    std::uint64_t world,
    const std::vector<runtime_service_snapshot>& snapshots) noexcept
{
    for (const runtime_service_snapshot& snapshot : snapshots)
    {
        const auto found = index_.find(snapshot.service.value);
        if (found != index_.end())
            services_[found->second]->restore_deterministic_state(
                world, snapshot.version, snapshot.bytes);
    }
}

} // namespace arc

#pragma once

#include <arc/ecs/simulation.h>

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arc
{

using runtime_service_id = ecs::runtime_service_id;
using runtime_service_provider = ecs::runtime_service_provider;

constexpr runtime_service_id make_runtime_service_id(const char* name) noexcept
{
    return { ecs::stable_hash_64(name) };
}

struct runtime_service_snapshot
{
    runtime_service_id service{};
    std::uint32_t version{ 1 };
    std::vector<std::byte> bytes;
};

class runtime_service_context;

class runtime_service
{
public:
    virtual ~runtime_service() = default;
    virtual runtime_service_id id() const noexcept = 0;
    virtual std::string_view name() const noexcept = 0;
    virtual std::vector<runtime_service_id> dependencies() const { return {}; }
    virtual bool has_deterministic_state() const noexcept { return false; }
    virtual std::uint32_t snapshot_version() const noexcept { return 1; }
    virtual bool capture_deterministic_state(
        std::uint64_t,
        std::vector<std::byte>&,
        std::string& error) const
    {
        if (has_deterministic_state())
        {
            error = "runtime service does not implement deterministic snapshot capture";
            return false;
        }
        return true;
    }
    virtual bool validate_deterministic_state(
        std::uint64_t,
        std::uint32_t,
        const std::vector<std::byte>&,
        std::string& error) const
    {
        if (has_deterministic_state())
        {
            error = "runtime service does not implement deterministic snapshot validation";
            return false;
        }
        return true;
    }
    virtual void restore_deterministic_state(
        std::uint64_t,
        std::uint32_t,
        const std::vector<std::byte>&) noexcept
    {
    }
    virtual void on_start(runtime_service_context&) {}
    virtual void on_shutdown(runtime_service_context&) noexcept {}
};

class runtime_service_registry;

class runtime_service_context
{
public:
    explicit runtime_service_context(runtime_service_registry& services) noexcept
        : services_(&services)
    {
    }

    runtime_service_registry& services() const noexcept { return *services_; }

private:
    runtime_service_registry* services_{};
};

class runtime_service_registry final : public ecs::runtime_service_provider
{
public:
    runtime_service_registry() = default;
    ~runtime_service_registry();

    runtime_service_registry(const runtime_service_registry&) = delete;
    runtime_service_registry& operator=(const runtime_service_registry&) = delete;

    bool add(std::unique_ptr<runtime_service> service);

    template <class Service, class... Args>
    Service& emplace(Args&&... args)
    {
        auto service = std::make_unique<Service>(std::forward<Args>(args)...);
        Service& result = *service;
        if (!add(std::move(service)))
            throw std::invalid_argument("duplicate or invalid runtime service");
        return result;
    }

    void start();
    void shutdown() noexcept;
    bool started() const noexcept { return started_; }
    std::size_t size() const noexcept { return services_.size(); }

    void* find_service(runtime_service_id id) noexcept override;
    const void* find_service(runtime_service_id id) const noexcept override;
    bool capture_deterministic_state(
        std::uint64_t world,
        std::vector<runtime_service_snapshot>& snapshots,
        std::string& error) const;
    bool validate_deterministic_state(
        std::uint64_t world,
        const std::vector<runtime_service_snapshot>& snapshots,
        std::string& error) const;
    void restore_deterministic_state(
        std::uint64_t world,
        const std::vector<runtime_service_snapshot>& snapshots) noexcept;

    template <class Service>
    Service* find(runtime_service_id id) noexcept
    {
        return static_cast<Service*>(find_service(id));
    }

    template <class Service>
    const Service* find(runtime_service_id id) const noexcept
    {
        return static_cast<const Service*>(find_service(id));
    }

private:
    void resolve_order();

    std::vector<std::unique_ptr<runtime_service>> services_;
    std::unordered_map<std::uint64_t, std::size_t> index_;
    std::vector<std::size_t> order_;
    std::size_t started_count_{};
    bool started_{};
};

} // namespace arc

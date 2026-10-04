#pragma once

#include <arc/physics/physics.h>

#include <memory>
#include <string_view>

namespace arc::physics
{

struct backend_config
{
    std::uint32_t max_bodies = 65536;
    std::uint32_t max_body_pairs = 65536;
    std::uint32_t max_contact_constraints = 10240;
};

class backend
{
public:
    virtual ~backend() = default;

    backend(const backend&) = delete;
    backend& operator=(const backend&) = delete;
    backend(backend&&) = delete;
    backend& operator=(backend&&) = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual bool initialized() const noexcept = 0;

protected:
    backend() = default;
};

using backend_ptr = std::unique_ptr<backend>;
using backend_factory = backend_ptr (*)(const backend_config& config);

} // namespace arc::physics

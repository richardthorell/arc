#pragma once

#include <arc/input/input.h>

#include <cstdint>

namespace arc::platform::windows
{

/**
 * @brief Internal Windows controller backend contract.
 *
 * Only the provider selected by windows_controller_manager is polled. This
 * prevents one physical controller from being surfaced through multiple Windows
 * input APIs at the same time.
 */
class windows_controller_provider
{
public:
    virtual ~windows_controller_provider() = default;

    [[nodiscard]] virtual input::input_backend_type backend() const noexcept = 0;
    [[nodiscard]] virtual std::uint32_t priority() const noexcept = 0;
    [[nodiscard]] virtual bool available() const noexcept = 0;
    virtual void poll() = 0;
};

} // namespace arc::platform::windows

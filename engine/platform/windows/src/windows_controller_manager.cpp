#include "windows_controller_manager.h"

#include "windows_controller_provider.h"
#include "windows_gamepad_backend.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace arc::platform::windows
{
namespace
{

std::vector<std::unique_ptr<windows_controller_provider>> default_providers(input::input_system& input)
{
    std::vector<std::unique_ptr<windows_controller_provider>> providers;
    providers.push_back(std::make_unique<windows_gamepad_backend>(input));
    return providers;
}

} // namespace

windows_controller_manager::windows_controller_manager(input::input_system& input)
    : windows_controller_manager(default_providers(input))
{
}

windows_controller_manager::windows_controller_manager(
    std::vector<std::unique_ptr<windows_controller_provider>> providers) noexcept
    : providers_(std::move(providers))
{
    select_provider();
}

windows_controller_manager::~windows_controller_manager() = default;

void windows_controller_manager::poll()
{
    if (active_) active_->poll();
}

input::input_backend_type windows_controller_manager::active_backend() const noexcept
{
    return active_ ? active_->backend() : input::input_backend_type::unknown;
}

bool windows_controller_manager::available() const noexcept
{
    return active_ != nullptr;
}

void windows_controller_manager::select_provider() noexcept
{
    const auto best =
        std::max_element(providers_.begin(), providers_.end(),
                         [](const auto& lhs, const auto& rhs)
                         {
                             const std::uint32_t lhs_priority = lhs && lhs->available() ? lhs->priority() : 0;
                             const std::uint32_t rhs_priority = rhs && rhs->available() ? rhs->priority() : 0;
                             return lhs_priority < rhs_priority;
                         });

    if (best == providers_.end() || !*best || !(*best)->available())
    {
        active_ = nullptr;
        return;
    }

    active_ = best->get();
}

} // namespace arc::platform::windows

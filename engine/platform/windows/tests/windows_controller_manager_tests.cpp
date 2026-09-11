#include "windows_controller_manager.h"
#include "windows_controller_provider.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace
{

class fake_provider final : public arc::platform::windows::windows_controller_provider
{
public:
    fake_provider(arc::input::input_backend_type backend, std::uint32_t priority, bool available) noexcept
        : backend_(backend), priority_(priority), available_(available)
    {
    }

    [[nodiscard]] arc::input::input_backend_type backend() const noexcept override
    {
        return backend_;
    }

    [[nodiscard]] std::uint32_t priority() const noexcept override
    {
        return priority_;
    }

    [[nodiscard]] bool available() const noexcept override
    {
        return available_;
    }

    void poll() override
    {
        ++poll_count_;
    }

    [[nodiscard]] int poll_count() const noexcept
    {
        return poll_count_;
    }

private:
    arc::input::input_backend_type backend_{arc::input::input_backend_type::unknown};
    std::uint32_t priority_{};
    bool available_{};
    int poll_count_{};
};

} // namespace

int main()
{
    using arc::input::input_backend_type;
    using arc::platform::windows::windows_controller_manager;
    using arc::platform::windows::windows_controller_provider;

    auto fallback = std::make_unique<fake_provider>(input_backend_type::xinput, 100, true);
    fake_provider* fallback_ptr = fallback.get();
    auto preferred = std::make_unique<fake_provider>(input_backend_type::game_input, 200, true);
    fake_provider* preferred_ptr = preferred.get();

    std::vector<std::unique_ptr<windows_controller_provider>> providers;
    providers.push_back(std::move(fallback));
    providers.push_back(std::move(preferred));

    windows_controller_manager manager(std::move(providers));
    assert(manager.available());
    assert(manager.active_backend() == input_backend_type::game_input);
    manager.poll();
    assert(preferred_ptr->poll_count() == 1);
    assert(fallback_ptr->poll_count() == 0);

    std::vector<std::unique_ptr<windows_controller_provider>> fallback_providers;
    fallback_providers.push_back(std::make_unique<fake_provider>(input_backend_type::game_input, 200, false));
    fallback_providers.push_back(std::make_unique<fake_provider>(input_backend_type::xinput, 100, true));
    windows_controller_manager fallback_manager(std::move(fallback_providers));
    assert(fallback_manager.available());
    assert(fallback_manager.active_backend() == input_backend_type::xinput);

    std::vector<std::unique_ptr<windows_controller_provider>> unavailable_providers;
    unavailable_providers.push_back(std::make_unique<fake_provider>(input_backend_type::game_input, 200, false));
    unavailable_providers.push_back(std::make_unique<fake_provider>(input_backend_type::xinput, 100, false));
    windows_controller_manager unavailable_manager(std::move(unavailable_providers));
    assert(!unavailable_manager.available());
    assert(unavailable_manager.active_backend() == input_backend_type::unknown);

    return 0;
}

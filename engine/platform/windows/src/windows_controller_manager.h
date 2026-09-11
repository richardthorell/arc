#pragma once

#include <arc/input/input.h>

#include <memory>
#include <vector>

namespace arc::platform::windows
{

class windows_controller_provider;

/**
 * @brief Selects one Windows controller provider for the process.
 *
 * Providers are ranked by priority. Only the highest-priority available backend
 * is polled, avoiding duplicate logical controllers when several Windows APIs
 * expose the same physical device.
 */
class windows_controller_manager final
{
public:
    explicit windows_controller_manager(input::input_system& input);
    explicit windows_controller_manager(std::vector<std::unique_ptr<windows_controller_provider>> providers) noexcept;
    ~windows_controller_manager();

    windows_controller_manager(const windows_controller_manager&) = delete;
    windows_controller_manager& operator=(const windows_controller_manager&) = delete;

    void poll();

    [[nodiscard]] input::input_backend_type active_backend() const noexcept;
    [[nodiscard]] bool available() const noexcept;

private:
    void select_provider() noexcept;

    std::vector<std::unique_ptr<windows_controller_provider>> providers_;
    windows_controller_provider* active_{};
};

} // namespace arc::platform::windows

#pragma once

#include <arc/input/input.h>

#include <windows.h>

#include <memory>
#include <vector>

namespace arc::platform::windows
{

class windows_controller_extension_host;
class windows_controller_provider;
class windows_hid_extension_manager;

/**
 * @brief Selects one Windows controller provider for the process.
 *
 * Providers are ranked by priority. Only the highest-priority available backend
 * is polled, avoiding duplicate logical controllers when several Windows APIs
 * expose the same physical device. Device extensions are polled afterward and
 * may augment those logical devices without registering additional controllers.
 */
class windows_controller_manager final
{
public:
    explicit windows_controller_manager(input::input_system& input);
    explicit windows_controller_manager(std::vector<std::unique_ptr<windows_controller_provider>> providers) noexcept;
    ~windows_controller_manager();

    windows_controller_manager(const windows_controller_manager&) = delete;
    windows_controller_manager& operator=(const windows_controller_manager&) = delete;

    [[nodiscard]] bool attach(HWND window);
    void process_message(UINT message, WPARAM wparam, LPARAM lparam);
    void poll();

    [[nodiscard]] input::input_backend_type active_backend() const noexcept;
    [[nodiscard]] bool available() const noexcept;

private:
    void select_provider() noexcept;

    std::vector<std::unique_ptr<windows_controller_provider>> providers_;
    windows_controller_provider* active_{};
    std::unique_ptr<windows_controller_extension_host> extension_host_;
    std::unique_ptr<windows_hid_extension_manager> hid_extensions_;
};

} // namespace arc::platform::windows

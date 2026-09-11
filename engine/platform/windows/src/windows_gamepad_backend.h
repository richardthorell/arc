#pragma once

#include <arc/input/input.h>

#include <windows.h>
#include <Xinput.h>

#include <array>
#include <cstdint>

namespace arc::platform::windows
{

/**
 * @brief Windows XInput gamepad adapter feeding normalized state and rumble through ARC input.
 *
 * XInput is loaded dynamically so the platform target does not acquire an
 * additional link-time SDK dependency. Device slots are kept stable for the
 * process lifetime, preserving ARC player assignments across reconnects.
 */
class windows_gamepad_backend final : public input::input_output_sink
{
public:
    explicit windows_gamepad_backend(input::input_system& input) noexcept;
    ~windows_gamepad_backend() override;

    windows_gamepad_backend(const windows_gamepad_backend&) = delete;
    windows_gamepad_backend& operator=(const windows_gamepad_backend&) = delete;

    [[nodiscard]] bool available() const noexcept;
    void poll();

    bool set_rumble(input::input_device_id device, input::input_rumble_state state) override;

private:
    using get_state_fn = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
    using set_state_fn = DWORD(WINAPI*)(DWORD, XINPUT_VIBRATION*);

    [[nodiscard]] static HMODULE load_xinput() noexcept;
    [[nodiscard]] static input::input_device_id stable_device_id(DWORD user_index) noexcept;
    [[nodiscard]] DWORD user_index(input::input_device_id device) const noexcept;
    void connect(DWORD user_index);
    void disconnect(DWORD user_index);
    void submit_state(DWORD user_index, const XINPUT_GAMEPAD& state);

    input::input_system* input_{};
    HMODULE module_{};
    get_state_fn get_state_{};
    set_state_fn set_state_{};
    std::array<input::input_device_id, XUSER_MAX_COUNT> devices_{};
};

} // namespace arc::platform::windows

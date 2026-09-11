#pragma once

#include "windows_controller_provider.h"

#include <arc/input/input.h>

#include <cstdint>
#include <memory>

namespace arc::platform::windows
{

/**
 * @brief Preferred Windows GameInput controller provider.
 *
 * The implementation is compiled when the Microsoft GameInput package is
 * available. Otherwise the provider remains unavailable and the controller
 * manager falls back to XInput without changing gameplay-facing APIs.
 */
class windows_game_input_backend final : public windows_controller_provider, public input::input_output_sink
{
public:
    explicit windows_game_input_backend(input::input_system& input);
    ~windows_game_input_backend() override;

    windows_game_input_backend(const windows_game_input_backend&) = delete;
    windows_game_input_backend& operator=(const windows_game_input_backend&) = delete;

    [[nodiscard]] input::input_backend_type backend() const noexcept override
    {
        return input::input_backend_type::game_input;
    }

    [[nodiscard]] std::uint32_t priority() const noexcept override
    {
        return 200;
    }

    [[nodiscard]] bool available() const noexcept override;
    void poll() override;

    bool set_rumble(input::input_device_id device, input::input_rumble_state state) override;

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace arc::platform::windows

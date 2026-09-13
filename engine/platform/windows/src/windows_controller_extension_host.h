#pragma once

#include <arc/input/input.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace arc::platform::windows
{

struct windows_controller_extension_id
{
    std::uint64_t value{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return value != 0;
    }

    friend bool operator==(const windows_controller_extension_id&, const windows_controller_extension_id&) = default;
};

struct windows_controller_extension_descriptor
{
    input::input_backend_type backend{input::input_backend_type::unknown};
    std::string backend_id;
    input::input_device_capabilities capabilities{};
};

/**
 * @brief Applies platform extensions to existing logical controller devices.
 *
 * Extensions never create player-visible devices. Their capabilities are merged
 * onto the owning controller and removed again when the extension detaches.
 */
class windows_controller_extension_host final
{
public:
    explicit windows_controller_extension_host(input::input_system& input) noexcept;

    [[nodiscard]] windows_controller_extension_id attach(input::input_device_id device,
                                                         windows_controller_extension_descriptor descriptor);
    bool update(input::input_device_id device, windows_controller_extension_id extension,
                windows_controller_extension_descriptor descriptor);
    bool detach(input::input_device_id device, windows_controller_extension_id extension);
    void detach_all(input::input_device_id device);
    void refresh(input::input_device_id device);

    [[nodiscard]] std::size_t extension_count(input::input_device_id device) const noexcept;

private:
    struct extension_record
    {
        windows_controller_extension_descriptor descriptor;
    };

    struct device_state
    {
        input::input_device_descriptor base_descriptor;
        std::unordered_map<std::uint64_t, extension_record> extensions;
    };

    [[nodiscard]] static input::input_device_descriptor snapshot(const input::input_device& device);
    [[nodiscard]] static input::input_device_capabilities
    merge_capabilities(input::input_device_capabilities base,
                       const input::input_device_capabilities& extension) noexcept;
    [[nodiscard]] input::input_device_capabilities extension_capabilities(const device_state& state) const noexcept;
    void sync_base_descriptor(input::input_device_id device, device_state& state);
    void apply(input::input_device_id device, device_state& state);

    input::input_system* input_{};
    std::unordered_map<std::uint64_t, device_state> devices_;
    std::uint64_t next_extension_id_{1};
};

} // namespace arc::platform::windows

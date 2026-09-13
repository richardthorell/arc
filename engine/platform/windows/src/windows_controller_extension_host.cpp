#include "windows_controller_extension_host.h"

#include <algorithm>
#include <utility>

namespace arc::platform::windows
{

windows_controller_extension_host::windows_controller_extension_host(input::input_system& input) noexcept
    : input_(&input)
{
}

windows_controller_extension_id
windows_controller_extension_host::attach(input::input_device_id device,
                                          windows_controller_extension_descriptor descriptor)
{
    const input::input_device* current = input_->device(device);
    if (!current || !current->connected()) return {};

    auto [state_it, inserted] = devices_.try_emplace(device.value);
    device_state& state = state_it->second;
    if (inserted)
        state.base_descriptor = snapshot(*current);
    else
        sync_base_descriptor(device, state);

    while (next_extension_id_ == 0)
        ++next_extension_id_;
    const windows_controller_extension_id id{.value = next_extension_id_++};
    state.extensions.emplace(id.value, extension_record{.descriptor = std::move(descriptor)});
    apply(device, state);
    return id;
}

bool windows_controller_extension_host::update(input::input_device_id device, windows_controller_extension_id extension,
                                               windows_controller_extension_descriptor descriptor)
{
    const auto state_it = devices_.find(device.value);
    if (state_it == devices_.end()) return false;

    auto extension_it = state_it->second.extensions.find(extension.value);
    if (extension_it == state_it->second.extensions.end()) return false;

    sync_base_descriptor(device, state_it->second);
    extension_it->second.descriptor = std::move(descriptor);
    if (const input::input_device* current = input_->device(device); current && current->connected())
        apply(device, state_it->second);
    return true;
}

bool windows_controller_extension_host::detach(input::input_device_id device, windows_controller_extension_id extension)
{
    const auto state_it = devices_.find(device.value);
    if (state_it == devices_.end()) return false;

    device_state& state = state_it->second;
    if (!state.extensions.contains(extension.value)) return false;

    sync_base_descriptor(device, state);
    state.extensions.erase(extension.value);

    const input::input_device* current = input_->device(device);
    if (current && current->connected())
    {
        if (state.extensions.empty())
            input_->connect_device(state.base_descriptor);
        else
            apply(device, state);
    }

    if (state.extensions.empty()) devices_.erase(state_it);
    return true;
}

void windows_controller_extension_host::detach_all(input::input_device_id device)
{
    const auto found = devices_.find(device.value);
    if (found == devices_.end()) return;

    sync_base_descriptor(device, found->second);
    const input::input_device* current = input_->device(device);
    if (current && current->connected()) input_->connect_device(found->second.base_descriptor);
    devices_.erase(found);
}

void windows_controller_extension_host::refresh(input::input_device_id device)
{
    const auto found = devices_.find(device.value);
    if (found == devices_.end()) return;

    const input::input_device* current = input_->device(device);
    if (!current || !current->connected()) return;

    sync_base_descriptor(device, found->second);
    apply(device, found->second);
}

std::size_t windows_controller_extension_host::extension_count(input::input_device_id device) const noexcept
{
    const auto found = devices_.find(device.value);
    return found == devices_.end() ? 0 : found->second.extensions.size();
}

input::input_device_descriptor windows_controller_extension_host::snapshot(const input::input_device& device)
{
    return {.id = device.id(),
            .type = device.type(),
            .subtype = device.subtype(),
            .connectivity = device.connectivity(),
            .backend = device.backend(),
            .hardware_id = device.hardware_id(),
            .backend_id = std::string(device.backend_id()),
            .name = std::string(device.name()),
            .capabilities = device.capabilities()};
}

input::input_device_capabilities
windows_controller_extension_host::merge_capabilities(input::input_device_capabilities base,
                                                      const input::input_device_capabilities& extension) noexcept
{
    base.buttons = base.buttons || extension.buttons;
    base.axes = base.axes || extension.axes;
    base.pointer = base.pointer || extension.pointer;
    base.scroll = base.scroll || extension.scroll;
    base.rumble = base.rumble || extension.rumble;
    base.trigger_rumble = base.trigger_rumble || extension.trigger_rumble;
    base.haptics = base.haptics || extension.haptics;
    base.gyroscope = base.gyroscope || extension.gyroscope;
    base.accelerometer = base.accelerometer || extension.accelerometer;
    base.touchpad = base.touchpad || extension.touchpad;
    base.light = base.light || extension.light;
    base.adaptive_triggers = base.adaptive_triggers || extension.adaptive_triggers;
    base.battery = base.battery || extension.battery;
    base.button_count = std::max(base.button_count, extension.button_count);
    base.axis_count = std::max(base.axis_count, extension.axis_count);
    return base;
}

input::input_device_capabilities
windows_controller_extension_host::extension_capabilities(const device_state& state) const noexcept
{
    input::input_device_capabilities result{};
    for (const auto& [_, extension] : state.extensions)
        result = merge_capabilities(result, extension.descriptor.capabilities);
    return result;
}

void windows_controller_extension_host::sync_base_descriptor(input::input_device_id device, device_state& state)
{
    const input::input_device* current = input_->device(device);
    if (!current || !current->connected()) return;

    input::input_device_descriptor observed = snapshot(*current);
    const input::input_device_capabilities overlays = extension_capabilities(state);
    input::input_device_capabilities base = state.base_descriptor.capabilities;
    const input::input_device_capabilities& effective = observed.capabilities;

    if (!overlays.buttons) base.buttons = effective.buttons;
    if (!overlays.axes) base.axes = effective.axes;
    if (!overlays.pointer) base.pointer = effective.pointer;
    if (!overlays.scroll) base.scroll = effective.scroll;
    if (!overlays.rumble) base.rumble = effective.rumble;
    if (!overlays.trigger_rumble) base.trigger_rumble = effective.trigger_rumble;
    if (!overlays.haptics) base.haptics = effective.haptics;
    if (!overlays.gyroscope) base.gyroscope = effective.gyroscope;
    if (!overlays.accelerometer) base.accelerometer = effective.accelerometer;
    if (!overlays.touchpad) base.touchpad = effective.touchpad;
    if (!overlays.light) base.light = effective.light;
    if (!overlays.adaptive_triggers) base.adaptive_triggers = effective.adaptive_triggers;
    if (!overlays.battery) base.battery = effective.battery;
    if (overlays.button_count == 0) base.button_count = effective.button_count;
    if (overlays.axis_count == 0) base.axis_count = effective.axis_count;

    observed.capabilities = base;
    state.base_descriptor = std::move(observed);
}

void windows_controller_extension_host::apply(input::input_device_id device, device_state& state)
{
    const input::input_device* current = input_->device(device);
    if (!current || !current->connected()) return;

    input::input_device_descriptor effective = state.base_descriptor;
    for (const auto& [_, extension] : state.extensions)
        effective.capabilities = merge_capabilities(effective.capabilities, extension.descriptor.capabilities);
    input_->connect_device(std::move(effective));
}

} // namespace arc::platform::windows

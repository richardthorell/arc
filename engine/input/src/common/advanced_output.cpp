#include <arc/input/input.h>

#include <algorithm>
#include <cmath>

namespace arc::input
{
namespace
{

input_light_state normalize_light(input_light_state state) noexcept
{
    auto normalize = [](float value)
    {
        if (!std::isfinite(value)) return 0.0f;
        return std::clamp(value, 0.0f, 1.0f);
    };

    state.red = normalize(state.red);
    state.green = normalize(state.green);
    state.blue = normalize(state.blue);
    return state;
}

input_adaptive_trigger_effect normalize_trigger_effect(input_adaptive_trigger_effect effect) noexcept
{
    auto normalize = [](float value)
    {
        if (!std::isfinite(value)) return 0.0f;
        return std::clamp(value, 0.0f, 1.0f);
    };

    effect.start_position = normalize(effect.start_position);
    effect.end_position = std::max(effect.start_position, normalize(effect.end_position));
    effect.strength = normalize(effect.strength);
    if (!std::isfinite(effect.frequency_hz))
        effect.frequency_hz = 0.0f;
    else
        effect.frequency_hz = std::clamp(effect.frequency_hz, 0.0f, 255.0f);
    return effect;
}

} // namespace

bool input_player::set_light(input_light_state state) const
{
    bool applied = false;
    for (input_device_id device : system_->devices_for_player(id_))
        applied = system_->set_light(device, state) || applied;
    return applied;
}

bool input_player::set_adaptive_triggers(input_adaptive_trigger_state state) const
{
    bool applied = false;
    for (input_device_id device : system_->devices_for_player(id_))
        applied = system_->set_adaptive_triggers(device, state) || applied;
    return applied;
}

bool input_system::register_advanced_output_sink(input_device_id device, input_advanced_output_sink& sink) noexcept
{
    if (!devices_.contains(device.value)) return false;
    advanced_output_sinks_[device.value] = &sink;
    return true;
}

bool input_system::unregister_advanced_output_sink(input_device_id device, input_advanced_output_sink& sink) noexcept
{
    const auto found = advanced_output_sinks_.find(device.value);
    if (found == advanced_output_sinks_.end() || found->second != &sink) return false;
    advanced_output_sinks_.erase(found);
    return true;
}

bool input_system::set_light(input_device_id device_value, input_light_state state)
{
    const auto device_found = devices_.find(device_value.value);
    if (device_found == devices_.end() || !device_found->second.connected_ ||
        !device_found->second.descriptor_.capabilities.light)
        return false;

    const auto sink_found = advanced_output_sinks_.find(device_value.value);
    if (sink_found == advanced_output_sinks_.end() || !sink_found->second) return false;
    return sink_found->second->set_light(device_value, normalize_light(state));
}

bool input_system::set_adaptive_triggers(input_device_id device_value, input_adaptive_trigger_state state)
{
    const auto device_found = devices_.find(device_value.value);
    if (device_found == devices_.end() || !device_found->second.connected_ ||
        !device_found->second.descriptor_.capabilities.adaptive_triggers)
        return false;

    const auto sink_found = advanced_output_sinks_.find(device_value.value);
    if (sink_found == advanced_output_sinks_.end() || !sink_found->second) return false;

    state.left = normalize_trigger_effect(state.left);
    state.right = normalize_trigger_effect(state.right);
    return sink_found->second->set_adaptive_triggers(device_value, state);
}

} // namespace arc::input

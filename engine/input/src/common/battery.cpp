#include <arc/input/input.h>

#include <algorithm>
#include <cmath>

namespace arc::input
{

const input_battery_state& input_device::battery_state() const noexcept
{
    return battery_state_;
}

bool input_system::submit_battery_state(input_device_id id, input_battery_state state)
{
    const auto found = devices_.find(id.value);
    if (found == devices_.end() || !found->second.connected_) return false;

    if (!state.level_available || !std::isfinite(state.level))
    {
        state.level = 0.0f;
        state.level_available = false;
    }
    else
    {
        state.level = std::clamp(state.level, 0.0f, 1.0f);
    }

    if (state.status == input_battery_status::not_present)
        found->second.descriptor_.capabilities.battery = false;
    else if (state.status != input_battery_status::unknown || state.level_available)
        found->second.descriptor_.capabilities.battery = true;

    found->second.battery_state_ = state;
    return true;
}

} // namespace arc::input
#include <arc/input/input.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace arc::input
{
namespace
{

constexpr float action_threshold = 0.5f;

bool contains_device(const std::vector<input_device_id>& devices, input_device_id id)
{
    return std::find(devices.begin(), devices.end(), id) != devices.end();
}

bool contains_player(const std::vector<player_id>& players, player_id id)
{
    return std::find(players.begin(), players.end(), id) != players.end();
}

} // namespace

input_device_id input_device::id() const noexcept
{
    return descriptor_.id;
}

input_device_type input_device::type() const noexcept
{
    return descriptor_.type;
}

input_connectivity_type input_device::connectivity() const noexcept
{
    return descriptor_.connectivity;
}

std::string_view input_device::name() const noexcept
{
    return descriptor_.name;
}

const input_device_capabilities& input_device::capabilities() const noexcept
{
    return descriptor_.capabilities;
}

bool input_device::connected() const noexcept
{
    return connected_;
}

input_player::input_player(player_id id, input_system& system) noexcept : id_(id), system_(&system) {}

player_id input_player::id() const noexcept
{
    return id_;
}

std::vector<input_device_id> input_player::devices() const
{
    return system_->devices_for_player(id_);
}

input_player::context_state* input_player::find_context(std::string_view name) noexcept
{
    const auto found = std::find_if(contexts_.begin(), contexts_.end(),
                                    [name](const context_state& state) { return state.context.name == name; });
    return found == contexts_.end() ? nullptr : &*found;
}

input_player::context_state& input_player::ensure_context(std::string_view name)
{
    if (context_state* existing = find_context(name)) return *existing;

    context_state state{};
    state.context.name = std::string(name);
    contexts_.push_back(std::move(state));
    return contexts_.back();
}

void input_player::add_context(std::string_view name, int priority, bool enabled)
{
    context_state& state = ensure_context(name);
    state.context.priority = priority;
    state.context.enabled = enabled;
}

bool input_player::set_context_enabled(std::string_view name, bool enabled)
{
    context_state* state = find_context(name);
    if (!state) return false;

    state->context.enabled = enabled;
    return true;
}

bool input_player::set_context_priority(std::string_view name, int priority)
{
    context_state* state = find_context(name);
    if (!state) return false;

    state->context.priority = priority;
    return true;
}

void input_player::bind_action(std::string_view context, std::string_view action, input_binding binding)
{
    ensure_context(context).actions[std::string(action)].push_back(std::move(binding));
}

void input_player::bind_axis(std::string_view context, std::string_view axis, input_binding binding, float contribution)
{
    ensure_context(context).axes[std::string(axis)].push_back(
        axis_contribution{.binding = std::move(binding), .contribution = contribution});
}

void input_player::bind_axis2d(std::string_view context, std::string_view axis, input_binding binding,
                               math::vector2f contribution)
{
    ensure_context(context).axes2d[std::string(axis)].push_back(
        axis2d_contribution{.binding = std::move(binding), .contribution = contribution});
}

const input_player::context_state* input_player::active_action_context(std::string_view name) const noexcept
{
    const context_state* best = nullptr;
    for (const context_state& state : contexts_)
    {
        if (!state.context.enabled || !state.actions.contains(std::string(name))) continue;
        if (!best || state.context.priority >= best->context.priority) best = &state;
    }
    return best;
}

const input_player::context_state* input_player::active_axis_context(std::string_view name) const noexcept
{
    const context_state* best = nullptr;
    for (const context_state& state : contexts_)
    {
        if (!state.context.enabled || !state.axes.contains(std::string(name))) continue;
        if (!best || state.context.priority >= best->context.priority) best = &state;
    }
    return best;
}

const input_player::context_state* input_player::active_axis2d_context(std::string_view name) const noexcept
{
    const context_state* best = nullptr;
    for (const context_state& state : contexts_)
    {
        if (!state.context.enabled || !state.axes2d.contains(std::string(name))) continue;
        if (!best || state.context.priority >= best->context.priority) best = &state;
    }
    return best;
}

bool input_player::evaluate_action(std::string_view name, bool previous) const
{
    const context_state* state = active_action_context(name);
    if (!state) return false;

    const auto found = state->actions.find(std::string(name));
    if (found == state->actions.end()) return false;

    for (const input_binding& binding : found->second)
    {
        if (std::abs(system_->binding_value(id_, binding, previous)) >= action_threshold) return true;
    }
    return false;
}

bool input_player::pressed(std::string_view action) const
{
    return evaluate_action(action, false) && !evaluate_action(action, true);
}

bool input_player::released(std::string_view action) const
{
    return !evaluate_action(action, false) && evaluate_action(action, true);
}

bool input_player::down(std::string_view action) const
{
    return evaluate_action(action, false);
}

float input_player::axis(std::string_view name) const
{
    const context_state* state = active_axis_context(name);
    if (!state) return 0.0f;

    const auto found = state->axes.find(std::string(name));
    if (found == state->axes.end()) return 0.0f;

    float value = 0.0f;
    for (const axis_contribution& contribution : found->second)
        value += system_->binding_value(id_, contribution.binding, false) * contribution.contribution;
    return value;
}

math::vector2f input_player::axis2d(std::string_view name) const
{
    const context_state* state = active_axis2d_context(name);
    if (!state) return {};

    const auto found = state->axes2d.find(std::string(name));
    if (found == state->axes2d.end()) return {};

    math::vector2f value{};
    for (const axis2d_contribution& contribution : found->second)
    {
        const float source = system_->binding_value(id_, contribution.binding, false);
        value[0] += source * contribution.contribution[0];
        value[1] += source * contribution.contribution[1];
    }
    return value;
}

input_system::input_system()
{
    player(0);
}

void input_system::begin_frame()
{
    device_events_.clear();
    for (auto& [_, device] : devices_)
    {
        device.previous_values_ = device.current_values_;
        for (auto& [control, value] : device.current_values_)
        {
            if (transient_control(control)) value = 0.0f;
        }
    }
}

input_device_id input_system::connect_device(input_device_descriptor descriptor)
{
    if (!descriptor.id)
    {
        while (devices_.contains(next_device_id_))
            ++next_device_id_;
        descriptor.id = {.value = next_device_id_++};
    }

    const std::uint64_t value = descriptor.id.value;
    auto found = devices_.find(value);
    const bool was_connected = found != devices_.end() && found->second.connected_;

    if (found == devices_.end())
    {
        input_device device{};
        device.descriptor_ = std::move(descriptor);
        device.connected_ = true;
        found = devices_.emplace(value, std::move(device)).first;
    }
    else
    {
        found->second.descriptor_ = std::move(descriptor);
        found->second.connected_ = true;
    }

    if (!was_connected)
        device_events_.push_back({.type = input_device_event_type::connected, .device = found->second.descriptor_.id});

    if ((found->second.descriptor_.type == input_device_type::keyboard ||
         found->second.descriptor_.type == input_device_type::mouse) &&
        !contains_player(device_players_[value], 0))
    {
        assign_device(0, found->second.descriptor_.id);
    }

    return found->second.descriptor_.id;
}

bool input_system::disconnect_device(input_device_id id)
{
    const auto found = devices_.find(id.value);
    if (found == devices_.end() || !found->second.connected_) return false;

    found->second.connected_ = false;
    for (auto& [_, value] : found->second.current_values_)
        value = 0.0f;
    device_events_.push_back({.type = input_device_event_type::disconnected, .device = id});
    return true;
}

bool input_system::submit_button(input_device_id id, input_control control, bool down)
{
    const auto found = devices_.find(id.value);
    if (found == devices_.end() || !found->second.connected_) return false;

    found->second.current_values_[control_key(control)] = down ? 1.0f : 0.0f;
    return true;
}

bool input_system::submit_axis(input_device_id id, input_control control, float value)
{
    const auto found = devices_.find(id.value);
    if (found == devices_.end() || !found->second.connected_) return false;

    const std::uint32_t key = control_key(control);
    if (transient_control(key))
        found->second.current_values_[key] += value;
    else
        found->second.current_values_[key] = value;
    return true;
}

void input_system::release_all()
{
    const std::uint32_t position_x = control_key(make_mouse_axis_control(mouse_axis::position_x));
    const std::uint32_t position_y = control_key(make_mouse_axis_control(mouse_axis::position_y));

    for (auto& [_, device] : devices_)
    {
        for (auto& [control, value] : device.current_values_)
        {
            if (control != position_x && control != position_y) value = 0.0f;
        }
    }
}

const input_device* input_system::device(input_device_id id) const noexcept
{
    const auto found = devices_.find(id.value);
    return found == devices_.end() ? nullptr : &found->second;
}

std::vector<input_device_id> input_system::devices(bool connected_only) const
{
    std::vector<input_device_id> result;
    result.reserve(devices_.size());
    for (const auto& [_, device] : devices_)
    {
        if (!connected_only || device.connected_) result.push_back(device.descriptor_.id);
    }
    return result;
}

std::vector<input_device_id> input_system::devices(input_device_type type, bool connected_only) const
{
    std::vector<input_device_id> result;
    for (const auto& [_, device] : devices_)
    {
        if (device.descriptor_.type != type) continue;
        if (!connected_only || device.connected_) result.push_back(device.descriptor_.id);
    }
    return result;
}

const std::vector<input_device_event>& input_system::device_events() const noexcept
{
    return device_events_;
}

input_player& input_system::player(player_id id)
{
    const auto found = players_.find(id);
    if (found != players_.end()) return *found->second;

    auto value = std::unique_ptr<input_player>(new input_player(id, *this));
    input_player& result = *value;
    players_.emplace(id, std::move(value));
    return result;
}

const input_player* input_system::find_player(player_id id) const noexcept
{
    const auto found = players_.find(id);
    return found == players_.end() ? nullptr : found->second.get();
}

bool input_system::assign_device(player_id player_value, input_device_id device_value)
{
    if (!devices_.contains(device_value.value)) return false;

    player(player_value);
    std::vector<input_device_id>& player_devices = player_devices_[player_value];
    std::vector<player_id>& device_players = device_players_[device_value.value];

    if (!contains_device(player_devices, device_value)) player_devices.push_back(device_value);
    if (!contains_player(device_players, player_value)) device_players.push_back(player_value);
    return true;
}

bool input_system::unassign_device(player_id player_value, input_device_id device_value)
{
    bool changed = false;

    if (auto found = player_devices_.find(player_value); found != player_devices_.end())
    {
        auto& values = found->second;
        const auto end = std::remove(values.begin(), values.end(), device_value);
        changed = end != values.end();
        values.erase(end, values.end());
    }

    if (auto found = device_players_.find(device_value.value); found != device_players_.end())
    {
        auto& values = found->second;
        const auto end = std::remove(values.begin(), values.end(), player_value);
        values.erase(end, values.end());
    }

    return changed;
}

std::vector<input_device_id> input_system::devices_for_player(player_id player_value) const
{
    const auto found = player_devices_.find(player_value);
    return found == player_devices_.end() ? std::vector<input_device_id>{} : found->second;
}

std::vector<player_id> input_system::players_for_device(input_device_id device_value) const
{
    const auto found = device_players_.find(device_value.value);
    return found == device_players_.end() ? std::vector<player_id>{} : found->second;
}

float input_system::binding_value(player_id player_value, const input_binding& binding, bool previous) const
{
    const auto assigned = player_devices_.find(player_value);
    if (assigned == player_devices_.end()) return 0.0f;

    const std::uint32_t key = control_key(binding.control);
    float value = 0.0f;

    for (input_device_id id : assigned->second)
    {
        const auto found = devices_.find(id.value);
        if (found == devices_.end() || found->second.descriptor_.type != binding.device) continue;
        if (!previous && !found->second.connected_) continue;

        const auto& values = previous ? found->second.previous_values_ : found->second.current_values_;
        const auto control = values.find(key);
        if (control != values.end()) value += control->second;
    }

    return apply_processors(value, binding);
}

float input_system::apply_processors(float value, const input_binding& binding) noexcept
{
    for (const input_processor& processor : binding.processors)
    {
        switch (processor.type)
        {
            case input_processor_type::scale:
                value *= processor.value;
                break;
            case input_processor_type::invert:
                value = -value;
                break;
            case input_processor_type::clamp:
            {
                const float lower = std::min(processor.value, processor.secondary);
                const float upper = std::max(processor.value, processor.secondary);
                value = std::clamp(value, lower, upper);
                break;
            }
        }
    }
    return value;
}

std::uint32_t input_system::control_key(input_control control) noexcept
{
    return (static_cast<std::uint32_t>(control.kind) << 16U) | control.code;
}

bool input_system::transient_control(std::uint32_t key) noexcept
{
    const auto kind = static_cast<input_control_kind>((key >> 16U) & 0xffU);
    if (kind != input_control_kind::mouse_axis) return false;

    const auto axis = static_cast<mouse_axis>(key & 0xffffU);
    return axis == mouse_axis::delta_x || axis == mouse_axis::delta_y || axis == mouse_axis::wheel_x ||
           axis == mouse_axis::wheel_y;
}

} // namespace arc::input

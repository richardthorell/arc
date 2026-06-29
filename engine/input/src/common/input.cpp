#include <arc/input/input.h>

#include <utility>

namespace arc::input
{
namespace
{

std::uint64_t key_for(input_binding binding)
{
    const auto device = static_cast<std::uint64_t>(binding.device);
    return (static_cast<std::uint64_t>(binding.player) << 40U)
        | (device << 32U)
        | static_cast<std::uint32_t>(binding.code);
}

input_binding binding_from_event(const arc::event& event)
{
    switch (event.type)
    {
    case arc::event_type::key_down:
    case arc::event_type::key_up:
        return input_binding{
            .device = input_device_type::keyboard,
            .code = event.key_code,
            .player = 0
        };
    case arc::event_type::mouse_button_down:
    case arc::event_type::mouse_button_up:
        return input_binding{
            .device = input_device_type::mouse,
            .code = static_cast<int>(event.button),
            .player = 0
        };
    default:
        return {};
    }
}

} // namespace

void input_manager::begin_frame()
{
    pressed_.clear();
    released_.clear();
}

void input_manager::process_event(const arc::event& event)
{
    if (event.type != arc::event_type::key_down
        && event.type != arc::event_type::key_up
        && event.type != arc::event_type::mouse_button_down
        && event.type != arc::event_type::mouse_button_up)
    {
        return;
    }

    const auto key = key_for(binding_from_event(event));
    if (event.type == arc::event_type::key_down || event.type == arc::event_type::mouse_button_down)
    {
        const auto [_, inserted] = held_.insert(key);
        if (inserted && !event.repeat)
            pressed_.insert(key);
        return;
    }

    const auto erased = held_.erase(key);
    if (erased > 0)
        released_.insert(key);
}

input_action_id input_manager::bind_action(std::string_view name, input_binding binding)
{
    actions_[std::string(name)].push_back(binding);
    return next_action_id_++;
}

input_axis_id input_manager::bind_axis(std::string_view name, input_binding positive_binding, input_binding negative_binding)
{
    axes_[std::string(name)] = axis_bindings{ .positive = positive_binding, .negative = negative_binding };
    return next_axis_id_++;
}

bool input_manager::pressed(std::string_view name, player_id player) const
{
    const auto found = actions_.find(std::string(name));
    if (found == actions_.end())
        return false;

    for (auto binding : found->second)
    {
        binding.player = player;
        if (pressed_.contains(key_for(binding)))
            return true;
    }
    return false;
}

bool input_manager::released(std::string_view name, player_id player) const
{
    const auto found = actions_.find(std::string(name));
    if (found == actions_.end())
        return false;

    for (auto binding : found->second)
    {
        binding.player = player;
        if (released_.contains(key_for(binding)))
            return true;
    }
    return false;
}

bool input_manager::down(std::string_view name, player_id player) const
{
    const auto found = actions_.find(std::string(name));
    if (found == actions_.end())
        return false;

    for (auto binding : found->second)
    {
        binding.player = player;
        if (held_.contains(key_for(binding)))
            return true;
    }
    return false;
}

float input_manager::axis(std::string_view name, player_id player) const
{
    const auto found = axes_.find(std::string(name));
    if (found == axes_.end())
        return 0.0f;

    auto positive = found->second.positive;
    auto negative = found->second.negative;
    positive.player = player;
    negative.player = player;

    float value = 0.0f;
    if (held_.contains(key_for(positive)))
        value += 1.0f;
    if (held_.contains(key_for(negative)))
        value -= 1.0f;
    return value;
}

} // namespace arc::input

#include <arc/input/input.h>

namespace arc::input
{
namespace
{

constexpr std::string_view legacy_context = "legacy";
constexpr std::uint8_t legacy_key_down_event = 3;
constexpr std::uint8_t legacy_key_up_event = 4;
constexpr std::uint8_t legacy_mouse_button_down_event = 5;
constexpr std::uint8_t legacy_mouse_button_up_event = 6;
constexpr std::uint8_t legacy_focus_lost_event = 10;

mouse_button normalize_mouse_button_code(int code) noexcept
{
    switch (code)
    {
        case 1:
            return mouse_button::left;
        case 2:
            return mouse_button::right;
        case 3:
            return mouse_button::middle;
        case 4:
            return mouse_button::x1;
        case 5:
            return mouse_button::x2;
        default:
            return mouse_button::unknown;
    }
}

} // namespace

input_manager::input_manager()
{
    keyboard_ = system_.connect_device({.type = input_device_type::keyboard,
                                        .connectivity = input_connectivity_type::unknown,
                                        .name = "Legacy Keyboard",
                                        .capabilities = {.buttons = true}});
    mouse_ = system_.connect_device({.type = input_device_type::mouse,
                                     .connectivity = input_connectivity_type::unknown,
                                     .name = "Legacy Mouse",
                                     .capabilities = {.buttons = true, .axes = true, .pointer = true, .scroll = true}});
    system_.player(0).add_context(legacy_context);
}

void input_manager::begin_frame()
{
    system_.begin_frame();
}

void input_manager::process_legacy_event(std::uint8_t type, int key_code, int mouse_button_code, bool repeat)
{
    if (type == legacy_focus_lost_event)
    {
        system_.release_all();
        return;
    }

    if (type == legacy_key_down_event || type == legacy_key_up_event)
    {
        const key normalized = normalize_key_code(key_code);
        if (normalized == key::unknown) return;
        if (repeat && type == legacy_key_down_event) return;
        system_.submit_button(keyboard_, make_key_control(normalized), type == legacy_key_down_event);
        return;
    }

    if (type == legacy_mouse_button_down_event || type == legacy_mouse_button_up_event)
    {
        const mouse_button normalized = normalize_mouse_button_code(mouse_button_code);
        if (normalized == mouse_button::unknown) return;
        system_.submit_button(mouse_, make_mouse_button_control(normalized), type == legacy_mouse_button_down_event);
    }
}

void input_manager::bind_action(std::string_view name, input_manager_binding binding)
{
    system_.player(0).bind_action(legacy_context, name, normalize_binding(binding));
}

void input_manager::bind_axis(std::string_view name, input_manager_binding positive_binding,
                              input_manager_binding negative_binding)
{
    input_player& player = system_.player(0);
    player.bind_axis(legacy_context, name, normalize_binding(positive_binding), 1.0f);
    player.bind_axis(legacy_context, name, normalize_binding(negative_binding), -1.0f);
}

bool input_manager::pressed(std::string_view name, player_id player) const
{
    const input_player* mapped = system_.find_player(player);
    return mapped && mapped->pressed(name);
}

bool input_manager::released(std::string_view name, player_id player) const
{
    const input_player* mapped = system_.find_player(player);
    return mapped && mapped->released(name);
}

bool input_manager::down(std::string_view name, player_id player) const
{
    const input_player* mapped = system_.find_player(player);
    return mapped && mapped->down(name);
}

float input_manager::axis(std::string_view name, player_id player) const
{
    const input_player* mapped = system_.find_player(player);
    return mapped ? mapped->axis(name) : 0.0f;
}

input_binding input_manager::normalize_binding(input_manager_binding binding) noexcept
{
    switch (binding.device)
    {
        case input_device_type::keyboard:
            return {.device = binding.device, .control = make_key_control(normalize_key_code(binding.code))};
        case input_device_type::mouse:
            return {.device = binding.device,
                    .control = make_mouse_button_control(normalize_mouse_button_code(binding.code))};
        default:
            return {.device = binding.device};
    }
}

key input_manager::normalize_key_code(int code) noexcept
{
    if (code >= 'A' && code <= 'Z')
        return static_cast<key>(static_cast<std::uint16_t>(key::a) + static_cast<std::uint16_t>(code - 'A'));
    if (code >= 'a' && code <= 'z')
        return static_cast<key>(static_cast<std::uint16_t>(key::a) + static_cast<std::uint16_t>(code - 'a'));
    if (code >= '0' && code <= '9')
        return static_cast<key>(static_cast<std::uint16_t>(key::num0) + static_cast<std::uint16_t>(code - '0'));

    switch (code)
    {
        case 8:
            return key::backspace;
        case 9:
            return key::tab;
        case 13:
            return key::enter;
        case 27:
            return key::escape;
        case 32:
            return key::space;
        default:
            return key::unknown;
    }
}

} // namespace arc::input

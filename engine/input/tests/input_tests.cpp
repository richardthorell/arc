#include <arc/input.h>

#include <arc/event.h>

#include <cassert>

int main()
{
    arc::input::input_manager input;
    input.bind_action("jump", { .device = arc::input::input_device_type::keyboard, .code = 'F' });
    input.bind_axis(
        "move.x",
        { .device = arc::input::input_device_type::keyboard, .code = 'D' },
        { .device = arc::input::input_device_type::keyboard, .code = 'A' });

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = 'F' });
    assert(input.pressed("jump"));
    assert(input.down("jump"));

    input.begin_frame();
    assert(!input.pressed("jump"));
    assert(input.down("jump"));

    input.process_event({ .type = arc::event_type::key_up, .key_code = 'F' });
    assert(input.released("jump"));
    assert(!input.down("jump"));

    input.begin_frame();
    input.process_event({ .type = arc::event_type::key_down, .key_code = 'D' });
    assert(input.axis("move.x") == 1.0f);
    input.process_event({ .type = arc::event_type::key_down, .key_code = 'A' });
    assert(input.axis("move.x") == 0.0f);

    input.begin_frame();
    input.bind_action(
        "select",
        { .device = arc::input::input_device_type::mouse, .code = static_cast<int>(arc::mouse_button::left) });
    input.process_event({ .type = arc::event_type::mouse_button_down, .button = arc::mouse_button::left });
    assert(input.pressed("select"));
    assert(input.down("select"));

    return 0;
}

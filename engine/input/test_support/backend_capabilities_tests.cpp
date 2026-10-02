#include <arc/input/backend.h>

int main()
{
    using namespace arc::input;

    constexpr input_backend_descriptor android{
        .platform = input_platform::android,
        .backend = input_backend_type::native,
        .name = "Android native input",
        .capabilities = {.keyboard = true,
                         .mouse = true,
                         .gamepad = true,
                         .touch = true,
                         .motion_sensors = true,
                         .battery = true,
                         .output = true},
    };

    static_assert(supports_device(android.capabilities, input_device_type::keyboard));
    static_assert(supports_device(android.capabilities, input_device_type::gamepad));
    static_assert(supports_device(android.capabilities, input_device_type::touch));
    static_assert(!supports_device(android.capabilities, input_device_type::pen));
    static_assert(!supports_device(android.capabilities, input_device_type::unknown));

    if (android.platform != input_platform::android) return 1;
    if (android.backend != input_backend_type::native) return 2;
    if (!android.capabilities.motion_sensors) return 3;
    if (!android.capabilities.battery || !android.capabilities.output) return 4;
    if (supports_device(android.capabilities, input_device_type::wheel)) return 5;

    constexpr input_backend_capabilities unavailable{};
    if (supports_device(unavailable, input_device_type::gamepad)) return 6;
    if (unavailable.motion_sensors || unavailable.battery || unavailable.output) return 7;

    return 0;
}

#pragma once

/** @namespace arc::input
 * @brief Platform-neutral physical input, player assignment, and mapping contracts.
 */

#include <arc/math/vector.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::input
{

/**
 * @brief Physical input device family.
 */
enum class input_device_type : std::uint8_t
{
    unknown,
    keyboard,
    mouse,
    gamepad,
    touch,
    pen,
    wheel,
    flight_stick,
    motion_controller
};

/**
 * @brief Physical connection used by an input device.
 */
enum class input_connectivity_type : std::uint8_t
{
    unknown,
    builtin,
    usb,
    wireless
};

/**
 * @brief Backend that owns discovery/state for a physical input device.
 *
 * This is diagnostic/provenance metadata only. Gameplay should bind against ARC
 * device/control types instead of selecting behavior from the backend.
 */
enum class input_backend_type : std::uint8_t
{
    unknown,
    native,
    raw_input,
    xinput,
    game_input,
    hid,
    platform_sdk
};

/**
 * @brief Optional specialization of a physical input device family.
 *
 * Device type remains the broad gameplay-facing category. Subtype preserves
 * richer controller classification without leaking platform constants.
 */
enum class input_device_subtype : std::uint8_t
{
    unknown,
    standard_gamepad,
    wheel,
    flight_stick,
    arcade_stick,
    dance_pad,
    guitar,
    drum_kit,
    arcade_pad
};

/**
 * @brief Optional USB/HID-style hardware identifiers reported by a backend.
 *
 * Zero values mean the backend cannot provide that field. These identifiers are
 * descriptive metadata and are not used as ARC runtime device IDs.
 */
struct input_device_hardware_id
{
    std::uint16_t vendor_id{};
    std::uint16_t product_id{};
    std::uint16_t version{};

    friend bool operator==(const input_device_hardware_id&, const input_device_hardware_id&) = default;
};

/**
 * @brief Stable runtime identifier for a physical input device.
 */
struct input_device_id
{
    std::uint64_t value{};

    [[nodiscard]] constexpr explicit operator bool() const noexcept
    {
        return value != 0;
    }

    friend bool operator==(const input_device_id&, const input_device_id&) = default;
};

/**
 * @brief Player slot identifier used by local input assignment.
 */
using player_id = std::uint32_t;

/**
 * @brief Keyboard keys normalized by ARC before gameplay mapping.
 */
enum class key : std::uint16_t
{
    unknown,
    a,
    b,
    c,
    d,
    e,
    f,
    g,
    h,
    i,
    j,
    k,
    l,
    m,
    n,
    o,
    p,
    q,
    r,
    s,
    t,
    u,
    v,
    w,
    x,
    y,
    z,
    num0,
    num1,
    num2,
    num3,
    num4,
    num5,
    num6,
    num7,
    num8,
    num9,
    escape,
    space,
    enter,
    tab,
    backspace,
    left_shift,
    right_shift,
    left_control,
    right_control,
    left_alt,
    right_alt,
    left,
    right,
    up,
    down,
    insert,
    delete_key,
    home,
    end,
    page_up,
    page_down,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    f10,
    f11,
    f12
};

/**
 * @brief Mouse buttons normalized by ARC.
 */
enum class mouse_button : std::uint8_t
{
    unknown,
    left,
    right,
    middle,
    x1,
    x2
};

/**
 * @brief Mouse scalar channels available to mappings.
 */
enum class mouse_axis : std::uint8_t
{
    position_x,
    position_y,
    delta_x,
    delta_y,
    wheel_x,
    wheel_y
};

/**
 * @brief Sensor scalar channels reserved for motion-capable devices.
 */
enum class sensor_axis : std::uint8_t
{
    gyroscope_x,
    gyroscope_y,
    gyroscope_z,
    accelerometer_x,
    accelerometer_y,
    accelerometer_z
};

/**
 * @brief Namespace of a normalized physical control.
 */
enum class input_control_kind : std::uint8_t
{
    unknown,
    keyboard_key,
    mouse_button,
    mouse_axis,
    gamepad_button,
    gamepad_axis,
    sensor_axis,
    touch
};

/**
 * @brief Platform-neutral physical control identifier.
 */
struct input_control
{
    input_control_kind kind{input_control_kind::unknown};
    std::uint16_t code{};

    friend bool operator==(const input_control&, const input_control&) = default;
};

[[nodiscard]] constexpr input_control make_key_control(key value) noexcept
{
    return {.kind = input_control_kind::keyboard_key, .code = static_cast<std::uint16_t>(value)};
}

[[nodiscard]] constexpr input_control make_mouse_button_control(mouse_button value) noexcept
{
    return {.kind = input_control_kind::mouse_button, .code = static_cast<std::uint16_t>(value)};
}

[[nodiscard]] constexpr input_control make_mouse_axis_control(mouse_axis value) noexcept
{
    return {.kind = input_control_kind::mouse_axis, .code = static_cast<std::uint16_t>(value)};
}

[[nodiscard]] constexpr input_control make_sensor_axis_control(sensor_axis value) noexcept
{
    return {.kind = input_control_kind::sensor_axis, .code = static_cast<std::uint16_t>(value)};
}

/**
 * @brief Capabilities advertised by a physical device.
 *
 * Unsupported capabilities remain false/zero. Backends may refine this data
 * after connection without changing the device identifier.
 */
struct input_device_capabilities
{
    bool buttons{};
    bool axes{};
    bool pointer{};
    bool scroll{};
    bool rumble{};
    bool trigger_rumble{};
    bool haptics{};
    bool gyroscope{};
    bool accelerometer{};
    bool touchpad{};
    bool light{};
    bool adaptive_triggers{};
    bool battery{};
    std::uint16_t button_count{};
    std::uint16_t axis_count{};
};

/**
 * @brief Platform-neutral metadata for one physical device.
 */
struct input_device_descriptor
{
    input_device_id id{};
    input_device_type type{input_device_type::unknown};
    input_device_subtype subtype{input_device_subtype::unknown};
    input_connectivity_type connectivity{input_connectivity_type::unknown};
    input_backend_type backend{input_backend_type::unknown};
    input_device_hardware_id hardware_id{};
    std::string backend_id;
    std::string name;
    input_device_capabilities capabilities{};
};

/**
 * @brief Read-only physical device record owned by input_system.
 */
class input_device
{
public:
    [[nodiscard]] input_device_id id() const noexcept;
    [[nodiscard]] input_device_type type() const noexcept;
    [[nodiscard]] input_connectivity_type connectivity() const noexcept;
    [[nodiscard]] std::string_view name() const noexcept;
    [[nodiscard]] const input_device_capabilities& capabilities() const noexcept;
    [[nodiscard]] bool connected() const noexcept;

    [[nodiscard]] input_device_subtype subtype() const noexcept
    {
        return descriptor_.subtype;
    }

    [[nodiscard]] input_backend_type backend() const noexcept
    {
        return descriptor_.backend;
    }

    [[nodiscard]] const input_device_hardware_id& hardware_id() const noexcept
    {
        return descriptor_.hardware_id;
    }

    [[nodiscard]] std::string_view backend_id() const noexcept
    {
        return descriptor_.backend_id;
    }

private:
    friend class input_system;

    input_device_descriptor descriptor_{};
    bool connected_{};
    std::unordered_map<std::uint32_t, float> current_values_;
    std::unordered_map<std::uint32_t, float> previous_values_;
};

/**
 * @brief Device lifecycle event emitted by input_system.
 */
enum class input_device_event_type : std::uint8_t
{
    connected,
    disconnected
};

struct input_device_event
{
    input_device_event_type type{input_device_event_type::connected};
    input_device_id device{};
};

/**
 * @brief Normalized four-motor rumble strengths.
 *
 * Values are clamped to [0, 1] before they reach a platform backend. The low
 * frequency channel maps to the heavy motor and the high frequency channel maps
 * to the light motor on traditional gamepads. Trigger channels are independent
 * rumble motors when the device exposes them; they are not adaptive-trigger
 * resistance controls.
 */
struct input_rumble_state
{
    float low_frequency{};
    float high_frequency{};
    float left_trigger{};
    float right_trigger{};

    friend bool operator==(const input_rumble_state&, const input_rumble_state&) = default;
};

/**
 * @brief Platform output sink associated with one or more physical devices.
 *
 * Platform backends implement this interface and register themselves with the
 * input system. Gameplay never depends on the concrete platform implementation.
 */
class input_output_sink
{
public:
    virtual ~input_output_sink() = default;

    virtual bool set_rumble(input_device_id device, input_rumble_state state) = 0;
};

/**
 * @brief Small processing operations applied to one binding value.
 */
enum class input_processor_type : std::uint8_t
{
    scale,
    invert,
    clamp
};

struct input_processor
{
    input_processor_type type{input_processor_type::scale};
    float value{1.0f};
    float secondary{};
};

/**
 * @brief One physical source used by a player mapping.
 */
struct input_binding
{
    input_device_type device{input_device_type::unknown};
    input_control control{};
    std::vector<input_processor> processors;
};

/**
 * @brief Named mapping layer with priority and enabled state.
 */
struct input_context
{
    std::string name;
    int priority{};
    bool enabled{true};
};

class input_system;

/**
 * @brief Per-player device assignment view and gameplay mapping state.
 */
class input_player final
{
public:
    [[nodiscard]] player_id id() const noexcept;
    [[nodiscard]] std::vector<input_device_id> devices() const;

    void add_context(std::string_view name, int priority = 0, bool enabled = true);
    bool set_context_enabled(std::string_view name, bool enabled);
    bool set_context_priority(std::string_view name, int priority);

    void bind_action(std::string_view context, std::string_view action, input_binding binding);
    void bind_axis(std::string_view context, std::string_view axis, input_binding binding, float contribution = 1.0f);
    void bind_axis2d(std::string_view context, std::string_view axis, input_binding binding,
                     math::vector2f contribution);

    [[nodiscard]] bool pressed(std::string_view action) const;
    [[nodiscard]] bool released(std::string_view action) const;
    [[nodiscard]] bool down(std::string_view action) const;
    [[nodiscard]] float axis(std::string_view name) const;
    [[nodiscard]] math::vector2f axis2d(std::string_view name) const;

    /**
     * @brief Apply rumble to every assigned connected device that supports it.
     * @return True if at least one backend accepted the output.
     */
    [[nodiscard]] bool set_rumble(input_rumble_state state) const;

    /**
     * @brief Stop rumble on every assigned connected device that supports it.
     */
    [[nodiscard]] bool stop_rumble() const;

private:
    friend class input_system;

    struct axis_contribution
    {
        input_binding binding;
        float contribution{1.0f};
    };

    struct axis2d_contribution
    {
        input_binding binding;
        math::vector2f contribution{};
    };

    struct context_state
    {
        input_context context;
        std::unordered_map<std::string, std::vector<input_binding>> actions;
        std::unordered_map<std::string, std::vector<axis_contribution>> axes;
        std::unordered_map<std::string, std::vector<axis2d_contribution>> axes2d;
    };

    input_player(player_id id, input_system& system) noexcept;
    context_state& ensure_context(std::string_view name);
    [[nodiscard]] context_state* find_context(std::string_view name) noexcept;
    [[nodiscard]] const context_state* active_action_context(std::string_view name) const noexcept;
    [[nodiscard]] const context_state* active_axis_context(std::string_view name) const noexcept;
    [[nodiscard]] const context_state* active_axis2d_context(std::string_view name) const noexcept;
    [[nodiscard]] bool evaluate_action(std::string_view name, bool previous) const;

    player_id id_{};
    input_system* system_{};
    std::vector<context_state> contexts_;
};

/**
 * @brief Platform-neutral physical input registry, player assignment, state sampler, and device output router.
 *
 * Platform backends call the submit/connect methods. Gameplay normally queries
 * through input_player mappings instead of reading physical devices directly.
 */
class input_system final
{
public:
    input_system();

    /**
     * @brief Advance input state to a new frame.
     *
     * Persistent controls retain their values. Transient mouse deltas and wheel
     * channels are reset after current values are copied to previous state.
     */
    void begin_frame();

    /**
     * @brief Register or reconnect a physical device.
     *
     * A zero identifier is replaced with a runtime-generated identifier. Reusing
     * an existing identifier preserves player assignments across reconnection.
     */
    input_device_id connect_device(input_device_descriptor descriptor);
    bool disconnect_device(input_device_id id);

    bool submit_button(input_device_id id, input_control control, bool down);
    bool submit_axis(input_device_id id, input_control control, float value);
    void release_all();

    [[nodiscard]] const input_device* device(input_device_id id) const noexcept;
    [[nodiscard]] std::vector<input_device_id> devices(bool connected_only = true) const;
    [[nodiscard]] std::vector<input_device_id> devices(input_device_type type, bool connected_only = true) const;
    [[nodiscard]] const std::vector<input_device_event>& device_events() const noexcept;

    input_player& player(player_id id);
    [[nodiscard]] const input_player* find_player(player_id id) const noexcept;

    bool assign_device(player_id player, input_device_id device);
    bool unassign_device(player_id player, input_device_id device);
    [[nodiscard]] std::vector<input_device_id> devices_for_player(player_id player) const;
    [[nodiscard]] std::vector<player_id> players_for_device(input_device_id device) const;

    /**
     * @brief Associate a platform output sink with a connected or retained device record.
     */
    bool register_output_sink(input_device_id device, input_output_sink& sink) noexcept;

    /**
     * @brief Remove a platform output sink if it is still owned by the supplied backend.
     */
    bool unregister_output_sink(input_device_id device, input_output_sink& sink) noexcept;

    /**
     * @brief Apply normalized rumble to one physical device.
     */
    bool set_rumble(input_device_id device, input_rumble_state state);

    /**
     * @brief Stop rumble on one physical device.
     */
    bool stop_rumble(input_device_id device);

    /**
     * @brief Stop rumble on all connected rumble-capable devices.
     */
    void stop_all_rumble();

private:
    friend class input_player;

    [[nodiscard]] float binding_value(player_id player, const input_binding& binding, bool previous) const;
    [[nodiscard]] static float apply_processors(float value, const input_binding& binding) noexcept;
    [[nodiscard]] static float normalize_output(float value) noexcept;
    [[nodiscard]] static std::uint32_t control_key(input_control control) noexcept;
    [[nodiscard]] static bool transient_control(std::uint32_t key) noexcept;

    std::unordered_map<std::uint64_t, input_device> devices_;
    std::unordered_map<player_id, std::unique_ptr<input_player>> players_;
    std::unordered_map<player_id, std::vector<input_device_id>> player_devices_;
    std::unordered_map<std::uint64_t, std::vector<player_id>> device_players_;
    std::unordered_map<std::uint64_t, input_output_sink*> output_sinks_;
    std::vector<input_device_event> device_events_;
    std::uint64_t next_device_id_{1};
};

/**
 * @brief Transitional binding shape for the pre-M1 input_manager adapter.
 *
 * New code should use input_binding and input_player mappings. This type exists
 * so editor code can migrate independently without reintroducing player state
 * into the new physical binding model.
 */
struct input_manager_binding
{
    input_device_type device{input_device_type::keyboard};
    int code{};
};

/**
 * @brief Event type bridge used only by the pre-M1 compatibility adapter.
 *
 * The converting constructor intentionally accepts enum-class values from the
 * legacy framework event API without making arc-input depend on framework.
 */
struct input_manager_event_type
{
    std::uint8_t value{};

    constexpr input_manager_event_type() noexcept = default;

    template <class Value>
    constexpr input_manager_event_type(Value input) noexcept : value(static_cast<std::uint8_t>(input))
    {
    }
};

/**
 * @brief Concrete event shape so designated braced events can bind to process_event.
 */
struct input_manager_event
{
    input_manager_event_type type{};
    int key_code{};
    int button{};
    bool repeat{};
};

/**
 * @brief Compatibility adapter backed by the M1 input_system/player machinery.
 *
 * This preserves the narrow pre-M1 mapping surface used by editor tests while
 * the runtime and new gameplay API use input_system directly.
 */
class input_manager final
{
public:
    input_manager();

    void begin_frame();

    void process_event(const input_manager_event& event)
    {
        process_legacy_event(event.type.value, event.key_code, event.button, event.repeat);
    }

    template <class Event> void process_event(const Event& event)
    {
        process_legacy_event(static_cast<std::uint8_t>(event.type), event.key_code, static_cast<int>(event.button),
                             event.repeat);
    }

    void bind_action(std::string_view name, input_manager_binding binding);
    void bind_axis(std::string_view name, input_manager_binding positive_binding,
                   input_manager_binding negative_binding);

    [[nodiscard]] bool pressed(std::string_view name, player_id player = 0) const;
    [[nodiscard]] bool released(std::string_view name, player_id player = 0) const;
    [[nodiscard]] bool down(std::string_view name, player_id player = 0) const;
    [[nodiscard]] float axis(std::string_view name, player_id player = 0) const;

private:
    void process_legacy_event(std::uint8_t type, int key_code, int mouse_button_code, bool repeat);
    [[nodiscard]] static input_binding normalize_binding(input_manager_binding binding) noexcept;
    [[nodiscard]] static key normalize_key_code(int code) noexcept;

    input_system system_;
    input_device_id keyboard_{};
    input_device_id mouse_{};
};

} // namespace arc::input
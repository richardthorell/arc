#include <arc/input/gamepad.h>
#include <arc/input/touch.h>
#include <arc/project/input_config.h>

#include <cmath>
#include <fstream>
#include <optional>
#include <string_view>
#include <unordered_set>

#include <nlohmann/json.hpp>

namespace arc::project
{
namespace
{

using ordered_json = nlohmann::ordered_json;

std::optional<std::string> key_token(input::key value)
{
    if (value >= input::key::a && value <= input::key::z)
        return std::string(1, static_cast<char>('A' + static_cast<int>(value) - static_cast<int>(input::key::a)));
    if (value >= input::key::num0 && value <= input::key::num9)
        return std::string(1, static_cast<char>('0' + static_cast<int>(value) - static_cast<int>(input::key::num0)));

    switch (value)
    {
        case input::key::escape:
            return "Escape";
        case input::key::space:
            return "Space";
        case input::key::enter:
            return "Enter";
        case input::key::tab:
            return "Tab";
        case input::key::backspace:
            return "Backspace";
        case input::key::left_shift:
            return "LeftShift";
        case input::key::right_shift:
            return "RightShift";
        case input::key::left_control:
            return "LeftControl";
        case input::key::right_control:
            return "RightControl";
        case input::key::left_alt:
            return "LeftAlt";
        case input::key::right_alt:
            return "RightAlt";
        case input::key::left:
            return "Left";
        case input::key::right:
            return "Right";
        case input::key::up:
            return "Up";
        case input::key::down:
            return "Down";
        case input::key::insert:
            return "Insert";
        case input::key::delete_key:
            return "Delete";
        case input::key::home:
            return "Home";
        case input::key::end:
            return "End";
        case input::key::page_up:
            return "PageUp";
        case input::key::page_down:
            return "PageDown";
        case input::key::f1:
            return "F1";
        case input::key::f2:
            return "F2";
        case input::key::f3:
            return "F3";
        case input::key::f4:
            return "F4";
        case input::key::f5:
            return "F5";
        case input::key::f6:
            return "F6";
        case input::key::f7:
            return "F7";
        case input::key::f8:
            return "F8";
        case input::key::f9:
            return "F9";
        case input::key::f10:
            return "F10";
        case input::key::f11:
            return "F11";
        case input::key::f12:
            return "F12";
        default:
            return std::nullopt;
    }
}

std::optional<std::string_view> mouse_button_token(input::mouse_button value)
{
    switch (value)
    {
        case input::mouse_button::left:
            return "Left";
        case input::mouse_button::right:
            return "Right";
        case input::mouse_button::middle:
            return "Middle";
        case input::mouse_button::x1:
            return "X1";
        case input::mouse_button::x2:
            return "X2";
        default:
            return std::nullopt;
    }
}

std::optional<std::string_view> mouse_axis_token(input::mouse_axis value)
{
    switch (value)
    {
        case input::mouse_axis::position_x:
            return "PositionX";
        case input::mouse_axis::position_y:
            return "PositionY";
        case input::mouse_axis::delta_x:
            return "DeltaX";
        case input::mouse_axis::delta_y:
            return "DeltaY";
        case input::mouse_axis::wheel_x:
            return "WheelX";
        case input::mouse_axis::wheel_y:
            return "WheelY";
    }
    return std::nullopt;
}

std::optional<std::string_view> gamepad_button_token(input::gamepad_button value)
{
    switch (value)
    {
        case input::gamepad_button::south:
            return "South";
        case input::gamepad_button::east:
            return "East";
        case input::gamepad_button::west:
            return "West";
        case input::gamepad_button::north:
            return "North";
        case input::gamepad_button::auxiliary_1:
            return "Auxiliary1";
        case input::gamepad_button::auxiliary_2:
            return "Auxiliary2";
        case input::gamepad_button::dpad_up:
            return "DPadUp";
        case input::gamepad_button::dpad_down:
            return "DPadDown";
        case input::gamepad_button::dpad_left:
            return "DPadLeft";
        case input::gamepad_button::dpad_right:
            return "DPadRight";
        case input::gamepad_button::left_shoulder:
            return "LeftShoulder";
        case input::gamepad_button::right_shoulder:
            return "RightShoulder";
        case input::gamepad_button::left_trigger_button:
            return "LeftTriggerButton";
        case input::gamepad_button::right_trigger_button:
            return "RightTriggerButton";
        case input::gamepad_button::left_stick:
            return "LeftStick";
        case input::gamepad_button::right_stick:
            return "RightStick";
        case input::gamepad_button::left_stick_up:
            return "LeftStickUp";
        case input::gamepad_button::left_stick_down:
            return "LeftStickDown";
        case input::gamepad_button::left_stick_left:
            return "LeftStickLeft";
        case input::gamepad_button::left_stick_right:
            return "LeftStickRight";
        case input::gamepad_button::right_stick_up:
            return "RightStickUp";
        case input::gamepad_button::right_stick_down:
            return "RightStickDown";
        case input::gamepad_button::right_stick_left:
            return "RightStickLeft";
        case input::gamepad_button::right_stick_right:
            return "RightStickRight";
        case input::gamepad_button::paddle_left_1:
            return "PaddleLeft1";
        case input::gamepad_button::paddle_left_2:
            return "PaddleLeft2";
        case input::gamepad_button::paddle_right_1:
            return "PaddleRight1";
        case input::gamepad_button::paddle_right_2:
            return "PaddleRight2";
        case input::gamepad_button::view:
            return "View";
        case input::gamepad_button::menu:
            return "Menu";
        case input::gamepad_button::guide:
            return "Guide";
        case input::gamepad_button::share:
            return "Share";
    }
    return std::nullopt;
}

std::optional<std::string_view> gamepad_axis_token(input::gamepad_axis value)
{
    switch (value)
    {
        case input::gamepad_axis::left_x:
            return "LeftX";
        case input::gamepad_axis::left_y:
            return "LeftY";
        case input::gamepad_axis::right_x:
            return "RightX";
        case input::gamepad_axis::right_y:
            return "RightY";
        case input::gamepad_axis::left_trigger:
            return "LeftTrigger";
        case input::gamepad_axis::right_trigger:
            return "RightTrigger";
    }
    return std::nullopt;
}

std::optional<std::string_view> sensor_axis_token(input::sensor_axis value)
{
    switch (value)
    {
        case input::sensor_axis::gyroscope_x:
            return "GyroscopeX";
        case input::sensor_axis::gyroscope_y:
            return "GyroscopeY";
        case input::sensor_axis::gyroscope_z:
            return "GyroscopeZ";
        case input::sensor_axis::accelerometer_x:
            return "AccelerometerX";
        case input::sensor_axis::accelerometer_y:
            return "AccelerometerY";
        case input::sensor_axis::accelerometer_z:
            return "AccelerometerZ";
    }
    return std::nullopt;
}

std::optional<std::string_view> touch_control_token(input::touch_control value, bool gamepad)
{
    switch (value)
    {
        case input::touch_control::primary_down:
            return gamepad ? "TouchPrimaryDown" : "PrimaryDown";
        case input::touch_control::primary_x:
            return gamepad ? "TouchPrimaryX" : "PrimaryX";
        case input::touch_control::primary_y:
            return gamepad ? "TouchPrimaryY" : "PrimaryY";
        case input::touch_control::primary_pressure:
            return gamepad ? "TouchPrimaryPressure" : "PrimaryPressure";
    }
    return std::nullopt;
}

std::optional<ordered_json> processor_to_json(const input::input_processor& processor, std::string& error)
{
    ordered_json result = ordered_json::object();
    switch (processor.type)
    {
        case input::input_processor_type::scale:
            if (!std::isfinite(processor.value))
            {
                error = "input scale processor value must be finite";
                return std::nullopt;
            }
            result["type"] = "scale";
            result["value"] = processor.value;
            break;
        case input::input_processor_type::invert:
            result["type"] = "invert";
            break;
        case input::input_processor_type::clamp:
            if (!std::isfinite(processor.value) || !std::isfinite(processor.secondary))
            {
                error = "input clamp processor bounds must be finite";
                return std::nullopt;
            }
            if (processor.value > processor.secondary)
            {
                error = "input clamp processor minimum cannot exceed maximum";
                return std::nullopt;
            }
            result["type"] = "clamp";
            result["minimum"] = processor.value;
            result["maximum"] = processor.secondary;
            break;
    }
    return result;
}

std::optional<ordered_json> binding_to_json(const input::input_binding& binding, std::string& error)
{
    std::string_view device;
    std::optional<std::string> owned_control;
    std::optional<std::string_view> control;

    switch (binding.device)
    {
        case input::input_device_type::keyboard:
            if (binding.control.kind != input::input_control_kind::keyboard_key)
            {
                error = "keyboard binding requires a keyboard control";
                return std::nullopt;
            }
            device = "keyboard";
            owned_control = key_token(static_cast<input::key>(binding.control.code));
            break;
        case input::input_device_type::mouse:
            device = "mouse";
            if (binding.control.kind == input::input_control_kind::mouse_button)
                control = mouse_button_token(static_cast<input::mouse_button>(binding.control.code));
            else if (binding.control.kind == input::input_control_kind::mouse_axis)
                control = mouse_axis_token(static_cast<input::mouse_axis>(binding.control.code));
            else
            {
                error = "mouse binding requires a mouse button or axis control";
                return std::nullopt;
            }
            break;
        case input::input_device_type::gamepad:
            device = "gamepad";
            if (binding.control.kind == input::input_control_kind::gamepad_button)
                control = gamepad_button_token(static_cast<input::gamepad_button>(binding.control.code));
            else if (binding.control.kind == input::input_control_kind::gamepad_axis)
                control = gamepad_axis_token(static_cast<input::gamepad_axis>(binding.control.code));
            else if (binding.control.kind == input::input_control_kind::sensor_axis)
                control = sensor_axis_token(static_cast<input::sensor_axis>(binding.control.code));
            else if (binding.control.kind == input::input_control_kind::touch)
                control = touch_control_token(static_cast<input::touch_control>(binding.control.code), true);
            else
            {
                error = "gamepad binding requires a gamepad, sensor, or touch control";
                return std::nullopt;
            }
            break;
        case input::input_device_type::motion_controller:
            if (binding.control.kind != input::input_control_kind::sensor_axis)
            {
                error = "motion binding requires a sensor control";
                return std::nullopt;
            }
            device = "motion";
            control = sensor_axis_token(static_cast<input::sensor_axis>(binding.control.code));
            break;
        case input::input_device_type::touch:
            if (binding.control.kind != input::input_control_kind::touch)
            {
                error = "touch binding requires a touch control";
                return std::nullopt;
            }
            device = "touch";
            control = touch_control_token(static_cast<input::touch_control>(binding.control.code), false);
            break;
        default:
            error = "unsupported input binding device";
            return std::nullopt;
    }

    if (owned_control) control = *owned_control;
    if (!control)
    {
        error = "input binding contains an unknown control value";
        return std::nullopt;
    }

    ordered_json result = ordered_json::object();
    result["device"] = device;
    result["control"] = *control;
    if (!binding.processors.empty())
    {
        ordered_json processors = ordered_json::array();
        for (const auto& processor : binding.processors)
        {
            auto serialized = processor_to_json(processor, error);
            if (!serialized) return std::nullopt;
            processors.push_back(std::move(*serialized));
        }
        result["processors"] = std::move(processors);
    }
    return result;
}

bool validate_name(std::string_view kind, const std::string& name, std::unordered_set<std::string>& names,
                   std::string& error)
{
    if (name.empty())
    {
        error = "input " + std::string(kind) + " name cannot be empty";
        return false;
    }
    if (!names.emplace(name).second)
    {
        error = "duplicate input " + std::string(kind) + " '" + name + "'";
        return false;
    }
    return true;
}

} // namespace

input_config_save_result save_input_config(const input_config& config, const std::filesystem::path& path)
{
    if (config.version != input_config_version)
        return {.error = "unsupported input config version " + std::to_string(config.version)};

    ordered_json root = ordered_json::object();
    root["version"] = config.version;
    ordered_json contexts = ordered_json::array();
    std::unordered_set<std::string> context_names;

    for (const auto& context : config.contexts)
    {
        std::string error;
        if (!validate_name("context", context.name, context_names, error)) return {.error = std::move(error)};

        ordered_json context_json = ordered_json::object();
        context_json["name"] = context.name;
        context_json["priority"] = context.priority;
        context_json["enabled"] = context.enabled;

        if (!context.actions.empty())
        {
            ordered_json actions = ordered_json::array();
            std::unordered_set<std::string> names;
            for (const auto& action : context.actions)
            {
                if (!validate_name("action", action.name, names, error)) return {.error = std::move(error)};
                if (action.bindings.empty())
                    return {.error = "input action '" + action.name + "' requires at least one binding"};
                ordered_json action_json = ordered_json::object();
                action_json["name"] = action.name;
                ordered_json bindings = ordered_json::array();
                for (const auto& binding : action.bindings)
                {
                    auto serialized = binding_to_json(binding, error);
                    if (!serialized) return {.error = "input action '" + action.name + "': " + std::move(error)};
                    bindings.push_back(std::move(*serialized));
                }
                action_json["bindings"] = std::move(bindings);
                actions.push_back(std::move(action_json));
            }
            context_json["actions"] = std::move(actions);
        }

        if (!context.axes.empty())
        {
            ordered_json axes = ordered_json::array();
            std::unordered_set<std::string> names;
            for (const auto& axis : context.axes)
            {
                if (!validate_name("axis", axis.name, names, error)) return {.error = std::move(error)};
                if (axis.bindings.empty())
                    return {.error = "input axis '" + axis.name + "' requires at least one binding"};
                ordered_json axis_json = ordered_json::object();
                axis_json["name"] = axis.name;
                ordered_json bindings = ordered_json::array();
                for (const auto& entry : axis.bindings)
                {
                    if (!std::isfinite(entry.contribution))
                        return {.error = "input axis '" + axis.name + "' binding contribution must be finite"};
                    auto serialized = binding_to_json(entry.binding, error);
                    if (!serialized) return {.error = "input axis '" + axis.name + "': " + std::move(error)};
                    (*serialized)["contribution"] = entry.contribution;
                    bindings.push_back(std::move(*serialized));
                }
                axis_json["bindings"] = std::move(bindings);
                axes.push_back(std::move(axis_json));
            }
            context_json["axes"] = std::move(axes);
        }

        if (!context.axes2d.empty())
        {
            ordered_json axes = ordered_json::array();
            std::unordered_set<std::string> names;
            for (const auto& axis : context.axes2d)
            {
                if (!validate_name("2D axis", axis.name, names, error)) return {.error = std::move(error)};
                if (axis.bindings.empty())
                    return {.error = "input 2D axis '" + axis.name + "' requires at least one binding"};
                ordered_json axis_json = ordered_json::object();
                axis_json["name"] = axis.name;
                ordered_json bindings = ordered_json::array();
                for (const auto& entry : axis.bindings)
                {
                    if (!std::isfinite(entry.contribution[0]) || !std::isfinite(entry.contribution[1]))
                        return {.error = "input 2D axis '" + axis.name + "' binding contribution must be finite"};
                    auto serialized = binding_to_json(entry.binding, error);
                    if (!serialized) return {.error = "input 2D axis '" + axis.name + "': " + std::move(error)};
                    (*serialized)["contribution"] = ordered_json::array({entry.contribution[0], entry.contribution[1]});
                    bindings.push_back(std::move(*serialized));
                }
                axis_json["bindings"] = std::move(bindings);
                axes.push_back(std::move(axis_json));
            }
            context_json["axes2d"] = std::move(axes);
        }

        contexts.push_back(std::move(context_json));
    }
    root["contexts"] = std::move(contexts);

    std::error_code filesystem_error;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error)
        return {.error = "input config directory could not be created: " + filesystem_error.message()};

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) return {.error = "input config could not be written: " + path.generic_string()};
    const std::string serialized = root.dump(2) + '\n';
    stream.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
    if (!stream) return {.error = "input config write failed: " + path.generic_string()};
    return {.succeeded = true};
}

} // namespace arc::project

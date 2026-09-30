#include <arc/input/gamepad.h>
#include <arc/input/touch.h>
#include <arc/project/input_config.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

namespace arc::project
{
namespace
{

std::string normalized_token(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char ch) { return ch == '_' || ch == '-'; }),
                value.end());
    return value;
}

std::optional<std::uint32_t> parse_version_value(const nlohmann::json& value)
{
    if (value.is_number_unsigned())
    {
        const auto raw = value.get<std::uint64_t>();
        if (raw <= std::numeric_limits<std::uint32_t>::max()) return static_cast<std::uint32_t>(raw);
        return std::nullopt;
    }
    if (value.is_number_integer())
    {
        const auto raw = value.get<std::int64_t>();
        if (raw >= 0 && static_cast<std::uint64_t>(raw) <= std::numeric_limits<std::uint32_t>::max())
            return static_cast<std::uint32_t>(raw);
    }
    return std::nullopt;
}

std::optional<std::uint32_t> config_version_from_json(const nlohmann::json& root, std::string& error)
{
    std::optional<std::uint32_t> version;
    std::optional<std::uint32_t> legacy_version;
    if (root.contains("version"))
    {
        version = parse_version_value(root.at("version"));
        if (!version)
        {
            error = "input config version must be an unsigned integer";
            return std::nullopt;
        }
    }
    if (root.contains("formatVersion"))
    {
        legacy_version = parse_version_value(root.at("formatVersion"));
        if (!legacy_version)
        {
            error = "input config formatVersion must be an unsigned integer";
            return std::nullopt;
        }
    }
    if (version && legacy_version && *version != *legacy_version)
    {
        error = "input config version and legacy formatVersion disagree";
        return std::nullopt;
    }
    return version.value_or(legacy_version.value_or(input_config_version));
}

bool finite_number(const nlohmann::json& value, float& result)
{
    if (!value.is_number()) return false;
    result = value.get<float>();
    return std::isfinite(result);
}

std::optional<input::key> key_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token.size() == 1)
    {
        const char value = token.front();
        if (value >= 'a' && value <= 'z')
            return static_cast<input::key>(static_cast<unsigned>(input::key::a) + static_cast<unsigned>(value - 'a'));
        if (value >= '0' && value <= '9')
            return static_cast<input::key>(static_cast<unsigned>(input::key::num0) +
                                           static_cast<unsigned>(value - '0'));
    }

    const std::pair<std::string_view, input::key> named[] = {
        {"escape", input::key::escape},
        {"esc", input::key::escape},
        {"space", input::key::space},
        {"spacebar", input::key::space},
        {"enter", input::key::enter},
        {"return", input::key::enter},
        {"tab", input::key::tab},
        {"backspace", input::key::backspace},
        {"leftshift", input::key::left_shift},
        {"rightshift", input::key::right_shift},
        {"shift", input::key::left_shift},
        {"leftcontrol", input::key::left_control},
        {"leftctrl", input::key::left_control},
        {"rightcontrol", input::key::right_control},
        {"rightctrl", input::key::right_control},
        {"control", input::key::left_control},
        {"ctrl", input::key::left_control},
        {"leftalt", input::key::left_alt},
        {"rightalt", input::key::right_alt},
        {"alt", input::key::left_alt},
        {"left", input::key::left},
        {"arrowleft", input::key::left},
        {"right", input::key::right},
        {"arrowright", input::key::right},
        {"up", input::key::up},
        {"arrowup", input::key::up},
        {"down", input::key::down},
        {"arrowdown", input::key::down},
        {"insert", input::key::insert},
        {"delete", input::key::delete_key},
        {"home", input::key::home},
        {"end", input::key::end},
        {"pageup", input::key::page_up},
        {"pagedown", input::key::page_down},
        {"f1", input::key::f1},
        {"f2", input::key::f2},
        {"f3", input::key::f3},
        {"f4", input::key::f4},
        {"f5", input::key::f5},
        {"f6", input::key::f6},
        {"f7", input::key::f7},
        {"f8", input::key::f8},
        {"f9", input::key::f9},
        {"f10", input::key::f10},
        {"f11", input::key::f11},
        {"f12", input::key::f12},
    };
    const auto found =
        std::find_if(std::begin(named), std::end(named), [&token](const auto& entry) { return entry.first == token; });
    return found == std::end(named) ? std::nullopt : std::optional<input::key>{found->second};
}

std::optional<input::mouse_button> mouse_button_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token == "left") return input::mouse_button::left;
    if (token == "right") return input::mouse_button::right;
    if (token == "middle") return input::mouse_button::middle;
    if (token == "x1" || token == "button4") return input::mouse_button::x1;
    if (token == "x2" || token == "button5") return input::mouse_button::x2;
    return std::nullopt;
}

std::optional<input::mouse_axis> mouse_axis_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token == "positionx") return input::mouse_axis::position_x;
    if (token == "positiony") return input::mouse_axis::position_y;
    if (token == "deltax") return input::mouse_axis::delta_x;
    if (token == "deltay") return input::mouse_axis::delta_y;
    if (token == "wheelx") return input::mouse_axis::wheel_x;
    if (token == "wheely" || token == "wheel") return input::mouse_axis::wheel_y;
    return std::nullopt;
}

std::optional<input::gamepad_button> gamepad_button_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    const std::pair<std::string_view, input::gamepad_button> named[] = {
        {"south", input::gamepad_button::south},
        {"east", input::gamepad_button::east},
        {"west", input::gamepad_button::west},
        {"north", input::gamepad_button::north},
        {"auxiliary1", input::gamepad_button::auxiliary_1},
        {"auxiliary2", input::gamepad_button::auxiliary_2},
        {"dpadup", input::gamepad_button::dpad_up},
        {"dpaddown", input::gamepad_button::dpad_down},
        {"dpadleft", input::gamepad_button::dpad_left},
        {"dpadright", input::gamepad_button::dpad_right},
        {"leftshoulder", input::gamepad_button::left_shoulder},
        {"rightshoulder", input::gamepad_button::right_shoulder},
        {"lefttriggerbutton", input::gamepad_button::left_trigger_button},
        {"righttriggerbutton", input::gamepad_button::right_trigger_button},
        {"leftstick", input::gamepad_button::left_stick},
        {"rightstick", input::gamepad_button::right_stick},
        {"leftstickup", input::gamepad_button::left_stick_up},
        {"leftstickdown", input::gamepad_button::left_stick_down},
        {"leftstickleft", input::gamepad_button::left_stick_left},
        {"leftstickright", input::gamepad_button::left_stick_right},
        {"rightstickup", input::gamepad_button::right_stick_up},
        {"rightstickdown", input::gamepad_button::right_stick_down},
        {"rightstickleft", input::gamepad_button::right_stick_left},
        {"rightstickright", input::gamepad_button::right_stick_right},
        {"paddleleft1", input::gamepad_button::paddle_left_1},
        {"paddleleft2", input::gamepad_button::paddle_left_2},
        {"paddleright1", input::gamepad_button::paddle_right_1},
        {"paddleright2", input::gamepad_button::paddle_right_2},
        {"view", input::gamepad_button::view},
        {"menu", input::gamepad_button::menu},
        {"guide", input::gamepad_button::guide},
        {"share", input::gamepad_button::share},
    };
    const auto found =
        std::find_if(std::begin(named), std::end(named), [&token](const auto& entry) { return entry.first == token; });
    return found == std::end(named) ? std::nullopt : std::optional<input::gamepad_button>{found->second};
}

std::optional<input::gamepad_axis> gamepad_axis_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token == "leftx" || token == "leftstickx") return input::gamepad_axis::left_x;
    if (token == "lefty" || token == "leftsticky") return input::gamepad_axis::left_y;
    if (token == "rightx" || token == "rightstickx") return input::gamepad_axis::right_x;
    if (token == "righty" || token == "rightsticky") return input::gamepad_axis::right_y;
    if (token == "lefttrigger") return input::gamepad_axis::left_trigger;
    if (token == "righttrigger") return input::gamepad_axis::right_trigger;
    return std::nullopt;
}

std::optional<input::sensor_axis> sensor_axis_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token == "gyroscopex" || token == "gyrox") return input::sensor_axis::gyroscope_x;
    if (token == "gyroscopey" || token == "gyroy") return input::sensor_axis::gyroscope_y;
    if (token == "gyroscopez" || token == "gyroz") return input::sensor_axis::gyroscope_z;
    if (token == "accelerometerx" || token == "accelx") return input::sensor_axis::accelerometer_x;
    if (token == "accelerometery" || token == "accely") return input::sensor_axis::accelerometer_y;
    if (token == "accelerometerz" || token == "accelz") return input::sensor_axis::accelerometer_z;
    return std::nullopt;
}

std::optional<input::touch_control> touch_control_from_token(std::string token)
{
    token = normalized_token(std::move(token));
    if (token == "primarydown" || token == "touchprimarydown") return input::touch_control::primary_down;
    if (token == "primaryx" || token == "touchprimaryx") return input::touch_control::primary_x;
    if (token == "primaryy" || token == "touchprimaryy") return input::touch_control::primary_y;
    if (token == "primarypressure" || token == "touchprimarypressure") return input::touch_control::primary_pressure;
    return std::nullopt;
}

bool parse_processor_array(const nlohmann::json& processors, std::vector<input::input_processor>& destination,
                           std::string& error)
{
    if (!processors.is_array())
    {
        error = "input processors must be an array";
        return false;
    }
    for (const auto& value : processors)
    {
        if (!value.is_object() || !value.contains("type") || !value.at("type").is_string())
        {
            error = "input processor requires a string type";
            return false;
        }
        const std::string type = normalized_token(value.at("type").get<std::string>());
        input::input_processor processor;
        if (type == "scale")
        {
            processor.type = input::input_processor_type::scale;
            if (value.contains("value") && !finite_number(value.at("value"), processor.value))
            {
                error = "input scale processor value must be a finite number";
                return false;
            }
        }
        else if (type == "invert")
        {
            processor.type = input::input_processor_type::invert;
        }
        else if (type == "clamp")
        {
            processor.type = input::input_processor_type::clamp;
            processor.value = -1.0f;
            processor.secondary = 1.0f;
            if (value.contains("minimum") && !finite_number(value.at("minimum"), processor.value))
            {
                error = "input clamp processor minimum must be a finite number";
                return false;
            }
            if (value.contains("maximum") && !finite_number(value.at("maximum"), processor.secondary))
            {
                error = "input clamp processor maximum must be a finite number";
                return false;
            }
            if (processor.value > processor.secondary)
            {
                error = "input clamp processor minimum cannot exceed maximum";
                return false;
            }
        }
        else
        {
            error = "unknown input processor type '" + value.at("type").get<std::string>() + "'";
            return false;
        }
        destination.push_back(processor);
    }
    return true;
}

bool parse_processors(const nlohmann::json& source, input::input_binding& binding, std::string& error)
{
    if (source.contains("processors") && !parse_processor_array(source.at("processors"), binding.processors, error))
        return false;
    if (source.contains("compositeProcessors") &&
        !parse_processor_array(source.at("compositeProcessors"), binding.composite_processors, error))
        return false;
    return true;
}

std::optional<input::input_binding> parse_binding(const nlohmann::json& source, std::string& error)
{
    if (!source.is_object() || !source.contains("device") || !source.at("device").is_string() ||
        !source.contains("control") || !source.at("control").is_string())
    {
        error = "input binding requires string device and control fields";
        return std::nullopt;
    }

    input::input_binding result;
    const std::string device = normalized_token(source.at("device").get<std::string>());
    const std::string control = source.at("control").get<std::string>();
    if (device == "keyboard")
    {
        const auto key = key_from_token(control);
        if (!key)
        {
            error = "unknown keyboard control '" + control + "'";
            return std::nullopt;
        }
        result.device = input::input_device_type::keyboard;
        result.control = input::make_key_control(*key);
    }
    else if (device == "mouse")
    {
        if (const auto button = mouse_button_from_token(control))
        {
            result.device = input::input_device_type::mouse;
            result.control = input::make_mouse_button_control(*button);
        }
        else if (const auto axis = mouse_axis_from_token(control))
        {
            result.device = input::input_device_type::mouse;
            result.control = input::make_mouse_axis_control(*axis);
        }
        else
        {
            error = "unknown mouse control '" + control + "'";
            return std::nullopt;
        }
    }
    else if (device == "gamepad" || device == "controller")
    {
        result.device = input::input_device_type::gamepad;
        if (const auto button = gamepad_button_from_token(control))
            result.control = input::make_gamepad_button_control(*button);
        else if (const auto axis = gamepad_axis_from_token(control))
            result.control = input::make_gamepad_axis_control(*axis);
        else if (const auto sensor = sensor_axis_from_token(control))
            result.control = input::make_sensor_axis_control(*sensor);
        else if (const auto touch = touch_control_from_token(control))
            result.control = input::make_touch_control(*touch);
        else
        {
            error = "unknown gamepad control '" + control + "'";
            return std::nullopt;
        }
    }
    else if (device == "motion" || device == "sensor" || device == "motioncontroller")
    {
        const auto axis = sensor_axis_from_token(control);
        if (!axis)
        {
            error = "unknown motion sensor control '" + control + "'";
            return std::nullopt;
        }
        result.device = input::input_device_type::motion_controller;
        result.control = input::make_sensor_axis_control(*axis);
    }
    else if (device == "touch" || device == "touchscreen")
    {
        const auto touch = touch_control_from_token(control);
        if (!touch)
        {
            error = "unknown touch control '" + control + "'";
            return std::nullopt;
        }
        result.device = input::input_device_type::touch;
        result.control = input::make_touch_control(*touch);
    }
    else
    {
        error = "unsupported input binding device '" + source.at("device").get<std::string>() + "'";
        return std::nullopt;
    }

    if (!parse_processors(source, result, error)) return std::nullopt;
    if (source.contains("modifiers"))
    {
        const auto& modifiers = source.at("modifiers");
        if (!modifiers.is_array() || modifiers.empty())
        {
            error = "input binding modifiers must be a non-empty array";
            return std::nullopt;
        }
        result.modifiers.reserve(modifiers.size());
        for (const auto& modifier_json : modifiers)
        {
            std::string modifier_error;
            auto modifier = parse_binding(modifier_json, modifier_error);
            if (!modifier)
            {
                error = "input binding modifier: " + std::move(modifier_error);
                return std::nullopt;
            }
            result.modifiers.push_back(std::move(*modifier));
        }
    }
    return result;
}

bool parse_priority(const nlohmann::json& source, int& priority, std::string& error)
{
    if (!source.contains("priority")) return true;
    const auto& value = source.at("priority");
    if (!value.is_number_integer())
    {
        error = "input context priority must be an integer";
        return false;
    }
    const auto raw = value.get<std::int64_t>();
    if (raw < std::numeric_limits<int>::min() || raw > std::numeric_limits<int>::max())
    {
        error = "input context priority is out of range";
        return false;
    }
    priority = static_cast<int>(raw);
    return true;
}

bool parse_scalar_contribution(const nlohmann::json& source, float& contribution, std::string& error)
{
    if (!source.contains("contribution")) return true;
    if (!finite_number(source.at("contribution"), contribution))
    {
        error = "input axis binding contribution must be a finite number";
        return false;
    }
    return true;
}

bool parse_axis2d_contribution(const nlohmann::json& source, math::vector2f& contribution, std::string& error)
{
    if (!source.contains("contribution") || !source.at("contribution").is_array() ||
        source.at("contribution").size() != 2)
    {
        error = "input 2D axis binding contribution must be [x, y]";
        return false;
    }
    float x{};
    float y{};
    if (!finite_number(source.at("contribution").at(0), x) || !finite_number(source.at("contribution").at(1), y))
    {
        error = "input 2D axis binding contribution values must be finite numbers";
        return false;
    }
    contribution = {x, y};
    return true;
}

} // namespace

input_config_load_result load_input_config(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {.error = "input config could not be read: " + path.generic_string()};
    try
    {
        nlohmann::json root;
        stream >> root;
        if (!root.is_object()) return {.error = "input config root must be an object"};

        std::string version_error;
        const auto version = config_version_from_json(root, version_error);
        if (!version) return {.error = std::move(version_error)};
        if (*version != input_config_version)
            return {.error = "unsupported input config version " + std::to_string(*version)};

        input_config config;
        config.version = *version;
        if (!root.contains("contexts")) return {.succeeded = true, .config = std::move(config)};
        if (!root.at("contexts").is_array()) return {.error = "input config contexts must be an array"};

        std::unordered_set<std::string> context_names;
        for (const auto& context_json : root.at("contexts"))
        {
            if (!context_json.is_object() || !context_json.contains("name") || !context_json.at("name").is_string())
                return {.error = "input context requires a string name"};
            input_context_config context;
            context.name = context_json.at("name").get<std::string>();
            if (context.name.empty()) return {.error = "input context name cannot be empty"};
            if (!context_names.emplace(context.name).second)
                return {.error = "duplicate input context '" + context.name + "'"};
            std::string priority_error;
            if (!parse_priority(context_json, context.priority, priority_error))
                return {.error = std::move(priority_error)};
            if (context_json.contains("enabled"))
            {
                if (!context_json.at("enabled").is_boolean())
                    return {.error = "input context enabled must be a boolean"};
                context.enabled = context_json.at("enabled").get<bool>();
            }

            if (context_json.contains("actions"))
            {
                if (!context_json.at("actions").is_array()) return {.error = "input context actions must be an array"};
                std::unordered_set<std::string> action_names;
                for (const auto& action_json : context_json.at("actions"))
                {
                    if (!action_json.is_object() || !action_json.contains("name") ||
                        !action_json.at("name").is_string())
                        return {.error = "input action requires a string name"};
                    input_action_config action;
                    action.name = action_json.at("name").get<std::string>();
                    if (action.name.empty()) return {.error = "input action name cannot be empty"};
                    if (!action_names.emplace(action.name).second)
                        return {.error =
                                    "duplicate input action '" + action.name + "' in context '" + context.name + "'"};
                    if (!action_json.contains("bindings") || !action_json.at("bindings").is_array() ||
                        action_json.at("bindings").empty())
                        return {.error = "input action '" + action.name + "' requires at least one binding"};
                    for (const auto& binding_json : action_json.at("bindings"))
                    {
                        std::string error;
                        auto binding = parse_binding(binding_json, error);
                        if (!binding) return {.error = "input action '" + action.name + "': " + std::move(error)};
                        action.bindings.push_back(std::move(*binding));
                    }
                    context.actions.push_back(std::move(action));
                }
            }
            if (context_json.contains("axes"))
            {
                if (!context_json.at("axes").is_array()) return {.error = "input context axes must be an array"};
                std::unordered_set<std::string> names;
                for (const auto& axis_json : context_json.at("axes"))
                {
                    if (!axis_json.is_object() || !axis_json.contains("name") || !axis_json.at("name").is_string())
                        return {.error = "input axis requires a string name"};
                    input_axis_config axis;
                    axis.name = axis_json.at("name").get<std::string>();
                    if (axis.name.empty()) return {.error = "input axis name cannot be empty"};
                    if (!names.emplace(axis.name).second) return {.error = "duplicate input axis '" + axis.name + "'"};
                    if (!axis_json.contains("bindings") || !axis_json.at("bindings").is_array() ||
                        axis_json.at("bindings").empty())
                        return {.error = "input axis '" + axis.name + "' requires at least one binding"};
                    for (const auto& binding_json : axis_json.at("bindings"))
                    {
                        std::string error;
                        auto binding = parse_binding(binding_json, error);
                        if (!binding) return {.error = "input axis '" + axis.name + "': " + std::move(error)};
                        float contribution = 1.0f;
                        if (!parse_scalar_contribution(binding_json, contribution, error))
                            return {.error = "input axis '" + axis.name + "': " + std::move(error)};
                        axis.bindings.push_back({.binding = std::move(*binding), .contribution = contribution});
                    }
                    context.axes.push_back(std::move(axis));
                }
            }
            if (context_json.contains("axes2d"))
            {
                if (!context_json.at("axes2d").is_array()) return {.error = "input context axes2d must be an array"};
                std::unordered_set<std::string> names;
                for (const auto& axis_json : context_json.at("axes2d"))
                {
                    if (!axis_json.is_object() || !axis_json.contains("name") || !axis_json.at("name").is_string())
                        return {.error = "input 2D axis requires a string name"};
                    input_axis2d_config axis;
                    axis.name = axis_json.at("name").get<std::string>();
                    if (axis.name.empty()) return {.error = "input 2D axis name cannot be empty"};
                    if (!names.emplace(axis.name).second)
                        return {.error = "duplicate input 2D axis '" + axis.name + "'"};
                    if (!axis_json.contains("bindings") || !axis_json.at("bindings").is_array() ||
                        axis_json.at("bindings").empty())
                        return {.error = "input 2D axis '" + axis.name + "' requires at least one binding"};
                    for (const auto& binding_json : axis_json.at("bindings"))
                    {
                        std::string error;
                        auto binding = parse_binding(binding_json, error);
                        if (!binding) return {.error = "input 2D axis '" + axis.name + "': " + std::move(error)};
                        math::vector2f contribution{};
                        if (!parse_axis2d_contribution(binding_json, contribution, error))
                            return {.error = "input 2D axis '" + axis.name + "': " + std::move(error)};
                        axis.bindings.push_back({.binding = std::move(*binding), .contribution = contribution});
                    }
                    context.axes2d.push_back(std::move(axis));
                }
            }
            config.contexts.push_back(std::move(context));
        }
        return {.succeeded = true, .config = std::move(config)};
    }
    catch (const std::exception& error)
    {
        return {.error = "input config parse failed: " + std::string(error.what())};
    }
}

input_config_apply_result apply_input_config(const input_config& config, input::input_system& system,
                                             input::player_id player_id)
{
    if (config.version != input_config_version)
        return {.error = "unsupported input config version " + std::to_string(config.version)};
    auto& player = system.player(player_id);
    for (const auto& context : config.contexts)
    {
        if (context.name.empty()) return {.error = "input context name cannot be empty"};
        player.add_context(context.name, context.priority, context.enabled);
        for (const auto& action : context.actions)
        {
            if (action.name.empty() || action.bindings.empty())
                return {.error = "input action requires a name and at least one binding"};
            for (const auto& binding : action.bindings)
                player.bind_action(context.name, action.name, binding);
        }
        for (const auto& axis : context.axes)
        {
            if (axis.name.empty() || axis.bindings.empty())
                return {.error = "input axis requires a name and at least one binding"};
            for (const auto& binding : axis.bindings)
                player.bind_axis(context.name, axis.name, binding.binding, binding.contribution);
        }
        for (const auto& axis : context.axes2d)
        {
            if (axis.name.empty() || axis.bindings.empty())
                return {.error = "input 2D axis requires a name and at least one binding"};
            for (const auto& binding : axis.bindings)
                player.bind_axis2d(context.name, axis.name, binding.binding, binding.contribution);
        }
    }
    return {.succeeded = true};
}

std::vector<std::string> input_action_names(const input_config& config)
{
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    for (const auto& context : config.contexts)
        for (const auto& action : context.actions)
            if (seen.emplace(action.name).second) result.push_back(action.name);
    return result;
}

} // namespace arc::project

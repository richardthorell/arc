#include <arc/project/input_config.h>

#include <algorithm>
#include <cctype>
#include <fstream>
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

bool parse_processors(const nlohmann::json& source, input::input_binding& binding, std::string& error)
{
    if (!source.contains("processors")) return true;
    const auto& processors = source.at("processors");
    if (!processors.is_array())
    {
        error = "input binding processors must be an array";
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
            processor.value = value.value("value", 1.0f);
        }
        else if (type == "invert")
        {
            processor.type = input::input_processor_type::invert;
        }
        else if (type == "clamp")
        {
            processor.type = input::input_processor_type::clamp;
            processor.value = value.value("minimum", -1.0f);
            processor.secondary = value.value("maximum", 1.0f);
        }
        else
        {
            error = "unknown input processor type '" + value.at("type").get<std::string>() + "'";
            return false;
        }
        binding.processors.push_back(processor);
    }
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
        const auto button = mouse_button_from_token(control);
        if (!button)
        {
            error = "unknown mouse button control '" + control + "'";
            return std::nullopt;
        }
        result.device = input::input_device_type::mouse;
        result.control = input::make_mouse_button_control(*button);
    }
    else
    {
        error = "unsupported action binding device '" + source.at("device").get<std::string>() + "'";
        return std::nullopt;
    }

    if (!parse_processors(source, result, error)) return std::nullopt;
    return result;
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

        input_config config;
        config.version = root.value("version", input_config_version);
        if (config.version != input_config_version)
            return {.error = "unsupported input config version " + std::to_string(config.version)};
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
            context.priority = context_json.value("priority", 0);
            context.enabled = context_json.value("enabled", true);

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

#include <arc/project/input_rebinding.h>

#include <algorithm>
#include <cmath>
#include <system_error>
#include <utility>

namespace arc::project
{
namespace
{

const input_context_config* find_context(const input_config& config, std::string_view name)
{
    const auto found = std::find_if(config.contexts.begin(), config.contexts.end(),
                                    [name](const input_context_config& context) { return context.name == name; });
    return found == config.contexts.end() ? nullptr : &*found;
}

input_context_config* find_context(input_config& config, std::string_view name)
{
    const auto found = std::find_if(config.contexts.begin(), config.contexts.end(),
                                    [name](const input_context_config& context) { return context.name == name; });
    return found == config.contexts.end() ? nullptr : &*found;
}

input_context_config& ensure_context(input_config& config, std::string_view name)
{
    if (auto* existing = find_context(config, name)) return *existing;
    input_context_config context{};
    context.name = std::string(name);
    config.contexts.push_back(std::move(context));
    return config.contexts.back();
}

std::optional<input_mapping_view> mapping_view(const input_context_config& context, const input_mapping_target& target)
{
    if (context.name != target.context) return std::nullopt;

    input_mapping_view view{.target = target};
    switch (target.kind)
    {
        case input_mapping_kind::action:
        {
            const auto found =
                std::find_if(context.actions.begin(), context.actions.end(),
                             [&target](const input_action_config& action) { return action.name == target.name; });
            if (found == context.actions.end()) return std::nullopt;
            view.bindings.reserve(found->bindings.size());
            for (const auto& binding : found->bindings)
                view.bindings.push_back({.binding = binding});
            return view;
        }
        case input_mapping_kind::axis:
        {
            const auto found =
                std::find_if(context.axes.begin(), context.axes.end(),
                             [&target](const input_axis_config& axis) { return axis.name == target.name; });
            if (found == context.axes.end()) return std::nullopt;
            view.bindings.reserve(found->bindings.size());
            for (const auto& binding : found->bindings)
                view.bindings.push_back({.binding = binding.binding, .contribution = binding.contribution});
            return view;
        }
        case input_mapping_kind::axis2d:
        {
            const auto found =
                std::find_if(context.axes2d.begin(), context.axes2d.end(),
                             [&target](const input_axis2d_config& axis) { return axis.name == target.name; });
            if (found == context.axes2d.end()) return std::nullopt;
            view.bindings.reserve(found->bindings.size());
            for (const auto& binding : found->bindings)
                view.bindings.push_back({.binding = binding.binding, .contribution2d = binding.contribution});
            return view;
        }
    }
    return std::nullopt;
}

std::optional<input_mapping_view> mapping_view(const input_config& config, const input_mapping_target& target)
{
    const auto* context = find_context(config, target.context);
    return context ? mapping_view(*context, target) : std::nullopt;
}

bool same_processors(const std::vector<input::input_processor>& lhs, const std::vector<input::input_processor>& rhs)
{
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index)
    {
        const auto& left = lhs[index];
        const auto& right = rhs[index];
        if (left.type != right.type || left.value != right.value || left.secondary != right.secondary) return false;
    }
    return true;
}

bool same_binding(const input::input_binding& lhs, const input::input_binding& rhs)
{
    if (lhs.device != rhs.device || lhs.control != rhs.control || !same_processors(lhs.processors, rhs.processors) ||
        lhs.modifiers.size() != rhs.modifiers.size() ||
        !same_processors(lhs.composite_processors, rhs.composite_processors))
        return false;
    for (std::size_t index = 0; index < lhs.modifiers.size(); ++index)
    {
        if (!same_binding(lhs.modifiers[index], rhs.modifiers[index])) return false;
    }
    return true;
}

bool same_mapping_binding(const input_mapping_binding& lhs, const input_mapping_binding& rhs, input_mapping_kind kind)
{
    if (!same_binding(lhs.binding, rhs.binding)) return false;
    if (kind == input_mapping_kind::axis) return lhs.contribution == rhs.contribution;
    if (kind == input_mapping_kind::axis2d)
        return lhs.contribution2d[0] == rhs.contribution2d[0] && lhs.contribution2d[1] == rhs.contribution2d[1];
    return true;
}

bool same_mapping(const input_mapping_view& lhs, const input_mapping_view& rhs)
{
    if (lhs.target.kind != rhs.target.kind || lhs.bindings.size() != rhs.bindings.size()) return false;
    for (std::size_t index = 0; index < lhs.bindings.size(); ++index)
    {
        if (!same_mapping_binding(lhs.bindings[index], rhs.bindings[index], lhs.target.kind)) return false;
    }
    return true;
}

void set_override_mapping(input_config& overrides, const input_mapping_target& target,
                          const std::vector<input_mapping_binding>& bindings)
{
    auto& context = ensure_context(overrides, target.context);
    switch (target.kind)
    {
        case input_mapping_kind::action:
        {
            auto found =
                std::find_if(context.actions.begin(), context.actions.end(),
                             [&target](const input_action_config& action) { return action.name == target.name; });
            if (found == context.actions.end())
            {
                context.actions.push_back({.name = target.name});
                found = std::prev(context.actions.end());
            }
            found->bindings.clear();
            found->bindings.reserve(bindings.size());
            for (const auto& binding : bindings)
                found->bindings.push_back(binding.binding);
            break;
        }
        case input_mapping_kind::axis:
        {
            auto found = std::find_if(context.axes.begin(), context.axes.end(),
                                      [&target](const input_axis_config& axis) { return axis.name == target.name; });
            if (found == context.axes.end())
            {
                context.axes.push_back({.name = target.name});
                found = std::prev(context.axes.end());
            }
            found->bindings.clear();
            found->bindings.reserve(bindings.size());
            for (const auto& binding : bindings)
                found->bindings.push_back({.binding = binding.binding, .contribution = binding.contribution});
            break;
        }
        case input_mapping_kind::axis2d:
        {
            auto found = std::find_if(context.axes2d.begin(), context.axes2d.end(),
                                      [&target](const input_axis2d_config& axis) { return axis.name == target.name; });
            if (found == context.axes2d.end())
            {
                context.axes2d.push_back({.name = target.name});
                found = std::prev(context.axes2d.end());
            }
            found->bindings.clear();
            found->bindings.reserve(bindings.size());
            for (const auto& binding : bindings)
                found->bindings.push_back({.binding = binding.binding, .contribution = binding.contribution2d});
            break;
        }
    }
}

bool remove_override_mapping(input_config& overrides, const input_mapping_target& target)
{
    auto* context = find_context(overrides, target.context);
    if (!context) return false;

    bool removed = false;
    switch (target.kind)
    {
        case input_mapping_kind::action:
        {
            const auto end =
                std::remove_if(context->actions.begin(), context->actions.end(),
                               [&target](const input_action_config& action) { return action.name == target.name; });
            removed = end != context->actions.end();
            context->actions.erase(end, context->actions.end());
            break;
        }
        case input_mapping_kind::axis:
        {
            const auto end =
                std::remove_if(context->axes.begin(), context->axes.end(),
                               [&target](const input_axis_config& axis) { return axis.name == target.name; });
            removed = end != context->axes.end();
            context->axes.erase(end, context->axes.end());
            break;
        }
        case input_mapping_kind::axis2d:
        {
            const auto end =
                std::remove_if(context->axes2d.begin(), context->axes2d.end(),
                               [&target](const input_axis2d_config& axis) { return axis.name == target.name; });
            removed = end != context->axes2d.end();
            context->axes2d.erase(end, context->axes2d.end());
            break;
        }
    }

    if (context->actions.empty() && context->axes.empty() && context->axes2d.empty())
    {
        const auto end =
            std::remove_if(overrides.contexts.begin(), overrides.contexts.end(),
                           [&target](const input_context_config& value) { return value.name == target.context; });
        overrides.contexts.erase(end, overrides.contexts.end());
    }
    return removed;
}

void apply_overrides(input_config& effective, const input_config& overrides)
{
    for (const auto& override_context : overrides.contexts)
    {
        auto* context = find_context(effective, override_context.name);
        if (!context) continue;

        for (const auto& override_action : override_context.actions)
        {
            const auto found = std::find_if(context->actions.begin(), context->actions.end(),
                                            [&override_action](const input_action_config& action)
                                            { return action.name == override_action.name; });
            if (found != context->actions.end()) found->bindings = override_action.bindings;
        }
        for (const auto& override_axis : override_context.axes)
        {
            const auto found =
                std::find_if(context->axes.begin(), context->axes.end(), [&override_axis](const input_axis_config& axis)
                             { return axis.name == override_axis.name; });
            if (found != context->axes.end()) found->bindings = override_axis.bindings;
        }
        for (const auto& override_axis : override_context.axes2d)
        {
            const auto found = std::find_if(context->axes2d.begin(), context->axes2d.end(),
                                            [&override_axis](const input_axis2d_config& axis)
                                            { return axis.name == override_axis.name; });
            if (found != context->axes2d.end()) found->bindings = override_axis.bindings;
        }
    }
}

std::vector<input_mapping_target> override_targets(const input_config& overrides)
{
    std::vector<input_mapping_target> result;
    for (const auto& context : overrides.contexts)
    {
        for (const auto& action : context.actions)
            result.push_back({.context = context.name, .name = action.name, .kind = input_mapping_kind::action});
        for (const auto& axis : context.axes)
            result.push_back({.context = context.name, .name = axis.name, .kind = input_mapping_kind::axis});
        for (const auto& axis : context.axes2d)
            result.push_back({.context = context.name, .name = axis.name, .kind = input_mapping_kind::axis2d});
    }
    return result;
}

bool same_physical_control(const input::input_binding& lhs, const input::input_binding& rhs)
{
    return lhs.device == rhs.device && lhs.control == rhs.control;
}

void append_unique(std::vector<input_mapping_target>& targets, input_mapping_target target)
{
    if (std::find(targets.begin(), targets.end(), target) == targets.end()) targets.push_back(std::move(target));
}

input::input_binding inert_binding()
{
    return {.device = input::input_device_type::unknown,
            .control = {.kind = input::input_control_kind::unknown, .code = 0}};
}

} // namespace

void input_rebind_capture::begin(const input::input_system& system, input::player_id player,
                                 input_rebind_capture_filter filter)
{
    if (!std::isfinite(filter.actuation_threshold) || filter.actuation_threshold <= 0.0f)
        filter.actuation_threshold = 0.5f;
    system_ = &system;
    player_ = player;
    filter_ = filter;
    active_ = true;
    canceled_ = false;
}

void input_rebind_capture::cancel() noexcept
{
    active_ = false;
    canceled_ = true;
}

bool input_rebind_capture::active() const noexcept
{
    return active_;
}

bool input_rebind_capture::canceled() const noexcept
{
    return canceled_;
}

std::optional<input_rebind_capture_result> input_rebind_capture::offer(input::input_device_id device,
                                                                       input::input_control control, float value)
{
    if (!active_ || !system_ || control.kind == input::input_control_kind::unknown || !std::isfinite(value) ||
        std::abs(value) < filter_.actuation_threshold)
        return std::nullopt;

    const auto* descriptor = system_->device(device);
    if (!descriptor || !descriptor->connected()) return std::nullopt;
    if (filter_.device != input::input_device_type::unknown && descriptor->type() != filter_.device)
        return std::nullopt;

    const auto assigned = system_->devices_for_player(player_);
    if (std::find(assigned.begin(), assigned.end(), device) == assigned.end()) return std::nullopt;

    active_ = false;
    canceled_ = false;
    return input_rebind_capture_result{
        .device = device, .binding = {.device = descriptor->type(), .control = control}, .value = value};
}

input_rebinding_profile::input_rebinding_profile(input_config project_defaults, input::player_id player)
    : project_defaults_(std::move(project_defaults)), player_(player)
{
    project_defaults_.version = input_config_version;
    user_overrides_.version = input_config_version;
}

input::player_id input_rebinding_profile::player() const noexcept
{
    return player_;
}

const input_config& input_rebinding_profile::project_defaults() const noexcept
{
    return project_defaults_;
}

const input_config& input_rebinding_profile::user_overrides() const noexcept
{
    return user_overrides_;
}

void input_rebinding_profile::set_user_overrides(input_config overrides)
{
    if (installed_system_)
    {
        auto& runtime_player = installed_system_->player(player_);
        for (const auto& published : published_)
            runtime_player.set_context_enabled(published.context_name, false);
    }
    published_.clear();

    overrides.version = input_config_version;
    user_overrides_ = std::move(overrides);

    if (installed_system_)
    {
        for (const auto& target : override_targets(user_overrides_))
            publish_override(target);
    }
}

input_config input_rebinding_profile::effective_config() const
{
    input_config result = project_defaults_;
    apply_overrides(result, user_overrides_);
    return result;
}

std::optional<input_mapping_view> input_rebinding_profile::default_mapping(const input_mapping_target& target) const
{
    return mapping_view(project_defaults_, target);
}

std::optional<input_mapping_view> input_rebinding_profile::effective_mapping(const input_mapping_target& target) const
{
    if (!mapping_view(project_defaults_, target)) return std::nullopt;
    if (const auto override = mapping_view(user_overrides_, target)) return override;
    return mapping_view(project_defaults_, target);
}

bool input_rebinding_profile::replace_bindings(const input_mapping_target& target,
                                               std::vector<input_mapping_binding> bindings)
{
    const auto defaults = default_mapping(target);
    if (!defaults) return false;

    input_mapping_view replacement{.target = target, .bindings = bindings};
    if (same_mapping(*defaults, replacement))
        remove_override_mapping(user_overrides_, target);
    else
        set_override_mapping(user_overrides_, target, bindings);

    publish_override(target);
    return true;
}

bool input_rebinding_profile::add_binding(const input_mapping_target& target, input_mapping_binding binding)
{
    auto effective = effective_mapping(target);
    if (!effective) return false;
    effective->bindings.push_back(std::move(binding));
    return replace_bindings(target, std::move(effective->bindings));
}

bool input_rebinding_profile::remove_binding(const input_mapping_target& target, std::size_t index)
{
    auto effective = effective_mapping(target);
    if (!effective || index >= effective->bindings.size()) return false;
    effective->bindings.erase(effective->bindings.begin() + static_cast<std::ptrdiff_t>(index));
    return replace_bindings(target, std::move(effective->bindings));
}

bool input_rebinding_profile::reset_binding(const input_mapping_target& target, std::size_t index)
{
    const auto defaults = default_mapping(target);
    auto effective = effective_mapping(target);
    if (!defaults || !effective || index >= effective->bindings.size()) return false;

    if (index < defaults->bindings.size())
        effective->bindings[index] = defaults->bindings[index];
    else
        effective->bindings.erase(effective->bindings.begin() + static_cast<std::ptrdiff_t>(index));

    return replace_bindings(target, std::move(effective->bindings));
}

bool input_rebinding_profile::reset_mapping(const input_mapping_target& target)
{
    if (!remove_override_mapping(user_overrides_, target)) return false;
    publish_override(target);
    return true;
}

bool input_rebinding_profile::reset_context(std::string_view context)
{
    auto found = std::find_if(user_overrides_.contexts.begin(), user_overrides_.contexts.end(),
                              [context](const input_context_config& value) { return value.name == context; });
    if (found == user_overrides_.contexts.end()) return false;

    disable_published_context(context);
    user_overrides_.contexts.erase(found);
    return true;
}

void input_rebinding_profile::reset_all()
{
    if (installed_system_)
    {
        auto& runtime_player = installed_system_->player(player_);
        for (const auto& published : published_)
            runtime_player.set_context_enabled(published.context_name, false);
    }
    published_.clear();
    user_overrides_.contexts.clear();
}

std::vector<input_mapping_target> input_rebinding_profile::conflicts_for(const input::input_binding& binding) const
{
    std::vector<input_mapping_target> result;
    const auto effective = effective_config();
    for (const auto& context : effective.contexts)
    {
        for (const auto& action : context.actions)
        {
            for (const auto& candidate : action.bindings)
            {
                if (same_physical_control(candidate, binding))
                    append_unique(result,
                                  {.context = context.name, .name = action.name, .kind = input_mapping_kind::action});
            }
        }
        for (const auto& axis : context.axes)
        {
            for (const auto& candidate : axis.bindings)
            {
                if (same_physical_control(candidate.binding, binding))
                    append_unique(result,
                                  {.context = context.name, .name = axis.name, .kind = input_mapping_kind::axis});
            }
        }
        for (const auto& axis : context.axes2d)
        {
            for (const auto& candidate : axis.bindings)
            {
                if (same_physical_control(candidate.binding, binding))
                    append_unique(result,
                                  {.context = context.name, .name = axis.name, .kind = input_mapping_kind::axis2d});
            }
        }
    }
    return result;
}

input_config_apply_result input_rebinding_profile::install(input::input_system& system)
{
    if (installed_system_ == &system)
    {
        auto& runtime_player = system.player(player_);
        for (const auto& published : published_)
            runtime_player.set_context_enabled(published.context_name, false);
        published_.clear();
        for (const auto& target : override_targets(user_overrides_))
            publish_override(target);
        return {.succeeded = true};
    }

    if (installed_system_)
    {
        auto& old_player = installed_system_->player(player_);
        for (const auto& published : published_)
            old_player.set_context_enabled(published.context_name, false);
    }
    published_.clear();
    installed_system_ = nullptr;

    auto applied = apply_input_config(project_defaults_, system, player_);
    if (!applied.succeeded) return applied;

    installed_system_ = &system;
    for (const auto& target : override_targets(user_overrides_))
        publish_override(target);
    return applied;
}

void input_rebinding_profile::publish_override(const input_mapping_target& target)
{
    disable_published(target);
    if (!installed_system_) return;

    const auto override = mapping_view(user_overrides_, target);
    if (!override) return;
    const auto* default_context = find_context(project_defaults_, target.context);
    if (!default_context || !default_mapping(target)) return;

    auto& runtime_player = installed_system_->player(player_);
    const std::string shadow_context =
        "__arc_user_override/" + std::to_string(player_) + "/" + std::to_string(++publish_generation_);
    runtime_player.add_context(shadow_context, default_context->priority, default_context->enabled);

    if (override->bindings.empty())
    {
        const auto inert = inert_binding();
        switch (target.kind)
        {
            case input_mapping_kind::action:
                runtime_player.bind_action(shadow_context, target.name, inert);
                break;
            case input_mapping_kind::axis:
                runtime_player.bind_axis(shadow_context, target.name, inert, 0.0f);
                break;
            case input_mapping_kind::axis2d:
                runtime_player.bind_axis2d(shadow_context, target.name, inert, {});
                break;
        }
    }
    else
    {
        for (const auto& binding : override->bindings)
        {
            switch (target.kind)
            {
                case input_mapping_kind::action:
                    runtime_player.bind_action(shadow_context, target.name, binding.binding);
                    break;
                case input_mapping_kind::axis:
                    runtime_player.bind_axis(shadow_context, target.name, binding.binding, binding.contribution);
                    break;
                case input_mapping_kind::axis2d:
                    runtime_player.bind_axis2d(shadow_context, target.name, binding.binding, binding.contribution2d);
                    break;
            }
        }
    }

    published_.push_back({.target = target, .context_name = shadow_context});
}

void input_rebinding_profile::disable_published(const input_mapping_target& target)
{
    if (!installed_system_) return;
    auto& runtime_player = installed_system_->player(player_);
    for (const auto& published : published_)
    {
        if (published.target == target) runtime_player.set_context_enabled(published.context_name, false);
    }
    const auto end = std::remove_if(published_.begin(), published_.end(),
                                    [&target](const published_override& value) { return value.target == target; });
    published_.erase(end, published_.end());
}

void input_rebinding_profile::disable_published_context(std::string_view context)
{
    if (!installed_system_) return;
    auto& runtime_player = installed_system_->player(player_);
    for (const auto& published : published_)
    {
        if (published.target.context == context) runtime_player.set_context_enabled(published.context_name, false);
    }
    const auto end = std::remove_if(published_.begin(), published_.end(), [context](const published_override& value)
                                    { return value.target.context == context; });
    published_.erase(end, published_.end());
}

input_user_overrides_io_result load_input_user_overrides(input_rebinding_profile& profile,
                                                         const std::filesystem::path& path)
{
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (error) return {.error = "failed to inspect input user overrides: " + error.message()};
    if (!exists)
    {
        profile.set_user_overrides({.version = input_config_version});
        return {.succeeded = true};
    }

    auto loaded = load_input_config(path);
    if (!loaded.succeeded) return {.error = std::move(loaded.error)};
    profile.set_user_overrides(std::move(loaded.config));
    return {.succeeded = true};
}

input_user_overrides_io_result save_input_user_overrides(const input_rebinding_profile& profile,
                                                         const std::filesystem::path& path)
{
    std::error_code error;
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path(), error);
        if (error) return {.error = "failed to create input user override directory: " + error.message()};
    }

    const auto saved = save_input_config(profile.user_overrides(), path);
    return {.succeeded = saved.succeeded, .error = saved.error};
}

std::filesystem::path input_user_overrides_path(const std::filesystem::path& user_settings_root,
                                                input::player_id player)
{
    return user_settings_root / "Input" / ("Player" + std::to_string(player) + ".json");
}

} // namespace arc::project

#pragma once

#include <arc/input/action_trigger.h>
#include <arc/input/input.h>

#include <string>
#include <string_view>
#include <unordered_map>

namespace arc::input
{

/**
 * @brief Runtime-owned trigger history for semantic actions.
 *
 * State is isolated per player and action. The active mapping context is part of
 * the runtime identity: switching the context that resolves an action discards
 * the previous trigger history before evaluating the new mapping, preventing a
 * hold/tap sequence from leaking across priority or enable changes.
 */
class input_action_trigger_runtime final
{
public:
    [[nodiscard]] input_trigger_result evaluate(player_id player, std::string_view action, std::string_view context,
                                                const input_trigger_config& config, float value,
                                                float delta_seconds)
    {
        player_state& player_runtime = players_[player];
        action_state& action_runtime = player_runtime.actions[std::string(action)];
        if (action_runtime.context != context)
        {
            action_runtime.context = std::string(context);
            action_runtime.trigger = {};
        }

        input_trigger_result result =
            evaluate_action_trigger(action_runtime.trigger, config, value, delta_seconds);
        action_runtime.trigger = result.state;
        return result;
    }

    /** Reset one player's actions sourced from a context that was disabled or reprioritized. */
    void reset_context(player_id player, std::string_view context)
    {
        const auto found = players_.find(player);
        if (found == players_.end()) return;

        auto& actions = found->second.actions;
        for (auto it = actions.begin(); it != actions.end();)
        {
            if (it->second.context == context)
                it = actions.erase(it);
            else
                ++it;
        }

        if (actions.empty()) players_.erase(found);
    }

    /** Reset all semantic trigger history for one local player. */
    void reset_player(player_id player)
    {
        players_.erase(player);
    }

    /** Reset every player's trigger history, for example after a mapping reload. */
    void reset()
    {
        players_.clear();
    }

    [[nodiscard]] const input_trigger_state* state(player_id player, std::string_view action) const noexcept
    {
        const auto player_found = players_.find(player);
        if (player_found == players_.end()) return nullptr;
        const auto action_found = player_found->second.actions.find(std::string(action));
        return action_found == player_found->second.actions.end() ? nullptr : &action_found->second.trigger;
    }

private:
    struct action_state
    {
        std::string context;
        input_trigger_state trigger{};
    };

    struct player_state
    {
        std::unordered_map<std::string, action_state> actions;
    };

    std::unordered_map<player_id, player_state> players_;
};

} // namespace arc::input

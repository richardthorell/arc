#pragma once

#include <arc/framework/event.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace arc::input
{

/**
 * @brief Device family that can drive input bindings.
 */
enum class input_device_type : std::uint8_t
{
    keyboard,
    mouse,
    gamepad
};

/**
 * @brief Player slot identifier. Keyboard and mouse map to player 0 for now.
 */
using player_id = std::uint32_t;

/**
 * @brief Stable action binding identifier.
 */
using input_action_id = std::uint32_t;

/**
 * @brief Stable axis binding identifier.
 */
using input_axis_id = std::uint32_t;

/**
 * @brief One physical input source.
 */
struct input_binding
{
    input_device_type device{ input_device_type::keyboard };
    int code{};
    player_id player{};

    /**
     * @brief Return whether this binding targets the same physical input.
     */
    friend bool operator==(const input_binding&, const input_binding&) = default;
};

/**
 * @brief A named set of bindings, useful for future editor/game mode stacks.
 */
struct input_context
{
    std::string name;
    bool enabled{ true };
};

/**
 * @brief Runtime action/axis mapper for platform-neutral ARC events.
 */
class input_manager
{
public:
    /**
     * @brief Clear per-frame pressed/released state while preserving held buttons.
     */
    void begin_frame();

    /**
     * @brief Consume one ARC platform event.
     */
    void process_event(const arc::event& event);

    /**
     * @brief Bind an action name to a physical input.
     */
    input_action_id bind_action(std::string_view name, input_binding binding);

    /**
     * @brief Bind an axis name to positive and negative physical inputs.
     */
    input_axis_id bind_axis(std::string_view name, input_binding positive_binding, input_binding negative_binding);

    /**
     * @brief Return true on the frame the action became held.
     */
    bool pressed(std::string_view name, player_id player = 0) const;

    /**
     * @brief Return true on the frame the action stopped being held.
     */
    bool released(std::string_view name, player_id player = 0) const;

    /**
     * @brief Return true while any bound input for the action is held.
     */
    bool down(std::string_view name, player_id player = 0) const;

    /**
     * @brief Return -1, 0, or 1 for the named digital axis.
     */
    float axis(std::string_view name, player_id player = 0) const;

private:
    struct axis_bindings
    {
        input_binding positive;
        input_binding negative;
    };

    std::unordered_map<std::string, std::vector<input_binding>> actions_;
    std::unordered_map<std::string, axis_bindings> axes_;
    std::unordered_set<std::uint64_t> held_;
    std::unordered_set<std::uint64_t> pressed_;
    std::unordered_set<std::uint64_t> released_;
    input_action_id next_action_id_{ 1 };
    input_axis_id next_axis_id_{ 1 };
};

} // namespace arc::input

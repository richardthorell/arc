#pragma once

#include <arc/project/input_config.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::project
{

enum class input_mapping_kind : std::uint8_t
{
    action,
    axis,
    axis2d
};

struct input_mapping_target
{
    std::string context;
    std::string name;
    input_mapping_kind kind{input_mapping_kind::action};

    friend bool operator==(const input_mapping_target&, const input_mapping_target&) = default;
};

struct input_mapping_binding
{
    input::input_binding binding;
    float contribution{1.0f};
    math::vector2f contribution2d{};
};

struct input_mapping_view
{
    input_mapping_target target;
    std::vector<input_mapping_binding> bindings;
};

struct input_rebind_capture_filter
{
    input::input_device_type device{input::input_device_type::unknown};
    float actuation_threshold{0.5f};
};

struct [[nodiscard]] input_rebind_capture_result
{
    input::input_device_id device{};
    input::input_binding binding;
    float value{};
};

/**
 * @brief Event-fed capture session for runtime rebinding.
 *
 * Platform/input event code offers normalized physical samples while the session
 * is active. The first compatible sample from a connected device assigned to the
 * selected player completes the capture without mutating mappings.
 */
class input_rebind_capture final
{
public:
    void begin(const input::input_system& system, input::player_id player, input_rebind_capture_filter filter = {});
    void cancel() noexcept;

    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] bool canceled() const noexcept;
    [[nodiscard]] std::optional<input_rebind_capture_result> offer(input::input_device_id device,
                                                                   input::input_control control, float value);

private:
    const input::input_system* system_{};
    input::player_id player_{};
    input_rebind_capture_filter filter_{};
    bool active_{};
    bool canceled_{};
};

struct [[nodiscard]] input_user_overrides_io_result
{
    bool succeeded{};
    std::string error;
};

/**
 * @brief Per-player semantic rebinding state layered over immutable project defaults.
 *
 * Overrides replace the complete binding list for one semantic mapping while
 * project defaults stay unchanged. The profile can be installed onto an input
 * system, where changes are published through transient shadow contexts so reset
 * operations take effect immediately without rewriting Config/Input.json.
 */
class input_rebinding_profile final
{
public:
    explicit input_rebinding_profile(input_config project_defaults, input::player_id player = 0);

    [[nodiscard]] input::player_id player() const noexcept;
    [[nodiscard]] const input_config& project_defaults() const noexcept;
    [[nodiscard]] const input_config& user_overrides() const noexcept;

    void set_user_overrides(input_config overrides);

    [[nodiscard]] input_config effective_config() const;
    [[nodiscard]] std::optional<input_mapping_view> default_mapping(const input_mapping_target& target) const;
    [[nodiscard]] std::optional<input_mapping_view> effective_mapping(const input_mapping_target& target) const;

    bool replace_bindings(const input_mapping_target& target, std::vector<input_mapping_binding> bindings);
    bool add_binding(const input_mapping_target& target, input_mapping_binding binding);
    bool remove_binding(const input_mapping_target& target, std::size_t index);
    bool reset_binding(const input_mapping_target& target, std::size_t index);
    bool reset_mapping(const input_mapping_target& target);
    bool reset_context(std::string_view context);
    void reset_all();

    /** Return semantic mappings that currently use the same physical control. */
    [[nodiscard]] std::vector<input_mapping_target> conflicts_for(const input::input_binding& binding) const;

    /** Apply project defaults and the current user override layer to one runtime player. */
    [[nodiscard]] input_config_apply_result install(input::input_system& system);

private:
    struct published_override
    {
        input_mapping_target target;
        std::string context_name;
    };

    void publish_override(const input_mapping_target& target);
    void disable_published(const input_mapping_target& target);
    void disable_published_context(std::string_view context);

    input_config project_defaults_;
    input_config user_overrides_;
    input::player_id player_{};
    input::input_system* installed_system_{};
    std::vector<published_override> published_;
    std::uint64_t publish_generation_{};
};

/** Missing override files are treated as an empty override layer. */
[[nodiscard]] input_user_overrides_io_result load_input_user_overrides(input_rebinding_profile& profile,
                                                                       const std::filesystem::path& path);
[[nodiscard]] input_user_overrides_io_result save_input_user_overrides(const input_rebinding_profile& profile,
                                                                       const std::filesystem::path& path);

/** Stable per-local-player filename under the caller-owned user settings root. */
[[nodiscard]] std::filesystem::path input_user_overrides_path(const std::filesystem::path& user_settings_root,
                                                              input::player_id player);

} // namespace arc::project

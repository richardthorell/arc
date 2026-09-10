from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"pattern not found in {path}: {old[:120]!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"pattern not unique in {path}: {text.count(old)} matches")
    file.write_text(text.replace(old, new, 1))


# Stable project-module input ABI.
replace_once(
    "engine/project/inc/arc/project/project_module.h",
    '''struct game_system_component_access_v1
{
    const char* component_id{};
    game_system_component_access_mode_v1 mode{game_system_component_access_mode_v1::read};
};

/** @brief ABI-safe transient entity handle used by project runtime callbacks. */''',
    '''struct game_system_component_access_v1
{
    const char* component_id{};
    game_system_component_access_mode_v1 mode{game_system_component_access_mode_v1::read};
};

/** @brief Platform-neutral input categories sampled for one simulation tick. */
enum class game_input_kind_v1 : std::uint8_t
{
    key,
    mouse_button,
    mouse_position,
    mouse_wheel,
    focus
};

/** @brief State transition carried by one sampled gameplay input command. */
enum class game_input_action_v1 : std::uint8_t
{
    pressed,
    released,
    changed
};

inline constexpr std::uint32_t game_input_modifier_alt_v1 = 1u << 0u;
inline constexpr std::uint32_t game_input_modifier_shift_v1 = 1u << 1u;
inline constexpr std::uint32_t game_input_modifier_control_v1 = 1u << 2u;

/** @brief ABI-safe input command visible to project ECS systems for the current simulation tick. */
struct game_input_command_v1
{
    game_input_kind_v1 kind{game_input_kind_v1::key};
    game_input_action_v1 action{game_input_action_v1::changed};
    std::int32_t code{};
    std::uint32_t modifiers{};
    std::int32_t x{};
    std::int32_t y{};
    float value{};
    bool repeat{};
};

/** @brief ABI-safe transient entity handle used by project runtime callbacks. */''',
)
replace_once(
    "engine/project/inc/arc/project/project_module.h",
    '''    bool presentation{};
    void* project_component_user_data{}; ///< Host-owned bridge valid only for this system invocation.''',
    '''    bool presentation{};
    std::uint64_t input_revision{};                     ///< Revision of the sampled input batch for this tick.
    const game_input_command_v1* input_commands{};      ///< Borrowed commands valid only for this invocation.
    std::size_t input_command_count{};                  ///< Number of entries in @ref input_commands.
    void* project_component_user_data{}; ///< Host-owned bridge valid only for this system invocation.''',
)

# Translate native ECS input snapshots into the project ABI.
replace_once(
    "editor/native/src/project_module_loader.cpp",
    '''bool valid_component_access_mode(project::game_system_component_access_mode_v1 mode) noexcept
{
    return mode == project::game_system_component_access_mode_v1::read ||
           mode == project::game_system_component_access_mode_v1::write;
}

ecs::system_phase to_system_phase''',
    '''bool valid_component_access_mode(project::game_system_component_access_mode_v1 mode) noexcept
{
    return mode == project::game_system_component_access_mode_v1::read ||
           mode == project::game_system_component_access_mode_v1::write;
}

project::game_input_kind_v1 to_game_input_kind(ecs::simulation_input_kind kind) noexcept
{
    switch (kind)
    {
        case ecs::simulation_input_kind::key:
            return project::game_input_kind_v1::key;
        case ecs::simulation_input_kind::mouse_button:
            return project::game_input_kind_v1::mouse_button;
        case ecs::simulation_input_kind::mouse_position:
            return project::game_input_kind_v1::mouse_position;
        case ecs::simulation_input_kind::mouse_wheel:
            return project::game_input_kind_v1::mouse_wheel;
        case ecs::simulation_input_kind::focus:
            return project::game_input_kind_v1::focus;
    }
    return project::game_input_kind_v1::key;
}

project::game_input_action_v1 to_game_input_action(ecs::simulation_input_action action) noexcept
{
    switch (action)
    {
        case ecs::simulation_input_action::pressed:
            return project::game_input_action_v1::pressed;
        case ecs::simulation_input_action::released:
            return project::game_input_action_v1::released;
        case ecs::simulation_input_action::changed:
            return project::game_input_action_v1::changed;
    }
    return project::game_input_action_v1::changed;
}

ecs::system_phase to_system_phase''',
)
replace_once(
    "editor/native/src/project_module_loader.cpp",
    '''            {
                project_component_bridge_context bridge{.native_context = &native_context,
                                                        .declared_accesses = &declared_accesses};
                project::game_system_context_v1 context{
                    .native_context = unrestricted_native_world_access ? &native_context : nullptr,''',
    '''            {
                project_component_bridge_context bridge{.native_context = &native_context,
                                                        .declared_accesses = &declared_accesses};
                const auto& native_input = native_context.input();
                std::vector<project::game_input_command_v1> input_commands;
                input_commands.reserve(native_input.commands.size());
                for (const auto& command : native_input.commands)
                    input_commands.push_back({.kind = to_game_input_kind(command.kind),
                                              .action = to_game_input_action(command.action),
                                              .code = command.code,
                                              .modifiers = command.modifiers,
                                              .x = command.x,
                                              .y = command.y,
                                              .value = command.value,
                                              .repeat = command.repeat});
                project::game_system_context_v1 context{
                    .native_context = unrestricted_native_world_access ? &native_context : nullptr,''',
)
replace_once(
    "editor/native/src/project_module_loader.cpp",
    '''                    .interpolation_alpha = native_context.interpolation_alpha(),
                    .presentation = native_context.presentation(),
                    .project_component_user_data = &bridge,''',
    '''                    .interpolation_alpha = native_context.interpolation_alpha(),
                    .presentation = native_context.presentation(),
                    .input_revision = native_input.revision,
                    .input_commands = input_commands.data(),
                    .input_command_count = input_commands.size(),
                    .project_component_user_data = &bridge,''',
)

# Route editor viewport protocol input into the Play runtime instead of treating it as a host no-op.
replace_once(
    "editor/native/src/arc_host_base.inc",
    '''constexpr std::string_view base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

template <class Component>''',
    '''constexpr std::string_view base64_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::uint32_t viewport_input_modifiers(bool alt, bool shift, bool control) noexcept
{
    return (alt ? project::game_input_modifier_alt_v1 : 0u) |
           (shift ? project::game_input_modifier_shift_v1 : 0u) |
           (control ? project::game_input_modifier_control_v1 : 0u);
}

framework::mouse_button viewport_mouse_button(std::int32_t button) noexcept
{
    switch (button)
    {
        case 0:
            return framework::mouse_button::left;
        case 1:
            return framework::mouse_button::middle;
        case 2:
            return framework::mouse_button::right;
        case 3:
            return framework::mouse_button::x1;
        case 4:
            return framework::mouse_button::x2;
        default:
            return framework::mouse_button::unknown;
    }
}

std::int32_t viewport_key_code(std::string_view key) noexcept
{
    if (key.size() == 1)
        return static_cast<std::int32_t>(std::toupper(static_cast<unsigned char>(key.front())));
    if (key == "Escape" || key == "Esc") return 27;
    if (key == "Enter") return 13;
    if (key == "Tab") return 9;
    if (key == "Backspace") return 8;
    if (key == "Space" || key == "Spacebar") return 32;
    if (key == "Delete") return 127;
    if (key == "ArrowLeft") return 0x1001;
    if (key == "ArrowRight") return 0x1002;
    if (key == "ArrowUp") return 0x1003;
    if (key == "ArrowDown") return 0x1004;
    if (key == "Home") return 0x1005;
    if (key == "End") return 0x1006;
    if (key == "PageUp") return 0x1007;
    if (key == "PageDown") return 0x1008;
    if (key == "Shift") return 0x1010;
    if (key == "Control") return 0x1011;
    if (key == "Alt") return 0x1012;
    return 0;
}

template <class Component>''',
)
replace_once(
    "editor/native/src/arc_host_base.inc",
    '''            else if constexpr (std::is_same_v<command_type, host_viewport_frame_released_command> ||
                               std::is_same_v<command_type, host_viewport_set_visibility_command> ||
                               std::is_same_v<command_type, host_viewport_pointer_command> ||
                               std::is_same_v<command_type, host_viewport_key_command>)
            {
                return success();
            }''',
    '''            else if constexpr (std::is_same_v<command_type, host_viewport_frame_released_command> ||
                               std::is_same_v<command_type, host_viewport_set_visibility_command>)
            {
                return success();
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_pointer_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                if (state_->preview_stopped) return success();
                const auto modifiers = viewport_input_modifiers(payload.alt, payload.shift, payload.control);
                framework::event input{.x = payload.x, .y = payload.y, .modifiers = modifiers};
                switch (payload.phase)
                {
                    case host_viewport_pointer_phase::down:
                        state_->simulation.dispatch(framework::event{.type = framework::event_type::focus_gained});
                        input.type = framework::event_type::mouse_button_down;
                        input.button = viewport_mouse_button(payload.button);
                        break;
                    case host_viewport_pointer_phase::move:
                        input.type = framework::event_type::mouse_moved;
                        break;
                    case host_viewport_pointer_phase::up:
                        input.type = framework::event_type::mouse_button_up;
                        input.button = viewport_mouse_button(payload.button);
                        break;
                    case host_viewport_pointer_phase::wheel:
                        input.type = framework::event_type::mouse_wheel;
                        input.wheel_delta = payload.wheel;
                        break;
                    case host_viewport_pointer_phase::leave:
                    case host_viewport_pointer_phase::cancel:
                        input.type = framework::event_type::focus_lost;
                        input.x = 0;
                        input.y = 0;
                        break;
                }
                state_->simulation.dispatch(input);
                return success(R"({"playSession":true,"input":true})");
            }
            else if constexpr (std::is_same_v<command_type, host_viewport_key_command>)
            {
                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");
                if (state_->preview_stopped) return success();
                state_->simulation.dispatch({.type = payload.down ? framework::event_type::key_down
                                                                  : framework::event_type::key_up,
                                             .key_code = viewport_key_code(payload.key),
                                             .modifiers = viewport_input_modifiers(payload.alt, payload.shift,
                                                                                  payload.control),
                                             .repeat = payload.repeat});
                return success(R"({"playSession":true,"input":true})");
            }''',
)

# Shared-texture input is already forwarded to arc_host; keep native editor manipulation from also consuming it during Play.
replace_once(
    "editor/native/src/arc_host_process_main.cpp",
    '''    void process_pointer(const arc::editor::host_viewport_pointer_command& pointer)
    {
        activate_interaction_surface(pointer.viewport_id);''',
    '''    void process_pointer(const arc::editor::host_viewport_pointer_command& pointer)
    {
        {
            std::lock_guard lock(host_mutex_);
            if (host_->runtime_snapshot().state != arc::editor::host_runtime_state::stopped) return;
        }
        activate_interaction_surface(pointer.viewport_id);''',
)
replace_once(
    "editor/native/src/arc_host_process_main.cpp",
    '''    void process_key(const arc::editor::host_viewport_key_command& key)
    {
        activate_interaction_surface(key.viewport_id);''',
    '''    void process_key(const arc::editor::host_viewport_key_command& key)
    {
        {
            std::lock_guard lock(host_mutex_);
            if (host_->runtime_snapshot().state != arc::editor::host_runtime_state::stopped) return;
        }
        activate_interaction_surface(key.viewport_id);''',
)

# Let the React viewport know when Play owns its input surface.
replace_once(
    "editor/src/renderer/src/app/Workbench.tsx",
    '''            startupState={startupState}
            onCommand={runCommand}''',
    '''            startupState={startupState}
            playSessionActive={runtimeState.state !== 'stopped'}
            onCommand={runCommand}''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  startupState: StartupState | null;
  onCommand: (command: CommandId) => void;''',
    '''  startupState: StartupState | null;
  playSessionActive?: boolean;
  onCommand: (command: CommandId) => void;''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  project,
  startupState,
  onCommand,''',
    '''  project,
  startupState,
  playSessionActive = false,
  onCommand,''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const [viewportError, setViewportError] = useState('');
  const [sharedFailure, setSharedFailure] = useState('');''',
    '''  const [viewportError, setViewportError] = useState('');
  const [playInputCaptured, setPlayInputCaptured] = useState(false);
  const [sharedFailure, setSharedFailure] = useState('');''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  useEffect(() => {
    const nextTransport = startupState?.viewportMode ?? 'unavailable';''',
    '''  useEffect(() => {
    if (!playSessionActive) setPlayInputCaptured(false);
  }, [playSessionActive]);

  useEffect(() => {
    const nextTransport = startupState?.viewportMode ?? 'unavailable';''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''    event.currentTarget.focus();
    onFocusChange?.(true);
    if (!viewportActive) return;
    if (cameraSourceId !== 'editor' && event.button !== 0) return;''',
    '''    event.currentTarget.focus();
    onFocusChange?.(true);
    if (!viewportActive) return;
    if (playSessionActive) {
      event.preventDefault();
      event.stopPropagation();
      setPlayInputCaptured(true);
      event.currentTarget.setPointerCapture(event.pointerId);
      if (streamedAvailable) sendPointer(event, 'down');
      return;
    }
    if (cameraSourceId !== 'editor' && event.button !== 0) return;''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const click = clickRef.current;''',
    '''  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (playSessionActive) {
      if (playInputCaptured && streamedAvailable) sendPointer(event, 'move');
      return;
    }
    const click = clickRef.current;''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (streamedAvailable) sendPointer(event, 'up');''',
    '''  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (playSessionActive) {
      if (playInputCaptured && streamedAvailable) sendPointer(event, 'up');
      return;
    }
    if (streamedAvailable) sendPointer(event, 'up');''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const onViewportKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (cameraSourceId === 'editor' && flyNavigationActiveRef.current && viewportFlyMovementCodes.has(event.code)) {''',
    '''  const onViewportKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (playSessionActive) {
      if (!playInputCaptured) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.key === 'Escape') {
        setPlayInputCaptured(false);
        if (streamedAvailable)
          void window.arc.viewport.pointer({ viewportId, phase: 'cancel', x: 0, y: 0, button: 0 });
        return;
      }
      if (streamedAvailable)
        void window.arc.viewport.key({
          viewportId,
          key: event.key,
          down: true,
          repeat: event.repeat,
          alt: event.altKey,
          shift: event.shiftKey,
          control: event.ctrlKey,
        });
      return;
    }
    if (cameraSourceId === 'editor' && flyNavigationActiveRef.current && viewportFlyMovementCodes.has(event.code)) {''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const onViewportKeyUp = (event: KeyboardEvent<HTMLDivElement>) => {
    movementKeysRef.current.delete(event.code);''',
    '''  const onViewportKeyUp = (event: KeyboardEvent<HTMLDivElement>) => {
    if (playSessionActive) {
      if (!playInputCaptured) return;
      event.preventDefault();
      event.stopPropagation();
      if (streamedAvailable)
        void window.arc.viewport.key({
          viewportId,
          key: event.key,
          down: false,
          repeat: false,
          alt: event.altKey,
          shift: event.shiftKey,
          control: event.ctrlKey,
        });
      return;
    }
    movementKeysRef.current.delete(event.code);''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!viewportActive || cameraSourceId !== 'editor') return;
    event.preventDefault();
    const zoom = normalizeViewportWheel(event.deltaY, event.deltaMode);
    if (zoom === 0) return;

    // Wheel zoom does not need pointer coordinates. Route every transport
    // through the same signed camera-input command instead of the streamed
    // pointer queue so zoom-in and zoom-out have identical semantics.
    sendCameraInput({ zoom });
  };''',
    '''  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!viewportActive) return;
    const zoom = normalizeViewportWheel(event.deltaY, event.deltaMode);
    if (zoom === 0) return;
    if (playSessionActive) {
      if (!playInputCaptured || !streamedAvailable) return;
      event.preventDefault();
      const rect = bodyRef.current?.getBoundingClientRect();
      const position = rect
        ? pointerCoordinates(rect.left + rect.width * 0.5, rect.top + rect.height * 0.5)
        : { x: 0, y: 0 };
      void window.arc.viewport.pointer({ viewportId, phase: 'wheel', ...position, wheel: zoom });
      return;
    }
    if (cameraSourceId !== 'editor') return;
    event.preventDefault();

    // Wheel zoom does not need pointer coordinates. Route every transport
    // through the same signed camera-input command instead of the streamed
    // pointer queue so zoom-in and zoom-out have identical semantics.
    sendCameraInput({ zoom });
  };''',
)
replace_once(
    "editor/src/renderer/src/viewport/ViewportPanel.tsx",
    '''            movementKeysRef.current.clear();
            consumedMovementKeysRef.current.clear();
            movementLastTickRef.current = null;
            onFocusChange?.(false);''',
    '''            movementKeysRef.current.clear();
            consumedMovementKeysRef.current.clear();
            movementLastTickRef.current = null;
            if (playSessionActive && playInputCaptured && streamedAvailable)
              void window.arc.viewport.pointer({ viewportId, phase: 'cancel', x: 0, y: 0, button: 0 });
            setPlayInputCaptured(false);
            onFocusChange?.(false);''',
)

# Native end-to-end fixture: first tick requires W + pointer movement, second tick requires focus loss.
Path("editor/native/tests/game_module_fixture_input.cpp").write_text(r'''#include <arc/project/project_module.h>

#include <cstddef>

namespace
{
std::uint32_t phase{};

bool execute(void*, arc::project::game_system_context_v1* context)
{
    if (!context) return false;
    bool saw_key{};
    bool saw_pointer{};
    bool saw_focus_lost{};
    for (std::size_t index = 0; index < context->input_command_count; ++index)
    {
        const auto& input = context->input_commands[index];
        if (input.kind == arc::project::game_input_kind_v1::key &&
            input.action == arc::project::game_input_action_v1::pressed && input.code == 'W' &&
            (input.modifiers & arc::project::game_input_modifier_shift_v1) != 0u)
            saw_key = true;
        if (input.kind == arc::project::game_input_kind_v1::mouse_position && input.x == 12 && input.y == 34)
            saw_pointer = true;
        if (input.kind == arc::project::game_input_kind_v1::focus &&
            input.action == arc::project::game_input_action_v1::changed && input.value == 0.0f)
            saw_focus_lost = true;
    }
    if (phase == 0)
    {
        if (!saw_key || !saw_pointer || context->input_revision == 0) return false;
        ++phase;
    }
    else if (phase == 1)
    {
        if (!saw_focus_lost) return false;
        ++phase;
    }
    return true;
}

bool start(const arc::project::game_module_host_v1*)
{
    phase = 0;
    return true;
}
void stop() {}

constexpr arc::project::game_system_descriptor_v1 input_system{
    .phase = arc::project::game_system_phase_v1::input,
    .priority = arc::project::game_system_priority_v1::critical,
    .unrestricted_native_world_access = false,
    .execute = execute,
};
constexpr arc::project::game_registration_descriptor_v1 registrations[]{
    {arc::project::game_registration_kind_v1::ecs_system, "fixture.runtime.input", "Fixture Runtime Input",
     &input_system},
};
constexpr arc::project::game_module_descriptor_v1 descriptor{
    .engine_version = "0.1.0",
    .project_guid = "12345678-1234-4234-8234-123456789abc",
    .module_id = "fixture.editor",
    .kind = arc::project::game_module_kind_v1::editor,
    .generation = 10,
    .registrations = registrations,
    .registration_count = std::size(registrations),
    .start = start,
    .stop = stop,
};
} // namespace

extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1()
{
    return &descriptor;
}
''')

replace_once(
    "editor/native/tests/CMakeLists.txt",
    '''add_library(arc_test_game_module_access_violation MODULE game_module_fixture_access_violation.cpp)
add_library(arc_test_game_module_rejected MODULE game_module_fixture_rejected.cpp)''',
    '''add_library(arc_test_game_module_access_violation MODULE game_module_fixture_access_violation.cpp)
add_library(arc_test_game_module_input MODULE game_module_fixture_input.cpp)
add_library(arc_test_game_module_rejected MODULE game_module_fixture_rejected.cpp)''',
)
replace_once(
    "editor/native/tests/CMakeLists.txt",
    '''                               arc_test_game_module_access arc_test_game_module_invalid_access
                               arc_test_game_module_access_violation arc_test_game_module_rejected)''',
    '''                               arc_test_game_module_access arc_test_game_module_invalid_access
                               arc_test_game_module_access_violation arc_test_game_module_input arc_test_game_module_rejected)''',
)
replace_once(
    "editor/native/tests/CMakeLists.txt",
    '''  "ARC_TEST_GAME_MODULE_ACCESS_VIOLATION=\\"$<TARGET_FILE:arc_test_game_module_access_violation>\\""
  "ARC_TEST_GAME_MODULE_REJECTED=\\"$<TARGET_FILE:arc_test_game_module_rejected>\\"")''',
    '''  "ARC_TEST_GAME_MODULE_ACCESS_VIOLATION=\\"$<TARGET_FILE:arc_test_game_module_access_violation>\\""
  "ARC_TEST_GAME_MODULE_INPUT=\\"$<TARGET_FILE:arc_test_game_module_input>\\""
  "ARC_TEST_GAME_MODULE_REJECTED=\\"$<TARGET_FILE:arc_test_game_module_rejected>\\"")''',
)
replace_once(
    "editor/native/tests/CMakeLists.txt",
    '''                 arc_test_game_module_access arc_test_game_module_invalid_access
                 arc_test_game_module_access_violation arc_test_game_module_rejected)''',
    '''                 arc_test_game_module_access arc_test_game_module_invalid_access
                 arc_test_game_module_access_violation arc_test_game_module_input arc_test_game_module_rejected)''',
)

Path("editor/native/tests/project_system_input_tests.cpp").write_text(r'''#include <arc/editor/arc_host.h>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <string>

namespace
{
std::filesystem::path stage_input_module(const std::filesystem::path& root)
{
    std::filesystem::create_directories(root / "Content");
    std::filesystem::create_directories(root / "Build");
    const auto destination = root / "Build" / std::filesystem::path(ARC_TEST_GAME_MODULE_INPUT).filename();
    std::filesystem::copy_file(ARC_TEST_GAME_MODULE_INPUT, destination, std::filesystem::copy_options::overwrite_existing);
    return destination;
}
} // namespace

TEST_CASE("Play viewport input is sampled by project ECS systems and focus loss clears the next tick")
{
    const auto root =
        std::filesystem::temp_directory_path() /
        ("arc-play-input-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto module_path = stage_input_module(root);

    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    arc::editor::editor_asset_state assets;
    assets.root = root / "Content";
    REQUIRE(host->open_project({.name = "Play Input",
                                .root = root,
                                .project_guid = "12345678-1234-4234-8234-123456789abc",
                                .engine_version = "0.1.0",
                                .editor_module_id = "fixture.editor",
                                .editor_module_path = module_path},
                               assets)
                .succeeded);

    REQUIRE(host->execute(arc::editor::host_runtime_resume_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_pause_command{}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_key_command{.key = "w", .down = true, .shift = true}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_pointer_command{
                              .phase = arc::editor::host_viewport_pointer_phase::move, .x = 12, .y = 34})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    REQUIRE(host->execute(arc::editor::host_viewport_pointer_command{
                              .phase = arc::editor::host_viewport_pointer_phase::cancel})
                .succeeded);
    REQUIRE(host->execute(arc::editor::host_runtime_step_command{.ticks = 1}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::paused);

    REQUIRE(host->execute(arc::editor::host_runtime_stop_command{}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);
    REQUIRE(host->execute(arc::editor::host_viewport_key_command{.key = "w", .down = true}).succeeded);
    CHECK(host->runtime_snapshot().state == arc::editor::host_runtime_state::stopped);

    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
}
''')

# Frontend regression: Play capture routes W to gameplay instead of editor fly navigation, Escape releases capture.
test_path = Path("editor/src/renderer/src/viewport/ViewportPanel.test.tsx")
test_text = test_path.read_text()
append = r'''

describe('ViewportPanel Play input', () => {
  it('captures gameplay keys on click and releases capture with Escape', async () => {
    const pointer = vi.fn().mockResolvedValue({ succeeded: true });
    const key = vi.fn().mockResolvedValue({ succeeded: true });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn().mockResolvedValue({
            succeeded: true,
            payload: { width: 640, height: 480, fps: 60, frameTimeMs: 16, drawCalls: 1, frameIndex: 1, submitted: true },
          }),
          command: vi.fn().mockResolvedValue({ succeeded: true }),
        },
        viewport: {
          create: vi.fn().mockResolvedValue({ succeeded: true }),
          attach: vi.fn(),
          resize: vi.fn().mockResolvedValue({ succeeded: true }),
          detach: vi.fn().mockResolvedValue({ succeeded: true }),
          cameraInput: vi.fn().mockResolvedValue({ succeeded: true }),
          pointer,
          key,
          registerSurface: vi.fn(),
          unregisterSurface: vi.fn(),
          setVisibility: vi.fn(),
        },
      },
    });

    const view = render(
      <ViewportPanel
        project={null}
        startupState={{ appVersion: '0.1.0', engineHostConnected: true, viewportMode: 'streamed' }}
        playSessionActive
        onCommand={vi.fn()}
        onReconnect={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    const surface = view.getByLabelText('ARC 3D viewport').parentElement!;
    fireEvent.pointerDown(surface, { pointerId: 1, button: 0, clientX: 20, clientY: 20 });
    fireEvent.keyDown(surface, { key: 'w', code: 'KeyW' });
    await waitFor(() => expect(key).toHaveBeenCalledWith(expect.objectContaining({ key: 'w', down: true })));

    fireEvent.keyDown(surface, { key: 'Escape', code: 'Escape' });
    await waitFor(() => expect(pointer).toHaveBeenCalledWith(expect.objectContaining({ phase: 'cancel' })));
    key.mockClear();
    fireEvent.keyDown(surface, { key: 'w', code: 'KeyW' });
    expect(key).not.toHaveBeenCalled();
  });
});
'''
if "describe('ViewportPanel Play input'" not in test_text:
    test_path.write_text(test_text + append)

print("Applied Play M3.3 input routing changes")

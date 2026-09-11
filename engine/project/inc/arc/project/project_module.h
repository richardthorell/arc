#pragma once

#include <arc/project/runtime_world_api.h>

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define ARC_PROJECT_MODULE_EXPORT __declspec(dllexport)
#else
#define ARC_PROJECT_MODULE_EXPORT __attribute__((visibility("default")))
#endif

namespace arc::project
{

/** @brief Stable binary interface version shared by ARC and project game modules. */
inline constexpr std::uint32_t game_module_abi_version = 1;

/** @brief Project module role. */
enum class game_module_kind_v1 : std::uint8_t
{
    editor,
    runtime,
    server
};

/** @brief Registration category advertised by a project module. */
enum class game_registration_kind_v1 : std::uint8_t
{
    ecs_system,
    service,
    asset_type,
    importer,
    cook_processor,
    console_command,
    editor_extension,
    play_lifecycle
};

/** @brief Fixed/presentation phase used by an executable project ECS system. */
enum class game_system_phase_v1 : std::uint8_t
{
    input,
    network_receive,
    gameplay_commands,
    movement,
    physics,
    abilities,
    ai,
    replication,
    presentation_extraction
};

/** @brief Scheduling priority requested by an executable project ECS system. */
enum class game_system_priority_v1 : std::uint8_t
{
    critical,
    high,
    normal,
    low,
    background
};

/** @brief Access mode declared by a project ECS system for one reflected project component. */
enum class game_system_component_access_mode_v1 : std::uint8_t
{
    read,
    write
};

/** @brief Stable project-component access used by ARC's dependency-aware scheduler. */
struct game_system_component_access_v1
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

/** @brief Query whether an entity owns an authored project component in the active runtime world. */
using game_has_project_component_v1 = bool (*)(void* user_data, game_entity_v1 entity, const char* component_id);

/**
 * @brief Read canonical JSON for one runtime project component.
 * @return Borrowed UTF-8 JSON valid until that component is patched or the system callback returns.
 */
using game_read_project_component_json_v1 = const char* (*)(void* user_data, game_entity_v1 entity,
                                                            const char* component_id);

/** @brief Merge a JSON object into one runtime project component. */
using game_patch_project_component_json_v1 = bool (*)(void* user_data, game_entity_v1 entity, const char* component_id,
                                                      const char* patch_json);

/** @brief Visitor invoked for each entity carrying a requested project component. */
using game_visit_project_component_v1 = bool (*)(void* visitor_user_data, game_entity_v1 entity, const char* json);

/** @brief Iterate entities carrying a declared project component in the active runtime world. */
using game_for_each_project_component_v1 = bool (*)(void* user_data, const char* component_id, void* visitor_user_data,
                                                    game_visit_project_component_v1 visitor);

/**
 * @brief Per-invocation context passed to project ECS systems.
 *
 * @ref native_context is an opaque pointer to ARC's native ecs::system_context for
 * modules compiled against the exact engine version accepted by the host. The
 * stable ABI keeps the pointer opaque so the C-facing descriptor does not expose
 * C++ scheduler types.
 */
struct game_system_context_v1
{
    std::size_t structure_size{sizeof(game_system_context_v1)};
    void* native_context{};
    std::uint64_t tick_id{};
    std::uint64_t world_id{};
    float delta_seconds{};
    float fixed_delta_seconds{};
    float frame_delta_seconds{};
    float interpolation_alpha{};
    bool presentation{};
    std::uint64_t input_revision{};                ///< Revision of the sampled input batch for this tick.
    const game_input_command_v1* input_commands{}; ///< Borrowed commands valid only for this invocation.
    std::size_t input_command_count{};             ///< Number of entries in @ref input_commands.
    void* project_component_user_data{};           ///< Host-owned bridge valid only for this system invocation.
    game_has_project_component_v1 has_project_component{};
    game_read_project_component_json_v1 read_project_component_json{};
    game_patch_project_component_json_v1 patch_project_component_json{};
    game_for_each_project_component_v1 for_each_project_component{};
    const game_world_api_v1* world{}; ///< Stable runtime-world bridge valid only for this invocation.
};

/** @brief Executable callback for a project ECS system. Return false to fault the runtime world. */
using game_system_execute_v1 = bool (*)(void* user_data, game_system_context_v1* context);

/** @brief Kind-specific descriptor referenced by an ecs_system registration. */
struct game_system_descriptor_v1
{
    std::size_t structure_size{sizeof(game_system_descriptor_v1)};
    game_system_phase_v1 phase{game_system_phase_v1::gameplay_commands};
    game_system_priority_v1 priority{game_system_priority_v1::normal};
    const game_system_component_access_v1* component_accesses{}; ///< Declared reflected project-component access.
    std::size_t component_access_count{};                        ///< Number of entries in @ref component_accesses.
    bool unrestricted_native_world_access{true}; ///< Preserve legacy native_context access and serialize this system.
    const char* const* before{};                 ///< Stable project-system IDs that must execute after this system.
    std::size_t before_count{};
    const char* const* after{}; ///< Stable project-system IDs that must execute before this system.
    std::size_t after_count{};
    void* user_data{};                ///< Module-owned state valid for the loaded generation.
    game_system_execute_v1 execute{}; ///< Called by ARC's ECS scheduler.
    const game_core_component_access_v1* core_component_accesses{}; ///< Stable engine-component access declarations.
    std::size_t core_component_access_count{}; ///< Number of entries in @ref core_component_accesses.
};

/** @brief Per-session context passed to project BeginPlay/EndPlay callbacks. */
struct game_play_context_v1
{
    std::size_t structure_size{sizeof(game_play_context_v1)};
    std::uint64_t world_id{};         ///< Runtime world that owns this Play session.
    const game_world_api_v1* world{}; ///< Stable runtime-world bridge for this lifecycle callback.
};

/** @brief Called exactly once after a Play World is assembled and before its first simulation tick. */
using game_begin_play_v1 = bool (*)(void* user_data, const game_play_context_v1* context);

/** @brief Called exactly once for a successfully begun Play session before its module generation can unload. */
using game_end_play_v1 = void (*)(void* user_data, const game_play_context_v1* context);

/** @brief Kind-specific descriptor referenced by a play_lifecycle registration. */
struct game_play_lifecycle_descriptor_v1
{
    std::size_t structure_size{sizeof(game_play_lifecycle_descriptor_v1)};
    void* user_data{};               ///< Module-owned state valid for the loaded generation.
    game_begin_play_v1 begin_play{}; ///< Required per-session startup callback.
    game_end_play_v1 end_play{};     ///< Required matching per-session teardown callback.
};

/** @brief Reflected field representation understood by the native editor host. */
enum class game_field_kind_v1 : std::uint8_t
{
    boolean,
    signed_integer,
    unsigned_integer,
    floating_point,
    string,
    enumeration,
    vector2,
    vector3,
    vector4,
    quaternion,
    entity_reference,
    asset_reference,
    structure,
    sequence
};

/** @brief Stable reflected-property behavior flags. */
enum class game_field_flags_v1 : std::uint32_t
{
    none = 0,
    editable = 1u << 0u,
    read_only = 1u << 1u,
    transient = 1u << 2u,
    save_game = 1u << 3u,
    prefab_override = 1u << 4u,
    replicated = 1u << 5u,
    serialized = 1u << 6u
};

/** @brief Return the bitwise union of reflected property flags. */
constexpr game_field_flags_v1 operator|(game_field_flags_v1 lhs, game_field_flags_v1 rhs) noexcept
{
    return static_cast<game_field_flags_v1>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

/** @brief Immutable reflected field metadata owned by a loaded module generation. */
struct game_field_descriptor_v1
{
    std::uint64_t stable_id{};                                   ///< Explicit field ID retained across C++ renames.
    const char* name{};                                          ///< Canonical persistence name.
    const char* display_name{};                                  ///< Inspector-facing name.
    const char* category{};                                      ///< Inspector grouping category.
    const char* tooltip{};                                       ///< User-facing documentation.
    game_field_kind_v1 kind{game_field_kind_v1::floating_point}; ///< Serialized/control kind.
    game_field_flags_v1 flags{game_field_flags_v1::editable | game_field_flags_v1::serialized |
                              game_field_flags_v1::prefab_override}; ///< Authored behavior.
    const char* default_json{};                                      ///< Canonical JSON default value.
    double minimum{};                                                ///< Numeric minimum when @ref has_minimum is true.
    double maximum{};                                                ///< Numeric maximum when @ref has_maximum is true.
    bool has_minimum{};                                              ///< Whether @ref minimum is active.
    bool has_maximum{};                                              ///< Whether @ref maximum is active.
    const char* asset_type_restriction{};                            ///< Stable allowed asset type, or null.
    const char* entity_component_restriction{};                      ///< Required target component ID, or null.
};

/** @brief Immutable component schema exported by a project module generation. */
struct game_component_descriptor_v1
{
    const char* stable_id{};                  ///< Explicit 128-bit hexadecimal component ID.
    const char* canonical_name{};             ///< Persistence-facing component name.
    const char* display_name{};               ///< Inspector-facing component name.
    const char* category{};                   ///< Add Component menu category.
    const char* tooltip{};                    ///< User-facing documentation.
    std::uint32_t schema_version{1};          ///< Monotonically increasing component schema.
    const game_field_descriptor_v1* fields{}; ///< Borrowed field array.
    std::size_t field_count{};                ///< Number of entries in @ref fields.
};

/** @brief One non-component facility registered by the module. */
struct game_registration_descriptor_v1
{
    game_registration_kind_v1 kind{game_registration_kind_v1::ecs_system}; ///< Facility category.
    const char* stable_id{};                                               ///< Stable registration ID.
    const char* name{};                                                    ///< Diagnostic/display name.
    const void* descriptor{};                                              ///< Kind-specific borrowed descriptor.
};

/** @brief Services exposed to a loaded project module. */
struct game_module_host_v1
{
    std::uint32_t abi_version{game_module_abi_version};                        ///< Host ABI version.
    std::size_t structure_size{sizeof(game_module_host_v1)};                   ///< Forward-compatible structure size.
    void* user_data{};                                                         ///< Opaque host-owned callback context.
    void (*log)(void* user_data, const char* category, const char* message){}; ///< Borrowed diagnostic callback.
};

/** @brief Immutable identity, schema, registrations, and lifecycle exported by a game module. */
struct game_module_descriptor_v1
{
    std::uint32_t abi_version{game_module_abi_version};            ///< ARC game-module ABI version.
    std::size_t structure_size{sizeof(game_module_descriptor_v1)}; ///< Descriptor byte size.
    const char* engine_version{};                                  ///< Exact ARC version used for compilation.
    const char* project_guid{};                                    ///< Persistent owner project GUID.
    const char* module_id{};                                       ///< Stable module ID from the project descriptor.
    game_module_kind_v1 kind{game_module_kind_v1::editor};         ///< Module role.
    std::uint64_t generation{};                                    ///< Monotonic build generation.
    const game_component_descriptor_v1* components{};              ///< Borrowed component schema array.
    std::size_t component_count{};                                 ///< Number of component schemas.
    const game_registration_descriptor_v1* registrations{};        ///< Borrowed registration array.
    std::size_t registration_count{};                              ///< Number of non-component registrations.
    bool (*start)(const game_module_host_v1* host){};              ///< Start after validation and registration.
    bool (*prepare_reload)(){}; ///< Quiesce module work before state capture and unload.
    void (*stop)(){};           ///< Idempotent shutdown before unloading the library.
};

/** @brief Function type for the stable game-module descriptor export. */
using query_game_module_v1 = const game_module_descriptor_v1* (*)();

} // namespace arc::project

/**
 * @brief Return the immutable descriptor exported by a project game module.
 * @return Borrowed descriptor valid until the module generation is unloaded.
 */
extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1();
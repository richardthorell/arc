#pragma once

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
    editor_extension
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
    return static_cast<game_field_flags_v1>(static_cast<std::uint32_t>(lhs) |
                                            static_cast<std::uint32_t>(rhs));
}

/** @brief Immutable reflected field metadata owned by a loaded module generation. */
struct game_field_descriptor_v1
{
    std::uint64_t stable_id{}; ///< Explicit field ID retained across C++ renames.
    const char* name{}; ///< Canonical persistence name.
    const char* display_name{}; ///< Inspector-facing name.
    const char* category{}; ///< Inspector grouping category.
    const char* tooltip{}; ///< User-facing documentation.
    game_field_kind_v1 kind{game_field_kind_v1::floating_point}; ///< Serialized/control kind.
    game_field_flags_v1 flags{game_field_flags_v1::editable | game_field_flags_v1::serialized |
                              game_field_flags_v1::prefab_override}; ///< Authored behavior.
    const char* default_json{}; ///< Canonical JSON default value.
    double minimum{}; ///< Numeric minimum when @ref has_minimum is true.
    double maximum{}; ///< Numeric maximum when @ref has_maximum is true.
    bool has_minimum{}; ///< Whether @ref minimum is active.
    bool has_maximum{}; ///< Whether @ref maximum is active.
    const char* asset_type_restriction{}; ///< Stable allowed asset type, or null.
    const char* entity_component_restriction{}; ///< Required target component ID, or null.
};

/** @brief Immutable component schema exported by a project module generation. */
struct game_component_descriptor_v1
{
    const char* stable_id{}; ///< Explicit 128-bit hexadecimal component ID.
    const char* canonical_name{}; ///< Persistence-facing component name.
    const char* display_name{}; ///< Inspector-facing component name.
    const char* category{}; ///< Add Component menu category.
    const char* tooltip{}; ///< User-facing documentation.
    std::uint32_t schema_version{1}; ///< Monotonically increasing component schema.
    const game_field_descriptor_v1* fields{}; ///< Borrowed field array.
    std::size_t field_count{}; ///< Number of entries in @ref fields.
};

/** @brief One non-component facility registered by the module. */
struct game_registration_descriptor_v1
{
    game_registration_kind_v1 kind{game_registration_kind_v1::ecs_system}; ///< Facility category.
    const char* stable_id{}; ///< Stable registration ID.
    const char* name{}; ///< Diagnostic/display name.
    const void* descriptor{}; ///< Kind-specific borrowed descriptor.
};

/** @brief Services exposed to a loaded project module. */
struct game_module_host_v1
{
    std::uint32_t abi_version{game_module_abi_version}; ///< Host ABI version.
    std::size_t structure_size{sizeof(game_module_host_v1)}; ///< Forward-compatible structure size.
    void* user_data{}; ///< Opaque host-owned callback context.
    void (*log)(void* user_data, const char* category, const char* message){}; ///< Borrowed diagnostic callback.
};

/** @brief Immutable identity, schema, registrations, and lifecycle exported by a game module. */
struct game_module_descriptor_v1
{
    std::uint32_t abi_version{game_module_abi_version}; ///< ARC game-module ABI version.
    std::size_t structure_size{sizeof(game_module_descriptor_v1)}; ///< Descriptor byte size.
    const char* engine_version{}; ///< Exact ARC version used for compilation.
    const char* project_guid{}; ///< Persistent owner project GUID.
    const char* module_id{}; ///< Stable module ID from the project descriptor.
    game_module_kind_v1 kind{game_module_kind_v1::editor}; ///< Module role.
    std::uint64_t generation{}; ///< Monotonic build generation.
    const game_component_descriptor_v1* components{}; ///< Borrowed component schema array.
    std::size_t component_count{}; ///< Number of component schemas.
    const game_registration_descriptor_v1* registrations{}; ///< Borrowed registration array.
    std::size_t registration_count{}; ///< Number of non-component registrations.
    bool (*start)(const game_module_host_v1* host){}; ///< Start after validation and registration.
    bool (*prepare_reload)(){}; ///< Quiesce module work before state capture and unload.
    void (*stop)(){}; ///< Idempotent shutdown before unloading the library.
};

/** @brief Function type for the stable game-module descriptor export. */
using query_game_module_v1 = const game_module_descriptor_v1* (*)();

} // namespace arc::project

/**
 * @brief Return the immutable descriptor exported by a project game module.
 * @return Borrowed descriptor valid until the module generation is unloaded.
 */
extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::game_module_descriptor_v1* arc_query_game_module_v1();

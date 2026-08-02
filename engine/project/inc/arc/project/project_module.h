#pragma once

#include <cstdint>

#if defined(_WIN32)
#define ARC_PROJECT_MODULE_EXPORT __declspec(dllexport)
#else
#define ARC_PROJECT_MODULE_EXPORT __attribute__((visibility("default")))
#endif

namespace arc::project
{

/** @brief Current binary interface version for project editor modules. */
inline constexpr std::uint32_t editor_module_abi_version = 1;

/** @brief Services exposed by the editor to a loaded version-1 project module. */
struct editor_module_host_v1
{
    /// ABI version supplied by the host.
    std::uint32_t abi_version{editor_module_abi_version};
    /// Opaque host-owned pointer passed to callbacks.
    void* user_data{};
    /// Optional host logging callback; strings are borrowed for the duration of the call.
    void (*log)(void* user_data, const char* category, const char* message){};
};

/** @brief Identity and lifecycle entry points exported by a project editor module. */
struct editor_module_descriptor_v1
{
    /// Module ABI version.
    std::uint32_t abi_version{editor_module_abi_version};
    /// Exact ARC engine version used to build the module.
    const char* engine_version{};
    /// Persistent GUID of the owning project.
    const char* project_guid{};
    /// Stable module ID declared by the project descriptor.
    const char* module_id{};
    /// Start callback invoked after identity validation; returns false on startup failure.
    bool (*start)(const editor_module_host_v1* host){};
    /// Idempotent shutdown callback invoked before the library is unloaded.
    void (*stop)(){};
};

/** @brief Function type for the version-1 module descriptor export. */
using query_editor_module_v1 = const editor_module_descriptor_v1* (*)();

} // namespace arc::project

/**
 * @brief Return the immutable descriptor exported by a project editor module.
 * @return Borrowed descriptor that remains valid until the module is unloaded.
 */
extern "C" ARC_PROJECT_MODULE_EXPORT const arc::project::editor_module_descriptor_v1* arc_query_editor_module_v1();

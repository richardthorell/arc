#pragma once

/** @namespace arc::project
 * @brief External project descriptors, templates, installations, and toolchains.
 */

#include <arc/core/result.h>
#include <arc/project/project_module.h>
#include <arc/project/reflection.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::project
{

inline constexpr std::string_view project_format = "arc-project"; ///< Project document format identifier.
inline constexpr std::uint32_t project_format_version = 2; ///< Current project document schema version.
inline constexpr std::string_view installation_format = "arc-installation"; ///< Installation manifest format.
inline constexpr std::uint32_t installation_format_version = 1; ///< Installation manifest schema version.
inline constexpr std::string_view template_format = "arc-project-template"; ///< Project-template manifest format.
inline constexpr std::uint32_t template_format_version = 1; ///< Project-template manifest schema version.

/** @brief Stable categories returned by project-platform operations. */
enum class project_error_code : std::uint8_t
{
    none,
    not_found,
    invalid_json,
    unsupported_version,
    invalid_descriptor,
    unsafe_path,
    incompatible_engine,
    missing_module,
    missing_plugin,
    missing_sdk,
    unsupported_platform,
    invalid_scene,
    io_failed,
    destination_not_empty,
    template_not_found,
    tool_not_found,
    process_failed
};

/** @brief Structured failure returned by project-platform operations. */
struct project_error
{
    project_error_code code{project_error_code::none}; ///< Machine-readable failure category.
    std::filesystem::path path; ///< Related descriptor, source, or output path when available.
    std::string field; ///< Related descriptor field or module/plugin ID when available.
    std::string message; ///< Human-readable diagnostic.
};

/** @brief Status returned by project operations without a value. */
using project_status = core::status<project_error>;

/** @brief Role of a project-owned native module. */
enum class module_kind : std::uint8_t
{
    editor,
    runtime,
    server
};

/** @brief Authority that owns a module dependency. */
enum class dependency_kind : std::uint8_t
{
    engine,
    project,
    plugin
};

/** @brief Renderer selected by a project. */
enum class renderer_backend : std::uint8_t
{
    none,
    vulkan
};

/** @brief GUID-authoritative persistent reference with a refreshable path hint. */
struct project_asset_reference
{
    std::string guid; ///< Stable asset GUID.
    std::string expected_type; ///< Required logical asset type.
    std::string path_hint; ///< Project-relative path used for presentation and repair.
};

/** @brief Typed dependency declared by a project module. */
struct module_dependency
{
    dependency_kind kind{dependency_kind::engine}; ///< Dependency authority.
    std::string id; ///< Stable module or plugin ID.
    std::string version; ///< Exact required version when applicable.
};

/** @brief Build and dependency metadata for one project-owned module. */
struct project_module_descriptor
{
    std::string id; ///< Stable module ID.
    module_kind kind{module_kind::runtime}; ///< Module role.
    std::string target; ///< CMake target name.
    std::filesystem::path source_root; ///< Project-relative source directory.
    bool enabled{true}; ///< Whether the module participates in the workspace.
    std::vector<module_dependency> dependencies; ///< Typed direct dependencies.
};

/** @brief Exact plugin requirement declared by a project. */
struct project_plugin_descriptor
{
    std::string id; ///< Stable plugin ID.
    std::string version; ///< Exact plugin version.
    std::string origin{"engine"}; ///< Installation, project, or other provider identity.
    bool required{true}; ///< Whether absence prevents writable opening.
    bool enabled{true}; ///< Whether the plugin is enabled.
    std::filesystem::path path; ///< Optional path contained by the project Plugins directory.
};

/** @brief Project-relative source-controlled and transient directory layout. */
struct project_path_layout
{
    std::filesystem::path source{"Source"}; ///< Project source root.
    std::filesystem::path content{"Content"}; ///< Primary authored-content root.
    std::filesystem::path config{"Config"}; ///< Source-controlled configuration root.
    std::filesystem::path plugins{"Plugins"}; ///< Project plugin root.
    std::filesystem::path saved{"Saved"}; ///< Ignored per-project editor/user state root.
    std::filesystem::path intermediate{"Intermediate"}; ///< Ignored caches and intermediates root.
    std::filesystem::path build{"Build"}; ///< Ignored build and package output root.
};

/** @brief One build/cook target platform exposed by a project. */
struct target_platform_descriptor
{
    std::string id; ///< Stable platform profile ID.
    bool enabled{true}; ///< Whether this platform participates in project workflows.
};

/** @brief Minimum host toolchain capabilities required by a project. */
struct toolchain_requirements
{
    std::string compiler{"auto"}; ///< Compiler family or automatic selection.
    std::string minimum_compiler_version; ///< Minimum compiler version, if constrained.
    std::string generator{"auto"}; ///< CMake generator or automatic selection.
    std::string architecture{"x86_64"}; ///< Target architecture.
    std::uint32_t cpp_standard{20}; ///< Required C++ language standard.
};

/** @brief Project-wide renderer selection. */
struct project_renderer_settings
{
    renderer_backend backend{renderer_backend::vulkan}; ///< Selected renderer backend.
    std::string api{"1.2"}; ///< Minimum graphics API version.
    std::string quality{"standard"}; ///< Initial quality profile.
};

/** @brief Inline cooker target profile stored in the project descriptor. */
struct cook_profile_descriptor
{
    std::string id; ///< Stable cook-profile ID.
    std::string platform; ///< Target operating system.
    std::string architecture{"x86_64"}; ///< Target architecture.
    std::string renderer{"vulkan"}; ///< Runtime renderer or none.
    std::string api{"1.2"}; ///< Target graphics API baseline.
    std::string texture_family{"bc"}; ///< Runtime texture compression family.
    std::string configuration{"Shipping"}; ///< Build configuration used for cooking.
};

/** @brief Package naming, output, and chunking policy. */
struct package_settings
{
    std::string application_name; ///< Product name embedded in packages.
    std::string company_name; ///< Optional publisher identity.
    std::filesystem::path output{"Build/Packages"}; ///< Project-relative package output.
    bool region_chunks{true}; ///< Whether world regions receive independent chunks.
};

/** @brief Source-controlled setting-document locations. */
struct project_settings_paths
{
    std::filesystem::path editor{"Config/Editor.json"}; ///< Editor/project setting document.
    std::filesystem::path renderer{"Config/Renderer.json"}; ///< Renderer setting document.
    std::filesystem::path input{"Config/Input.json"}; ///< Input setting document.
};

/** @brief Complete in-memory representation of a version-2 ARC project. */
struct project_descriptor
{
    std::string guid; ///< Persistent project GUID.
    std::string name; ///< Human-readable project name.
    std::string engine_version; ///< Exact required ARC version.
    project_path_layout paths; ///< Project-relative directory layout.
    std::vector<std::filesystem::path> asset_roots{"Content"}; ///< Declared content roots.
    std::vector<project_module_descriptor> modules; ///< Project-native modules.
    std::vector<project_plugin_descriptor> plugins; ///< Exact plugin requirements.
    std::optional<project_asset_reference> default_scene; ///< Scene opened by the editor.
    std::vector<project_asset_reference> startup_scenes; ///< Runtime startup-scene order.
    std::vector<target_platform_descriptor> target_platforms; ///< Enabled target platforms.
    toolchain_requirements toolchain; ///< Required build tools.
    std::vector<std::string> build_configurations{"Debug", "RelWithDebInfo", "Shipping"}; ///< Supported configurations.
    project_renderer_settings renderer; ///< Renderer selection.
    std::vector<cook_profile_descriptor> cook_profiles; ///< Inline cook profiles.
    package_settings package; ///< Package policy.
    project_settings_paths settings; ///< Source-controlled setting paths.
};

/** @brief Optional validation policy supplied by an editor or CLI caller. */
struct project_validation_options
{
    std::string engine_version; ///< Running engine version used for compatibility checks.
    bool require_exact_engine{}; ///< Reject engine-version mismatches when true.
    bool require_paths{}; ///< Validate source, content, module, plugin, and scene paths.
    bool allow_read_only{}; ///< Return a non-writable result instead of failing compatibility.
};

/** @brief Validated writability and non-fatal diagnostics. */
struct [[nodiscard]] project_validation_result
{
    bool writable{true}; ///< Whether mutation and native module loading are safe.
    std::vector<project_error> diagnostics; ///< Non-fatal compatibility diagnostics.
};

/** @brief Canonical absolute roots derived from a validated descriptor. */
struct project_context
{
    std::filesystem::path descriptor_path; ///< Absolute descriptor path.
    std::filesystem::path root; ///< Absolute project root.
    std::vector<std::filesystem::path> asset_roots; ///< Absolute content roots.
    std::filesystem::path config_root; ///< Absolute configuration root.
    std::filesystem::path plugin_root; ///< Absolute plugin root.
    std::filesystem::path saved_root; ///< Absolute per-project editor state root.
    std::filesystem::path intermediate_root; ///< Absolute intermediate root.
    std::filesystem::path build_root; ///< Absolute build root.
    std::filesystem::path asset_cache_root; ///< Absolute asset registry and DDC root.
    std::filesystem::path recovery_root; ///< Absolute recovery root.
};

/** @brief Discoverable installed template. */
struct project_template_snapshot
{
    std::string id; ///< Stable template ID.
    std::string name; ///< Display name.
    std::string description; ///< User-facing purpose and content summary.
    std::string engine_version; ///< Template engine version, if fixed.
    std::filesystem::path root; ///< Absolute template directory.
};

/** @brief Inputs for atomic template-based project creation. */
struct create_project_request
{
    std::string name; ///< New project display name.
    std::filesystem::path destination; ///< Empty destination directory.
    std::string template_id; ///< Installed template ID.
    std::filesystem::path templates_root; ///< Root containing installed templates.
    std::string engine_version; ///< Exact engine version written to the descriptor.
};

/** @brief Plugin advertised by an engine installation. */
struct engine_plugin_snapshot
{
    std::string id; ///< Stable plugin ID.
    std::string version; ///< Installed exact version.
    std::vector<std::string> platforms; ///< Supported target-platform IDs.
};

/** @brief Resolved installation manifest with root-relative paths made absolute. */
struct engine_installation_manifest
{
    std::string installation_id; ///< Stable installation identity.
    std::string engine_version; ///< Installed ARC version.
    std::filesystem::path manifest_path; ///< Absolute authority manifest path.
    std::filesystem::path root; ///< Absolute installation root.
    std::filesystem::path editor; ///< Installed editor executable, if available.
    std::filesystem::path sdk; ///< Installed CMake SDK root.
    std::filesystem::path cooker; ///< Installed asset cooker executable.
    std::filesystem::path project_tool; ///< Installed project CLI executable.
    std::vector<std::string> platforms; ///< Supported target-platform IDs.
    std::vector<std::string> configurations; ///< Installed configurations.
    std::vector<engine_plugin_snapshot> plugins; ///< Available engine plugins.
    std::vector<project_template_snapshot> templates; ///< Available project templates.
    toolchain_requirements toolchain; ///< Installation toolchain requirements.
};

/** @brief Serialized installation-registry entries. */
struct engine_installation_registry
{
    std::vector<std::filesystem::path> manifests; ///< Absolute registered manifest paths.
};

/** @brief Detected compiler, SDK, generator, or IDE tool. */
struct tool_snapshot
{
    std::string id; ///< Stable tool ID.
    std::filesystem::path executable; ///< Resolved executable or SDK root.
    std::string version; ///< Detected version text.
    bool available{}; ///< Whether detection succeeded.
};

using descriptor_result = core::result<project_descriptor, project_error>;
using validation_result = core::result<project_validation_result, project_error>;
using context_result = core::result<project_context, project_error>;
using templates_result = core::result<std::vector<project_template_snapshot>, project_error>;
using installation_result = core::result<engine_installation_manifest, project_error>;
using installations_result = core::result<std::vector<engine_installation_manifest>, project_error>;
using tools_result = core::result<std::vector<tool_snapshot>, project_error>;

/** @brief Load a version-2 project descriptor. @param descriptor_path Descriptor file. @return Parsed descriptor or failure. */
[[nodiscard]] descriptor_result load_descriptor(const std::filesystem::path& descriptor_path);
/** @brief Atomically save a validated descriptor. @param descriptor_path Destination file. @param descriptor Value to save. @return Success or failure. */
[[nodiscard]] project_status save_descriptor(const std::filesystem::path& descriptor_path,
                                             const project_descriptor& descriptor);
/** @brief Validate descriptor structure, compatibility, and optional filesystem state. @param descriptor_path Descriptor identity. @param descriptor Value to validate. @param options Validation policy. @return Writability result or failure. */
[[nodiscard]] validation_result validate_descriptor(const std::filesystem::path& descriptor_path,
                                                    const project_descriptor& descriptor,
                                                    const project_validation_options& options = {});
/** @brief Resolve canonical project-local roots. @param descriptor_path Descriptor file. @param descriptor Parsed descriptor. @return Contained absolute roots or failure. */
[[nodiscard]] context_result resolve_context(const std::filesystem::path& descriptor_path,
                                            const project_descriptor& descriptor);
/** @brief Upgrade a legacy descriptor with a validated backup. @param descriptor_path Legacy descriptor. @param target_engine_version Exact target ARC version. @return Success or failure without partial publication. */
[[nodiscard]] project_status upgrade_descriptor(const std::filesystem::path& descriptor_path,
                                                std::string_view target_engine_version);

/** @brief Discover declarative templates. @param templates_root Installed template root. @return Sorted template snapshots or failure. */
[[nodiscard]] templates_result discover_templates(const std::filesystem::path& templates_root);
/** @brief Generate a complete project atomically through a staging directory. @param request Creation inputs. @return Success or failure. */
[[nodiscard]] project_status create_project(const create_project_request& request);

/** @brief Return the platform-specific per-user installation-registry path. */
[[nodiscard]] std::filesystem::path default_installation_registry_path();
/** @brief Load and resolve an installation manifest. @param manifest_path Manifest file. @return Resolved installation or failure. */
[[nodiscard]] installation_result load_installation_manifest(const std::filesystem::path& manifest_path);
/** @brief Atomically register an installation manifest. @param registry_path Registry file. @param manifest_path Authority manifest. @return Success or failure. */
[[nodiscard]] project_status register_installation(const std::filesystem::path& registry_path,
                                                   const std::filesystem::path& manifest_path);
/** @brief Remove an installation identity. @param registry_path Registry file. @param installation_id Stable installation ID. @return Success or failure. */
[[nodiscard]] project_status unregister_installation(const std::filesystem::path& registry_path,
                                                     std::string_view installation_id);
/** @brief Resolve every valid registered installation. @param registry_path Registry file. @return Sorted installations or failure. */
[[nodiscard]] installations_result discover_installations(const std::filesystem::path& registry_path);
/** @brief Rebuild a registry from surviving entries and explicit search roots. @param registry_path Registry file. @param search_roots Roots searched for manifests. @return Repaired installations or failure. */
[[nodiscard]] installations_result repair_installations(const std::filesystem::path& registry_path,
                                                        const std::vector<std::filesystem::path>& search_roots = {});

/** @brief Detect ARC-supported compilers, generators, IDE support, and Vulkan SDK state. @return Tool snapshots. */
[[nodiscard]] tools_result detect_toolchains();

} // namespace arc::project

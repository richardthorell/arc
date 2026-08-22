#pragma once

#include <arc/core/id.h>
#include <arc/core/result.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::render
{

/** @brief Backend shader output format. */
enum class shader_target : std::uint8_t
{
    spirv,
    dxil,
    msl
};

/** @brief Authoring and execution domain for a shader module. */
enum class shader_domain : std::uint8_t
{
    surface,
    post_process,
    compute
};

/** @brief Programmable pipeline stage containing an entry point. */
enum class shader_stage : std::uint8_t
{
    vertex,
    fragment,
    compute,
    ray_generation,
    closest_hit,
    any_hit,
    miss
};

/** @brief Optimization policy used by the offline compiler. */
enum class shader_optimization : std::uint8_t
{
    disabled,
    development,
    performance
};

/** @brief Renderer pass that may consume a surface shader contract. */
enum class material_pass : std::uint8_t
{
    depth,
    shadow,
    gbuffer,
    forward,
    motion,
    object_id,
    selection,
    ray_hit
};

/** @brief Type of one reflected material or shader parameter. */
enum class shader_parameter_type : std::uint8_t
{
    boolean,
    int32,
    uint32,
    float32,
    float2,
    float3,
    float4,
    matrix4x4,
    texture_2d,
    texture_cube,
    sampler
};

/** @brief Kind of a reflected shader resource binding. */
enum class shader_resource_kind : std::uint8_t
{
    constant_buffer,
    structured_buffer,
    read_write_buffer,
    sampled_texture,
    storage_texture,
    sampler,
    acceleration_structure
};

struct shader_entry_point_id_tag;
struct shader_parameter_id_tag;
struct shader_permutation_id_tag;
struct shader_generation_id_tag;
struct shader_package_id_tag;

/** Stable identifier for an entry point within an ARC shader package. */
using shader_entry_point_id = core::strong_id<shader_entry_point_id_tag, std::uint64_t, 0>;
/** Stable identifier for a material/shader parameter. */
using shader_parameter_id = core::strong_id<shader_parameter_id_tag, std::uint64_t, 0>;
/** Stable identifier for a compiled static permutation. */
using shader_permutation_id = core::strong_id<shader_permutation_id_tag, std::uint64_t, 0>;
/** Monotonic runtime generation of a published shader package. */
using shader_generation_id = core::strong_id<shader_generation_id_tag, std::uint64_t, 0>;
/** Stable authored identity of a shader package. */
using shader_package_id = core::uuid<shader_package_id_tag>;

/** @brief SHA-256 content identity used by shader packages and dependencies. */
struct shader_content_hash
{
    std::array<std::byte, 32> bytes{};

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        for (const std::byte value : bytes)
            if (value != std::byte{}) return false;
        return true;
    }

    friend constexpr auto operator<=>(const shader_content_hash&, const shader_content_hash&) noexcept = default;
};

/** @brief Source location associated with compiler output or generated graph code. */
struct shader_source_location
{
    std::string path;
    std::uint32_t line{};
    std::uint32_t column{};
    std::string graph_node_id;
};

/** @brief Severity of one structured compiler diagnostic. */
enum class shader_diagnostic_severity : std::uint8_t
{
    information,
    warning,
    error
};

/** @brief Actionable diagnostic emitted while compiling or validating a shader. */
struct shader_diagnostic
{
    shader_diagnostic_severity severity{shader_diagnostic_severity::error};
    std::string code;
    std::string message;
    shader_source_location location;
    std::vector<shader_source_location> include_stack;
    std::optional<shader_permutation_id> permutation;
};

/** @brief Reflected entry-point metadata. */
struct shader_entry_point_descriptor
{
    shader_entry_point_id id{};
    std::string name{"main"};
    shader_stage stage{shader_stage::fragment};
    std::string profile;
    std::array<std::uint32_t, 3> thread_group_size{1, 1, 1};
};

/** @brief Reflected parameter layout entry. */
struct shader_parameter_descriptor
{
    shader_parameter_id id{};
    std::string name;
    shader_parameter_type type{shader_parameter_type::float32};
    std::uint32_t offset{};
    std::uint32_t size{};
    std::vector<std::byte> default_value;
};

/** @brief Reflected resource binding independent of a graphics backend. */
struct shader_resource_descriptor
{
    std::string name;
    shader_resource_kind kind{shader_resource_kind::constant_buffer};
    std::uint32_t set{};
    std::uint32_t binding{};
    std::uint32_t count{1};
    bool writable{};
};

/** @brief Declared static switch and its selected value for a permutation. */
struct shader_static_switch
{
    shader_parameter_id id{};
    std::string name;
    bool value{};
};

/** @brief Surface-pass coverage and routing metadata. */
struct material_pass_support
{
    material_pass pass{material_pass::forward};
    shader_entry_point_id entry_point{};
    bool generated{};
};

/** @brief Shader compilation input. */
struct shader_compile_request
{
    std::string source_path;
    std::string source_override;
    std::string entry_point{"main"};
    std::string profile;
    std::string library_version{"arc-shader-library/1"};
    shader_domain domain{shader_domain::surface};
    shader_stage stage{shader_stage::fragment};
    shader_target target{shader_target::spirv};
    shader_optimization optimization{shader_optimization::development};
    std::vector<std::string> defines;
    std::vector<std::filesystem::path> include_directories;
    std::vector<shader_static_switch> static_switches;
    std::vector<material_pass> required_passes;
    std::unordered_map<std::uint32_t, std::string> generated_line_nodes;
    bool generate_debug_information{};
};

/** @brief Complete shader reflection data consumed by render backends and tools. */
struct shader_reflection
{
    shader_domain domain{shader_domain::surface};
    std::vector<shader_entry_point_descriptor> entry_points;
    std::vector<shader_parameter_descriptor> parameters;
    std::vector<shader_resource_descriptor> resources;
    std::vector<material_pass_support> passes;
    std::uint32_t parameter_block_size{};
    bool custom_lighting{};
    bool vertex_deformation{};
    bool previous_vertex_deformation{};
};

/** @brief One source/include dependency captured in a compiled package. */
struct shader_dependency
{
    std::string path;
    shader_content_hash content_hash{};
};

/** @brief Mapping from a generated Slang line to its authored source or graph node. */
struct shader_source_map_entry
{
    std::uint32_t generated_line{};
    shader_source_location source;
};

/** @brief Successfully compiled shader package slice. */
struct shader_compile_output
{
    std::vector<std::uint8_t> bytecode;
    shader_reflection reflection;
    shader_content_hash build_hash{};
    std::vector<shader_dependency> dependencies;
    std::vector<shader_source_map_entry> source_map;
    std::vector<shader_diagnostic> diagnostics;
    std::string compiler_fingerprint;
};

/** @brief Immutable cooked shader package containing one target slice. */
struct shader_package
{
    static constexpr std::uint32_t current_version = 2;

    std::uint32_t version{current_version};
    shader_package_id id{};
    shader_generation_id generation{};
    shader_target target{shader_target::spirv};
    shader_permutation_id permutation{};
    shader_compile_output compiled;
};

/** @brief Shader compilation or package failure categories. */
enum class shader_compile_error_code : std::uint8_t
{
    invalid_request,
    source_unavailable,
    dependency_cycle,
    compiler_unavailable,
    compilation_failed,
    reflection_failed,
    validation_failed,
    package_corrupt,
    permutation_limit_exceeded
};

/** @brief Structured shader compilation failure. */
struct shader_compile_error
{
    shader_compile_error_code code{shader_compile_error_code::compilation_failed};
    std::string source_path;
    std::string message;
    std::vector<shader_diagnostic> diagnostics;
};

using shader_compile_result = core::result<shader_compile_output, shader_compile_error>;
using shader_package_result = core::result<shader_package, shader_compile_error>;
using shader_package_bytes_result = core::result<std::vector<std::byte>, shader_compile_error>;

/** @brief Return a deterministic ID for a stable shader symbol name. */
[[nodiscard]] shader_parameter_id make_shader_parameter_id(std::string_view stable_name) noexcept;
/** @brief Return a deterministic ID for an entry point and stage. */
[[nodiscard]] shader_entry_point_id make_shader_entry_point_id(std::string_view stable_name,
                                                               shader_stage stage) noexcept;
/** @brief Return lowercase hexadecimal text for a shader content hash. */
[[nodiscard]] std::string to_string(const shader_content_hash& hash);

/** @brief Serialize a validated shader package to deterministic ARC_SHADER_2 bytes. */
[[nodiscard]] shader_package_bytes_result serialize_shader_package(const shader_package& package);

/** @brief Decode and validate deterministic ARC_SHADER_2 bytes. */
[[nodiscard]] shader_package_result deserialize_shader_package(std::span<const std::byte> bytes);

/**
 * @brief Backend-neutral shader compiler interface.
 *
 * Implementations belong to editor/cooker tooling. Shipping runtimes consume
 * `shader_package` data and do not need a source compiler.
 */
class shader_compiler
{
public:
    virtual ~shader_compiler() = default;
    [[nodiscard]] virtual shader_compile_result compile(const shader_compile_request& request) = 0;
    [[nodiscard]] virtual std::string_view fingerprint() const noexcept = 0;
};

/** @brief Shader library with content-addressed request caching. */
class shader_library_cache
{
public:
    [[nodiscard]] shader_compile_result compile_or_get(shader_compiler& compiler,
                                                       const shader_compile_request& request);
    [[nodiscard]] bool source_changed(const shader_compile_request& request) const;
    void clear() noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

private:
    struct cached_shader
    {
        shader_compile_result result;
        shader_content_hash request_hash{};
    };

    std::unordered_map<std::string, cached_shader> cache_;
};

/** @brief Result of atomically publishing a cooked shader generation. */
enum class shader_publication_status : std::uint8_t
{
    published,
    unchanged,
    rejected_stale_generation,
    rejected_incompatible_layout
};

/** @brief Snapshot of a runtime shader package and its last publication failure. */
struct shader_package_snapshot
{
    shader_package_id id{};
    shader_permutation_id permutation{};
    shader_generation_id generation{};
    shader_content_hash build_hash{};
    std::optional<shader_compile_error> last_error;
};

/**
 * @brief Thread-safe runtime store for validated cooked shader generations.
 *
 * Publication is atomic. Failed compilation can be reported separately and
 * never replaces the last-good package. Retired generations remain available
 * until the caller's completed-frame value reaches their retirement frame.
 */
class shader_package_library
{
public:
    [[nodiscard]] shader_publication_status publish(shader_package package, std::uint64_t retire_after_frame);
    void report_failure(shader_package_id id, shader_permutation_id permutation, shader_compile_error error);
    [[nodiscard]] std::optional<shader_package> find(shader_package_id id, shader_permutation_id permutation) const;
    [[nodiscard]] std::optional<shader_package_snapshot> snapshot(shader_package_id id,
                                                                  shader_permutation_id permutation) const;
    void collect(std::uint64_t completed_frame);
    void clear();
    [[nodiscard]] std::size_t active_count() const;
    [[nodiscard]] std::size_t retired_count() const;

private:
    struct entry
    {
        shader_package active;
        std::optional<shader_compile_error> last_error;
    };
    struct retired_entry
    {
        shader_package package;
        std::uint64_t retire_after_frame{};
    };

    [[nodiscard]] static std::string key(shader_package_id id, shader_permutation_id permutation);

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, entry> active_;
    std::vector<retired_entry> retired_;
};

} // namespace arc::render

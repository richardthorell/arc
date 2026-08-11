#pragma once

#include <arc/assets/assets.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace arc::assets
{

struct cook_processor_id
{
    std::uint64_t high{};
    std::uint64_t low{};

    constexpr bool valid() const noexcept
    {
        return high != 0 || low != 0;
    }
    friend constexpr auto operator<=>(const cook_processor_id&, const cook_processor_id&) noexcept = default;
};

struct artifact_schema_id
{
    std::uint64_t high{};
    std::uint64_t low{};

    constexpr bool valid() const noexcept
    {
        return high != 0 || low != 0;
    }
    friend constexpr auto operator<=>(const artifact_schema_id&, const artifact_schema_id&) noexcept = default;
};

using asset_build_key = asset_hash;
using content_hash = asset_hash;

std::string to_string(cook_processor_id value);
std::string to_string(artifact_schema_id value);

namespace cook_processor_ids
{
inline constexpr cook_processor_id source{0xa7ca55e700000003ull, 0x0000000000000001ull};
inline constexpr cook_processor_id mesh{0xa7ca55e700000003ull, 0x0000000000000002ull};
inline constexpr cook_processor_id texture{0xa7ca55e700000003ull, 0x0000000000000003ull};
inline constexpr cook_processor_id shader{0xa7ca55e700000003ull, 0x0000000000000004ull};
inline constexpr cook_processor_id material{0xa7ca55e700000003ull, 0x0000000000000005ull};
inline constexpr cook_processor_id scene{0xa7ca55e700000003ull, 0x0000000000000006ull};
inline constexpr cook_processor_id environment{0xa7ca55e700000003ull, 0x0000000000000007ull};
inline constexpr cook_processor_id animation{0xa7ca55e700000003ull, 0x0000000000000008ull};
inline constexpr cook_processor_id collision{0xa7ca55e700000003ull, 0x0000000000000009ull};
inline constexpr cook_processor_id navigation{0xa7ca55e700000003ull, 0x000000000000000aull};
inline constexpr cook_processor_id audio{0xa7ca55e700000003ull, 0x000000000000000bull};
} // namespace cook_processor_ids

namespace artifact_schemas
{
inline constexpr artifact_schema_id source{0xa7ca55e700000004ull, 0x0000000000000001ull};
inline constexpr artifact_schema_id mesh{0xa7ca55e700000004ull, 0x0000000000000002ull};
inline constexpr artifact_schema_id texture{0xa7ca55e700000004ull, 0x0000000000000003ull};
inline constexpr artifact_schema_id shader{0xa7ca55e700000004ull, 0x0000000000000004ull};
inline constexpr artifact_schema_id material{0xa7ca55e700000004ull, 0x0000000000000005ull};
inline constexpr artifact_schema_id scene{0xa7ca55e700000004ull, 0x0000000000000006ull};
inline constexpr artifact_schema_id package_manifest{0xa7ca55e700000004ull, 0x0000000000000007ull};
inline constexpr artifact_schema_id virtual_geometry{0xa7ca55e700000004ull, 0x0000000000000008ull};
inline constexpr artifact_schema_id surface_cards{0xa7ca55e700000004ull, 0x0000000000000009ull};
inline constexpr artifact_schema_id mesh_distance_field{0xa7ca55e700000004ull, 0x000000000000000aull};
} // namespace artifact_schemas

enum class cook_platform : std::uint8_t
{
    windows,
    linux_os,
    macos
};
enum class cook_architecture : std::uint8_t
{
    x86_64,
    arm64
};
enum class cook_renderer : std::uint8_t
{
    none,
    vulkan,
    direct3d12,
    metal
};
enum class cook_texture_family : std::uint8_t
{
    bc,
    astc,
    etc2,
    portable
};
enum class cook_configuration : std::uint8_t
{
    development,
    shipping
};

struct cook_target
{
    std::string name{"windows-x64-vulkan"};
    cook_platform platform{cook_platform::windows};
    cook_architecture architecture{cook_architecture::x86_64};
    cook_renderer renderer{cook_renderer::vulkan};
    cook_texture_family textures{cook_texture_family::bc};
    cook_configuration configuration{cook_configuration::shipping};
    std::uint32_t api_major{1};
    std::uint32_t api_minor{2};
    bool little_endian{true};
    std::vector<std::string> features;

    friend bool operator==(const cook_target&, const cook_target&) = default;
};

cook_target windows_vulkan_cook_target();
cook_target linux_vulkan_cook_target();
std::string canonical_cook_target(const cook_target& target);

struct asset_build_key_descriptor
{
    asset_hash source_hash{};
    std::vector<asset_hash> dependency_hashes;
    asset_importer_id importer{};
    std::uint32_t importer_version{};
    cook_processor_id processor{};
    std::uint32_t processor_version{};
    artifact_schema_id schema{};
    std::uint32_t schema_version{};
    std::string canonical_settings{"{}"};
    std::string toolchain_fingerprint;
    std::vector<asset_hash> shader_include_hashes;
    std::string shader_compiler_fingerprint;
    std::string shader_entry_point;
    std::vector<std::string> shader_defines;
    cook_target target{};
};

asset_build_key make_asset_build_key(const asset_build_key_descriptor& description);

enum class cache_layer : std::uint8_t
{
    none,
    local,
    shared
};
enum class cache_access : std::uint8_t
{
    read_write,
    read_only,
    offline
};

struct cache_error
{
    std::string message;
    explicit operator bool() const noexcept
    {
        return !message.empty();
    }
};

struct cache_blob
{
    content_hash hash{};
    std::vector<std::byte> bytes;
    cache_layer layer{cache_layer::none};
};

struct cache_action
{
    asset_build_key key{};
    std::vector<content_hash> artifacts;
    std::string metadata;
};

struct cache_statistics
{
    std::uint64_t local_hits{};
    std::uint64_t local_misses{};
    std::uint64_t shared_hits{};
    std::uint64_t shared_misses{};
    std::uint64_t bytes_read{};
    std::uint64_t bytes_written{};
    std::uint64_t bytes_downloaded{};
    std::uint64_t bytes_uploaded{};
    std::uint64_t corrupt_entries{};
    std::uint64_t evictions{};
    std::uint64_t avoided_processor_runs{};
    std::uint64_t local_bytes{};
    double hit_rate() const noexcept;
};

struct cache_cleanup_policy
{
    std::uint64_t maximum_bytes{50ull * 1024ull * 1024ull * 1024ull};
    float prune_threshold{0.90f};
    float prune_target{0.75f};
    std::chrono::hours temporary_lifetime{24};
    std::chrono::hours action_lifetime{24 * 30};
};

class shared_cache_backend
{
public:
    virtual ~shared_cache_backend() = default;
    virtual std::optional<std::vector<std::byte>> get_blob(content_hash hash, cache_error& error) = 0;
    virtual bool put_blob(content_hash hash, std::span<const std::byte> bytes, cache_error& error) = 0;
    virtual std::optional<cache_action> get_action(asset_build_key key, cache_error& error) = 0;
    virtual bool put_action(const cache_action& action, cache_error& error) = 0;
};

class filesystem_shared_cache final : public shared_cache_backend
{
public:
    explicit filesystem_shared_cache(std::filesystem::path root, bool read_only = false);
    ~filesystem_shared_cache() override;
    std::optional<std::vector<std::byte>> get_blob(content_hash hash, cache_error& error) override;
    bool put_blob(content_hash hash, std::span<const std::byte> bytes, cache_error& error) override;
    std::optional<cache_action> get_action(asset_build_key key, cache_error& error) override;
    bool put_action(const cache_action& action, cache_error& error) override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

enum class http_cache_method : std::uint8_t
{
    head,
    get,
    put
};

struct http_cache_request
{
    http_cache_method method{http_cache_method::get};
    std::string url;
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::byte> body;
};

struct http_cache_response
{
    std::uint16_t status{};
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::byte> body;
    std::string error;
};

using http_cache_transport = std::function<http_cache_response(const http_cache_request&)>;

struct http_shared_cache_config
{
    std::string endpoint;
    std::string bearer_token;
    bool read_only{};
    http_cache_transport transport;
};

class http_shared_cache final : public shared_cache_backend
{
public:
    explicit http_shared_cache(http_shared_cache_config config);
    ~http_shared_cache() override;
    std::optional<std::vector<std::byte>> get_blob(content_hash hash, cache_error& error) override;
    bool put_blob(content_hash hash, std::span<const std::byte> bytes, cache_error& error) override;
    std::optional<cache_action> get_action(asset_build_key key, cache_error& error) override;
    bool put_action(const cache_action& action, cache_error& error) override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

struct derived_data_cache_config
{
    std::filesystem::path root;
    cache_access access{cache_access::read_write};
    cache_cleanup_policy cleanup{};
    std::shared_ptr<shared_cache_backend> shared;
    bool require_shared{};
};

class derived_data_cache
{
public:
    explicit derived_data_cache(derived_data_cache_config config);
    ~derived_data_cache();
    derived_data_cache(derived_data_cache&&) noexcept;
    derived_data_cache& operator=(derived_data_cache&&) noexcept;
    derived_data_cache(const derived_data_cache&) = delete;
    derived_data_cache& operator=(const derived_data_cache&) = delete;

    std::optional<cache_blob> get_blob(content_hash hash, cache_error& error);
    std::optional<cache_action> get_action(asset_build_key key, cache_error& error);
    bool put_blob(content_hash hash, std::span<const std::byte> bytes, cache_error& error);
    bool put_action(const cache_action& action, cache_error& error);
    bool pin(content_hash hash);
    bool unpin(content_hash hash);
    std::size_t verify(std::vector<std::string>* diagnostics = nullptr);
    std::uint64_t prune(bool force = false);
    void note_avoided_processor_run();
    cache_statistics statistics() const;
    const derived_data_cache_config& config() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

struct cooked_artifact
{
    std::string name;
    std::string extension;
    artifact_schema_id schema{};
    std::uint32_t schema_version{1};
    content_hash hash{};
    std::uint64_t size{};
    bool gpu_compressed{};
    std::vector<std::byte> bytes;
};

struct asset_cook_context
{
    asset_snapshot asset;
    source_asset_data source;
    cook_target target;
    std::string canonical_settings{"{}"};
    std::vector<asset_snapshot> dependencies;
    jobs::cancellation_token cancellation;
};

struct [[nodiscard]] asset_cook_result
{
    std::vector<cooked_artifact> artifacts;
    std::vector<asset_diagnostic> diagnostics;
    asset_error error;
    bool succeeded() const noexcept
    {
        return !error && !artifacts.empty();
    }
};

struct asset_cook_processor_descriptor
{
    cook_processor_id id{};
    std::string name;
    std::uint32_t version{1};
    artifact_schema_id schema{};
    std::uint32_t schema_version{1};
    jobs::job_affinity affinity{jobs::job_affinity::any_worker};
    std::vector<asset_type_id> input_types;
};

class asset_cook_processor
{
public:
    virtual ~asset_cook_processor() = default;
    virtual const asset_cook_processor_descriptor& descriptor() const noexcept = 0;
    virtual std::string toolchain_fingerprint() const = 0;
    virtual asset_cook_result cook(const asset_cook_context& context) = 0;
};

struct cook_manifest_artifact
{
    asset_guid asset{};
    asset_type_id type{};
    std::string name;
    artifact_schema_id schema{};
    std::uint32_t schema_version{};
    content_hash hash{};
    std::uint64_t size{};
    std::string chunk{"startup"};
    std::uint64_t offset{};
    std::uint64_t stored_size{};
    bool compressed{};
};

struct cook_manifest
{
    static constexpr std::uint32_t current_version = 1;
    std::uint32_t version{current_version};
    std::string build_id;
    cook_target target{};
    std::vector<asset_guid> roots;
    std::vector<asset_guid> dependency_closure;
    std::vector<cook_manifest_artifact> artifacts;
};

struct cook_request
{
    std::vector<asset_guid> roots;
    cook_target target{windows_vulkan_cook_target()};
    std::filesystem::path output;
    bool fail_on_warning{};
    jobs::cancellation_token cancellation;
};

struct [[nodiscard]] cook_result
{
    cook_manifest manifest;
    std::size_t cooked{};
    std::size_t cache_hits{};
    std::vector<asset_diagnostic> diagnostics;
    asset_error error;
    bool succeeded() const noexcept
    {
        return !error;
    }
};

class asset_cooker
{
public:
    asset_cooker(asset_manager& assets, derived_data_cache& cache);
    ~asset_cooker();
    bool register_processor(std::unique_ptr<asset_cook_processor> processor);
    cook_result cook(const cook_request& request);

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

[[nodiscard]] asset_status save_cook_manifest(const std::filesystem::path& path, const cook_manifest& manifest);
using cook_manifest_result = core::result<cook_manifest, asset_error>;
[[nodiscard]] cook_manifest_result load_cook_manifest(const std::filesystem::path& path);
[[nodiscard]] asset_status verify_cook_manifest(const cook_manifest& manifest, derived_data_cache& cache);

struct [[nodiscard]] package_build_result
{
    std::filesystem::path manifest_path;
    std::vector<std::filesystem::path> chunks;
    std::uint64_t stored_bytes{};
    std::uint64_t source_bytes{};
    std::string error;
    bool succeeded() const noexcept
    {
        return error.empty();
    }
};

package_build_result build_asset_packages(cook_manifest manifest, derived_data_cache& cache,
                                          const std::filesystem::path& output);

class asset_package_mount
{
public:
    asset_package_mount();
    ~asset_package_mount();
    asset_package_mount(asset_package_mount&&) noexcept;
    asset_package_mount& operator=(asset_package_mount&&) noexcept;
    asset_package_mount(const asset_package_mount&) = delete;
    asset_package_mount& operator=(const asset_package_mount&) = delete;

    [[nodiscard]] asset_status mount(const std::filesystem::path& manifest_path);
    [[nodiscard]] core::result<std::vector<std::byte>, asset_error> read(asset_guid asset,
                                                                         artifact_schema_id schema) const;
    jobs::job_future<io::file_result<io::file_buffer>> read_async(asset_guid asset, artifact_schema_id schema,
                                                                  io::async_file_service& files,
                                                                  jobs::cancellation_token cancellation = {}) const;
    const cook_manifest& manifest() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace arc::assets

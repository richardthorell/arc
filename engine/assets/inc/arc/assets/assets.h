#pragma once

/** @namespace arc::assets
 * @brief Persistent asset identity, importing, loading, residency, and diagnostics.
 */

#include <arc/framework/service.h>
#include <arc/core/id.h>
#include <arc/io/io.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>

#include <array>
#include <atomic>
#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

namespace arc::assets
{

struct asset_guid_tag;
struct asset_type_id_tag;
struct asset_importer_id_tag;

using asset_guid = core::uuid<asset_guid_tag>;
using asset_type_id = core::uuid<asset_type_id_tag>;
using asset_importer_id = core::uuid<asset_importer_id_tag>;

static_assert(sizeof(asset_guid) == 16);
static_assert(sizeof(asset_type_id) == 16);
static_assert(sizeof(asset_importer_id) == 16);
static_assert(std::is_standard_layout_v<asset_guid>);
static_assert(std::is_trivially_copyable_v<asset_guid>);

struct asset_guid_hash
{
    [[nodiscard]] std::size_t operator()(asset_guid value) const noexcept;
};

struct asset_type_id_hash
{
    [[nodiscard]] std::size_t operator()(asset_type_id value) const noexcept;
};

struct asset_importer_id_hash
{
    [[nodiscard]] std::size_t operator()(asset_importer_id value) const noexcept;
};

[[nodiscard]] std::string to_string(asset_guid value);
[[nodiscard]] std::string to_string(asset_type_id value);
[[nodiscard]] std::string to_string(asset_importer_id value);
[[nodiscard]] std::optional<asset_guid> parse_asset_guid(std::string_view text) noexcept;
[[nodiscard]] std::optional<asset_type_id> parse_asset_type_id(std::string_view text) noexcept;
[[nodiscard]] std::optional<asset_importer_id> parse_asset_importer_id(std::string_view text) noexcept;
[[nodiscard]] asset_guid generate_asset_guid() noexcept;

struct asset_hash
{
    std::array<std::byte, 32> bytes{};

    [[nodiscard]] constexpr bool empty() const noexcept
    {
        for (std::byte value : bytes)
            if (value != std::byte{}) return false;
        return true;
    }

    friend constexpr auto operator<=>(const asset_hash&, const asset_hash&) noexcept = default;
};

[[nodiscard]] std::string to_string(const asset_hash& value);
[[nodiscard]] std::optional<asset_hash> parse_asset_hash(std::string_view text) noexcept;
[[nodiscard]] asset_hash hash_bytes(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] asset_hash combine_hashes(std::span<const asset_hash> hashes) noexcept;

namespace asset_types
{
inline constexpr asset_type_id scene{0xa7ca55e700000001ull, 0x0000000000000001ull};
inline constexpr asset_type_id prefab{0xa7ca55e700000001ull, 0x0000000000000002ull};
inline constexpr asset_type_id material{0xa7ca55e700000001ull, 0x0000000000000003ull};
inline constexpr asset_type_id shader{0xa7ca55e700000001ull, 0x0000000000000004ull};
inline constexpr asset_type_id texture_2d{0xa7ca55e700000001ull, 0x0000000000000005ull};
inline constexpr asset_type_id environment{0xa7ca55e700000001ull, 0x0000000000000006ull};
inline constexpr asset_type_id imported_scene{0xa7ca55e700000001ull, 0x0000000000000007ull};
inline constexpr asset_type_id static_mesh{0xa7ca55e700000001ull, 0x0000000000000008ull};
inline constexpr asset_type_id binary_blob{0xa7ca55e700000001ull, 0x0000000000000009ull};
inline constexpr asset_type_id animation_clip{0xa7ca55e700000001ull, 0x000000000000000aull};
inline constexpr asset_type_id collision{0xa7ca55e700000001ull, 0x000000000000000bull};
inline constexpr asset_type_id navigation{0xa7ca55e700000001ull, 0x000000000000000cull};
inline constexpr asset_type_id audio_clip{0xa7ca55e700000001ull, 0x000000000000000dull};
inline constexpr asset_type_id material_instance{0xa7ca55e700000001ull, 0x000000000000000eull};
inline constexpr asset_type_id unknown{0xa7ca55e700000001ull, 0xffffffffffffffffull};
} // namespace asset_types

namespace importer_ids
{
inline constexpr asset_importer_id scene{0xa7ca55e700000002ull, 0x0000000000000001ull};
inline constexpr asset_importer_id prefab{0xa7ca55e700000002ull, 0x0000000000000002ull};
inline constexpr asset_importer_id material{0xa7ca55e700000002ull, 0x0000000000000003ull};
inline constexpr asset_importer_id shader{0xa7ca55e700000002ull, 0x0000000000000004ull};
inline constexpr asset_importer_id texture{0xa7ca55e700000002ull, 0x0000000000000005ull};
inline constexpr asset_importer_id environment{0xa7ca55e700000002ull, 0x0000000000000006ull};
inline constexpr asset_importer_id gltf{0xa7ca55e700000002ull, 0x0000000000000007ull};
inline constexpr asset_importer_id fbx{0xa7ca55e700000002ull, 0x0000000000000008ull};
inline constexpr asset_importer_id binary{0xa7ca55e700000002ull, 0x0000000000000009ull};
inline constexpr asset_importer_id animation{0xa7ca55e700000002ull, 0x000000000000000aull};
inline constexpr asset_importer_id collision{0xa7ca55e700000002ull, 0x000000000000000bull};
inline constexpr asset_importer_id navigation{0xa7ca55e700000002ull, 0x000000000000000cull};
inline constexpr asset_importer_id audio{0xa7ca55e700000002ull, 0x000000000000000dull};
inline constexpr asset_importer_id material_instance{0xa7ca55e700000002ull, 0x000000000000000eull};
} // namespace importer_ids

namespace fallback_assets
{
inline constexpr asset_guid missing_mesh{0xa7ca55e7f0000001ull, 0x0000000000000001ull};
inline constexpr asset_guid error_material{0xa7ca55e7f0000001ull, 0x0000000000000002ull};
inline constexpr asset_guid white_texture{0xa7ca55e7f0000001ull, 0x0000000000000003ull};
inline constexpr asset_guid black_texture{0xa7ca55e7f0000001ull, 0x0000000000000004ull};
inline constexpr asset_guid normal_texture{0xa7ca55e7f0000001ull, 0x0000000000000005ull};
inline constexpr asset_guid neutral_environment{0xa7ca55e7f0000001ull, 0x0000000000000006ull};
} // namespace fallback_assets

enum class asset_state : std::uint8_t
{
    unknown,
    queued,
    importing,
    ready,
    stale,
    failed
};

enum class asset_streaming_priority : std::uint8_t
{
    background,
    low,
    normal,
    high,
    critical
};

enum class asset_residency : std::uint8_t
{
    metadata_only,
    source,
    derived,
    cpu,
    device
};

enum class asset_error_code : std::uint8_t
{
    none,
    not_found,
    type_mismatch,
    invalid_metadata,
    duplicate_guid,
    dependency_cycle,
    dependency_failed,
    importer_missing,
    import_failed,
    cancelled,
    io_failed,
    database_failed,
    budget_exceeded,
    invalid_request
};

enum class asset_diagnostic_severity : std::uint8_t
{
    information,
    warning,
    error
};

struct asset_reference
{
    asset_guid guid{};
    asset_type_id expected_type{};
    std::string path_hint;

    [[nodiscard]] constexpr bool resolved() const noexcept
    {
        return guid.valid();
    }
    friend bool operator==(const asset_reference&, const asset_reference&) = default;
};

struct asset_error
{
    asset_error_code code{asset_error_code::none};
    asset_guid guid{};
    std::filesystem::path path;
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return code != asset_error_code::none;
    }
};

struct asset_source_metadata;

/** @brief Hash one source file, returning structured I/O failure context. */
using asset_hash_result = core::result<asset_hash, asset_error>;

/** @brief Status returned by asset metadata writes. */
using asset_status = core::status<asset_error>;

[[nodiscard]] asset_hash_result hash_file(const std::filesystem::path& path);

struct asset_subasset_metadata
{
    std::string persistent_key;
    asset_guid guid{};
    asset_type_id type{};
    std::string name;
    bool tombstoned{};
};

struct asset_source_metadata
{
    static constexpr std::uint32_t current_format_version = 1;

    std::uint32_t format_version{current_format_version};
    asset_guid guid{};
    asset_type_id type{};
    asset_importer_id importer{};
    std::uint32_t settings_version{1};
    std::string canonical_settings{"{}"};
    std::vector<asset_subasset_metadata> subassets;
};

/** @brief Result of loading authored asset metadata. */
using asset_metadata_result = core::result<asset_source_metadata, asset_error>;

struct asset_importer_snapshot
{
    asset_importer_id id{};
    std::string name;
    std::uint32_t version{};
    std::uint32_t settings_version{};
    std::vector<std::string> extensions;
    std::vector<asset_type_id> output_types;
};

struct asset_artifact_snapshot
{
    std::string name;
    std::filesystem::path path;
    asset_hash content_hash{};
    std::uint64_t size{};
    asset_residency residency{asset_residency::derived};
};

struct asset_diagnostic
{
    std::uint64_t sequence{};
    asset_diagnostic_severity severity{asset_diagnostic_severity::information};
    asset_guid guid{};
    std::string category;
    std::string message;
};

struct missing_asset_reference
{
    asset_reference reference;
    std::string owner;
    std::string field;
    std::vector<asset_guid> repair_candidates;
    std::string reason;
};

struct asset_snapshot
{
    asset_guid guid{};
    asset_type_id type{};
    asset_importer_id importer{};
    std::uint32_t importer_version{};
    std::uint32_t imported_version{};
    std::filesystem::path source_path;
    asset_hash source_hash{};
    asset_hash dependency_hash{};
    asset_state state{asset_state::unknown};
    asset_residency residency{asset_residency::metadata_only};
    std::uint64_t generation{};
    std::uint64_t revision{};
    std::uint32_t strong_references{};
    std::uint32_t pins{};
    bool source_missing{};
    bool has_last_good{};
    bool read_only{};
    std::vector<asset_guid> dependencies;
    std::vector<asset_guid> reverse_dependencies;
    std::vector<asset_subasset_metadata> subassets;
    std::vector<asset_artifact_snapshot> artifacts;
    std::vector<asset_diagnostic> diagnostics;
};

struct asset_registry_snapshot
{
    std::uint64_t revision{};
    std::filesystem::path project_root;
    std::filesystem::path asset_root;
    std::filesystem::path database_path;
    std::filesystem::path derived_data_root;
    std::vector<asset_snapshot> assets;
    std::vector<missing_asset_reference> missing_references;
};

struct asset_importer_descriptor
{
    asset_importer_id id{};
    std::string name;
    std::uint32_t version{1};
    std::uint32_t settings_version{1};
    jobs::job_affinity affinity{jobs::job_affinity::any_worker};
    std::vector<std::string> extensions;
    std::vector<asset_type_id> output_types;
};

struct asset_import_artifact
{
    std::string name;
    std::string extension;
    std::vector<std::byte> bytes;
    asset_residency residency{asset_residency::derived};
};

struct source_asset_data
{
    std::filesystem::path source_path;
    asset_hash source_hash{};
    std::vector<std::byte> bytes;
};

class asset_payload
{
public:
    asset_payload() = default;

    template <class T>
    static asset_payload make(asset_type_id type, std::shared_ptr<const T> value,
                              std::size_t resident_bytes = sizeof(T))
    {
        asset_payload result;
        result.type_ = type;
        result.value_ = std::move(value);
        result.cpp_type_ = &typeid(T);
        result.resident_bytes_ = resident_bytes;
        return result;
    }

    template <class T> const T* get() const noexcept
    {
        return cpp_type_ && *cpp_type_ == typeid(T) ? static_cast<const T*>(value_.get()) : nullptr;
    }

    asset_type_id type() const noexcept
    {
        return type_;
    }
    std::size_t resident_bytes() const noexcept
    {
        return resident_bytes_;
    }
    explicit operator bool() const noexcept
    {
        return value_ != nullptr;
    }

private:
    asset_type_id type_{};
    std::shared_ptr<const void> value_;
    const std::type_info* cpp_type_{};
    std::size_t resident_bytes_{};
};

struct asset_import_context
{
    asset_reference reference;
    asset_source_metadata metadata;
    std::filesystem::path project_root;
    std::filesystem::path source_path;
    std::filesystem::path derived_data_root;
    std::span<const std::byte> source_bytes;
    asset_hash source_hash{};
    asset_streaming_priority priority{asset_streaming_priority::normal};
    asset_residency requested_residency{asset_residency::cpu};
    jobs::cancellation_token cancellation;
};

struct [[nodiscard]] asset_import_result
{
    asset_payload payload;
    std::vector<asset_reference> dependencies;
    // Importers own their dependency list. An empty authoritative list removes
    // dependencies reported by a previous generation.
    bool dependencies_authoritative{true};
    std::vector<asset_import_artifact> artifacts;
    std::vector<asset_subasset_metadata> subassets;
    std::vector<asset_diagnostic> diagnostics;
    asset_residency residency{asset_residency::cpu};
    asset_error error;

    bool succeeded() const noexcept
    {
        return !error && static_cast<bool>(payload);
    }
};

class asset_importer
{
public:
    virtual ~asset_importer() = default;
    virtual const asset_importer_descriptor& descriptor() const noexcept = 0;
    virtual asset_import_result import(const asset_import_context& context) = 0;
};

struct asset_manager_config
{
    std::filesystem::path project_root;
    std::filesystem::path asset_root;
    std::vector<std::filesystem::path> additional_source_roots;
    std::vector<std::filesystem::path> read_only_source_roots;
    std::filesystem::path cache_root;
    std::string target_profile{"desktop"};
    std::size_t streaming_heap_bytes{256u * 1024u * 1024u};
    bool create_missing_metadata{true};
    bool enable_source_monitor{
#if defined(ARC_BUILD_SHIPPING)
        false
#else
        true
#endif
    };
    std::chrono::milliseconds source_poll_interval{500};
    std::chrono::milliseconds change_debounce{200};
};

struct [[nodiscard]] asset_scan_result
{
    std::size_t discovered{};
    std::size_t updated{};
    std::size_t missing{};
    std::size_t metadata_created{};
    std::vector<asset_diagnostic> diagnostics;
    asset_error error;

    bool succeeded() const noexcept
    {
        return !error;
    }
};

struct [[nodiscard]] asset_move_result
{
    asset_guid guid{};
    std::filesystem::path previous_path;
    std::filesystem::path current_path;
    asset_error error;

    bool succeeded() const noexcept
    {
        return !error;
    }
};

struct asset_load_request
{
    asset_reference reference;
    asset_streaming_priority priority{asset_streaming_priority::normal};
    asset_residency residency{asset_residency::cpu};
    jobs::cancellation_token cancellation;
    bool allow_fallback{true};
};

namespace detail
{
struct asset_slot
{
    asset_guid requested_guid{};
    asset_guid resolved_guid{};
    asset_type_id type{};
    std::atomic<std::uint64_t> generation{};
    std::atomic<std::uint32_t> pins{};
    std::atomic<std::shared_ptr<const asset_payload>> payload;
};
} // namespace detail

template <class T> class [[nodiscard]] asset_handle
{
public:
    asset_handle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept
    {
        return slot_ && slot_->payload.load(std::memory_order_acquire) != nullptr;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return valid();
    }
    [[nodiscard]] asset_guid requested_guid() const noexcept
    {
        return requested_guid_.valid() ? requested_guid_ : slot_ ? slot_->requested_guid : asset_guid{};
    }
    [[nodiscard]] asset_guid resolved_guid() const noexcept
    {
        return slot_ ? slot_->resolved_guid : asset_guid{};
    }
    [[nodiscard]] asset_type_id type() const noexcept
    {
        return slot_ ? slot_->type : asset_type_id{};
    }
    [[nodiscard]] std::uint64_t generation() const noexcept
    {
        return slot_ ? slot_->generation.load(std::memory_order_acquire) : 0;
    }
    [[nodiscard]] bool using_fallback() const noexcept
    {
        return requested_guid() != resolved_guid();
    }

    [[nodiscard]] const T* get() const noexcept
    {
        snapshot_ = slot_ ? slot_->payload.load(std::memory_order_acquire) : nullptr;
        return snapshot_ ? snapshot_->template get<T>() : nullptr;
    }
    [[nodiscard]] const T& operator*() const noexcept
    {
        return *get();
    }
    [[nodiscard]] const T* operator->() const noexcept
    {
        return get();
    }

private:
    explicit asset_handle(std::shared_ptr<detail::asset_slot> slot, asset_guid requested_guid = {})
        : slot_(std::move(slot)), requested_guid_(requested_guid)
    {
    }

    std::shared_ptr<detail::asset_slot> slot_;
    asset_guid requested_guid_{};
    mutable std::shared_ptr<const asset_payload> snapshot_;
    friend class asset_manager;
};

class asset_pin
{
public:
    asset_pin() = default;
    ~asset_pin();
    asset_pin(asset_pin&& other) noexcept;
    asset_pin& operator=(asset_pin&& other) noexcept;
    asset_pin(const asset_pin&) = delete;
    asset_pin& operator=(const asset_pin&) = delete;

    [[nodiscard]] bool valid() const noexcept
    {
        return slot_ != nullptr;
    }

private:
    explicit asset_pin(std::shared_ptr<detail::asset_slot> slot);
    void reset() noexcept;

    std::shared_ptr<detail::asset_slot> slot_;
    friend class asset_manager;
};

template <class T> struct [[nodiscard]] asset_load_result
{
    asset_handle<T> asset;
    asset_error error;

    [[nodiscard]] bool succeeded() const noexcept
    {
        return !error && asset.valid();
    }
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return succeeded();
    }
};

template <class T> class [[nodiscard]] asset_load_handle
{
public:
    asset_load_handle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept
    {
        return future_.valid();
    }
    [[nodiscard]] bool ready() const noexcept
    {
        return future_.ready();
    }
    [[nodiscard]] jobs::job_status status() const noexcept
    {
        return future_.status();
    }
    [[nodiscard]] float progress() const noexcept
    {
        return progress_ ? progress_->load(std::memory_order_relaxed) : 0.0f;
    }
    bool cancel() noexcept
    {
        return cancellation_ && cancellation_->request_cancel();
    }
    [[nodiscard]] asset_load_result<T> get() const
    {
        return future_.get();
    }
    [[nodiscard]] const jobs::job_handle& job() const noexcept
    {
        return future_.handle();
    }

#if defined(ARC_ENABLE_JOB_COROUTINES)
    auto operator co_await() const
    {
        return future_.operator co_await();
    }
#endif

private:
    explicit asset_load_handle(jobs::job_future<asset_load_result<T>> future,
                               std::shared_ptr<jobs::cancellation_source> cancellation,
                               std::shared_ptr<std::atomic<float>> progress)
        : future_(std::move(future)), cancellation_(std::move(cancellation)), progress_(std::move(progress))
    {
    }
    jobs::job_future<asset_load_result<T>> future_;
    std::shared_ptr<jobs::cancellation_source> cancellation_;
    std::shared_ptr<std::atomic<float>> progress_;
    friend class asset_manager;
};

enum class asset_event_kind : std::uint8_t
{
    discovered,
    state_changed,
    progress,
    dependencies_changed,
    published,
    evicted,
    missing_reference,
    failed,
    moved
};

struct asset_event
{
    std::uint64_t sequence{};
    std::uint64_t registry_revision{};
    asset_event_kind kind{asset_event_kind::discovered};
    asset_guid guid{};
    asset_state state{asset_state::unknown};
    float progress{};
    std::string message;
};

using asset_event_callback = std::function<void(const asset_event&)>;

class asset_manager final : public framework::runtime_service
{
public:
    static constexpr framework::runtime_service_id service_id = framework::make_runtime_service_id("arc.assets");

    using framework::runtime_service::dependencies;

    asset_manager(asset_manager_config config, jobs::job_system& jobs, io::async_file_service& files,
                  memory::memory_system& memory);
    ~asset_manager() override;

    asset_manager(const asset_manager&) = delete;
    asset_manager& operator=(const asset_manager&) = delete;

    framework::runtime_service_id id() const noexcept override
    {
        return service_id;
    }
    std::string_view name() const noexcept override
    {
        return "ARC Assets";
    }
    void on_start(framework::runtime_service_context&) override;
    void on_shutdown(framework::runtime_service_context&) noexcept override;

    bool register_importer(std::unique_ptr<asset_importer> importer);
    bool register_virtual_asset(asset_guid guid, asset_type_id type, asset_payload payload, std::string name,
                                bool pin = true);
    bool set_fallback(asset_type_id type, asset_guid guid);
    [[nodiscard]] asset_guid fallback_for(asset_type_id type) const noexcept;
    [[nodiscard]] std::vector<asset_importer_snapshot> importers() const;
    [[nodiscard]] asset_scan_result scan();
    void poll();

    [[nodiscard]] std::optional<asset_snapshot> find(asset_guid guid) const;
    [[nodiscard]] std::optional<asset_snapshot> find(std::string_view project_relative_path) const;
    [[nodiscard]] std::vector<asset_snapshot> search(std::string_view text = {},
                                                     std::optional<asset_type_id> type = std::nullopt) const;
    [[nodiscard]] asset_registry_snapshot snapshot() const;
    [[nodiscard]] std::vector<asset_guid> dependencies(asset_guid guid) const;
    [[nodiscard]] std::vector<asset_guid> reverse_dependencies(asset_guid guid) const;

    [[nodiscard]] asset_reference resolve(std::string_view project_relative_path,
                                          asset_type_id expected_type = {}) const;
    [[nodiscard]] missing_asset_reference audit_reference(const asset_reference& reference, std::string owner,
                                                          std::string field);

    bool set_dependencies(asset_guid guid, std::span<const asset_reference> dependencies);
    bool mark_stale(asset_guid guid, std::string reason);
    [[nodiscard]] jobs::job_handle reimport(asset_guid guid,
                                            asset_streaming_priority priority = asset_streaming_priority::normal,
                                            jobs::cancellation_token cancellation = {});
    bool cancel_import(asset_guid guid);
    [[nodiscard]] asset_move_result move(asset_guid guid, std::filesystem::path destination);
    [[nodiscard]] asset_move_result rename(asset_guid guid, std::string filename);

    template <class T> [[nodiscard]] asset_load_handle<T> load(asset_load_request request)
    {
        std::shared_ptr<jobs::cancellation_source> owned_cancellation;
        if (!request.cancellation.valid())
        {
            owned_cancellation = std::make_shared<jobs::cancellation_source>();
            request.cancellation = owned_cancellation->token();
        }
        auto progress = std::make_shared<std::atomic<float>>(0.0f);
        auto untyped = load_untyped(request);
        auto future = jobs().submit_future(
            {.name = "assets.load.typed", .priority = to_job_priority(request.priority)},
            [untyped = std::move(untyped), progress]() mutable
            {
                auto result = untyped.get();
                asset_load_result<T> typed;
                typed.error = result.error;
                if (result.slot)
                {
                    const auto payload = result.slot->payload.load(std::memory_order_acquire);
                    if (payload && !payload->template get<T>())
                    {
                        typed.error = {.code = asset_error_code::type_mismatch,
                                       .guid = result.slot->requested_guid,
                                       .message = "Loaded asset payload does not match the requested C++ type"};
                    }
                    else
                    {
                        typed.asset = asset_handle<T>(std::move(result.slot), result.error.guid);
                    }
                }
                progress->store(1.0f, std::memory_order_relaxed);
                return typed;
            });
        return asset_load_handle<T>(std::move(future), std::move(owned_cancellation), std::move(progress));
    }

    [[nodiscard]] jobs::job_handle prefetch(asset_load_request request);
    [[nodiscard]] asset_pin pin(asset_guid guid);
    std::size_t evict_unused(asset_residency maximum_residency = asset_residency::device);

    [[nodiscard]] std::uint64_t subscribe(asset_event_callback callback);
    bool unsubscribe(std::uint64_t token);
    [[nodiscard]] std::vector<asset_event> events_since(std::uint64_t sequence) const;

    [[nodiscard]] const asset_manager_config& config() const noexcept;
    [[nodiscard]] jobs::job_system& jobs() const noexcept;
    [[nodiscard]] static jobs::job_priority to_job_priority(asset_streaming_priority priority) noexcept;

private:
    struct [[nodiscard]] untyped_load_result
    {
        std::shared_ptr<detail::asset_slot> slot;
        asset_error error;
    };

    jobs::job_future<untyped_load_result> load_untyped(asset_load_request request);

    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

[[nodiscard]] std::filesystem::path metadata_path_for(const std::filesystem::path& source_path);
[[nodiscard]] asset_metadata_result load_asset_metadata(const std::filesystem::path& path);
[[nodiscard]] asset_status save_asset_metadata(const std::filesystem::path& path,
                                               const asset_source_metadata& metadata);
[[nodiscard]] std::string normalize_asset_path(const std::filesystem::path& path);
std::optional<std::pair<asset_type_id, asset_importer_id>>
classify_asset_path(const std::filesystem::path& path) noexcept;

} // namespace arc::assets

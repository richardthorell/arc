#pragma once

/** @file
 * @brief Backend-neutral asset search document, query, result and index contracts.
 *
 * Search consumes logical Asset Database records. It never discovers asset
 * identity by scanning storage paths. Concrete lexical/vector backends live
 * behind asset_search_index.
 */

#include <arc/assets/asset_database.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace arc::assets
{

inline constexpr std::uint32_t asset_search_document_schema_version = 1;
inline constexpr std::uint32_t asset_search_index_schema_version = 1;

enum class asset_search_metadata_origin : std::uint8_t
{
    intrinsic,
    authored,
    generated
};

struct asset_search_metadata_value
{
    std::string key;
    std::string value;
    asset_search_metadata_origin origin{asset_search_metadata_origin::intrinsic};

    friend bool operator==(const asset_search_metadata_value&, const asset_search_metadata_value&) = default;
};

/**
 * Stable, backend-neutral representation of one searchable logical asset.
 *
 * Embeddings intentionally do not live inline here. Later semantic/visual
 * providers can attach versioned derived records keyed by GUID + source
 * revision without changing the logical document contract.
 */
struct asset_search_document
{
    asset_guid guid{};
    asset_type_id type{};
    std::string display_name;
    std::string logical_path;
    std::string active_provider;
    std::string title;
    std::string description;
    std::vector<std::string> authored_tags;
    std::vector<std::string> generated_tags;
    std::vector<std::string> categories;
    std::vector<asset_search_metadata_value> metadata;
    std::uint64_t source_generation{};
    std::uint64_t source_revision{};
    std::uint32_t schema_version{asset_search_document_schema_version};

    friend bool operator==(const asset_search_document&, const asset_search_document&) = default;
};

struct asset_search_filters
{
    std::optional<asset_type_id> type;
    std::vector<std::string> provider_ids;
};

struct asset_search_query
{
    std::string text;
    asset_search_filters filters;
    std::size_t limit{100};
    std::size_t offset{};
};

enum class asset_search_match_kind : std::uint8_t
{
    exact_name,
    prefix,
    lexical,
    fuzzy,
    metadata,
    semantic,
    visual,
    relation,
    relaxed
};

struct asset_search_match_reason
{
    asset_search_match_kind kind{asset_search_match_kind::lexical};
    float score{};
    std::string detail;

    friend bool operator==(const asset_search_match_reason&, const asset_search_match_reason&) = default;
};

struct [[nodiscard]] asset_search_result
{
    asset_guid guid{};
    float score{};
    std::vector<asset_search_match_reason> reasons;
};

/**
 * Search backend boundary.
 *
 * Implementations may use SQLite/FTS, in-memory structures, HNSW, or another
 * backend. Callers must not depend on the concrete index. For equal ranking
 * scores implementations must provide deterministic ordering by GUID.
 */
class asset_search_index
{
public:
    virtual ~asset_search_index() = default;

    [[nodiscard]] virtual asset_status reset(std::span<const asset_search_document> documents) = 0;
    [[nodiscard]] virtual asset_status upsert(asset_search_document document) = 0;
    [[nodiscard]] virtual asset_status erase(asset_guid guid) = 0;
    [[nodiscard]] virtual std::optional<std::uint64_t> source_change_cursor() const = 0;
    [[nodiscard]] virtual asset_status set_source_change_cursor(std::uint64_t cursor) = 0;
    [[nodiscard]] virtual std::vector<asset_search_result> query(const asset_search_query& query) const = 0;
};

/**
 * Persistent local search-document cache backed by SQLite.
 *
 * This backend intentionally provides only the S1 baseline matching needed to
 * validate persistence and the query contract. S2 owns FTS/fuzzy search,
 * facets, structured predicates and richer ranking.
 *
 * The SQLite connection uses serialized mode, but compound synchronization
 * ownership belongs to asset_search_service. Production callers should prefer
 * the service rather than interleave direct backend mutations and queries.
 */
class persistent_asset_search_index final : public asset_search_index
{
public:
    using open_result = core::result<std::unique_ptr<persistent_asset_search_index>, asset_error>;

    [[nodiscard]] static open_result open(std::filesystem::path database_path);
    ~persistent_asset_search_index() override;

    persistent_asset_search_index(const persistent_asset_search_index&) = delete;
    persistent_asset_search_index& operator=(const persistent_asset_search_index&) = delete;

    [[nodiscard]] asset_status reset(std::span<const asset_search_document> documents) override;
    [[nodiscard]] asset_status upsert(asset_search_document document) override;
    [[nodiscard]] asset_status erase(asset_guid guid) override;
    [[nodiscard]] std::optional<std::uint64_t> source_change_cursor() const override;
    [[nodiscard]] asset_status set_source_change_cursor(std::uint64_t cursor) override;
    [[nodiscard]] std::vector<asset_search_result> query(const asset_search_query& query) const override;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    struct implementation;
    explicit persistent_asset_search_index(std::unique_ptr<implementation> implementation);
    std::unique_ptr<implementation> implementation_;
};

/** Build the searchable logical document for an Asset Database record. */
[[nodiscard]] asset_search_document make_asset_search_document(const asset_database_record& record);

/**
 * Keeps one search index synchronized with an Asset Database change cursor.
 *
 * rebuild() is the recovery path for first use, cache loss, schema migration,
 * catalog reset, or lost change history. refresh() applies incremental changes.
 *
 * The indexer is a single-writer synchronization primitive. Its cursor is
 * persisted by the index only after document changes succeed, making replay
 * idempotent after interruption. asset_search_service provides serialization
 * for normal multi-threaded callers.
 */
class asset_search_indexer
{
public:
    asset_search_indexer(const asset_database& database, asset_search_index& index)
        : database_(&database), index_(&index), cursor_(index.source_change_cursor().value_or(0))
    {
    }

    [[nodiscard]] asset_status rebuild();
    [[nodiscard]] asset_status refresh();

    [[nodiscard]] std::uint64_t cursor() const noexcept
    {
        return cursor_;
    }

private:
    const asset_database* database_{};
    asset_search_index* index_{};
    std::uint64_t cursor_{};
};

/**
 * Thread-safe owner for one persistent search cache and its Asset Database
 * synchronization state.
 *
 * synchronize() resumes from the persisted change cursor when possible and
 * performs a full rebuild when no valid checkpoint exists or history was lost.
 * The service owns the backend/indexer lifetimes and serializes public
 * query/synchronization operations so callers never need to know that the
 * initial backend is SQLite.
 */
class asset_search_service final
{
public:
    using open_result = core::result<std::unique_ptr<asset_search_service>, asset_error>;

    [[nodiscard]] static open_result open(const asset_database& database, std::filesystem::path cache_path);
    ~asset_search_service();

    asset_search_service(const asset_search_service&) = delete;
    asset_search_service& operator=(const asset_search_service&) = delete;

    [[nodiscard]] asset_status synchronize();
    [[nodiscard]] asset_status rebuild();
    [[nodiscard]] asset_status refresh();
    [[nodiscard]] std::vector<asset_search_result> query(const asset_search_query& query) const;

    [[nodiscard]] std::uint64_t cursor() const;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    asset_search_service(const asset_database& database, std::unique_ptr<persistent_asset_search_index> index);

    const asset_database* database_{};
    std::unique_ptr<persistent_asset_search_index> index_;
    std::unique_ptr<asset_search_indexer> indexer_;
    mutable std::mutex mutex_;
};

} // namespace arc::assets

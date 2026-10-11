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
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace arc::assets
{

inline constexpr std::uint32_t asset_search_document_schema_version = 1;

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

struct asset_search_result
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

    virtual void reset(std::span<const asset_search_document> documents) = 0;
    virtual void upsert(asset_search_document document) = 0;
    virtual void erase(asset_guid guid) = 0;
    [[nodiscard]] virtual std::vector<asset_search_result> query(const asset_search_query& query) const = 0;
};

/** Build the searchable logical document for an Asset Database record. */
[[nodiscard]] asset_search_document make_asset_search_document(const asset_database_record& record);

/**
 * Keeps one search index synchronized with an Asset Database change cursor.
 *
 * rebuild() is the recovery path for first use, cache loss, schema migration,
 * catalog reset, or lost change history. refresh() applies incremental changes.
 */
class asset_search_indexer
{
public:
    asset_search_indexer(const asset_database& database, asset_search_index& index) noexcept
        : database_(&database), index_(&index)
    {
    }

    void rebuild();
    void refresh();

    [[nodiscard]] std::uint64_t cursor() const noexcept
    {
        return cursor_;
    }

private:
    const asset_database* database_{};
    asset_search_index* index_{};
    std::uint64_t cursor_{};
};

} // namespace arc::assets

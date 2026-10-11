#include <arc/assets/asset_search.h>

#include <algorithm>

namespace arc::assets
{
namespace
{

asset_status apply_changes(const asset_database& database, asset_search_index& index,
                           const asset_database_change_batch& batch)
{
    for (const auto& change : batch.events)
    {
        if (change.kind == asset_database_change_kind::catalog_reset || !change.guid.valid()) continue;

        asset_status status = asset_status::success();
        if (change.kind == asset_database_change_kind::removed)
            status = index.erase(change.guid);
        else if (const auto record = database.query(change.guid))
            status = index.upsert(make_asset_search_document(*record));
        else
            status = index.erase(change.guid);

        if (!status) return status;
    }
    return asset_status::success();
}

} // namespace

asset_search_document make_asset_search_document(const asset_database_record& record)
{
    asset_search_document document;
    document.guid = record.guid;
    document.type = record.type;
    document.display_name = record.title;
    document.logical_path = record.source_hint;
    document.active_provider = record.active_provider;
    document.title = record.title;
    document.description = record.description;
    document.source_generation = record.generation;
    document.source_revision = record.revision;
    return document;
}

asset_status asset_search_indexer::rebuild()
{
    if (!database_ || !index_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search indexer is not initialized"});

    const auto records = database_->records();
    std::vector<asset_search_document> documents;
    documents.reserve(records.size());
    for (const auto& record : records)
        documents.push_back(make_asset_search_document(record));

    auto status = index_->reset(documents);
    if (!status) return status;

    // Replaying retained events after the snapshot is intentionally idempotent.
    // It closes the race between taking the database snapshot and observing the
    // latest change cursor. history_lost is harmless here because the complete
    // record snapshot is already the baseline.
    const auto batch = database_->changes_since(0);
    status = apply_changes(*database_, *index_, batch);
    if (!status) return status;
    status = index_->set_source_change_cursor(batch.next_sequence);
    if (!status) return status;
    cursor_ = batch.next_sequence;
    return asset_status::success();
}

asset_status asset_search_indexer::refresh()
{
    if (!database_ || !index_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search indexer is not initialized"});

    const auto batch = database_->changes_since(cursor_);
    if (batch.history_lost ||
        std::any_of(batch.events.begin(), batch.events.end(), [](const asset_database_change& change)
                    { return change.kind == asset_database_change_kind::catalog_reset; }))
    {
        return rebuild();
    }

    auto status = apply_changes(*database_, *index_, batch);
    if (!status) return status;
    status = index_->set_source_change_cursor(batch.next_sequence);
    if (!status) return status;
    cursor_ = batch.next_sequence;
    return asset_status::success();
}

asset_search_service::asset_search_service(const asset_database& database,
                                           std::unique_ptr<persistent_asset_search_index> index)
    : database_(&database), index_(std::move(index)),
      indexer_(std::make_unique<asset_search_indexer>(database, *index_))
{
}

asset_search_service::~asset_search_service() = default;

asset_search_service::open_result asset_search_service::open(const asset_database& database,
                                                             std::filesystem::path cache_path)
{
    auto opened = persistent_asset_search_index::open(std::move(cache_path));
    if (!opened) return open_result::failure(std::move(opened).error());

    auto service = std::unique_ptr<asset_search_service>(new asset_search_service(database, std::move(opened).value()));
    auto status = service->synchronize();
    if (!status) return open_result::failure(std::move(status).error());
    return open_result::success(std::move(service));
}

asset_status asset_search_service::synchronize()
{
    std::scoped_lock lock(mutex_);
    if (!database_ || !index_ || !indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});

    if (!index_->source_change_cursor()) return indexer_->rebuild();
    return indexer_->refresh();
}

asset_status asset_search_service::rebuild()
{
    std::scoped_lock lock(mutex_);
    if (!indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});
    return indexer_->rebuild();
}

asset_status asset_search_service::refresh()
{
    std::scoped_lock lock(mutex_);
    if (!indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});
    return indexer_->refresh();
}

std::vector<asset_search_result> asset_search_service::query(const asset_search_query& query_value) const
{
    std::scoped_lock lock(mutex_);
    return index_ ? index_->query(query_value) : std::vector<asset_search_result>{};
}

std::uint64_t asset_search_service::cursor() const
{
    std::scoped_lock lock(mutex_);
    return indexer_ ? indexer_->cursor() : 0;
}

const std::filesystem::path& asset_search_service::path() const noexcept
{
    static const std::filesystem::path empty;
    return index_ ? index_->path() : empty;
}

} // namespace arc::assets

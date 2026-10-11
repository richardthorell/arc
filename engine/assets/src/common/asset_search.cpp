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
    cursor_ = batch.next_sequence;
    return asset_status::success();
}

} // namespace arc::assets

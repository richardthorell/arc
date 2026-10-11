#include <arc/assets/asset_search.h>

#include <algorithm>

namespace arc::assets
{
namespace
{

void apply_changes(const asset_database& database, asset_search_index& index, const asset_database_change_batch& batch)
{
    for (const auto& change : batch.events)
    {
        if (change.kind == asset_database_change_kind::catalog_reset || !change.guid.valid()) continue;

        if (change.kind == asset_database_change_kind::removed)
        {
            index.erase(change.guid);
            continue;
        }

        if (const auto record = database.query(change.guid))
            index.upsert(make_asset_search_document(*record));
        else
            index.erase(change.guid);
    }
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

void asset_search_indexer::rebuild()
{
    if (!database_ || !index_) return;

    const auto records = database_->records();
    std::vector<asset_search_document> documents;
    documents.reserve(records.size());
    for (const auto& record : records)
        documents.push_back(make_asset_search_document(record));

    index_->reset(documents);

    // Replaying retained events after the snapshot is intentionally idempotent.
    // It closes the race between taking the database snapshot and observing the
    // latest change cursor. history_lost is harmless here because the complete
    // record snapshot is already the baseline.
    const auto batch = database_->changes_since(0);
    apply_changes(*database_, *index_, batch);
    cursor_ = batch.next_sequence;
}

void asset_search_indexer::refresh()
{
    if (!database_ || !index_) return;

    const auto batch = database_->changes_since(cursor_);
    if (batch.history_lost ||
        std::any_of(batch.events.begin(), batch.events.end(), [](const asset_database_change& change)
                    { return change.kind == asset_database_change_kind::catalog_reset; }))
    {
        rebuild();
        return;
    }

    apply_changes(*database_, *index_, batch);
    cursor_ = batch.next_sequence;
}

} // namespace arc::assets

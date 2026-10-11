#include <arc/assets/asset_search.h>

#include <algorithm>

namespace arc::assets
{

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

    // Replaying from zero after the snapshot is intentionally idempotent. It
    // closes the race between taking the database snapshot and observing the
    // latest change cursor without requiring storage-specific transactions.
    cursor_ = 0;
    refresh();
}

void asset_search_indexer::refresh()
{
    if (!database_ || !index_) return;

    const auto batch = database_->changes_since(cursor_);
    if (batch.history_lost ||
        std::any_of(batch.events.begin(), batch.events.end(),
                    [](const asset_database_change& change)
                    { return change.kind == asset_database_change_kind::catalog_reset; }))
    {
        rebuild();
        return;
    }

    for (const auto& change : batch.events)
    {
        if (!change.guid.valid()) continue;

        if (change.kind == asset_database_change_kind::removed)
        {
            index_->erase(change.guid);
            continue;
        }

        if (const auto record = database_->query(change.guid))
            index_->upsert(make_asset_search_document(*record));
        else
            index_->erase(change.guid);
    }

    cursor_ = batch.next_sequence;
}

} // namespace arc::assets

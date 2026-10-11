#include <arc/assets/asset_search.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

namespace
{

class fake_asset_database final : public arc::assets::asset_database
{
public:
    std::optional<arc::assets::asset_database_record>
    query(arc::assets::asset_guid guid, arc::assets::asset_type_id expected_type = {}) const override
    {
        const auto it = std::find_if(records_.begin(), records_.end(),
                                     [&](const auto& record) { return record.guid == guid; });
        if (it == records_.end()) return std::nullopt;
        if (expected_type.valid() && it->type != expected_type) return std::nullopt;
        return *it;
    }

    std::vector<arc::assets::asset_guid> dependencies(arc::assets::asset_guid guid) const override
    {
        if (const auto record = query(guid)) return record->dependencies;
        return {};
    }

    std::vector<arc::assets::asset_guid> reverse_dependencies(arc::assets::asset_guid guid) const override
    {
        if (const auto record = query(guid)) return record->reverse_dependencies;
        return {};
    }

    std::vector<arc::assets::asset_database_record> records() const override
    {
        return records_;
    }

    std::uint64_t revision() const override
    {
        return revision_;
    }

    arc::assets::asset_database_change_batch changes_since(std::uint64_t cursor) const override
    {
        arc::assets::asset_database_change_batch batch;
        batch.next_sequence = cursor;
        batch.history_lost = history_lost_ && cursor != 0;
        for (const auto& event : events_)
        {
            batch.next_sequence = std::max(batch.next_sequence, event.sequence);
            if (event.sequence > cursor) batch.events.push_back(event);
        }
        return batch;
    }

    std::vector<arc::assets::asset_database_record> records_;
    std::vector<arc::assets::asset_database_change> events_;
    std::uint64_t revision_{};
    bool history_lost_{};
};

class recording_search_index final : public arc::assets::asset_search_index
{
public:
    void reset(std::span<const arc::assets::asset_search_document> documents) override
    {
        ++reset_count;
        documents_.assign(documents.begin(), documents.end());
    }

    void upsert(arc::assets::asset_search_document document) override
    {
        const auto it = std::find_if(documents_.begin(), documents_.end(),
                                     [&](const auto& existing) { return existing.guid == document.guid; });
        if (it == documents_.end())
            documents_.push_back(std::move(document));
        else
            *it = std::move(document);
    }

    void erase(arc::assets::asset_guid guid) override
    {
        std::erase_if(documents_, [&](const auto& document) { return document.guid == guid; });
    }

    std::vector<arc::assets::asset_search_result>
    query(const arc::assets::asset_search_query&) const override
    {
        return {};
    }

    int reset_count{};
    std::vector<arc::assets::asset_search_document> documents_;
};

arc::assets::asset_database_record make_record(std::string title, std::uint64_t revision = 1)
{
    using namespace arc::assets;
    return {.guid = generate_asset_guid(),
            .type = asset_types::binary_blob,
            .title = std::move(title),
            .description = "searchable description",
            .source_hint = "assets/content/example.bin",
            .active_provider = std::string(asset_provider_ids::project_source),
            .generation = 3,
            .revision = revision};
}

} // namespace

TEST_CASE("asset search document preserves logical identity and source revision")
{
    using namespace arc::assets;
    const auto record = make_record("Old Bicycle", 17);

    const auto document = make_asset_search_document(record);

    CHECK(document.schema_version == asset_search_document_schema_version);
    CHECK(document.guid == record.guid);
    CHECK(document.type == record.type);
    CHECK(document.display_name == "Old Bicycle");
    CHECK(document.logical_path == "assets/content/example.bin");
    CHECK(document.active_provider == asset_provider_ids::project_source);
    CHECK(document.description == "searchable description");
    CHECK(document.source_generation == 3);
    CHECK(document.source_revision == 17);
}

TEST_CASE("asset search indexer rebuilds from Asset Database records and replays changes")
{
    using namespace arc::assets;
    fake_asset_database database;
    recording_search_index index;

    auto bicycle = make_record("Bicycle");
    auto barrel = make_record("Barrel");
    database.records_ = {bicycle, barrel};
    database.events_ = {{.sequence = 1, .kind = asset_database_change_kind::added, .guid = bicycle.guid},
                        {.sequence = 2, .kind = asset_database_change_kind::added, .guid = barrel.guid}};

    asset_search_indexer indexer(database, index);
    indexer.rebuild();

    CHECK(index.reset_count == 1);
    CHECK(index.documents_.size() == 2);
    CHECK(indexer.cursor() == 2);
}

TEST_CASE("asset search indexer incrementally upserts and removes logical assets")
{
    using namespace arc::assets;
    fake_asset_database database;
    recording_search_index index;

    auto bicycle = make_record("Bicycle");
    database.records_ = {bicycle};
    database.events_ = {{.sequence = 1, .kind = asset_database_change_kind::added, .guid = bicycle.guid}};

    asset_search_indexer indexer(database, index);
    indexer.rebuild();

    auto chair = make_record("Chair", 4);
    database.records_.push_back(chair);
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::added, .guid = chair.guid});
    indexer.refresh();

    REQUIRE(index.documents_.size() == 2);
    CHECK(std::any_of(index.documents_.begin(), index.documents_.end(),
                      [&](const auto& document) { return document.guid == chair.guid; }));
    CHECK(indexer.cursor() == 2);

    std::erase_if(database.records_, [&](const auto& record) { return record.guid == bicycle.guid; });
    database.events_.push_back({.sequence = 3, .kind = asset_database_change_kind::removed, .guid = bicycle.guid});
    indexer.refresh();

    REQUIRE(index.documents_.size() == 1);
    CHECK(index.documents_.front().guid == chair.guid);
    CHECK(indexer.cursor() == 3);
}

TEST_CASE("asset search indexer recovers from lost incremental history with a full rebuild")
{
    using namespace arc::assets;
    fake_asset_database database;
    recording_search_index index;

    auto bicycle = make_record("Bicycle");
    database.records_ = {bicycle};
    database.events_ = {{.sequence = 1, .kind = asset_database_change_kind::added, .guid = bicycle.guid}};

    asset_search_indexer indexer(database, index);
    indexer.rebuild();
    REQUIRE(index.reset_count == 1);

    auto updated = bicycle;
    updated.title = "Weathered Bicycle";
    updated.revision = 2;
    database.records_ = {updated};
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::updated, .guid = bicycle.guid});
    database.history_lost_ = true;

    indexer.refresh();

    CHECK(index.reset_count == 2);
    REQUIRE(index.documents_.size() == 1);
    CHECK(index.documents_.front().display_name == "Weathered Bicycle");
    CHECK(index.documents_.front().source_revision == 2);
    CHECK(indexer.cursor() == 2);
}

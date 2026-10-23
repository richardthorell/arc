#include <arc/assets/asset_search.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace
{

class fake_asset_database final : public arc::assets::asset_database
{
public:
    std::optional<arc::assets::asset_database_record>
    query(arc::assets::asset_guid guid, arc::assets::asset_type_id expected_type = {}) const override
    {
        const auto it =
            std::find_if(records_.begin(), records_.end(), [&](const auto& record) { return record.guid == guid; });
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
        ++records_calls_;
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
    mutable int records_calls_{};
};

class recording_search_index final : public arc::assets::asset_search_index
{
public:
    arc::assets::asset_status reset(std::span<const arc::assets::asset_search_document> documents) override
    {
        ++reset_count;
        documents_.assign(documents.begin(), documents.end());
        source_cursor_.reset();
        return arc::assets::asset_status::success();
    }

    arc::assets::asset_status upsert(arc::assets::asset_search_document document) override
    {
        const auto it = std::find_if(documents_.begin(), documents_.end(),
                                     [&](const auto& existing) { return existing.guid == document.guid; });
        if (it == documents_.end())
            documents_.push_back(std::move(document));
        else
            *it = std::move(document);
        return arc::assets::asset_status::success();
    }

    arc::assets::asset_status erase(arc::assets::asset_guid guid) override
    {
        std::erase_if(documents_, [&](const auto& document) { return document.guid == guid; });
        return arc::assets::asset_status::success();
    }

    std::optional<std::uint64_t> source_change_cursor() const override
    {
        return source_cursor_;
    }

    arc::assets::asset_status set_source_change_cursor(std::uint64_t cursor) override
    {
        source_cursor_ = cursor;
        return arc::assets::asset_status::success();
    }

    std::vector<arc::assets::asset_search_result> query(const arc::assets::asset_search_query&) const override
    {
        return {};
    }

    int reset_count{};
    std::optional<std::uint64_t> source_cursor_;
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
    auto record = make_record("Old Bicycle", 17);
    record.authored_tags = {"bicycle", "vintage"};
    record.authored_categories = {"Vehicle/Bicycle", "Style/Rustic"};

    const auto document = make_asset_search_document(record);

    CHECK(document.schema_version == asset_search_document_schema_version);
    CHECK(document.guid == record.guid);
    CHECK(document.type == record.type);
    CHECK(document.display_name == "Old Bicycle");
    CHECK(document.logical_path == "assets/content/example.bin");
    CHECK(document.active_provider == asset_provider_ids::project_source);
    CHECK(document.description == "searchable description");
    CHECK(document.authored_tags == record.authored_tags);
    CHECK(document.categories == record.authored_categories);
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
    REQUIRE(indexer.rebuild());

    CHECK(index.reset_count == 1);
    CHECK(index.documents_.size() == 2);
    CHECK(indexer.cursor() == 2);
    REQUIRE(index.source_change_cursor());
    CHECK(*index.source_change_cursor() == 2);
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
    REQUIRE(indexer.rebuild());

    auto chair = make_record("Chair", 4);
    database.records_.push_back(chair);
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::added, .guid = chair.guid});
    REQUIRE(indexer.refresh());

    REQUIRE(index.documents_.size() == 2);
    CHECK(std::any_of(index.documents_.begin(), index.documents_.end(),
                      [&](const auto& document) { return document.guid == chair.guid; }));
    CHECK(indexer.cursor() == 2);

    std::erase_if(database.records_, [&](const auto& record) { return record.guid == bicycle.guid; });
    database.events_.push_back({.sequence = 3, .kind = asset_database_change_kind::removed, .guid = bicycle.guid});
    REQUIRE(indexer.refresh());

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
    REQUIRE(indexer.rebuild());
    REQUIRE(index.reset_count == 1);

    auto updated = bicycle;
    updated.title = "Weathered Bicycle";
    updated.revision = 2;
    database.records_ = {updated};
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::updated, .guid = bicycle.guid});
    database.history_lost_ = true;

    REQUIRE(indexer.refresh());

    CHECK(index.reset_count == 2);
    REQUIRE(index.documents_.size() == 1);
    CHECK(index.documents_.front().display_name == "Weathered Bicycle");
    CHECK(index.documents_.front().source_revision == 2);
    CHECK(indexer.cursor() == 2);
}

TEST_CASE("persistent asset search index survives reopen and ranks deterministic baseline matches")
{
    using namespace arc::assets;
    const auto root = std::filesystem::temp_directory_path() / ("arc-search-index-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    const auto exact_guid = generate_asset_guid();
    const auto prefix_guid = generate_asset_guid();
    const auto path_guid = generate_asset_guid();

    {
        auto opened = persistent_asset_search_index::open(database_path);
        REQUIRE(opened);
        auto index = std::move(opened).value();

        const std::vector<asset_search_document> documents{
            {.guid = exact_guid,
             .type = asset_types::static_mesh,
             .display_name = "Bike",
             .logical_path = "assets/props/bike.arcmesh",
             .active_provider = std::string(asset_provider_ids::project_source),
             .title = "Bike",
             .description = "Modern bicycle",
             .source_generation = 1,
             .source_revision = 1},
            {.guid = prefix_guid,
             .type = asset_types::static_mesh,
             .display_name = "Bike Rack",
             .logical_path = "assets/props/bike-rack.arcmesh",
             .active_provider = std::string(asset_provider_ids::project_source),
             .title = "Bike Rack",
             .description = "Metal rack",
             .source_generation = 1,
             .source_revision = 1},
            {.guid = path_guid,
             .type = asset_types::texture_2d,
             .display_name = "Surface 01",
             .logical_path = "assets/bikes/rusty-surface.png",
             .active_provider = std::string(asset_provider_ids::project_source),
             .title = "Surface 01",
             .description = "Weathered surface",
             .source_generation = 1,
             .source_revision = 1}};
        REQUIRE(index->reset(documents));

        const auto results = index->query({.text = "bike"});
        REQUIRE(results.size() == 3);
        CHECK(results[0].guid == exact_guid);
        CHECK(results[0].reasons.front().kind == asset_search_match_kind::exact_name);
        CHECK(results[1].guid == prefix_guid);
        CHECK(results[1].reasons.front().kind == asset_search_match_kind::prefix);
        CHECK(results[2].guid == path_guid);
    }

    {
        auto reopened = persistent_asset_search_index::open(database_path);
        REQUIRE(reopened);
        auto index = std::move(reopened).value();

        const auto filtered = index->query({.text = "bike", .filters = {.type = asset_types::static_mesh}, .limit = 1});
        REQUIRE(filtered.size() == 1);
        CHECK(filtered.front().guid == exact_guid);

        const auto second_page =
            index->query({.text = "bike", .filters = {.type = asset_types::static_mesh}, .limit = 1, .offset = 1});
        REQUIRE(second_page.size() == 1);
        CHECK(second_page.front().guid == prefix_guid);
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index rejects incompatible documents atomically")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-index-invalid-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    asset_search_document good{
        .guid = generate_asset_guid(), .type = asset_types::material, .display_name = "Stone", .title = "Stone"};
    REQUIRE(index->upsert(good));

    auto incompatible = good;
    incompatible.guid = generate_asset_guid();
    incompatible.schema_version = asset_search_document_schema_version + 1;
    CHECK_FALSE(index->upsert(incompatible));

    const std::vector documents{good, incompatible};
    CHECK_FALSE(index->reset(documents));

    const auto results = index->query({.text = "Stone"});
    REQUIRE(results.size() == 1);
    CHECK(results.front().guid == good.guid);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index removes documents and derived metadata together")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-index-erase-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    asset_search_document document{.guid = generate_asset_guid(),
                                   .type = asset_types::material,
                                   .display_name = "Rusty Metal",
                                   .title = "Rusty Metal",
                                   .authored_tags = {"metal"},
                                   .generated_tags = {"rusty"},
                                   .categories = {"Material/Metal"},
                                   .metadata = {{"condition", "weathered", asset_search_metadata_origin::generated}}};
    REQUIRE(index->upsert(document));
    REQUIRE(index->query({.text = "Rusty"}).size() == 1);

    REQUIRE(index->erase(document.guid));
    CHECK(index->query({.text = "Rusty"}).empty());

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index recreates a corrupt derived cache")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-index-corrupt-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";
    std::filesystem::create_directories(root);

    {
        std::ofstream output(database_path, std::ios::binary | std::ios::trunc);
        REQUIRE(output);
        output << "not a sqlite database";
    }

    {
        auto opened = persistent_asset_search_index::open(database_path);
        REQUIRE(opened);
        auto index = std::move(opened).value();

        asset_search_document document{.guid = generate_asset_guid(),
                                       .type = asset_types::static_mesh,
                                       .display_name = "Recovered Bike",
                                       .title = "Recovered Bike"};
        REQUIRE(index->upsert(document));
        const auto results = index->query({.text = "Recovered Bike"});
        REQUIRE(results.size() == 1);
        CHECK(results.front().guid == document.guid);
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index persists the Asset Database change checkpoint")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-checkpoint-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    {
        auto opened = persistent_asset_search_index::open(database_path);
        REQUIRE(opened);
        auto index = std::move(opened).value();
        CHECK_FALSE(index->source_change_cursor());
        REQUIRE(index->set_source_change_cursor(42));
        REQUIRE(index->source_change_cursor());
        CHECK(*index->source_change_cursor() == 42);
    }

    {
        auto reopened = persistent_asset_search_index::open(database_path);
        REQUIRE(reopened);
        auto index = std::move(reopened).value();
        REQUIRE(index->source_change_cursor());
        CHECK(*index->source_change_cursor() == 42);
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index rejects source revisions older than the cached document")
{
    using namespace arc::assets;
    const auto root = std::filesystem::temp_directory_path() / ("arc-search-stale-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    const auto guid = generate_asset_guid();
    asset_search_document current{.guid = guid,
                                  .type = asset_types::static_mesh,
                                  .display_name = "Current Bicycle",
                                  .title = "Current Bicycle",
                                  .source_generation = 7,
                                  .source_revision = 12};
    REQUIRE(index->upsert(current));

    auto stale = current;
    stale.display_name = "Stale Bicycle";
    stale.title = "Stale Bicycle";
    stale.source_generation = 99;
    stale.source_revision = 11;
    CHECK_FALSE(index->upsert(stale));

    auto stale_generation = current;
    stale_generation.display_name = "Older Generation Bicycle";
    stale_generation.title = "Older Generation Bicycle";
    stale_generation.source_generation = 6;
    CHECK_FALSE(index->upsert(stale_generation));

    const auto results = index->query({.text = "Current Bicycle"});
    REQUIRE(results.size() == 1);
    CHECK(results.front().guid == guid);
    CHECK(index->query({.text = "Stale Bicycle"}).empty());
    CHECK(index->query({.text = "Older Generation Bicycle"}).empty());

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("asset search service resumes incrementally from its persisted checkpoint")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-service-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    fake_asset_database database;
    auto bicycle = make_record("Bicycle", 1);
    database.records_ = {bicycle};
    database.events_ = {{.sequence = 1, .kind = asset_database_change_kind::added, .guid = bicycle.guid}};

    {
        auto opened = asset_search_service::open(database, database_path);
        REQUIRE(opened);
        auto service = std::move(opened).value();
        CHECK(service->cursor() == 1);
        REQUIRE(service->query({.text = "Bicycle"}).size() == 1);
        CHECK(database.records_calls_ == 1);
    }

    auto chair = make_record("Chair", 2);
    database.records_.push_back(chair);
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::added, .guid = chair.guid});

    {
        auto reopened = asset_search_service::open(database, database_path);
        REQUIRE(reopened);
        auto service = std::move(reopened).value();

        CHECK(service->cursor() == 2);
        REQUIRE(service->query({.text = "Bicycle"}).size() == 1);
        REQUIRE(service->query({.text = "Chair"}).size() == 1);
        // A second full records() snapshot would mean startup ignored the
        // persisted cursor and rebuilt rather than resuming incrementally.
        CHECK(database.records_calls_ == 1);
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("asset search service rebuilds when persisted change history is no longer available")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-service-history-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    fake_asset_database database;
    auto bicycle = make_record("Bicycle", 1);
    database.records_ = {bicycle};
    database.events_ = {{.sequence = 1, .kind = asset_database_change_kind::added, .guid = bicycle.guid}};

    {
        auto opened = asset_search_service::open(database, database_path);
        REQUIRE(opened);
        auto service = std::move(opened).value();
        CHECK(database.records_calls_ == 1);
    }

    auto replacement = make_record("Replacement", 2);
    database.records_ = {replacement};
    database.events_.push_back({.sequence = 2, .kind = asset_database_change_kind::catalog_reset, .guid = {}});
    database.history_lost_ = true;

    {
        auto reopened = asset_search_service::open(database, database_path);
        REQUIRE(reopened);
        auto service = std::move(reopened).value();

        CHECK(database.records_calls_ == 2);
        CHECK(service->query({.text = "Bicycle"}).empty());
        REQUIRE(service->query({.text = "Replacement"}).size() == 1);
    }

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search index finds tags categories and metadata through FTS")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-fts-metadata-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    const auto bicycle_guid = generate_asset_guid();
    const auto barrel_guid = generate_asset_guid();

    const std::vector<asset_search_document> documents{
        {.guid = bicycle_guid,
         .type = asset_types::static_mesh,
         .display_name = "SM_PROP_BCY_03_FINAL_v7",
         .logical_path = "assets/props/vehicle-03.arcmesh",
         .active_provider = std::string(asset_provider_ids::project_source),
         .title = "SM_PROP_BCY_03_FINAL_v7",
         .description = "Outdoor prop",
         .authored_tags = {"bicycle"},
         .generated_tags = {"weathered", "rusty"},
         .categories = {"Vehicle/Bicycle"},
         .metadata = {{"material", "metal", asset_search_metadata_origin::generated},
                      {"style", "rustic", asset_search_metadata_origin::authored}},
         .source_generation = 1,
         .source_revision = 1},
        {.guid = barrel_guid,
         .type = asset_types::static_mesh,
         .display_name = "Barrel",
         .logical_path = "assets/props/barrel.arcmesh",
         .active_provider = std::string(asset_provider_ids::project_source),
         .title = "Barrel",
         .description = "Industrial storage prop",
         .generated_tags = {"rusty"},
         .categories = {"Prop/Container"},
         .metadata = {{"material", "metal", asset_search_metadata_origin::generated}},
         .source_generation = 1,
         .source_revision = 1}};
    REQUIRE(index->reset(documents));

    const auto tag = index->query({.text = "bicycle"});
    REQUIRE(tag.size() == 1);
    CHECK(tag.front().guid == bicycle_guid);
    CHECK(tag.front().reasons.front().kind == asset_search_match_kind::metadata);
    CHECK(tag.front().reasons.front().detail == "authored tag");

    const auto category = index->query({.text = "Vehicle"});
    REQUIRE(category.size() == 1);
    CHECK(category.front().guid == bicycle_guid);
    CHECK(category.front().reasons.front().detail == "category");

    const auto metadata = index->query({.text = "rustic"});
    REQUIRE(metadata.size() == 1);
    CHECK(metadata.front().guid == bicycle_guid);
    CHECK(metadata.front().reasons.front().detail == "metadata");

    const auto multi_token = index->query({.text = "rusty bicycle"});
    REQUIRE(multi_token.size() == 1);
    CHECK(multi_token.front().guid == bicycle_guid);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search FTS supports token prefixes without a full document scan")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-fts-prefix-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    const auto bicycle_guid = generate_asset_guid();
    const auto chair_guid = generate_asset_guid();
    REQUIRE(index->reset(std::vector<asset_search_document>{{.guid = bicycle_guid,
                                                             .type = asset_types::static_mesh,
                                                             .display_name = "Bicycle",
                                                             .title = "Bicycle",
                                                             .source_generation = 1,
                                                             .source_revision = 1},
                                                            {.guid = chair_guid,
                                                             .type = asset_types::static_mesh,
                                                             .display_name = "Chair",
                                                             .title = "Chair",
                                                             .source_generation = 1,
                                                             .source_revision = 1}}));

    const auto results = index->query({.text = "bicy"});
    REQUIRE(results.size() == 1);
    CHECK(results.front().guid == bicycle_guid);
    CHECK(results.front().reasons.front().kind == asset_search_match_kind::prefix);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("asset search parser separates free text from structured predicates")
{
    using namespace arc::assets;

    auto parsed = parse_asset_search_query(
        R"(rustic chair type:mesh tag:"old wood" category:"Furniture/Chair" scope:project)", 25, 5);
    REQUIRE(parsed);

    const auto& query = parsed.value();
    CHECK(query.text == "rustic chair");
    REQUIRE(query.filters.type);
    CHECK(*query.filters.type == asset_types::static_mesh);
    CHECK(query.filters.required_tags == std::vector<std::string>{"old wood"});
    CHECK(query.filters.required_categories == std::vector<std::string>{"Furniture/Chair"});
    CHECK(query.filters.provider_ids == std::vector<std::string>{std::string(asset_provider_ids::project_source)});
    CHECK(query.limit == 25);
    CHECK(query.offset == 5);
}

TEST_CASE("asset search parser resolves built-in scope aliases")
{
    using namespace arc::assets;

    auto parsed = parse_asset_search_query("scope:builtin");
    REQUIRE(parsed);

    CHECK(parsed.value().filters.provider_ids ==
          std::vector<std::string>{std::string(asset_provider_ids::builtin_source),
                                   std::string(asset_provider_ids::virtual_builtin)});
}

TEST_CASE("asset search parser supports escaped quotes in quoted values")
{
    using namespace arc::assets;

    auto parsed = parse_asset_search_query(R"(tag:"designer \"classic\"")");
    REQUIRE(parsed);
    CHECK(parsed.value().filters.required_tags == std::vector<std::string>{R"(designer "classic")"});
}

TEST_CASE("asset search parser reports unknown predicates and malformed quotes")
{
    using namespace arc::assets;

    auto unknown = parse_asset_search_query("bike colour:red");
    REQUIRE_FALSE(unknown);
    CHECK(unknown.error().token == "colour:red");
    CHECK(unknown.error().message == "Unknown search predicate");
    CHECK(unknown.error().offset == 5);

    auto quote = parse_asset_search_query(R"(tag:"old wood)");
    REQUIRE_FALSE(quote);
    CHECK(quote.error().message == "Unterminated quoted search value");
}

TEST_CASE("asset search parser rejects conflicting type predicates")
{
    using namespace arc::assets;

    auto parsed = parse_asset_search_query("type:mesh type:material");
    REQUIRE_FALSE(parsed);
    CHECK(parsed.error().message == "Only one type predicate is supported");
}

TEST_CASE("persistent asset search applies structured predicates with free text")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-structured-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    const auto rustic_chair = generate_asset_guid();
    const auto modern_chair = generate_asset_guid();
    const auto rustic_texture = generate_asset_guid();

    REQUIRE(index->reset(
        std::vector<asset_search_document>{{.guid = rustic_chair,
                                            .type = asset_types::static_mesh,
                                            .display_name = "Chair_A",
                                            .active_provider = std::string(asset_provider_ids::project_source),
                                            .title = "Chair_A",
                                            .authored_tags = {"rustic", "old wood"},
                                            .categories = {"Furniture/Chair"},
                                            .source_generation = 1,
                                            .source_revision = 1},
                                           {.guid = modern_chair,
                                            .type = asset_types::static_mesh,
                                            .display_name = "Chair_B",
                                            .active_provider = std::string(asset_provider_ids::project_source),
                                            .title = "Chair_B",
                                            .authored_tags = {"modern"},
                                            .categories = {"Furniture/Chair"},
                                            .source_generation = 1,
                                            .source_revision = 1},
                                           {.guid = rustic_texture,
                                            .type = asset_types::texture_2d,
                                            .display_name = "Rustic_Wood",
                                            .active_provider = std::string(asset_provider_ids::builtin_source),
                                            .title = "Rustic_Wood",
                                            .authored_tags = {"rustic", "old wood"},
                                            .categories = {"Surface/Wood"},
                                            .source_generation = 1,
                                            .source_revision = 1}}));

    auto parsed = parse_asset_search_query(
        R"(chair type:mesh tag:rustic tag:"old wood" category:"Furniture/Chair" scope:project)");
    REQUIRE(parsed);
    const auto results = index->query(parsed.value());

    REQUIRE(results.size() == 1);
    CHECK(results.front().guid == rustic_chair);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

TEST_CASE("persistent asset search supports structured-only indexed filtering")
{
    using namespace arc::assets;
    const auto root =
        std::filesystem::temp_directory_path() / ("arc-search-structured-only-" + to_string(generate_asset_guid()));
    const auto database_path = root / "search.db";

    auto opened = persistent_asset_search_index::open(database_path);
    REQUIRE(opened);
    auto index = std::move(opened).value();

    const auto project_mesh = generate_asset_guid();
    const auto builtin_mesh = generate_asset_guid();

    REQUIRE(index->reset(
        std::vector<asset_search_document>{{.guid = project_mesh,
                                            .type = asset_types::static_mesh,
                                            .display_name = "Project Prop",
                                            .active_provider = std::string(asset_provider_ids::project_source),
                                            .title = "Project Prop",
                                            .authored_tags = {"hero"},
                                            .source_generation = 1,
                                            .source_revision = 1},
                                           {.guid = builtin_mesh,
                                            .type = asset_types::static_mesh,
                                            .display_name = "Built-in Prop",
                                            .active_provider = std::string(asset_provider_ids::builtin_source),
                                            .title = "Built-in Prop",
                                            .authored_tags = {"hero"},
                                            .source_generation = 1,
                                            .source_revision = 1}}));

    auto parsed = parse_asset_search_query("type:mesh tag:hero scope:project");
    REQUIRE(parsed);
    CHECK(parsed.value().text.empty());

    const auto results = index->query(parsed.value());
    REQUIRE(results.size() == 1);
    CHECK(results.front().guid == project_mesh);

    std::error_code error;
    std::filesystem::remove_all(root, error);
}

#include <arc/assets/asset_database.h>
#include <arc/assets/content_bundle.h>
#include <arc/assets/cook.h>

#include <map>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include <filesystem>
#include <fstream>

namespace
{

class temporary_database_project
{
public:
    temporary_database_project()
    {
        root = std::filesystem::temp_directory_path() /
               ("arc-asset-database-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
        assets = root / "assets";
        std::filesystem::create_directories(assets);
    }

    ~temporary_database_project()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void write(std::string_view relative, std::string_view contents)
    {
        const auto path = assets / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << contents;
    }

    std::filesystem::path root;
    std::filesystem::path assets;
};

struct database_fixture
{
    explicit database_fixture(temporary_database_project& project)
        : jobs({.worker_count = 2, .io_worker_count = 1, .enable_render_thread = false, .memory = &memory}),
          files(jobs), manager({.project_root = project.root,
                                .asset_root = project.assets,
                                .cache_root = project.root / ".arc" / "cache",
                                .enable_source_monitor = false},
                               jobs, files, memory),
          context(services)
    {
        manager.on_start(context);
    }

    ~database_fixture()
    {
        manager.on_shutdown(context);
    }

    arc::memory::memory_system memory;
    arc::jobs::job_system jobs;
    arc::io::async_file_service files;
    arc::assets::asset_manager manager;
    arc::framework::runtime_service_registry services;
    arc::framework::runtime_service_context context;
};

} // namespace

TEST_CASE("asset database provider vocabulary is stable across backends")
{
    using namespace arc::assets;

    CHECK(asset_provider_ids::project_source == "arc.project.source");
    CHECK(asset_provider_ids::builtin_source == "arc.builtin.source");
    CHECK(asset_provider_ids::virtual_builtin == "arc.builtin.virtual");
    CHECK(asset_provider_ids::cooked_package_prefix == "arc.package.");
    CHECK(asset_provider_ids::runtime_overlay == "arc.runtime.overlay");

    CHECK(asset_provider_kind_name(asset_provider_kind::project_source) == "project-source");
    CHECK(asset_provider_kind_name(asset_provider_kind::builtin_source) == "builtin-source");
    CHECK(asset_provider_kind_name(asset_provider_kind::virtual_builtin) == "virtual-builtin");
    CHECK(asset_provider_kind_name(asset_provider_kind::cooked_package) == "cooked-package");
    CHECK(asset_provider_kind_name(asset_provider_kind::runtime_overlay) == "runtime-overlay");
}

TEST_CASE("asset database change cursor reports authoring additions and updates")
{
    using namespace arc::assets;
    temporary_database_project project;
    database_fixture fixture(project);
    const auto initial = fixture.manager.database().changes_since(0).next_sequence;
    const auto guid = generate_asset_guid();
    auto payload =
        asset_payload::make(asset_types::binary_blob, std::make_shared<const source_asset_data>(source_asset_data{}));
    REQUIRE(fixture.manager.register_virtual_asset(guid, asset_types::binary_blob, std::move(payload), "event-test"));

    const auto changes = fixture.manager.database().changes_since(initial);
    CHECK(changes.next_sequence >= initial);
    CHECK(std::any_of(changes.events.begin(), changes.events.end(),
                      [&](const auto& event) { return event.guid == guid; }));
    CHECK(fixture.manager.database().changes_since(changes.next_sequence).events.empty());
}

TEST_CASE("asset database queries logical authoring records by GUID")
{
    using namespace arc::assets;
    temporary_database_project project;
    project.write("materials/stone.arcmat", R"({"version":4,"name":"Stone"})");
    database_fixture fixture(project);

    const auto source = fixture.manager.find("assets/materials/stone.arcmat");
    REQUIRE(source);

    const auto record = fixture.manager.database().query(source->guid, asset_types::material);
    REQUIRE(record);
    CHECK(record->guid == source->guid);
    CHECK(record->type == asset_types::material);
    CHECK(record->source_hint == "assets/materials/stone.arcmat");
    CHECK(record->active_provider == asset_provider_ids::project_source);
    REQUIRE(record->providers.size() == 1);
    CHECK(record->providers.front().id == asset_provider_ids::project_source);
    CHECK(record->providers.front().kind == asset_provider_kind::project_source);
    CHECK(record->providers.front().active);
    CHECK(record->generation == source->generation);
    CHECK(record->revision == source->revision);

    CHECK_FALSE(fixture.manager.database().query(source->guid, asset_types::texture_2d));
}

TEST_CASE("asset database classifies built-in authoring providers")
{
    using namespace arc::assets;
    temporary_database_project project;
    const auto builtin_root = project.root / "builtin";
    std::filesystem::create_directories(builtin_root / "materials");
    const auto source_path = builtin_root / "materials" / "builtin.arcmat";
    {
        std::ofstream output(source_path, std::ios::binary | std::ios::trunc);
        output << R"({"version":4,"name":"Builtin"})";
    }
    const auto builtin_guid = generate_asset_guid();
    REQUIRE(
        save_asset_metadata(metadata_path_for(source_path),
                            {.guid = builtin_guid, .type = asset_types::material, .importer = importer_ids::material}));

    arc::memory::memory_system memory;
    arc::jobs::job_system jobs(
        {.worker_count = 2, .io_worker_count = 1, .enable_render_thread = false, .memory = &memory});
    arc::io::async_file_service files(jobs);
    asset_manager manager({.project_root = project.root,
                           .asset_root = project.assets,
                           .read_only_source_roots = {builtin_root},
                           .cache_root = project.root / ".arc" / "cache",
                           .enable_source_monitor = false},
                          jobs, files, memory);
    arc::framework::runtime_service_registry services;
    arc::framework::runtime_service_context context(services);
    manager.on_start(context);

    const auto record = manager.database().query(builtin_guid, asset_types::material);
    REQUIRE(record);
    REQUIRE(record->providers.size() == 1);
    CHECK(record->providers.front().id == asset_provider_ids::builtin_source);
    CHECK(record->providers.front().kind == asset_provider_kind::builtin_source);
    CHECK(record->providers.front().read_only);
    CHECK(record->active_provider == asset_provider_ids::builtin_source);

    manager.on_shutdown(context);
}

TEST_CASE("asset database classifies virtual built-in providers")
{
    using namespace arc::assets;
    temporary_database_project project;
    database_fixture fixture(project);
    const auto guid = generate_asset_guid();
    auto payload =
        asset_payload::make(asset_types::binary_blob, std::make_shared<const source_asset_data>(source_asset_data{}));
    REQUIRE(fixture.manager.register_virtual_asset(guid, asset_types::binary_blob, std::move(payload), "test"));

    const auto record = fixture.manager.database().query(guid, asset_types::binary_blob);
    REQUIRE(record);
    REQUIRE(record->providers.size() == 1);
    CHECK(record->providers.front().id == asset_provider_ids::virtual_builtin);
    CHECK(record->providers.front().kind == asset_provider_kind::virtual_builtin);
    CHECK(record->providers.front().read_only);
    CHECK(record->active_provider == asset_provider_ids::virtual_builtin);
}

TEST_CASE("asset database reference queries never recover identity from path hints")
{
    using namespace arc::assets;
    temporary_database_project project;
    project.write("materials/stone.arcmat", "{}");
    database_fixture fixture(project);

    const auto source = fixture.manager.find("assets/materials/stone.arcmat");
    REQUIRE(source);

    const asset_reference stable{source->guid, asset_types::material, "assets/materials/moved.arcmat"};
    REQUIRE(fixture.manager.database().query(stable));

    const asset_reference legacy{{}, asset_types::material, "assets/materials/stone.arcmat"};
    CHECK_FALSE(fixture.manager.database().query(legacy));
}

TEST_CASE("asset manager owns a stable live Asset Database view")
{
    using namespace arc::assets;
    temporary_database_project project;
    project.write("materials/live.arcmat", R"({"version":4,"name":"Live"})");
    database_fixture fixture(project);

    const asset_database* first = &fixture.manager.database();
    const asset_database* second = &fixture.manager.database();
    CHECK(first == second);

    const auto source = fixture.manager.find("assets/materials/live.arcmat");
    REQUIRE(source);
    const auto before = first->query(source->guid);
    REQUIRE(before);

    const auto loaded =
        fixture.manager
            .load<source_asset_data>({.reference = {source->guid, source->type, "assets/materials/live.arcmat"}})
            .get();
    REQUIRE(loaded.succeeded());

    const auto after = first->query(source->guid);
    REQUIRE(after);
    CHECK(after->generation >= before->generation);
    CHECK(after->state == asset_state::ready);
    CHECK(after->residency == asset_residency::cpu);
    CHECK(after->strong_references >= before->strong_references);
}

TEST_CASE("asset database exposes dependency graph through one logical contract")
{
    using namespace arc::assets;
    temporary_database_project project;
    project.write("materials/a.arcmat", "{}");
    project.write("materials/b.arcmat", "{}");
    database_fixture fixture(project);

    const auto a = fixture.manager.find("assets/materials/a.arcmat");
    const auto b = fixture.manager.find("assets/materials/b.arcmat");
    REQUIRE(a);
    REQUIRE(b);

    const asset_reference b_reference{b->guid, b->type, "assets/materials/b.arcmat"};
    REQUIRE(fixture.manager.set_dependencies(a->guid, std::span(&b_reference, 1)));

    CHECK(fixture.manager.database().dependencies(a->guid) == std::vector<asset_guid>{b->guid});
    CHECK(fixture.manager.database().reverse_dependencies(b->guid) == std::vector<asset_guid>{a->guid});

    const auto record = fixture.manager.database().query(a->guid);
    REQUIRE(record);
    CHECK(record->dependencies == std::vector<asset_guid>{b->guid});
    CHECK(fixture.manager.database().revision() >= record->revision);
}

namespace
{
class bundle_test_database final : public arc::assets::asset_database
{
public:
    std::map<arc::assets::asset_guid, std::vector<arc::assets::asset_guid>> entries;

    std::optional<arc::assets::asset_database_record> query(arc::assets::asset_guid guid,
                                                            arc::assets::asset_type_id = {}) const override
    {
        if (!entries.contains(guid)) return std::nullopt;
        arc::assets::asset_database_record record;
        record.guid = guid;
        record.dependencies = entries.at(guid);
        return record;
    }
    std::vector<arc::assets::asset_database_record> records() const override
    {
        std::vector<arc::assets::asset_database_record> result;
        result.reserve(entries.size());
        for (const auto& [guid, dependencies] : entries)
        {
            arc::assets::asset_database_record record;
            record.guid = guid;
            record.dependencies = dependencies;
            result.push_back(std::move(record));
        }
        return result;
    }
    std::vector<arc::assets::asset_guid> dependencies(arc::assets::asset_guid guid) const override
    {
        const auto it = entries.find(guid);
        return it == entries.end() ? std::vector<arc::assets::asset_guid>{} : it->second;
    }
    std::vector<arc::assets::asset_guid> reverse_dependencies(arc::assets::asset_guid) const override
    {
        return {};
    }
    std::uint64_t revision() const override
    {
        return 1;
    }
    arc::assets::asset_database_change_batch changes_since(std::uint64_t cursor) const override
    {
        return {.next_sequence = cursor};
    }
};
} // namespace

TEST_CASE("content bundles preserve shared GUIDs across nested dependency closures")
{
    using namespace arc::assets;
    bundle_test_database database;
    const auto scene = generate_asset_guid();
    const auto audio = generate_asset_guid();
    const auto shared = generate_asset_guid();
    database.entries.emplace(scene, std::vector<asset_guid>{shared});
    database.entries.emplace(audio, std::vector<asset_guid>{shared});
    database.entries.emplace(shared, std::vector<asset_guid>{});

    content_bundle_catalog catalog;
    content_bundle_descriptor common;
    common.id = std::string(builtin_bundle_ids::startup);
    common.roots = {shared};
    REQUIRE(catalog.add(std::move(common)) == bundle_error::none);

    content_bundle_descriptor forest;
    forest.id = "Level/Forest";
    forest.roots = {scene, audio};
    forest.bundle_dependencies = {std::string(builtin_bundle_ids::startup)};
    REQUIRE(catalog.add(std::move(forest)) == bundle_error::none);

    const auto resolved = catalog.closure("Level/Forest", database);
    REQUIRE(resolved.succeeded());
    CHECK(resolved.assets.size() == 3);
    CHECK(resolved.bundles.size() == 2);
    CHECK(catalog.memberships(shared, database) ==
          std::vector<content_bundle_id>{"Level/Forest", std::string(builtin_bundle_ids::startup)});
    CHECK(catalog.closure("Level/Forest", database).assets == resolved.assets);
}

TEST_CASE("content bundles reject dependency cycles and missing references")
{
    using namespace arc::assets;
    bundle_test_database database;
    content_bundle_catalog catalog;
    content_bundle_descriptor first;
    first.id = "A";
    first.bundle_dependencies = {"B"};
    content_bundle_descriptor second;
    second.id = "B";
    second.bundle_dependencies = {"A"};
    REQUIRE(catalog.add(std::move(first)) == bundle_error::none);
    REQUIRE(catalog.add(std::move(second)) == bundle_error::none);
    CHECK(catalog.closure("A", database).error == bundle_error::cyclic_bundle_dependency);
    CHECK(catalog.closure("Unknown", database).error == bundle_error::missing_bundle);

    content_bundle_catalog missing;
    content_bundle_descriptor root;
    root.id = "Only";
    root.roots = {generate_asset_guid()};
    REQUIRE(missing.add(std::move(root)) == bundle_error::none);
    CHECK(missing.closure("Only", database).error == bundle_error::missing_asset);
}

TEST_CASE("arcbundle descriptor round trips with GUID roots and policies")
{
    using namespace arc::assets;
    content_bundle_descriptor descriptor;
    descriptor.id = "Level/Forest";
    descriptor.display_name = "Forest";
    descriptor.roots = {generate_asset_guid(), generate_asset_guid()};
    descriptor.bundle_dependencies = {"Startup/Common"};
    descriptor.preload = bundle_preload_policy::metadata;
    descriptor.closure = bundle_closure_policy::roots_only;
    descriptor.tags = {"level", "forest"};
    const auto encoded = serialize_content_bundle(descriptor);
    REQUIRE_FALSE(encoded.empty());
    const auto decoded = parse_content_bundle(encoded);
    REQUIRE(decoded.succeeded());
    CHECK(decoded.descriptor.id == descriptor.id);
    CHECK(decoded.descriptor.roots == descriptor.roots);
    CHECK(decoded.descriptor.bundle_dependencies == descriptor.bundle_dependencies);
    CHECK(decoded.descriptor.preload == descriptor.preload);
    CHECK(decoded.descriptor.closure == descriptor.closure);
    CHECK(decoded.descriptor.tags == descriptor.tags);
}

TEST_CASE("arcbundle parser rejects malformed formats and references")
{
    using namespace arc::assets;
    CHECK_FALSE(parse_content_bundle("{}").succeeded());
    CHECK_FALSE(parse_content_bundle(R"({"format":"arc.bundle","version":2,"id":"A","roots":[]})").succeeded());
    CHECK_FALSE(
        parse_content_bundle(R"({"format":"arc.bundle","version":1,"id":"A","roots":["not-a-guid"]})").succeeded());
    CHECK_FALSE(
        parse_content_bundle(R"({"format":"arc.bundle","version":1,"id":"A","roots":[],"bundleDependencies":["A"]})")
            .succeeded());
    CHECK_FALSE(parse_content_bundle(R"({"format":"arc.bundle","version":1,"id":"A","roots":[],"preload":"invalid"})")
                    .succeeded());
}

TEST_CASE("bundle cooking resolves shared assets deterministically and rejects cycles")
{
    using namespace arc::assets;
    bundle_test_database database;
    const auto scene = generate_asset_guid();
    const auto audio = generate_asset_guid();
    const auto shared = generate_asset_guid();
    database.entries.emplace(scene, std::vector<asset_guid>{shared});
    database.entries.emplace(audio, std::vector<asset_guid>{shared});
    database.entries.emplace(shared, std::vector<asset_guid>{});

    content_bundle_catalog bundles;
    content_bundle_descriptor common;
    common.id = "Startup/Common";
    common.roots = {shared};
    REQUIRE(bundles.add(std::move(common)) == bundle_error::none);
    content_bundle_descriptor forest;
    forest.id = "Level/Forest";
    forest.roots = {audio, scene};
    forest.bundle_dependencies = {"Startup/Common"};
    REQUIRE(bundles.add(std::move(forest)) == bundle_error::none);

    const auto first = resolve_bundle_cook_roots(bundles, "Level/Forest", database);
    REQUIRE(first.succeeded());
    REQUIRE(first.roots.size() == 3);
    CHECK(std::is_sorted(first.roots.begin(), first.roots.end()));
    CHECK(first.roots == resolve_bundle_cook_roots(bundles, "Level/Forest", database).roots);
    CHECK_FALSE(resolve_bundle_cook_roots(bundles, "Unknown", database).succeeded());

    content_bundle_catalog cyclic;
    content_bundle_descriptor a;
    a.id = "A";
    a.bundle_dependencies = {"B"};
    content_bundle_descriptor b;
    b.id = "B";
    b.bundle_dependencies = {"A"};
    REQUIRE(cyclic.add(std::move(a)) == bundle_error::none);
    REQUIRE(cyclic.add(std::move(b)) == bundle_error::none);
    CHECK_FALSE(resolve_bundle_cook_roots(cyclic, "A", database).succeeded());
}

TEST_CASE("bundle membership database projects shared and nested memberships onto GUID records")
{
    using namespace arc::assets;
    bundle_test_database database;
    const auto root = generate_asset_guid();
    const auto shared = generate_asset_guid();
    database.entries.emplace(root, std::vector<asset_guid>{shared});
    database.entries.emplace(shared, std::vector<asset_guid>{});

    content_bundle_catalog catalog;
    content_bundle_descriptor common;
    common.id = "Startup/Common";
    common.roots = {shared};
    REQUIRE(catalog.add(std::move(common)) == bundle_error::none);
    content_bundle_descriptor level;
    level.id = "Level/Forest";
    level.roots = {root};
    level.bundle_dependencies = {"Startup/Common"};
    REQUIRE(catalog.add(std::move(level)) == bundle_error::none);

    const bundle_membership_database view(database, catalog);
    const auto shared_record = view.query(shared);
    REQUIRE(shared_record);
    CHECK(shared_record->guid == shared);
    CHECK(shared_record->bundle_memberships == std::vector<std::string>{"Level/Forest", "Startup/Common"});
    const auto root_record = view.query(root);
    REQUIRE(root_record);
    CHECK(root_record->bundle_memberships == std::vector<std::string>{"Level/Forest"});

    const auto records = view.records();
    REQUIRE(records.size() == 2);
    for (const auto& record : records)
    {
        const auto queried = view.query(record.guid);
        REQUIRE(queried);
        CHECK(record.bundle_memberships == queried->bundle_memberships);
    }
    CHECK_FALSE(view.query(generate_asset_guid()));
    CHECK(view.dependencies(root) == std::vector<asset_guid>{shared});
    CHECK(view.revision() == database.revision());
}

TEST_CASE("built-in boot and startup bundles retain legacy shader classification")
{
    using namespace arc::assets;
    bundle_test_database database;
    const auto shader = generate_asset_guid();
    const auto scene = generate_asset_guid();
    database.entries.emplace(shader, std::vector<asset_guid>{});
    database.entries.emplace(scene, std::vector<asset_guid>{});

    // The fake record type defaults to unknown; project shader type in records().
    // The adapter must classify by declared asset type, never by path.
    const auto empty = make_builtin_content_bundles(database);
    const auto boot = empty.find(builtin_bundle_ids::boot);
    const auto startup = empty.find(builtin_bundle_ids::startup);
    REQUIRE(boot);
    REQUIRE(startup);
    CHECK(boot->bundle_dependencies == std::vector<std::string>{std::string(builtin_bundle_ids::engine)});
    CHECK(startup->bundle_dependencies == std::vector<std::string>{std::string(builtin_bundle_ids::boot)});
    CHECK(startup->roots.size() == 2);
    const auto resolved = empty.closure(builtin_bundle_ids::startup, database);
    REQUIRE(resolved.succeeded());
    CHECK(resolved.assets.size() == 2);
}

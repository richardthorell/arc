#include <arc/assets/asset_database.h>

#include <catch2/catch_test_macros.hpp>

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
    CHECK(record->active_provider == "arc.project.source");
    REQUIRE(record->providers.size() == 1);
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
    REQUIRE(save_asset_metadata(metadata_path_for(source_path),
                                {.guid = builtin_guid,
                                 .type = asset_types::material,
                                 .importer = importer_ids::material}));

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
    CHECK(record->providers.front().kind == asset_provider_kind::builtin_source);
    CHECK(record->providers.front().read_only);
    CHECK(record->active_provider == "arc.builtin.source");

    manager.on_shutdown(context);
}

TEST_CASE("asset database classifies virtual built-in providers")
{
    using namespace arc::assets;
    temporary_database_project project;
    database_fixture fixture(project);
    const auto guid = generate_asset_guid();
    auto payload = asset_payload::make(
        asset_types::binary_blob, std::make_shared<const source_asset_data>(source_asset_data{}));
    REQUIRE(fixture.manager.register_virtual_asset(guid, asset_types::binary_blob, std::move(payload), "test"));

    const auto record = fixture.manager.database().query(guid, asset_types::binary_blob);
    REQUIRE(record);
    REQUIRE(record->providers.size() == 1);
    CHECK(record->providers.front().kind == asset_provider_kind::virtual_builtin);
    CHECK(record->providers.front().read_only);
    CHECK(record->active_provider == "arc.builtin.virtual");
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

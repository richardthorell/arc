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
          context(services), database(manager)
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
    arc::assets::asset_manager_database database;
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

    const auto record = fixture.database.query(source->guid, asset_types::material);
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

    CHECK_FALSE(fixture.database.query(source->guid, asset_types::texture_2d));
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
    REQUIRE(fixture.database.query(stable));

    const asset_reference legacy{{}, asset_types::material, "assets/materials/stone.arcmat"};
    CHECK_FALSE(fixture.database.query(legacy));
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

    CHECK(fixture.database.dependencies(a->guid) == std::vector<asset_guid>{b->guid});
    CHECK(fixture.database.reverse_dependencies(b->guid) == std::vector<asset_guid>{a->guid});

    const auto record = fixture.database.query(a->guid);
    REQUIRE(record);
    CHECK(record->dependencies == std::vector<asset_guid>{b->guid});
    CHECK(fixture.database.revision() >= record->revision);
}

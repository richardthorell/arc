#include <arc/assets/assets.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>

namespace
{

class temporary_project
{
public:
    temporary_project()
    {
        root = std::filesystem::temp_directory_path() /
            ("arc-assets-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
        assets = root / "assets";
        std::filesystem::create_directories(assets);
    }
    ~temporary_project()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    void write(std::string_view relative, std::string_view contents)
    {
        const auto path = assets / relative;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }

    std::filesystem::path root;
    std::filesystem::path assets;
};

struct asset_fixture
{
    asset_fixture(temporary_project& project)
        : jobs({ .worker_count = 2, .io_worker_count = 1, .enable_render_thread = false, .memory = &memory })
        , files(jobs)
        , manager({
            .project_root = project.root,
            .asset_root = project.assets,
            .cache_root = project.root / ".arc" / "cache",
            .enable_source_monitor = false
        }, jobs, files, memory)
        , context(services)
    {
        manager.on_start(context);
    }

    ~asset_fixture()
    {
        manager.on_shutdown(context);
    }

    arc::memory_system memory;
    arc::job_system jobs;
    arc::io::async_file_service files;
    arc::assets::asset_manager manager;
    arc::runtime_service_registry services;
    arc::runtime_service_context context;
};

}

TEST_CASE("asset identifiers and SHA-256 hashes are stable")
{
    using namespace arc::assets;
    const auto guid = generate_asset_guid();
    REQUIRE(guid.valid());
    REQUIRE(parse_asset_guid(to_string(guid)) == guid);
    REQUIRE(parse_asset_type_id(to_string(asset_types::material)) == asset_types::material);
    REQUIRE(parse_asset_importer_id(to_string(importer_ids::material)) == importer_ids::material);
    REQUIRE_FALSE(parse_asset_guid("not-a-guid"));

    constexpr std::string_view abc = "abc";
    const auto hash = hash_bytes(std::as_bytes(std::span(abc.data(), abc.size())));
    REQUIRE(to_string(hash) == "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad");
    REQUIRE(parse_asset_hash(to_string(hash)) == hash);
}

TEST_CASE("asset metadata round trips stable subasset identities")
{
    using namespace arc::assets;
    temporary_project project;
    asset_source_metadata source{
        .guid = generate_asset_guid(),
        .type = asset_types::imported_scene,
        .importer = importer_ids::gltf,
        .settings_version = 2,
        .canonical_settings = R"({"normalizeAxes":true})",
        .subassets = {{
            .persistent_key = "mesh:Cabin",
            .guid = generate_asset_guid(),
            .type = asset_types::static_mesh,
            .name = "Cabin"
        }, {
            .persistent_key = "mesh:Removed",
            .guid = generate_asset_guid(),
            .type = asset_types::static_mesh,
            .name = "Removed",
            .tombstoned = true
        }}
    };
    const auto path = project.assets / "cabin.glb.arcmeta";
    std::string error;
    REQUIRE(save_asset_metadata(path, source, error));
    asset_source_metadata loaded;
    REQUIRE(load_asset_metadata(path, loaded, error));
    REQUIRE(loaded.guid == source.guid);
    REQUIRE(loaded.type == source.type);
    REQUIRE(loaded.importer == source.importer);
    REQUIRE(loaded.settings_version == 2);
    REQUIRE(loaded.subassets.size() == 2);
    REQUIRE(loaded.subassets[1].tombstoned);
}

TEST_CASE("asset manager scans persists moves and loads source generations")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("materials/stone.arcmat", R"({"version":3,"name":"Stone"})");
    project.write("textures/stone.png", "not-a-real-png");

    asset_fixture fixture(project);
    const auto registry = fixture.manager.snapshot();
    REQUIRE(std::filesystem::exists(registry.database_path));
    REQUIRE(std::filesystem::exists(project.assets / "materials/stone.arcmat.arcmeta"));
    REQUIRE(std::filesystem::exists(project.assets / "textures/stone.png.arcmeta"));

    const auto material = fixture.manager.find("assets/materials/stone.arcmat");
    REQUIRE(material);
    REQUIRE(material->type == asset_types::material);
    REQUIRE(material->state == asset_state::stale);
    const auto reference = fixture.manager.resolve("assets/materials/stone.arcmat", asset_types::material);
    REQUIRE(reference.guid == material->guid);

    const auto loaded = fixture.manager.load<source_asset_data>({
        .reference = reference,
        .priority = asset_streaming_priority::high,
        .residency = asset_residency::cpu
    }).get();
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.asset->bytes.size() > 10);
    REQUIRE(fixture.manager.find(material->guid)->state == asset_state::ready);
    REQUIRE_FALSE(fixture.manager.find(material->guid)->artifacts.empty());

    const auto moved = fixture.manager.move(material->guid, "assets/materials/renamed.arcmat");
    REQUIRE(moved.succeeded());
    REQUIRE(std::filesystem::exists(project.assets / "materials/renamed.arcmat"));
    REQUIRE(std::filesystem::exists(project.assets / "materials/renamed.arcmat.arcmeta"));
    REQUIRE(fixture.manager.find("assets/materials/renamed.arcmat")->guid == material->guid);
}

TEST_CASE("asset dependency cycles are rejected and reverse dependencies become stale")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("a.arcmat", "{}");
    project.write("b.arcmat", "{}");
    asset_fixture fixture(project);
    const auto a = fixture.manager.find("assets/a.arcmat");
    const auto b = fixture.manager.find("assets/b.arcmat");
    REQUIRE(a);
    REQUIRE(b);

    const asset_reference a_reference{ a->guid, a->type, "assets/a.arcmat" };
    const asset_reference b_reference{ b->guid, b->type, "assets/b.arcmat" };
    REQUIRE(fixture.manager.load<source_asset_data>({ .reference = a_reference }).get().succeeded());
    REQUIRE(fixture.manager.load<source_asset_data>({ .reference = b_reference }).get().succeeded());

    REQUIRE(fixture.manager.set_dependencies(a->guid, std::span(&b_reference, 1)));
    REQUIRE_FALSE(fixture.manager.set_dependencies(b->guid, std::span(&a_reference, 1)));

    REQUIRE(fixture.manager.mark_stale(b->guid, "test edit"));
    REQUIRE(fixture.manager.find(a->guid)->state == asset_state::stale);
}

TEST_CASE("missing references report repair candidates without changing identity")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("replacement.arcmat", "{}");
    asset_fixture fixture(project);
    const auto replacement = fixture.manager.find("assets/replacement.arcmat");
    REQUIRE(replacement);

    const asset_reference missing{
        generate_asset_guid(),
        asset_types::material,
        "assets/replacement.arcmat"
    };
    const auto report = fixture.manager.audit_reference(missing, "scene.entity", "material");
    REQUIRE_FALSE(report.reason.empty());
    REQUIRE(report.repair_candidates == std::vector<asset_guid>{ replacement->guid });
    REQUIRE(fixture.manager.snapshot().missing_references.size() == 1);
}

TEST_CASE("asset registry rebuilds from sidecars after database corruption")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("materials/recover.arcmat", "{}");
    asset_guid original_guid;
    {
        asset_fixture fixture(project);
        const auto original = fixture.manager.find("assets/materials/recover.arcmat");
        REQUIRE(original);
        original_guid = original->guid;
    }

    const auto database = project.root / ".arc" / "cache" / "assets.db";
    {
        std::ofstream corrupt(database, std::ios::binary | std::ios::trunc);
        corrupt << "not a sqlite database";
    }
    {
        asset_fixture fixture(project);
        const auto recovered = fixture.manager.find("assets/materials/recover.arcmat");
        REQUIRE(recovered);
        REQUIRE(recovered->guid == original_guid);
        REQUIRE(std::filesystem::exists(database));
    }

    bool preserved_corrupt_registry{};
    for (const auto& entry : std::filesystem::directory_iterator(database.parent_path()))
        preserved_corrupt_registry = preserved_corrupt_registry ||
            entry.path().filename().string().starts_with("assets.db.corrupt-");
    REQUIRE(preserved_corrupt_registry);
}

TEST_CASE("first prefab sidecar adopts the authored prefab identity")
{
    using namespace arc::assets;
    temporary_project project;
    const auto authored = generate_asset_guid();
    project.write("prefabs/cabin.arcprefab",
        std::string(R"({"format":"arc.prefab","formatVersion":1,"prefab":{"id":")") +
        to_string(authored) + R"(","root":"00000000-0000-4000-8000-000000000001"},"entities":[]})");

    asset_fixture fixture(project);
    const auto prefab = fixture.manager.find("assets/prefabs/cabin.arcprefab");
    REQUIRE(prefab);
    REQUIRE(prefab->guid == authored);
    asset_source_metadata metadata;
    std::string error;
    REQUIRE(load_asset_metadata(project.assets / "prefabs/cabin.arcprefab.arcmeta", metadata, error));
    REQUIRE(metadata.guid == authored);
}

TEST_CASE("authored dependencies are extracted imported and reverse indexed")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("textures/albedo.png", "texture-bytes");
    project.write("materials/linked.arcmat",
        R"({"version":3,"textures":{"baseColor":"textures/albedo.png"}})");

    asset_fixture fixture(project);
    const auto material = fixture.manager.find("assets/materials/linked.arcmat");
    const auto texture = fixture.manager.find("assets/textures/albedo.png");
    REQUIRE(material);
    REQUIRE(texture);
    const auto loaded = fixture.manager.load<source_asset_data>({
        .reference = { material->guid, asset_types::material, "assets/materials/linked.arcmat" }
    }).get();
    REQUIRE(loaded.succeeded());
    REQUIRE(fixture.manager.dependencies(material->guid) == std::vector<asset_guid>{ texture->guid });
    REQUIRE(fixture.manager.reverse_dependencies(texture->guid) == std::vector<asset_guid>{ material->guid });
    REQUIRE(fixture.manager.find(texture->guid)->state == asset_state::ready);
}

TEST_CASE("asset handles pins cancellation and pressure eviction preserve residency contracts")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("textures/streamed.png", "streamed-texture-bytes");
    asset_fixture fixture(project);
    const auto asset = fixture.manager.find("assets/textures/streamed.png");
    REQUIRE(asset);
    const asset_reference reference{
        asset->guid, asset_types::texture_2d, "assets/textures/streamed.png"
    };

    {
        auto request = fixture.manager.load<source_asset_data>({ .reference = reference });
        const auto loaded = request.get();
        REQUIRE(loaded.succeeded());
        auto pin = fixture.manager.pin(asset->guid);
        REQUIRE(pin.valid());
        const auto resident = fixture.manager.find("assets/textures/streamed.png");
        REQUIRE(resident->strong_references >= 1);
        REQUIRE(resident->pins == 1);
        REQUIRE(fixture.manager.evict_unused() == 0);
    }
    REQUIRE(fixture.manager.evict_unused() == 1);
    REQUIRE(fixture.manager.find(asset->guid)->residency == asset_residency::derived);

    arc::cancellation_source cancelled;
    REQUIRE(cancelled.request_cancel());
    const auto cancelled_result = fixture.manager.load<source_asset_data>({
        .reference = reference,
        .cancellation = cancelled.token()
    }).get();
    REQUIRE_FALSE(cancelled_result.succeeded());
    REQUIRE(cancelled_result.error.code == asset_error_code::cancelled);
}

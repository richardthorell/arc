#include <arc/assets/assets.h>
#include <arc/assets/cook.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>

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

    ~asset_fixture()
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

class test_cook_processor final : public arc::assets::asset_cook_processor
{
public:
    test_cook_processor(arc::assets::asset_type_id type, arc::assets::cook_processor_id id,
                        arc::assets::artifact_schema_id schema, std::shared_ptr<std::size_t> runs)
        : runs_(std::move(runs))
    {
        descriptor_.id = id;
        descriptor_.name = "Test cook processor";
        descriptor_.schema = schema;
        descriptor_.input_types.push_back(type);
    }

    const arc::assets::asset_cook_processor_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }
    std::string toolchain_fingerprint() const override
    {
        return "test-toolchain/1";
    }
    arc::assets::asset_cook_result cook(const arc::assets::asset_cook_context& context) override
    {
        ++*runs_;
        return {.artifacts = {{.name = "test",
                               .extension = ".test",
                               .schema = descriptor_.schema,
                               .schema_version = descriptor_.schema_version,
                               .bytes = context.source.bytes}}};
    }

private:
    arc::assets::asset_cook_processor_descriptor descriptor_;
    std::shared_ptr<std::size_t> runs_;
};

} // namespace

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
    asset_source_metadata source{.guid = generate_asset_guid(),
                                 .type = asset_types::imported_scene,
                                 .importer = importer_ids::gltf,
                                 .settings_version = 2,
                                 .canonical_settings = R"({"normalizeAxes":true})",
                                 .subassets = {{.persistent_key = "mesh:Cabin",
                                                .guid = generate_asset_guid(),
                                                .type = asset_types::static_mesh,
                                                .name = "Cabin"},
                                               {.persistent_key = "mesh:Removed",
                                                .guid = generate_asset_guid(),
                                                .type = asset_types::static_mesh,
                                                .name = "Removed",
                                                .tombstoned = true}}};
    const auto path = project.assets / "cabin.glb.arcmeta";
    REQUIRE(save_asset_metadata(path, source));
    auto loaded_result = load_asset_metadata(path);
    REQUIRE(loaded_result);
    const asset_source_metadata& loaded = loaded_result.value();
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

    const auto loaded =
        fixture.manager
            .load<source_asset_data>(
                {.reference = reference, .priority = asset_streaming_priority::high, .residency = asset_residency::cpu})
            .get();
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

    const asset_reference a_reference{a->guid, a->type, "assets/a.arcmat"};
    const asset_reference b_reference{b->guid, b->type, "assets/b.arcmat"};
    REQUIRE(fixture.manager.load<source_asset_data>({.reference = a_reference}).get().succeeded());
    REQUIRE(fixture.manager.load<source_asset_data>({.reference = b_reference}).get().succeeded());

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

    const asset_reference missing{generate_asset_guid(), asset_types::material, "assets/replacement.arcmat"};
    const auto report = fixture.manager.audit_reference(missing, "scene.entity", "material");
    REQUIRE_FALSE(report.reason.empty());
    REQUIRE(report.repair_candidates == std::vector<asset_guid>{replacement->guid});
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
        preserved_corrupt_registry =
            preserved_corrupt_registry || entry.path().filename().string().starts_with("assets.db.corrupt-");
    REQUIRE(preserved_corrupt_registry);
}

TEST_CASE("first prefab sidecar adopts the authored prefab identity")
{
    using namespace arc::assets;
    temporary_project project;
    const auto authored = generate_asset_guid();
    project.write("prefabs/cabin.arcprefab",
                  std::string(R"({"format":"arc.prefab","formatVersion":1,"prefab":{"id":")") + to_string(authored) +
                      R"(","root":"00000000-0000-4000-8000-000000000001"},"entities":[]})");

    asset_fixture fixture(project);
    const auto prefab = fixture.manager.find("assets/prefabs/cabin.arcprefab");
    REQUIRE(prefab);
    REQUIRE(prefab->guid == authored);
    auto metadata_result = load_asset_metadata(project.assets / "prefabs/cabin.arcprefab.arcmeta");
    REQUIRE(metadata_result);
    const asset_source_metadata& metadata = metadata_result.value();
    REQUIRE(metadata.guid == authored);
}

TEST_CASE("authored dependencies are extracted imported and reverse indexed")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("textures/albedo.png", "texture-bytes");
    project.write("materials/linked.arcmat", R"({"version":3,"textures":{"baseColor":"textures/albedo.png"}})");

    asset_fixture fixture(project);
    const auto material = fixture.manager.find("assets/materials/linked.arcmat");
    const auto texture = fixture.manager.find("assets/textures/albedo.png");
    REQUIRE(material);
    REQUIRE(texture);
    const auto loaded = fixture.manager
                            .load<source_asset_data>({.reference = {material->guid, asset_types::material,
                                                                    "assets/materials/linked.arcmat"}})
                            .get();
    REQUIRE(loaded.succeeded());
    REQUIRE(fixture.manager.dependencies(material->guid) == std::vector<asset_guid>{texture->guid});
    REQUIRE(fixture.manager.reverse_dependencies(texture->guid) == std::vector<asset_guid>{material->guid});
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
    const asset_reference reference{asset->guid, asset_types::texture_2d, "assets/textures/streamed.png"};

    {
        auto request = fixture.manager.load<source_asset_data>({.reference = reference});
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

    arc::jobs::cancellation_source cancelled;
    REQUIRE(cancelled.request_cancel());
    const auto cancelled_result =
        fixture.manager.load<source_asset_data>({.reference = reference, .cancellation = cancelled.token()}).get();
    REQUIRE_FALSE(cancelled_result.succeeded());
    REQUIRE(cancelled_result.error.code == asset_error_code::cancelled);
}

TEST_CASE("cook build keys include processor shader and platform identities")
{
    using namespace arc::assets;
    constexpr std::string_view source_text = "source";
    asset_build_key_descriptor description{
        .source_hash = hash_bytes(std::as_bytes(std::span(source_text.data(), source_text.size()))),
        .importer = importer_ids::shader,
        .importer_version = 3,
        .processor = cook_processor_ids::shader,
        .processor_version = 4,
        .schema = artifact_schemas::shader,
        .schema_version = 2,
        .canonical_settings = R"({"optimize":true})",
        .toolchain_fingerprint = "arc-cook/test",
        .shader_compiler_fingerprint = "compiler/1",
        .shader_entry_point = "main",
        .shader_defines = {"STANDARD=1"},
        .target = windows_vulkan_cook_target()};
    const auto original = make_asset_build_key(description);
    REQUIRE(original == make_asset_build_key(description));
    description.processor_version++;
    REQUIRE(original != make_asset_build_key(description));
    description.processor_version--;
    description.shader_compiler_fingerprint = "compiler/2";
    REQUIRE(original != make_asset_build_key(description));
    description.shader_compiler_fingerprint = "compiler/1";
    description.target = linux_vulkan_cook_target();
    REQUIRE(original != make_asset_build_key(description));
}

TEST_CASE("derived data cache verifies read-through and immutable actions")
{
    using namespace arc::assets;
    temporary_project project;
    const auto local = project.root / "local";
    const auto shared_root = project.root / "shared";
    auto shared = std::make_shared<filesystem_shared_cache>(shared_root);
    derived_data_cache writer({.root = project.root / "writer", .shared = shared});
    constexpr std::string_view text = "content-addressed-artifact";
    const auto bytes = std::as_bytes(std::span(text.data(), text.size()));
    const auto hash = hash_bytes(bytes);
    const auto key = hash_bytes(std::as_bytes(std::span("action-key", 10)));
    cache_error error;
    REQUIRE(writer.put_blob(hash, bytes, error));
    REQUIRE(writer.put_action({.key = key, .artifacts = {hash}, .metadata = "[]"}, error));

    derived_data_cache reader({.root = local, .shared = shared});
    const auto action = reader.get_action(key, error);
    REQUIRE(action);
    REQUIRE(action->artifacts == std::vector<asset_hash>{hash});
    const auto blob = reader.get_blob(hash, error);
    REQUIRE(blob);
    REQUIRE(blob->layer == cache_layer::shared);
    REQUIRE(blob->bytes.size() == bytes.size());
    REQUIRE(reader.statistics().shared_hits == 2);
    REQUIRE(reader.verify() == 1);
}

TEST_CASE("incremental cooker reuses actions and packages mount without source state")
{
    using namespace arc::assets;
    temporary_project project;
    project.write("textures/albedo.png", "texture");
    project.write("materials/cooked.arcmat", R"({"version":3,"textures":{"baseColor":"textures/albedo.png"}})");
    asset_fixture fixture(project);
    const auto material = fixture.manager.find("assets/materials/cooked.arcmat");
    REQUIRE(material);

    derived_data_cache cache({.root = project.root / ".arc" / "cook-cache"});
    const auto runs = std::make_shared<std::size_t>();
    asset_cooker cooker(fixture.manager, cache);
    REQUIRE(cooker.register_processor(std::make_unique<test_cook_processor>(
        asset_types::texture_2d, cook_processor_ids::texture, artifact_schemas::texture, runs)));
    REQUIRE(cooker.register_processor(std::make_unique<test_cook_processor>(
        asset_types::material, cook_processor_ids::material, artifact_schemas::material, runs)));

    const cook_request request{
        .roots = {material->guid}, .target = windows_vulkan_cook_target(), .output = project.root / "cooked"};
    const auto first = cooker.cook(request);
    REQUIRE(first.succeeded());
    REQUIRE(first.cooked == 2);
    REQUIRE(first.cache_hits == 0);
    REQUIRE(*runs == 2);

    const auto second = cooker.cook(request);
    REQUIRE(second.succeeded());
    REQUIRE(second.cooked == 0);
    REQUIRE(second.cache_hits == 2);
    REQUIRE(*runs == 2);
    REQUIRE(second.manifest.artifacts.size() == 2);

    const auto package = build_asset_packages(second.manifest, cache, request.output);
    REQUIRE(package.succeeded());
    REQUIRE(package.chunks.size() == 1);
    asset_package_mount mount;
    REQUIRE(mount.mount(package.manifest_path));
    for (const auto& artifact : mount.manifest().artifacts)
    {
        const auto loaded = mount.read(artifact.asset, artifact.schema);
        REQUIRE(loaded);
        REQUIRE(hash_bytes(loaded.value()) == artifact.hash);

        arc::jobs::job_system jobs({.run_inline = true});
        arc::io::async_file_service files(jobs);
        const auto async_loaded = mount.read_async(artifact.asset, artifact.schema, files).get();
        REQUIRE(async_loaded);
        REQUIRE(hash_bytes(async_loaded.value()) == artifact.hash);
    }

    {
        std::ofstream corrupt(package.chunks.front(), std::ios::binary | std::ios::trunc);
        corrupt << "corrupt";
    }
    const auto repaired = build_asset_packages(second.manifest, cache, request.output);
    REQUIRE(repaired.succeeded());
    REQUIRE(repaired.chunks == package.chunks);
    REQUIRE(std::ranges::any_of(std::filesystem::directory_iterator(request.output), [](const auto& entry)
                                { return entry.path().string().find(".corrupt-") != std::string::npos; }));
    REQUIRE(mount.mount(repaired.manifest_path));
    for (const auto& artifact : mount.manifest().artifacts)
        REQUIRE(mount.read(artifact.asset, artifact.schema));
}

TEST_CASE("cache pruning preserves pinned blobs")
{
    using namespace arc::assets;
    temporary_project project;
    derived_data_cache cache({.root = project.root / "cache",
                              .cleanup = {.maximum_bytes = 8, .prune_threshold = 0.5f, .prune_target = 0.25f}});
    const std::array<std::byte, 8> first{};
    const std::array<std::byte, 8> second{std::byte{1}};
    const auto first_hash = hash_bytes(first);
    const auto second_hash = hash_bytes(second);
    cache_error error;
    REQUIRE(cache.put_blob(first_hash, first, error));
    REQUIRE(cache.put_blob(second_hash, second, error));
    REQUIRE(cache.pin(first_hash));
    REQUIRE(cache.prune() == second.size());
    REQUIRE(cache.get_blob(first_hash, error));
    REQUIRE_FALSE(cache.get_blob(second_hash, error));
}

TEST_CASE("HTTP shared cache authenticates and verifies immutable blob responses")
{
    using namespace arc::assets;
    constexpr std::string_view text = "remote-content";
    const std::vector<std::byte> bytes(std::as_bytes(std::span(text.data(), text.size())).begin(),
                                       std::as_bytes(std::span(text.data(), text.size())).end());
    const auto hash = hash_bytes(bytes);
    std::vector<http_cache_request> requests;
    http_shared_cache cache(
        {.endpoint = "https://cache.example/",
         .bearer_token = "test-token",
         .transport = [&](const http_cache_request& request)
         {
             requests.push_back(request);
             if (request.method == http_cache_method::get)
                 return http_cache_response{
                     .status = 200, .headers = {{"etag", std::string("\"") + to_string(hash) + "\""}}, .body = bytes};
             return http_cache_response{.status = 201};
         }});
    cache_error error;
    const auto loaded = cache.get_blob(hash, error);
    REQUIRE(loaded == bytes);
    REQUIRE(requests.front().url == "https://cache.example/v1/blobs/sha256/" + to_string(hash));
    REQUIRE(requests.front().headers ==
            std::vector<std::pair<std::string, std::string>>{{"authorization", "Bearer test-token"}});
    REQUIRE(cache.put_blob(hash, bytes, error));
    REQUIRE(requests.back().method == http_cache_method::put);
    REQUIRE(std::find(requests.back().headers.begin(), requests.back().headers.end(),
                      std::pair<std::string, std::string>{"if-none-match", "*"}) != requests.back().headers.end());

    http_shared_cache corrupt({.endpoint = "https://cache.example", .transport = [&](const http_cache_request&) {
                                   return http_cache_response{
                                       .status = 200, .headers = {{"etag", to_string(hash)}}, .body = {std::byte{42}}};
                               }});
    REQUIRE_FALSE(corrupt.get_blob(hash, error));
    REQUIRE(error);
}

TEST_CASE("read-only source roots mount built-in assets without allowing source mutation")
{
    using namespace arc::assets;
    temporary_project project;
    const auto builtin_root = std::filesystem::path(project.root.string() + "-builtin");
    std::filesystem::create_directories(builtin_root / "materials");
    const auto source_path = builtin_root / "materials" / "default_phong.arcmat";
    {
        std::ofstream output(source_path, std::ios::binary | std::ios::trunc);
        output << R"({"version":1,"name":"Default Phong"})";
    }
    const auto guid = generate_asset_guid();
    REQUIRE(save_asset_metadata(metadata_path_for(source_path),
                                {.guid = guid, .type = asset_types::material, .importer = importer_ids::material}));

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

    const auto builtin = manager.find("builtin/materials/default_phong.arcmat");
    REQUIRE(builtin);
    REQUIRE(builtin->guid == guid);
    REQUIRE(builtin->read_only);
    const auto loaded = manager
                            .load<source_asset_data>(
                                {.reference = {guid, asset_types::material, "builtin/materials/default_phong.arcmat"}})
                            .get();
    REQUIRE(loaded.succeeded());
    const auto moved = manager.move(guid, "assets/materials/copied.arcmat");
    REQUIRE_FALSE(moved.succeeded());
    REQUIRE(moved.error.code == asset_error_code::invalid_request);
    REQUIRE(moved.error.message == "Built-in assets are read-only");
    REQUIRE(std::filesystem::exists(source_path));

    manager.on_shutdown(context);
    std::error_code cleanup_error;
    std::filesystem::remove_all(builtin_root, cleanup_error);
}

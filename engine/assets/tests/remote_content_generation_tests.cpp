#include <arc/assets/remote_content_generation.h>
#include <arc/assets/terrain_types.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

arc::io::file_buffer bytes(std::string_view text)
{
    const auto span = std::as_bytes(std::span(text.data(), text.size()));
    return {span.begin(), span.end()};
}

arc::io::file_buffer read_bytes(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    std::string characters(std::istreambuf_iterator<char>{input}, {});
    return bytes(characters);
}

arc::io::virtual_path path(std::string_view text)
{
    auto parsed = arc::io::virtual_path::parse(text);
    REQUIRE(parsed);
    return std::move(parsed).value();
}

struct fixture
{
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-c9-generation-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    arc::assets::derived_data_cache cache{{.root = root / "cache"}};
    arc::jobs::job_system jobs{{.worker_count = 1, .run_inline = false, .io_worker_count = 2,
                                .enable_render_thread = false}};
    arc::io::async_file_service files{jobs};
    arc::io::virtual_file_system vfs{jobs};
    arc::assets::cooked_asset_catalog catalog{vfs, path("artifact://game")};
    arc::assets::asset_guid asset = arc::assets::generate_asset_guid();
    arc::io::file_buffer payload = bytes("a remotely cooked artifact");
    arc::assets::cooked_artifact_address address{asset, arc::assets::artifact_schemas::virtual_geometry, "terrain"};
    arc::assets::remote_content_release release;
    arc::io::file_buffer package_bytes;
    arc::io::file_buffer manifest_bytes;
    arc::assets::cook_manifest cooked;

    arc::assets::remote_content_object ingest(const arc::io::file_buffer& body)
    {
        using namespace arc;
        const auto hash = assets::to_string(assets::hash_bytes(body));
        assets::remote_content_object object{
            .path = "objects/sha256/" + hash, .sha256 = hash, .size = body.size()};
        io::content_http_request request{.origin = "https://content.example.org",
                                         .object_path = object.path,
                                         .object_size = object.size};
        io::content_http_response response{.status = 200,
                                           .content_length = std::to_string(body.size()),
                                           .body = body};
        const auto added = assets::ingest_verified_remote_object(cache, object, request, response, {});
        REQUIRE(added.succeeded());
        return object;
    }

    fixture()
    {
        using namespace arc::assets;
        cache_error error;
        const auto payload_hash = hash_bytes(payload);
        REQUIRE(cache.put_blob(payload_hash, payload, error));

        cook_manifest description;
        description.target = windows_vulkan_cook_target();
        description.build_id = "build-abc";
        description.roots.push_back(asset);
        description.dependency_closure.push_back(asset);
        description.artifacts.push_back({.asset = asset,
                                         .type = asset_types::terrain,
                                         .name = address.name,
                                         .schema = address.schema,
                                         .schema_version = 1,
                                         .hash = payload_hash,
                                         .size = payload.size(),
                                         .chunk = "startup"});
        auto packages = build_asset_packages(std::move(description), cache, root / "package");
        REQUIRE(packages.succeeded());
        REQUIRE(packages.chunks.size() == 1);
        manifest_bytes = read_bytes(packages.manifest_path);
        package_bytes = read_bytes(packages.chunks.front());
        auto parsed = parse_cook_manifest(manifest_bytes);
        REQUIRE(parsed);
        cooked = std::move(parsed).value();

        release.project_guid = "11111111-2222-4333-8444-555555555555";
        release.engine_version = "0.1.0";
        release.base_build_id = "build-abc";
        release.target = {.platform = "windows",
                          .architecture = "x86_64",
                          .renderer = "vulkan",
                          .cook_profile = cooked.target.name,
                          .texture_family = "bc"};
        release.generation = 5;
        release.cook_manifest_name = packages.manifest_path.filename().string();
        release.cook_manifest = ingest(manifest_bytes);
        release.chunks.push_back({.name = packages.chunks.front().filename().string(),
                                  .object = ingest(package_bytes)});
    }

    ~fixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

} // namespace

TEST_CASE("C9.2c2 verifies a full cook generation before exposing artifact VFS and Asset Database records")
{
    fixture f;
    auto provider = arc::assets::remote_content_artifact_provider::create(f.release, f.cache, f.files);
    REQUIRE(provider);
    CHECK(provider.value()->manifest().artifacts.size() == 1);
    CHECK(provider.value()->provider_generation() == 5);

    REQUIRE(f.catalog.mount_remote_generation(f.release, f.cache, f.files));
    auto record = f.catalog.database().query(f.asset, arc::assets::asset_types::terrain);
    REQUIRE(record);
    CHECK(record->active_provider.find(":remote:5:") != std::string::npos);
    CHECK(record->representations.front().content_hash == arc::assets::hash_bytes(f.payload));

    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    CHECK(resolved.value().size() == f.payload.size());
    auto read = f.vfs.read_range(resolved.value(), 2, 6).get();
    REQUIRE(read);
    CHECK(read.value() == arc::io::file_buffer(f.payload.begin() + 2, f.payload.begin() + 8));
}

TEST_CASE("C9.2c2 rejects missing and unreferenced chunks without replacing last good generation")
{
    fixture f;
    REQUIRE(f.catalog.mount_remote_generation(f.release, f.cache, f.files));
    auto candidate = f.release;
    candidate.generation = 6;
    candidate.chunks.clear();
    CHECK_FALSE(f.catalog.mount_remote_generation(candidate, f.cache, f.files));
    candidate.chunks = f.release.chunks;
    candidate.chunks.push_back({.name = "orphan.arcpak", .object = f.ingest(bytes("orphan"))});
    CHECK_FALSE(f.catalog.mount_remote_generation(candidate, f.cache, f.files));

    auto record = f.catalog.database().query(f.asset);
    REQUIRE(record);
    CHECK(record->active_provider.find(":remote:5:") != std::string::npos);
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}

TEST_CASE("C9.2c2 rejects valid-CAS package bytes with corrupt cooked artifact ranges")
{
    fixture f;
    auto corrupted = f.package_bytes;
    const auto offset = f.cooked.artifacts.front().offset;
    REQUIRE(offset < corrupted.size());
    corrupted[static_cast<std::size_t>(offset)] ^= std::byte{0x01};
    f.release.chunks.front().object = f.ingest(corrupted);
    auto result = arc::assets::remote_content_artifact_provider::create(f.release, f.cache, f.files);
    CHECK_FALSE(result);
    CHECK_FALSE(f.catalog.database().query(f.asset));
}

TEST_CASE("C9.2c2 rejects an incompatible cook target, mismatched base build and duplicate artifact mappings")
{
    fixture f;
    auto candidate = f.release;
    candidate.target.platform = "android";
    CHECK_FALSE(arc::assets::remote_content_artifact_provider::create(candidate, f.cache, f.files));
    candidate = f.release;
    candidate.base_build_id = "other-build";
    CHECK_FALSE(arc::assets::remote_content_artifact_provider::create(candidate, f.cache, f.files));

    candidate = f.release;
    candidate.chunks.front().object.sha256 = std::string(64, 'f');
    candidate.chunks.front().object.path = "objects/sha256/" + candidate.chunks.front().object.sha256;
    CHECK_FALSE(arc::assets::remote_content_artifact_provider::create(candidate, f.cache, f.files));
}

TEST_CASE("C9.2c2 rejects chunks whose manifest artifact byte ranges are truncated")
{
    fixture f;
    // Keep the cook manifest unchanged but publish a different, valid CAS
    // chunk object that is shorter than its declared artifact range.
    auto truncated = f.package_bytes;
    truncated.resize(static_cast<std::size_t>(f.cooked.artifacts.front().offset) + f.payload.size() - 1);
    f.release.chunks.front().object = f.ingest(truncated);
    CHECK_FALSE(arc::assets::remote_content_artifact_provider::create(f.release, f.cache, f.files));
}

#include <arc/assets/remote_content_download.h>
#include <arc/assets/remote_content_generation.h>
#include <arc/assets/remote_content_lazy.h>
#include <arc/assets/remote_runtime_generation.h>
#include <arc/assets/remote_content_transfer.h>
#include <arc/assets/terrain_types.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <filesystem>
#include <memory>
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
    arc::jobs::job_system jobs{
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false}};
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
        assets::remote_content_object object{.path = "objects/sha256/" + hash, .sha256 = hash, .size = body.size()};
        http::content_http_request request{
            .origin = "https://content.example.org", .object_path = object.path, .object_size = object.size};
        http::content_http_response response{
            .status = 200, .content_length = std::to_string(body.size()), .body = body};
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
        release.chunks.push_back(
            {.name = packages.chunks.front().filename().string(), .object = ingest(package_bytes)});
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
    f.release.base_build_id = "unchanged-runtime-package";
    f.release.cook_build_id = "build-abc";
    REQUIRE(arc::assets::remote_content_artifact_provider::create(f.release, f.cache, f.files));
    f.release.cook_build_id = "wrong-content-build";
    CHECK_FALSE(arc::assets::remote_content_artifact_provider::create(f.release, f.cache, f.files));
    f.release.cook_build_id = "build-abc";

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

namespace
{

arc::assets::remote_generation_fetch_executor mock_remote_objects(fixture& fixture, std::size_t& requests,
                                                                  bool corrupt_chunk = false)
{
    return [&fixture, &requests, corrupt_chunk](arc::jobs::job_system& jobs, arc::http::content_http_request request,
                                                arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.remote_content.test_fetch", .affinity = arc::jobs::job_affinity::io_thread},
            [&fixture, &requests, corrupt_chunk, request = std::move(request)]() -> arc::http::content_http_fetch_result
            {
                ++requests;
                auto body = request.object_path == fixture.release.cook_manifest.path ? fixture.manifest_bytes
                                                                                      : fixture.package_bytes;
                if (corrupt_chunk && request.object_path == fixture.release.chunks.front().object.path)
                    body.back() ^= std::byte{0x01};
                arc::http::content_http_response response{
                    .status = 200, .content_length = std::to_string(body.size()), .body = std::move(body)};
                return {.response = std::move(response), .validation = {}};
            });
    };
}

} // namespace

TEST_CASE("C9.2d1 downloads missing immutable objects then reuses cache and mounts on request")
{
    fixture f;
    arc::assets::derived_data_cache downloaded{{.root = f.root / "downloaded-cache"}};
    arc::assets::remote_generation_transfer_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    auto fetcher = mock_remote_objects(f, requests);

    auto first =
        arc::assets::acquire_remote_generation(f.jobs, f.files, downloaded, f.release, options, {}, fetcher).get();
    REQUIRE(first.succeeded());
    CHECK(first.downloaded_objects == 2);
    CHECK(first.reused_objects == 0);
    CHECK(first.downloaded_bytes == f.manifest_bytes.size() + f.package_bytes.size());
    CHECK(requests == 2);
    CHECK_FALSE(f.catalog.database().query(f.asset)); // acquisition never activates

    auto repeat =
        arc::assets::acquire_remote_generation(f.jobs, f.files, downloaded, f.release, options, {}, fetcher).get();
    REQUIRE(repeat.succeeded());
    CHECK(repeat.reused_objects == 2);
    CHECK(repeat.downloaded_objects == 0);
    CHECK(requests == 2);

    options.policy = arc::assets::remote_generation_transfer_policy::offline_only;
    options.origin.clear();
    auto offline = arc::assets::acquire_remote_generation(f.jobs, f.files, downloaded, f.release, options).get();
    REQUIRE(offline.succeeded());
    CHECK(offline.reused_objects == 2);

    REQUIRE(f.catalog.mount_remote_generation(f.release, downloaded, f.files));
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}

TEST_CASE("C9.2d1 refuses a corrupt downloaded chunk and preserves active content")
{
    fixture f;
    REQUIRE(f.catalog.mount_remote_generation(f.release, f.cache, f.files));
    arc::assets::derived_data_cache downloaded{{.root = f.root / "downloaded-cache"}};
    arc::assets::remote_generation_transfer_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    auto result = arc::assets::acquire_remote_generation(f.jobs, f.files, downloaded, f.release, options, {},
                                                         mock_remote_objects(f, requests, true))
                      .get();
    CHECK_FALSE(result.succeeded());
    CHECK(result.cache_error == arc::assets::remote_object_cache_error::hash_mismatch);
    CHECK(result.downloaded_objects == 1);
    CHECK(result.failed_object == f.release.chunks.front().object.path);
    CHECK(requests == 2);

    arc::assets::cache_error cache_error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(downloaded, f.release.chunks.front().object, cache_error));
    auto old = f.catalog.database().query(f.asset);
    REQUIRE(old);
    CHECK(old->active_provider.find(":remote:5:") != std::string::npos);
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}

TEST_CASE("C9.2d1 rejects offline cache misses, invalid origins and unsafe generation budgets")
{
    fixture f;
    arc::assets::derived_data_cache empty{{.root = f.root / "empty-cache"}};
    arc::assets::remote_generation_transfer_options options;
    options.policy = arc::assets::remote_generation_transfer_policy::offline_only;
    auto offline = arc::assets::acquire_remote_generation(f.jobs, f.files, empty, f.release, options).get();
    CHECK_FALSE(offline.succeeded());
    CHECK(offline.failed_object == f.release.cook_manifest.path);

    options.policy = arc::assets::remote_generation_transfer_policy::cache_first;
    options.origin = "file:///etc/passwd";
    auto origin = arc::assets::acquire_remote_generation(f.jobs, f.files, empty, f.release, options).get();
    CHECK_FALSE(origin.succeeded());
    CHECK(origin.transport_error == arc::http::content_http_error_code::invalid_origin);

    options.origin = "https://content.example.org";
    options.maximum_generation_bytes = f.manifest_bytes.size() + f.package_bytes.size() - 1;
    auto bounded = arc::assets::acquire_remote_generation(f.jobs, f.files, empty, f.release, options).get();
    CHECK_FALSE(bounded.succeeded());
    CHECK(bounded.failed_object == f.release.chunks.front().object.path);

    options.maximum_generation_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    auto duplicated = f.release;
    duplicated.chunks.push_back(duplicated.chunks.front());
    auto duplicate = arc::assets::acquire_remote_generation(f.jobs, f.files, empty, duplicated, options).get();
    CHECK_FALSE(duplicate.succeeded());
}

TEST_CASE("C9.2d1 propagates cancellation without doing network work")
{
    fixture f;
    arc::assets::derived_data_cache empty{{.root = f.root / "cancelled-cache"}};
    arc::assets::remote_generation_transfer_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    arc::jobs::cancellation_source source;
    REQUIRE(source.request_cancel());
    auto cancelled = arc::assets::acquire_remote_generation(f.jobs, f.files, empty, f.release, options, source.token(),
                                                            mock_remote_objects(f, requests))
                         .get();
    CHECK_FALSE(cancelled.succeeded());
    CHECK(cancelled.transport_error == arc::http::content_http_error_code::cancelled);
    CHECK(requests == 0);
}

TEST_CASE("C9.2f1 prepares only immutable metadata then downloads a missing chunk on first VFS read")
{
    fixture f;
    arc::assets::derived_data_cache local{{.root = f.root / "lazy-cache"}};
    arc::assets::lazy_remote_content_options options{.origin = "https://content.example.org"};
    std::size_t requests{};

    auto prepared = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options, {},
                                                                mock_remote_objects(f, requests))
                        .get();
    REQUIRE(prepared.succeeded());
    CHECK_FALSE(prepared.manifest_reused);
    CHECK(prepared.downloaded_bytes == f.manifest_bytes.size());
    CHECK(requests == 1); // Manifest only; no package chunk download.
    arc::assets::cache_error error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(local, f.release.chunks.front().object, error));
    CHECK_FALSE(f.catalog.database().query(f.asset)); // Preparation never mounts.

    REQUIRE(f.catalog.mount_remote_provider(prepared.provider));
    auto database_record = f.catalog.database().query(f.asset, arc::assets::asset_types::terrain);
    REQUIRE(database_record);
    CHECK(database_record->generation == f.release.generation);
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    CHECK(resolved.value().size() == f.payload.size());
    const auto first = f.vfs.read_range(resolved.value(), 2, 6).get();
    REQUIRE(first);
    CHECK(first.value() == arc::io::file_buffer(f.payload.begin() + 2, f.payload.begin() + 8));
    CHECK(requests == 2); // One full immutable chunk was fetched and verified into CAS.

    const auto repeated = f.vfs.read_all(resolved.value()).get();
    REQUIRE(repeated);
    CHECK(repeated.value() == f.payload);
    CHECK(requests == 2); // Verified cached chunk reused.

    auto prepared_again = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options, {},
                                                                      mock_remote_objects(f, requests))
                              .get();
    REQUIRE(prepared_again.succeeded());
    CHECK(prepared_again.manifest_reused);
    CHECK(prepared_again.downloaded_bytes == 0);
    CHECK(requests == 2);
}

TEST_CASE("C9.2f1 rejects corrupt lazy chunks without overwriting a previously mounted generation")
{
    fixture f;
    REQUIRE(f.catalog.mount_remote_generation(f.release, f.cache, f.files));
    arc::assets::derived_data_cache local{{.root = f.root / "lazy-corrupt-cache"}};
    auto newer = f.release;
    newer.generation = 6;
    arc::assets::lazy_remote_content_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    auto prepared = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, newer, options, {},
                                                                mock_remote_objects(f, requests, true))
                        .get();
    REQUIRE(prepared.succeeded());
    CHECK(requests == 1);
    REQUIRE(f.catalog.mount_remote_provider(prepared.provider, 50));

    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto rejected = f.vfs.read_all(resolved.value()).get();
    CHECK_FALSE(rejected);
    CHECK(rejected.error().code == arc::io::file_error_code::corrupt_content);
    CHECK(requests == 2);
    arc::assets::cache_error error;
    CHECK_FALSE(arc::assets::load_verified_remote_object(local, newer.chunks.front().object, error));

    auto current = f.catalog.database().query(f.asset);
    REQUIRE(current);
    CHECK(current->active_provider.find(":remote:6:") != std::string::npos);
    REQUIRE(f.catalog.unmount_provider(current->active_provider));
    auto last_good = f.catalog.resolve(f.address);
    REQUIRE(last_good);
    auto read = f.vfs.read_all(last_good.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}

TEST_CASE("C9.2f1 rejects invalid lazy chunk metadata and wrong targets before mounting")
{
    fixture f;
    arc::assets::derived_data_cache local{{.root = f.root / "lazy-invalid-cache"}};
    arc::assets::lazy_remote_content_options options{.origin = "https://content.example.org"};
    std::size_t requests{};

    auto bad_range = f.release;
    bad_range.chunks.front().object.size = f.cooked.artifacts.front().offset;
    auto invalid = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, bad_range, options, {},
                                                               mock_remote_objects(f, requests))
                       .get();
    CHECK_FALSE(invalid.succeeded());
    CHECK(requests == 1); // May cache verified manifest but never download a chunk.
    CHECK_FALSE(f.catalog.database().query(f.asset));

    auto bad_target = f.release;
    bad_target.target.platform = "android";
    auto wrong_target = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, bad_target, options, {},
                                                                    mock_remote_objects(f, requests))
                            .get();
    CHECK_FALSE(wrong_target.succeeded());
    CHECK(requests == 1);

    options.origin = "file:///etc/passwd";
    auto bad_origin = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options, {},
                                                                  mock_remote_objects(f, requests))
                          .get();
    CHECK_FALSE(bad_origin.succeeded());
    CHECK(requests == 1);
}

TEST_CASE("C9.2f1 metadata-only offline mount fails closed on a missing immutable chunk")
{
    fixture f;
    arc::assets::derived_data_cache local{{.root = f.root / "lazy-offline-cache"}};
    arc::assets::lazy_remote_content_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    auto initial = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options, {},
                                                               mock_remote_objects(f, requests))
                       .get();
    REQUIRE(initial.succeeded());
    CHECK(requests == 1);

    options.origin.clear();
    options.offline_only = true;
    auto offline = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options).get();
    REQUIRE(offline.succeeded());
    CHECK(offline.manifest_reused);
    REQUIRE(f.catalog.mount_remote_provider(offline.provider));
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto missing = f.vfs.read_all(resolved.value()).get();
    CHECK_FALSE(missing);
    CHECK(missing.error().code == arc::io::file_error_code::not_found);
    CHECK(requests == 1);

    arc::jobs::cancellation_source cancelled;
    REQUIRE(cancelled.request_cancel());
    auto cancelled_read = f.vfs.read_all(resolved.value(), cancelled.token()).get();
    CHECK_FALSE(cancelled_read);
    CHECK(cancelled_read.error().code == arc::io::file_error_code::cancelled);
}

TEST_CASE("C9.2f1 pre-cancelled preparation returns a typed error without scheduling HTTP")
{
    fixture f;
    arc::assets::derived_data_cache local{{.root = f.root / "lazy-pre-cancelled-cache"}};
    arc::assets::lazy_remote_content_options options{.origin = "https://content.example.org"};
    std::size_t requests{};
    arc::jobs::cancellation_source cancelled;
    REQUIRE(cancelled.request_cancel());
    auto prepared = arc::assets::prepare_lazy_remote_generation(f.jobs, f.files, local, f.release, options,
                                                                cancelled.token(), mock_remote_objects(f, requests))
                        .get();
    CHECK_FALSE(prepared.succeeded());
    CHECK(prepared.transport_error == arc::http::content_http_error_code::cancelled);
    CHECK(requests == 0);
}

TEST_CASE("C9.2f2 eager generation acquisition retries a transient GET without duplicating CAS publication")
{
    fixture f;
    arc::assets::derived_data_cache local{{.root = f.root / "eager-retry-cache"}};
    arc::assets::remote_generation_transfer_options options{.origin = "https://content.example.org"};
    arc::assets::remote_download_limits policy;
    policy.maximum_attempts = 2;
    policy.initial_backoff = std::chrono::milliseconds{0};
    policy.maximum_backoff = std::chrono::milliseconds{0};
    options.downloads = std::make_shared<arc::assets::remote_object_download_coordinator>(f.jobs, local, policy);

    std::size_t attempts{};
    arc::assets::remote_generation_fetch_executor transient =
        [&f, &attempts](arc::jobs::job_system& jobs, arc::http::content_http_request request,
                        arc::http::content_http_limits, arc::jobs::cancellation_token)
    {
        return jobs.submit_future(
            {.name = "assets.remote_content.test_retry", .affinity = arc::jobs::job_affinity::io_thread},
            [&f, &attempts, request = std::move(request)]() -> arc::http::content_http_fetch_result
            {
                ++attempts;
                if (attempts == 1)
                    return {.response = std::nullopt,
                            .validation = {.code = arc::http::content_http_error_code::timeout,
                                           .message = "Injected timeout"}};
                auto body = request.object_path == f.release.cook_manifest.path ? f.manifest_bytes : f.package_bytes;
                return {.response = arc::http::content_http_response{.status = 200,
                                                                     .content_length = std::to_string(body.size()),
                                                                     .body = std::move(body)},
                        .validation = {}};
            });
    };

    const auto result =
        arc::assets::acquire_remote_generation(f.jobs, f.files, local, f.release, options, {}, transient).get();
    REQUIRE(result.succeeded());
    CHECK(result.downloaded_objects == 2);
    CHECK(result.reused_objects == 0);
    CHECK(attempts == 3); // 1 failed manifest attempt, then manifest and package.
    REQUIRE(f.catalog.mount_remote_generation(f.release, local, f.files));
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    const auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}


namespace
{

arc::assets::remote_content_expectations expectations_for(const fixture& f)
{
    return {.project_guid = f.release.project_guid,
            .engine_version = f.release.engine_version,
            .base_build_id = f.release.base_build_id,
            .target = f.release.target};
}

} // namespace

TEST_CASE("C9.2g2 promotes verified remote bases over packaged content and preserves bounded last-good")
{
    fixture f;
    arc::assets::remote_runtime_generation_session runtime(f.catalog, f.cache, f.files, expectations_for(f));
    REQUIRE(runtime.activate_verified(f.release));
    auto first = runtime.state();
    CHECK(first.active_generation == 5);
    CHECK(first.last_good_generation == 0);
    CHECK(first.active_provider_id.find(":remote:5:") != std::string::npos);
    const auto scope = arc::assets::remote_content_retention_scope(f.release);
    CHECK(f.cache.persistent_pins(scope).size() == 2);

    auto next = f.release;
    next.generation = 6;
    REQUIRE(runtime.activate_verified(next));
    auto second = runtime.state();
    CHECK(second.active_generation == 6);
    CHECK(second.last_good_generation == 5);
    CHECK(second.active_provider_id != first.active_provider_id);
    auto record = f.catalog.database().query(f.asset);
    REQUIRE(record);
    CHECK(record->active_provider == second.active_provider_id);
    CHECK(record->providers.size() == 2);

    auto third = f.release;
    third.generation = 7;
    REQUIRE(runtime.activate_verified(third));
    const auto state = runtime.state();
    CHECK(state.active_generation == 7);
    CHECK(state.last_good_generation == 6);
    CHECK(state.highest_accepted_generation == 7);
    CHECK(state.retired_mount_cleanup_failures == 0);
    record = f.catalog.database().query(f.asset);
    REQUIRE(record);
    CHECK(record->providers.size() == 2); // Generation 5 has been retired.
    CHECK(record->active_provider == state.active_provider_id);

    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
    CHECK(f.cache.persistent_pins(scope).size() == 2);
}

TEST_CASE("C9.2g2 rejects corrupt or mismatched candidate without changing active VFS or cache retention")
{
    fixture f;
    arc::assets::remote_runtime_generation_session runtime(f.catalog, f.cache, f.files, expectations_for(f));
    REQUIRE(runtime.activate_verified(f.release));
    const auto current = runtime.state();
    const auto scope = arc::assets::remote_content_retention_scope(f.release);
    const auto retained = f.cache.persistent_pins(scope);

    auto wrong_target = f.release;
    wrong_target.generation = 6;
    wrong_target.target.platform = "android";
    CHECK_FALSE(runtime.activate_verified(wrong_target));

    auto wrong_build = f.release;
    wrong_build.generation = 6;
    wrong_build.base_build_id = "different-game-runtime";
    CHECK_FALSE(runtime.activate_verified(wrong_build));

    auto missing_chunk = f.release;
    missing_chunk.generation = 6;
    missing_chunk.chunks.front().object.sha256 = std::string(64, 'f');
    missing_chunk.chunks.front().object.path = "objects/sha256/" + missing_chunk.chunks.front().object.sha256;
    CHECK_FALSE(runtime.activate_verified(missing_chunk));

    auto invalid_manifest = f.release;
    invalid_manifest.generation = 6;
    invalid_manifest.cook_build_id = "invalid-cook-manifest-build";
    CHECK_FALSE(runtime.activate_verified(invalid_manifest));

    auto replay = f.release;
    CHECK_FALSE(runtime.activate_verified(replay));
    CHECK(runtime.state().rejected_activations == 5);
    CHECK(runtime.state().active_provider_id == current.active_provider_id);
    CHECK(runtime.state().active_generation == 5);
    CHECK(f.cache.persistent_pins(scope) == retained);
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto read = f.vfs.read_all(resolved.value()).get();
    REQUIRE(read);
    CHECK(read.value() == f.payload);
}

TEST_CASE("C9.2g2 explicit base rollback restores prior provider and retention but rejects replay")
{
    fixture f;
    arc::assets::remote_runtime_generation_session runtime(f.catalog, f.cache, f.files, expectations_for(f));
    REQUIRE(runtime.activate_verified(f.release));
    const auto previous_id = runtime.state().active_provider_id;
    auto newer = f.release;
    newer.generation = 6;
    REQUIRE(runtime.activate_verified(newer));
    REQUIRE(runtime.rollback());
    auto state = runtime.state();
    CHECK(state.active_generation == 5);
    CHECK(state.last_good_generation == 0);
    CHECK(state.highest_accepted_generation == 6);
    CHECK(state.active_provider_id == previous_id);
    auto record = f.catalog.database().query(f.asset);
    REQUIRE(record);
    CHECK(record->active_provider == previous_id);
    CHECK(record->providers.size() == 1);
    const auto scope = arc::assets::remote_content_retention_scope(f.release);
    CHECK(f.cache.persistent_pins(scope).size() == 2);
    CHECK_FALSE(runtime.activate_verified(newer)); // No stale replay after rollback.
    CHECK_FALSE(runtime.rollback()); // Only one active base remains.
    auto newest = newer;
    newest.generation = 7;
    REQUIRE(runtime.activate_verified(newest));
    CHECK(runtime.state().active_generation == 7);
    CHECK(runtime.state().last_good_generation == 5);
}

TEST_CASE("C9.2g2 reuses native runtime_update_receiver for atomic higher priority artifact updates")
{
    fixture f;
    arc::assets::remote_runtime_generation_session runtime(f.catalog, f.cache, f.files, expectations_for(f));
    CHECK_FALSE(runtime.attach_overlay(f.root / "live"));
    REQUIRE(runtime.activate_verified(f.release));
    REQUIRE(runtime.attach_overlay(f.root / "live"));
    CHECK(runtime.state().overlay_attached);
    CHECK_FALSE(runtime.attach_overlay(f.root / "live-again"));

    const auto new_bytes = bytes("an updated terrain payload");
    const auto replacement_hash = arc::assets::hash_bytes(new_bytes);
    arc::assets::runtime_update_manifest update{
        .target_profile = f.release.target.cook_profile,
        .base_build_id = f.release.base_build_id,
        .update_sequence = 1,
        .artifacts = {{.address = f.address,
                       .type = arc::assets::asset_types::terrain,
                       .schema_version = 1,
                       .hash = replacement_hash,
                       .size = new_bytes.size()}}};
    const std::array blobs{std::pair{replacement_hash, new_bytes}};
    REQUIRE(runtime.ingest_overlay(update, blobs));
    CHECK(runtime.state().overlay_sequence == 1);
    auto resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    auto body = f.vfs.read_all(resolved.value()).get();
    REQUIRE(body);
    CHECK(body.value() == new_bytes);
    CHECK_FALSE(runtime.rollback());

    auto stale = update;
    CHECK_FALSE(runtime.ingest_overlay(stale, blobs));
    auto forged = update;
    forged.update_sequence = 2;
    forged.base_build_id = "wrong-build";
    CHECK_FALSE(runtime.ingest_overlay(forged, blobs));
    CHECK(runtime.state().overlay_sequence == 1);
    resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    body = f.vfs.read_all(resolved.value()).get();
    REQUIRE(body);
    CHECK(body.value() == new_bytes);

    auto newer = f.release;
    newer.generation = 6;
    REQUIRE(runtime.activate_verified(newer));
    resolved = f.catalog.resolve(f.address);
    REQUIRE(resolved);
    body = f.vfs.read_all(resolved.value()).get();
    REQUIRE(body);
    CHECK(body.value() == new_bytes); // Existing overlay retains priority.
}

TEST_CASE("C9.2g2 refuses activation when persistent retention cannot be safely restored")
{
    fixture f;
    const auto journal = f.root / "cache" / "persistent-pins.json";
    {
        std::ofstream stream(journal, std::ios::binary | std::ios::trunc);
        REQUIRE(stream.good());
        stream << "{\"format\":\"arc.ddc-persistent-pins\",\"version\":1,\"scopes\":[]}";
    }
    arc::assets::derived_data_cache corrupt{{.root = f.root / "cache"}};
    REQUIRE_FALSE(corrupt.persistent_retention_ready());
    arc::assets::remote_runtime_generation_session runtime(f.catalog, corrupt, f.files, expectations_for(f));
    CHECK_FALSE(runtime.activate_verified(f.release));
    CHECK(runtime.state().active_generation == 0);
    CHECK_FALSE(f.catalog.database().query(f.asset));
}

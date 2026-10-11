#include <arc/assets/runtime_content.h>
#include <arc/assets/terrain_types.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

namespace
{

arc::io::file_buffer bytes(std::string_view text)
{
    const auto view = std::as_bytes(std::span(text.data(), text.size()));
    return {view.begin(), view.end()};
}

arc::io::virtual_path path(std::string_view text)
{
    auto parsed = arc::io::virtual_path::parse(text);
    REQUIRE(parsed.succeeded());
    return std::move(parsed.value());
}

struct package_fixture
{
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-runtime-content-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    arc::assets::asset_guid asset = arc::assets::generate_asset_guid();
    arc::assets::asset_guid dependency = arc::assets::generate_asset_guid();
    arc::io::file_buffer packaged = bytes("packaged-payload");
    arc::assets::cooked_artifact_address address{asset, arc::assets::artifact_schemas::virtual_geometry,
                                                 "terrain/regions/0/0/virtual-geometry"};
    std::filesystem::path manifest;

    package_fixture()
    {
        using namespace arc::assets;
        derived_data_cache cache({.root = root / "cache"});
        cache_error error;
        const auto hash = hash_bytes(packaged);
        REQUIRE(cache.put_blob(hash, packaged, error));
        cook_manifest description;
        description.target = windows_vulkan_cook_target();
        description.build_id = "base-build";
        description.roots.push_back(asset);
        description.dependency_closure.push_back(asset);
        description.dependency_edges.push_back({.asset = asset, .dependencies = {dependency}});
        description.artifacts.push_back({.asset = asset,
                                         .type = asset_types::terrain,
                                         .name = address.name,
                                         .schema = address.schema,
                                         .schema_version = 4,
                                         .hash = hash,
                                         .size = packaged.size(),
                                         .chunk = "terrain"});
        auto result = build_asset_packages(std::move(description), cache, root / "package");
        REQUIRE(result.succeeded());
        manifest = result.manifest_path;
    }

    ~package_fixture()
    {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }
};

} // namespace

TEST_CASE("package artifacts resolve and range-read through logical VFS handles")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto package_files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = package_files}).succeeded());
    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    REQUIRE(catalog.mount_package(path("package://base/" + fixture.manifest.filename().string())));

    auto artifact = catalog.resolve(fixture.address);
    REQUIRE(artifact.succeeded());
    CHECK(artifact.value().path().string().starts_with("artifact://game/"));
    CHECK(artifact.value().size() == fixture.packaged.size());
    auto range = vfs.read_range(artifact.value(), 2, 5).get();
    REQUIRE(range.succeeded());
    CHECK(range.value() == arc::io::file_buffer(fixture.packaged.begin() + 2, fixture.packaged.begin() + 7));
}

TEST_CASE("cooked package catalog exposes logical assets through Asset Database")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto package_files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = package_files}).succeeded());

    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    REQUIRE(catalog.mount_package(path("package://base/" + fixture.manifest.filename().string()), 25));

    const auto record = catalog.database().query(fixture.asset, arc::assets::asset_types::terrain);
    REQUIRE(record);
    CHECK(record->guid == fixture.asset);
    CHECK(record->type == arc::assets::asset_types::terrain);
    CHECK(record->state == arc::assets::asset_state::ready);
    CHECK(record->residency == arc::assets::asset_residency::metadata_only);
    CHECK(record->dependencies == std::vector<arc::assets::asset_guid>{fixture.dependency});
    CHECK(catalog.database().dependencies(fixture.asset) == record->dependencies);
    CHECK(catalog.database().reverse_dependencies(fixture.dependency) ==
          std::vector<arc::assets::asset_guid>{fixture.asset});
    CHECK(record->active_provider.starts_with(std::string(arc::assets::asset_provider_ids::cooked_package_prefix) +
                                              "base-build:"));
    REQUIRE(record->providers.size() == 1);
    CHECK(record->providers.front().kind == arc::assets::asset_provider_kind::cooked_package);
    CHECK(record->providers.front().priority == 25);
    CHECK(record->providers.front().active);
    CHECK(record->providers.front().read_only);
    REQUIRE(record->representations.size() == 1);
    CHECK(record->representations.front().name == fixture.address.name);
    CHECK(record->representations.front().content_hash == arc::assets::hash_bytes(fixture.packaged));
    CHECK(record->representations.front().size == fixture.packaged.size());
    CHECK(record->representations.front().locator_hint.starts_with("artifact://game/"));
    CHECK_FALSE(catalog.database().query(fixture.asset, arc::assets::asset_types::texture_2d));
    CHECK(catalog.database().revision() > 0);
}

TEST_CASE("cooked Asset Database change cursor tracks mounted providers")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = files}).succeeded());

    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    const auto manifest = path("package://base/" + fixture.manifest.filename().string());
    REQUIRE(catalog.mount_package(manifest, 10));
    const auto first = catalog.database().changes_since(0);
    REQUIRE(first.events.size() == 1);
    CHECK(first.events.front().kind == arc::assets::asset_database_change_kind::added);
    CHECK(first.events.front().guid == fixture.asset);

    REQUIRE(catalog.mount_package(manifest, 20));
    const auto second = catalog.database().changes_since(first.next_sequence);
    REQUIRE(second.events.size() == 1);
    CHECK(second.events.front().kind == arc::assets::asset_database_change_kind::provider_changed);
    CHECK(second.events.front().guid == fixture.asset);
    CHECK(catalog.database().changes_since(second.next_sequence).events.empty());
}

TEST_CASE("cooked Asset Database chooses the highest priority package provider")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto package_files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = package_files}).succeeded());

    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    const auto manifest_path = path("package://base/" + fixture.manifest.filename().string());
    REQUIRE(catalog.mount_package(manifest_path, 10));
    REQUIRE(catalog.mount_package(manifest_path, 100));

    const auto record = catalog.database().query(fixture.asset);
    REQUIRE(record);
    REQUIRE(record->providers.size() == 2);
    const auto active = std::find_if(record->providers.begin(), record->providers.end(),
                                     [](const auto& provider) { return provider.active; });
    REQUIRE(active != record->providers.end());
    CHECK(active->priority == 100);
    CHECK(active->id == record->active_provider);
    CHECK(record->representations.size() == 2);
}

TEST_CASE("CAS updates publish atomically persist and supersede resolved generations")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto package_files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = package_files}).succeeded());
    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    REQUIRE(catalog.mount_package(path("package://base/" + fixture.manifest.filename().string())));

    const auto overlay_root = fixture.root / "live";
    auto overlay = std::make_shared<arc::assets::cas_overlay_provider>(native_files, overlay_root);
    REQUIRE(overlay->load("windows-test", "base-build"));
    REQUIRE(catalog.mount_provider(overlay, 100, "live overlay"));
    arc::assets::runtime_update_receiver receiver(
        overlay, {.cache_root = overlay_root, .target_profile = "windows-test", .base_build_id = "base-build"});

    const auto replacement = bytes("live-payload");
    const auto replacement_hash = arc::assets::hash_bytes(replacement);
    arc::assets::runtime_update_manifest update{.target_profile = "windows-test",
                                                .base_build_id = "base-build",
                                                .update_sequence = 1,
                                                .artifacts = {{.address = fixture.address,
                                                               .type = arc::assets::asset_types::terrain,
                                                               .schema_version = 4,
                                                               .hash = replacement_hash,
                                                               .size = replacement.size()}}};
    REQUIRE(receiver.begin(update));
    REQUIRE(receiver.begin_blob(replacement_hash, replacement.size()));
    REQUIRE(receiver.stage_blob(std::span(replacement).first(4)));
    REQUIRE(receiver.stage_blob(std::span(replacement).subspan(4)));
    REQUIRE(receiver.finish_blob());
    REQUIRE(receiver.finish_verify());

    auto before_commit = catalog.resolve(fixture.address);
    REQUIRE(before_commit.succeeded());
    CHECK(vfs.read_all(before_commit.value()).get().value() == fixture.packaged);
    REQUIRE(receiver.commit());
    auto after_commit = catalog.resolve(fixture.address);
    REQUIRE(after_commit.succeeded());
    CHECK(vfs.read_all(after_commit.value()).get().value() == replacement);
    const auto changes = catalog.poll_changes();
    CHECK(std::any_of(
        changes.events.begin(), changes.events.end(), [&](const auto& event)
        { return event.address == fixture.address && event.kind == arc::assets::cooked_artifact_change_kind::added; }));

    const auto replacement_two = bytes("newer-payload");
    arc::assets::runtime_update_manifest update_two = update;
    update_two.update_sequence = 2;
    update_two.artifacts.front().hash = arc::assets::hash_bytes(replacement_two);
    update_two.artifacts.front().size = replacement_two.size();
    const std::array blobs{std::pair{update_two.artifacts.front().hash, replacement_two}};
    REQUIRE(receiver.ingest(update_two, blobs));
    const auto stale = vfs.read_all(after_commit.value()).get();
    REQUIRE_FALSE(stale.succeeded());
    CHECK(stale.error().code == arc::io::file_error_code::stale_handle);

    auto restored = std::make_shared<arc::assets::cas_overlay_provider>(native_files, overlay_root);
    REQUIRE(restored->load("windows-test", "base-build"));
    CHECK(restored->active_sequence() == 2);
    const auto logical = arc::assets::cooked_artifact_virtual_path(fixture.address);
    REQUIRE(logical.succeeded());
    auto restored_file = restored->resolve(logical.value().relative_path());
    CHECK(restored_file.status == arc::io::provider_lookup_status::found);

    CHECK_FALSE(receiver.begin(update_two));
    CHECK(overlay->active_sequence() == 2);
    auto corrupt_bytes = replacement_two;
    corrupt_bytes.front() ^= std::byte{1};
    auto corrupt_update = update_two;
    corrupt_update.update_sequence = 3;
    const std::array corrupt_blobs{std::pair{corrupt_update.artifacts.front().hash, corrupt_bytes}};
    CHECK_FALSE(receiver.ingest(corrupt_update, corrupt_blobs));
    CHECK(overlay->active_sequence() == 2);
    auto preserved = catalog.resolve(fixture.address);
    REQUIRE(preserved.succeeded());
    CHECK(vfs.read_all(preserved.value()).get().value() == replacement_two);
}

TEST_CASE("cooked database projects overlay records and provider unmount")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto files = std::make_shared<arc::io::filesystem_file_provider>(native_files, fixture.root / "package");
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = files}).succeeded());
    arc::assets::cooked_asset_catalog catalog(vfs, path("artifact://game"));
    REQUIRE(catalog.mount_package(path("package://base/" + fixture.manifest.filename().string()), 10));

    auto overlay = std::make_shared<arc::assets::cas_overlay_provider>(native_files, fixture.root / "overlay");
    REQUIRE(catalog.mount_overlay(overlay, 100));
    auto before = catalog.database().query(fixture.asset);
    REQUIRE(before);
    REQUIRE(before->providers.size() == 1);

    const auto overlay_only = arc::assets::generate_asset_guid();
    const auto payload = bytes("overlay-only");
    const arc::assets::runtime_update_manifest manifest{
        .target_profile = "windows-test",
        .base_build_id = "base-build",
        .update_sequence = 1,
        .artifacts = {{.address = {overlay_only, fixture.address.schema, "custom"},
                       .type = arc::assets::asset_types::terrain,
                       .schema_version = 1,
                       .hash = arc::assets::hash_bytes(payload),
                       .size = payload.size()}}};
    arc::assets::runtime_update_receiver receiver(
        overlay, {.cache_root = fixture.root / "overlay",
                  .target_profile = "windows-test", .base_build_id = "base-build"});
    const std::array blobs{std::pair{manifest.artifacts.front().hash, payload}};
    REQUIRE(receiver.ingest(manifest, blobs));
    static_cast<void>(catalog.poll_changes());

    const auto record = catalog.database().query(overlay_only, arc::assets::asset_types::terrain);
    REQUIRE(record);
    CHECK(record->providers.size() == 1);
    CHECK(record->providers.front().kind == arc::assets::asset_provider_kind::runtime_overlay);
    CHECK(record->providers.front().active);
    CHECK(record->active_provider == record->providers.front().id);
    CHECK(record->representations.front().content_hash == manifest.artifacts.front().hash);

    REQUIRE(catalog.unmount_provider(record->active_provider));
    CHECK_FALSE(catalog.database().query(overlay_only));
}

TEST_CASE("CAS rejects corrupt incompatible and stale updates without replacing active content")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    const auto overlay_root = fixture.root / "live";
    auto overlay = std::make_shared<arc::assets::cas_overlay_provider>(native_files, overlay_root);
    arc::assets::runtime_update_receiver receiver(
        overlay, {.cache_root = overlay_root, .target_profile = "windows-test", .base_build_id = "base-build"});

    const auto good = bytes("correct");
    const auto bad = bytes("corrupt");
    arc::assets::runtime_update_manifest update{.target_profile = "windows-test",
                                                .base_build_id = "base-build",
                                                .update_sequence = 1,
                                                .artifacts = {{.address = fixture.address,
                                                               .type = arc::assets::asset_types::terrain,
                                                               .schema_version = 4,
                                                               .hash = arc::assets::hash_bytes(good),
                                                               .size = good.size()}}};
    const std::array bad_blobs{std::pair{update.artifacts.front().hash, bad}};
    CHECK_FALSE(receiver.ingest(update, bad_blobs));
    CHECK(overlay->active_sequence() == 0);

    update.target_profile = "android-test";
    const std::array good_blobs{std::pair{update.artifacts.front().hash, good}};
    CHECK_FALSE(receiver.ingest(update, good_blobs));
    CHECK(overlay->active_sequence() == 0);
    CHECK(receiver.telemetry().rejected_updates >= 2);
}

TEST_CASE("CAS tombstones stop fallback to packaged artifacts")
{
    package_fixture fixture;
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service native_files(jobs);
    arc::io::virtual_file_system vfs(jobs);
    auto base = std::make_shared<arc::io::memory_file_provider>(jobs);
    const auto logical = arc::assets::cooked_artifact_virtual_path(fixture.address);
    REQUIRE(logical.succeeded());
    const std::array base_update{arc::io::memory_provider_update{
        .relative_path = std::string(logical.value().relative_path()), .bytes = fixture.packaged}};
    REQUIRE(base->publish(base_update).succeeded());
    REQUIRE(vfs.mount({.root = path("artifact://game"), .priority = 0, .provider = base}).succeeded());

    const auto overlay_root = fixture.root / "live";
    auto overlay = std::make_shared<arc::assets::cas_overlay_provider>(native_files, overlay_root);
    REQUIRE(vfs.mount({.root = path("artifact://game"), .priority = 100, .provider = overlay}).succeeded());
    arc::assets::runtime_update_receiver receiver(
        overlay, {.cache_root = overlay_root, .target_profile = "windows-test", .base_build_id = "base-build"});
    arc::assets::runtime_update_manifest removal{.target_profile = "windows-test",
                                                 .base_build_id = "base-build",
                                                 .update_sequence = 1,
                                                 .artifacts = {{.address = fixture.address,
                                                                .type = arc::assets::asset_types::terrain,
                                                                .schema_version = 4,
                                                                .tombstone = true}}};
    const std::span<const std::pair<arc::assets::content_hash, arc::io::file_buffer>> no_blobs;
    REQUIRE(receiver.ingest(removal, no_blobs));
    const auto removed = vfs.resolve(logical.value());
    REQUIRE_FALSE(removed.succeeded());
    CHECK(removed.error().code == arc::io::file_error_code::tombstoned);
}

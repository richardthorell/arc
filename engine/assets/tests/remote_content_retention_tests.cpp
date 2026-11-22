#include <arc/assets/remote_content_cache.h>
#include <arc/assets/remote_content_retention.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{

arc::io::file_buffer bytes(std::string_view text)
{
    const auto view = std::as_bytes(std::span(text.data(), text.size()));
    return {view.begin(), view.end()};
}

arc::assets::remote_content_object object(std::string_view text)
{
    const auto digest = arc::assets::to_string(arc::assets::hash_bytes(bytes(text)));
    return {.path = "objects/sha256/" + digest, .sha256 = digest, .size = text.size()};
}

arc::assets::remote_content_release release(std::uint64_t generation, std::string_view manifest,
                                            std::string_view shared_chunk = "chunk shared")
{
    arc::assets::remote_content_release result;
    result.project_guid = "11111111-2222-4333-8444-555555555555";
    result.engine_version = "0.1.0";
    result.base_build_id = "runtime-build";
    result.cook_build_id = "cook-build-" + std::to_string(generation);
    result.target = {.platform = "windows",
                     .architecture = "x86_64",
                     .renderer = "vulkan",
                     .cook_profile = "windows-vulkan",
                     .texture_family = "bc"};
    result.generation = generation;
    result.cook_manifest_name = "content.arccookmanifest";
    result.cook_manifest = object(manifest);
    result.chunks.push_back({.name = "startup.arcpak", .object = object(shared_chunk)});
    return result;
}

struct fixture
{
    std::filesystem::path root = std::filesystem::temp_directory_path() /
                                 ("arc-c9-retention-" + arc::assets::to_string(arc::assets::generate_asset_guid()));

    arc::assets::derived_data_cache_config config() const
    {
        arc::assets::derived_data_cache_config cfg;
        cfg.root = root;
        cfg.cleanup.maximum_bytes = 64;
        cfg.cleanup.prune_threshold = 0.5f;
        cfg.cleanup.prune_target = 0.25f;
        return cfg;
    }

    void store(arc::assets::derived_data_cache& cache, std::string_view text)
    {
        arc::assets::cache_error error;
        auto body = bytes(text);
        REQUIRE(cache.put_blob(arc::assets::hash_bytes(body), body, error));
    }

    bool has(arc::assets::derived_data_cache& cache, std::string_view text)
    {
        arc::assets::cache_error error;
        return arc::assets::load_verified_remote_object(cache, object(text), error).has_value();
    }

    ~fixture()
    {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }
};

} // namespace

TEST_CASE("C9.2g1 retains active and last-good SHA objects across pruning and process restart")
{
    fixture f;
    const auto active = release(5, "manifest active");
    const auto old = release(4, "manifest previous");
    const auto scope = arc::assets::remote_content_retention_scope(active);
    REQUIRE(scope == arc::assets::remote_content_retention_scope(old));

    {
        arc::assets::derived_data_cache cache(f.config());
        f.store(cache, "manifest active");
        f.store(cache, "manifest previous");
        f.store(cache, "chunk shared");
        f.store(cache, "unused expendable");
        const auto retained = arc::assets::retain_remote_content_generations(cache, active, old);
        REQUIRE(retained.succeeded());
        CHECK(retained.retained_objects == 3); // Shared chunk is pinned exactly once.
        CHECK(cache.persistent_pins(scope).size() == 3);
        CHECK(cache.prune(true) == std::string_view("unused expendable").size());
        CHECK(f.has(cache, "manifest active"));
        CHECK(f.has(cache, "manifest previous"));
        CHECK(f.has(cache, "chunk shared"));
        CHECK_FALSE(f.has(cache, "unused expendable"));
    }

    // Construct an entirely new cache instance: it must reconstruct pins
    // before cleanup, even with no editor server and no network available.
    {
        arc::assets::derived_data_cache cache(f.config());
        REQUIRE(cache.persistent_retention_ready());
        CHECK(cache.persistent_pins(scope).size() == 3);
        CHECK(cache.prune(true) == 0);
        CHECK(f.has(cache, "manifest active"));
        CHECK(f.has(cache, "manifest previous"));
        CHECK(f.has(cache, "chunk shared"));
    }
}

TEST_CASE("C9.2g1 generation rotation releases superseded blobs but preserves prior active as last-good")
{
    fixture f;
    const auto old = release(4, "manifest previous");
    const auto active = release(5, "manifest active");
    const auto next = release(6, "manifest next");
    const auto scope = arc::assets::remote_content_retention_scope(active);
    arc::assets::derived_data_cache cache(f.config());

    f.store(cache, "manifest previous");
    f.store(cache, "manifest active");
    f.store(cache, "manifest next");
    f.store(cache, "chunk shared");

    REQUIRE(arc::assets::retain_remote_content_generations(cache, active, old).succeeded());
    REQUIRE(arc::assets::retain_remote_content_generations(cache, next, active).succeeded());
    CHECK(cache.persistent_pins(scope).size() == 3);
    CHECK(cache.prune(true) == std::string_view("manifest previous").size());
    CHECK_FALSE(f.has(cache, "manifest previous"));
    CHECK(f.has(cache, "manifest active"));
    CHECK(f.has(cache, "manifest next"));
    CHECK(f.has(cache, "chunk shared"));

    arc::assets::cache_error error;
    REQUIRE(arc::assets::clear_remote_content_retention(cache, next, error));
    CHECK(cache.persistent_pins(scope).empty());
    CHECK(cache.prune(true) > 0);
    CHECK_FALSE(f.has(cache, "manifest active"));
    CHECK_FALSE(f.has(cache, "chunk shared"));
}

TEST_CASE("C9.2g1 cannot replace last-good retention with forged or incompatible release metadata")
{
    fixture f;
    const auto active = release(5, "manifest active");
    const auto old = release(4, "manifest previous");
    arc::assets::derived_data_cache cache(f.config());
    REQUIRE(arc::assets::retain_remote_content_generations(cache, active, old).succeeded());
    const auto scope = arc::assets::remote_content_retention_scope(active);
    const auto before = cache.persistent_pins(scope);

    auto incompatible = old;
    incompatible.target.platform = "android";
    const auto wrong = arc::assets::retain_remote_content_generations(cache, active, incompatible);
    CHECK_FALSE(wrong.succeeded());
    CHECK(cache.persistent_pins(scope) == before);

    auto reversed = old;
    reversed.generation = active.generation;
    CHECK_FALSE(arc::assets::retain_remote_content_generations(cache, active, reversed).succeeded());
    CHECK(cache.persistent_pins(scope) == before);

    auto tampered = active;
    tampered.chunks[0].object.path = "../private/source";
    CHECK_FALSE(arc::assets::retain_remote_content_generations(cache, tampered, old).succeeded());
    CHECK(cache.persistent_pins(scope) == before);

    auto inconsistent = active;
    inconsistent.chunks[0].object.sha256 = inconsistent.cook_manifest.sha256;
    inconsistent.chunks[0].object.path = inconsistent.cook_manifest.path;
    inconsistent.chunks[0].object.size += 1;
    CHECK_FALSE(arc::assets::retain_remote_content_generations(cache, inconsistent, old).succeeded());
    CHECK(cache.persistent_pins(scope) == before);
}

TEST_CASE("C9.2g1 rejects unreadable retention metadata and never evicts last-good on corruption")
{
    fixture f;
    const auto active = release(5, "manifest active");
    {
        arc::assets::derived_data_cache cache(f.config());
        f.store(cache, "manifest active");
        REQUIRE(arc::assets::retain_remote_content_generations(cache, active).succeeded());
    }
    // Simulates torn/crashed or malicious retention metadata. A startup
    // must not guess that it is safe to evict the last-good CAS objects.
    {
        std::ofstream broken(f.root / "persistent-pins.json", std::ios::trunc);
        REQUIRE(broken.good());
        broken << "{\"format\":\"arc.ddc-persistent-pins\",\"version\":1,\"scopes\":[]}";
    }
    arc::assets::derived_data_cache recovered(f.config());
    CHECK_FALSE(recovered.persistent_retention_ready());
    CHECK(recovered.prune(true) == 0);
    CHECK(f.has(recovered, "manifest active"));
    arc::assets::cache_error error;
    CHECK_FALSE(recovered.set_persistent_pins("c9.test", {}, error));
    CHECK_FALSE(error.message.empty());
}

TEST_CASE("C9.2g1 offline restart reuses retained CAS but does not authenticate corrupt blobs")
{
    fixture f;
    const auto active = release(5, "manifest active");
    const auto scope = arc::assets::remote_content_retention_scope(active);
    {
        arc::assets::derived_data_cache cache(f.config());
        f.store(cache, "manifest active");
        f.store(cache, "chunk shared");
        REQUIRE(arc::assets::retain_remote_content_generations(cache, active).succeeded());
    }
    auto offline_cfg = f.config();
    offline_cfg.access = arc::assets::cache_access::offline;
    arc::assets::derived_data_cache offline(offline_cfg);
    REQUIRE(offline.persistent_retention_ready());
    CHECK(offline.persistent_pins(scope).size() == 2);
    CHECK(offline.prune(true) == 0);
    CHECK(f.has(offline, "manifest active"));
    CHECK(f.has(offline, "chunk shared"));

    // Retention is merely an eviction barrier, not a trust or activation flag.
    // Even a pinned CAS file must pass SHA verification on every load.
    const auto digest = object("chunk shared").sha256;
    const auto blob = f.root / "cas" / "sha256" / digest.substr(0, 2) / digest;
    {
        std::ofstream corrupt(blob, std::ios::binary | std::ios::trunc);
        REQUIRE(corrupt.good());
        corrupt << "tampered bytes";
    }
    CHECK_FALSE(f.has(offline, "chunk shared"));
}

TEST_CASE("C9.2g1 persistent scope mutations fail closed for read-only cache and invalid owner")
{
    fixture f;
    const auto active = release(5, "manifest active");
    {
        arc::assets::derived_data_cache cache(f.config());
        REQUIRE(arc::assets::retain_remote_content_generations(cache, active).succeeded());
    }
    auto cfg = f.config();
    cfg.access = arc::assets::cache_access::read_only;
    arc::assets::derived_data_cache read_only(cfg);
    arc::assets::cache_error error;
    CHECK_FALSE(read_only.set_persistent_pins("../escape", {}, error));
    CHECK_FALSE(error.message.empty());
    error = {};
    CHECK_FALSE(arc::assets::clear_remote_content_retention(read_only, active, error));
    CHECK_FALSE(error.message.empty());
    CHECK(read_only.persistent_pins(arc::assets::remote_content_retention_scope(active)).size() == 2);
}

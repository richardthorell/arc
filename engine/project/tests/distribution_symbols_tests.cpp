#include <arc/project/distribution_symbols.h>

#include <arc/assets/assets.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;

struct fixture
{
    fs::path root = fs::temp_directory_path() / arc::assets::to_string(arc::assets::generate_asset_guid());
    fs::path release = root / "Shipping";
    fs::path symbols = root / "Symbols";
    fs::path pdb = root / "Build" / "Game.pdb";
    project::distribution_manifest manifest;

    fixture()
    {
        manifest.project_guid = "11111111-1111-1111-1111-111111111111";
        manifest.engine_version = "0.1.0";
        manifest.sdk_version = "0.1.0";
        manifest.target = {.id = "windows-x64-vulkan",
                           .platform = "windows",
                           .architecture = "x86_64",
                           .renderer = "vulkan",
                           .configuration = "Shipping",
                           .cook_profile = "windows-x64-vulkan"};
        manifest.role = "client";
        manifest.finalizer = "windows-directory";
        manifest.entrypoint = {.kind = "executable", .value = "bin/Game.exe"};
        manifest.runtime_config = "Config/runtime.json";
        manifest.cook_manifests = {"Content/game.arccookmanifest"};
        manifest.startup_scenes = {"22222222-2222-2222-2222-222222222222"};
        add("bin/Game.exe", "executable", "native executable");
        add("Config/runtime.json", "configuration", "{}");
        add("Content/game.arccookmanifest", "content", "cook manifest");
        const auto serialized = project::serialize_distribution_manifest(manifest);
        REQUIRE(serialized);
        std::ofstream out(release / "arc-distribution.json", std::ios::binary);
        out << serialized.value();
        out.close();
        fs::create_directories(pdb.parent_path());
        std::ofstream symbol_file(pdb, std::ios::binary);
        symbol_file << "test symbols";
    }

    ~fixture()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    void add(const std::string& path, const std::string& kind, const std::string& bytes)
    {
        const auto target = release / path;
        fs::create_directories(target.parent_path());
        {
            std::ofstream stream(target, std::ios::binary);
            stream << bytes;
        }
        const auto hash = arc::assets::hash_file(target);
        REQUIRE(hash);
        manifest.files.push_back({.path = path,
                                  .kind = kind,
                                  .size = fs::file_size(target),
                                  .sha256 = arc::assets::to_string(hash.value())});
    }

    static std::string read(const fs::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    }
};
} // namespace

TEST_CASE("D2.6 symbols sidecar is separate, deterministic and linked to Shipping manifest")
{
    fixture f;
    REQUIRE(project::write_distribution_symbols(f.release, f.symbols, {f.pdb}));
    CHECK(fs::is_regular_file(f.symbols / "symbols/game.pdb"));
    CHECK_FALSE(fs::exists(f.release / "symbols"));
    const auto first = fixture::read(f.symbols / "arc-symbols.json");
    CHECK(first.find("distributionManifestSha256") != std::string::npos);
    CHECK(first.find("bin/Game.exe") != std::string::npos);
    fs::remove_all(f.symbols);
    REQUIRE(project::write_distribution_symbols(f.release, f.symbols, {f.pdb}));
    CHECK(fixture::read(f.symbols / "arc-symbols.json") == first);
}

TEST_CASE("D2.6 sidecar rejects unmatched PDB and corrupt runtime without partial output")
{
    fixture f;
    const auto unmatched = f.root / "Build" / "Other.pdb";
    fs::copy_file(f.pdb, unmatched);
    auto status = project::write_distribution_symbols(f.release, f.symbols, {unmatched});
    REQUIRE_FALSE(status);
    CHECK(status.error().field == "symbols");
    CHECK_FALSE(fs::exists(f.symbols));

    {
        std::ofstream stream(f.release / "Config/runtime.json", std::ios::app);
        stream << "tampered";
    }
    status = project::write_distribution_symbols(f.release, f.symbols, {f.pdb});
    REQUIRE_FALSE(status);
    CHECK(status.error().field == "files.size");
    CHECK_FALSE(fs::exists(f.symbols));
}

TEST_CASE("D2.6 sidecar refuses duplicate PDB and existing output")
{
    fixture f;
    auto status = project::write_distribution_symbols(f.release, f.symbols, {f.pdb, f.pdb});
    REQUIRE_FALSE(status);
    CHECK(status.error().field == "symbols");
    REQUIRE(project::write_distribution_symbols(f.release, f.symbols, {f.pdb}));
    status = project::write_distribution_symbols(f.release, f.symbols, {f.pdb});
    REQUIRE_FALSE(status);
    CHECK(status.error().field == "symbolsRoot");
}

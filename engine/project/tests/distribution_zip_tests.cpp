#include <arc/project/distribution_zip.h>

#include <arc/assets/assets.h>

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;

struct zip_fixture
{
    fs::path root =
        fs::temp_directory_path() / ("arc-d2-zip-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    fs::path distribution = root / "Shipping";
    project::distribution_manifest manifest;

    zip_fixture()
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
        add("bin/Game.exe", "executable", "native-executable");
        add("Content/game.arccookmanifest", "content", "cooked-content-manifest");
        add("Config/runtime.json", "configuration", "{}");
        const auto serialized = project::serialize_distribution_manifest(manifest);
        REQUIRE(serialized);
        std::ofstream out(distribution / "arc-distribution.json", std::ios::binary);
        out << serialized.value();
    }

    void add(const std::string& name, const std::string& kind, const std::string& contents)
    {
        const auto path = distribution / name;
        fs::create_directories(path.parent_path());
        {
            std::ofstream out(path, std::ios::binary);
            out << contents;
        }
        const auto hash = arc::assets::hash_file(path);
        REQUIRE(hash);
        manifest.files.push_back(
            {.path = name, .kind = kind, .size = fs::file_size(path), .sha256 = arc::assets::to_string(hash.value())});
    }

    ~zip_fixture()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    static std::vector<char> read(const fs::path& path)
    {
        std::ifstream stream(path, std::ios::binary);
        return {std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    }
};
} // namespace

TEST_CASE("D2.5 ZIP output is reproducible with no timestamp or build-path variation")
{
    zip_fixture fixture;
    const auto first = fixture.root / "first.zip";
    const auto second = fixture.root / "second.zip";
    REQUIRE(project::write_distribution_zip(fixture.distribution, first));
    REQUIRE(project::write_distribution_zip(fixture.distribution, second));
    const auto a = zip_fixture::read(first);
    const auto b = zip_fixture::read(second);
    CHECK(a == b);
    REQUIRE(a.size() > 100);
    CHECK(a[0] == 'P');
    CHECK(a[1] == 'K');
    CHECK(fs::is_regular_file(fixture.distribution / "bin/Game.exe"));
}

TEST_CASE("D2.5 ZIP export fails closed for corrupt distribution bytes")
{
    zip_fixture fixture;
    {
        std::ofstream out(fixture.distribution / "Config/runtime.json", std::ios::app);
        out << "corrupt";
    }
    const auto output = fixture.root / "corrupt.zip";
    const auto result = project::write_distribution_zip(fixture.distribution, output);
    REQUIRE_FALSE(result);
    CHECK(result.error().field == "files.size");
    CHECK_FALSE(fs::exists(output));
    CHECK_FALSE(fs::exists(fs::path(output.string() + ".pending")));
}

TEST_CASE("D2.5 ZIP export rejects extra unlisted files and existing archive outputs")
{
    zip_fixture fixture;
    const auto output = fixture.root / "release.zip";
    {
        std::ofstream out(fixture.distribution / "Source.txt");
        out << "not published";
    }
    const auto unlisted = project::write_distribution_zip(fixture.distribution, output);
    REQUIRE_FALSE(unlisted);
    CHECK(unlisted.error().field == "files");
    fs::remove(fixture.distribution / "Source.txt");
    REQUIRE(project::write_distribution_zip(fixture.distribution, output));
    const auto duplicate = project::write_distribution_zip(fixture.distribution, output);
    REQUIRE_FALSE(duplicate);
    CHECK(duplicate.error().field == "zipPath");
}

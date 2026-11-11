#include <arc/project/windows_dependencies.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace fs = std::filesystem;
namespace project = arc::project;

void put16(std::vector<std::uint8_t>& data, std::size_t offset, std::uint16_t value)
{
    data[offset] = static_cast<std::uint8_t>(value);
    data[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}
void put32(std::vector<std::uint8_t>& data, std::size_t offset, std::uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i)
        data[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
void put_string(std::vector<std::uint8_t>& data, std::size_t offset, std::string_view value)
{
    std::copy(value.begin(), value.end(), data.begin() + static_cast<std::ptrdiff_t>(offset));
    data[offset + value.size()] = 0;
}

struct pe_fixture
{
    fs::path root = fs::temp_directory_path() / ("arc-d2-pe-" + std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()));

    ~pe_fixture()
    {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }

    fs::path make(std::string filename, std::vector<std::string> imports)
    {
        fs::create_directories(root);
        std::vector<std::uint8_t> data(4096);
        put16(data, 0, 0x5a4d);
        put32(data, 0x3c, 0x80);
        put32(data, 0x80, 0x00004550);
        put16(data, 0x86, 1);
        put16(data, 0x94, 0xf0);
        const std::size_t optional = 0x98;
        put16(data, optional, 0x20b);
        put32(data, optional + 60, 0x200);
        put32(data, optional + 112 + 8, 0x1000);
        put32(data, optional + 112 + 12, 0x200);
        const std::size_t section = optional + 0xf0;
        put32(data, section + 8, 0x400);
        put32(data, section + 12, 0x1000);
        put32(data, section + 16, 0x400);
        put32(data, section + 20, 0x200);
        for (std::size_t i = 0; i < imports.size(); ++i)
        {
            put32(data, 0x200 + i * 20 + 12, 0x1100 + static_cast<std::uint32_t>(i * 48));
            put_string(data, 0x300 + i * 48, imports[i]);
        }
        const auto path = root / filename;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        return path;
    }
};
} // namespace

TEST_CASE("D2.4 resolves transitive imported DLLs and excludes OS dependencies")
{
    pe_fixture f;
    const auto executable = f.make("Game.exe", {"KERNEL32.dll", "physics.dll"});
    f.make("physics.dll", {"graphics.dll", "USER32.dll"});
    f.make("graphics.dll", {"ntdll.dll"});
    const auto resolved = project::resolve_windows_dependencies(executable, {}, {});
    REQUIRE(resolved);
    REQUIRE(resolved.value().files.size() == 2);
    CHECK(resolved.value().files[0].destination == "bin/physics.dll");
    CHECK(resolved.value().files[1].destination == "bin/graphics.dll");
    CHECK(resolved.value().dependencies.size() == 2);
}

TEST_CASE("D2.4 missing imported DLL fails with an actionable dependency diagnostic")
{
    pe_fixture f;
    const auto executable = f.make("Game.exe", {"missing.dll"});
    const auto resolved = project::resolve_windows_dependencies(executable, {}, {});
    REQUIRE_FALSE(resolved);
    CHECK(resolved.error().field == "dependencies");
    CHECK(resolved.error().message.find("missing.dll") != std::string::npos);
}

TEST_CASE("D2.4 rejects malformed PE binaries instead of silently publishing missing libraries")
{
    pe_fixture f;
    const auto executable = f.make("Game.exe", {"physics.dll"});
    {
        std::ofstream stream(executable, std::ios::binary | std::ios::trunc);
        stream << "not a PE";
    }
    const auto resolved = project::resolve_windows_dependencies(executable, {}, {});
    REQUIRE_FALSE(resolved);
    CHECK(resolved.error().message.find("PE") != std::string::npos);
}

TEST_CASE("D2.4 plugin import closure is included without duplicate imported DLLs")
{
    pe_fixture f;
    const auto executable = f.make("Game.exe", {"physics.dll"});
    f.make("physics.dll", {"KERNEL32.dll"});
    const auto plugin = f.make("extra.dll", {"physics.dll"});
    const auto resolved = project::resolve_windows_dependencies(executable, {plugin}, {});
    REQUIRE(resolved);
    CHECK(resolved.value().files.size() == 2);
    CHECK(resolved.value().files[0].kind == "plugin");
    CHECK(resolved.value().files[1].kind == "library");
}

#include <arc/project/windows_dependencies.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::project
{
namespace
{
namespace fs = std::filesystem;
using bytes = std::vector<std::uint8_t>;

project_error error(std::string field, std::string message)
{
    return {.code = project_error_code::invalid_descriptor, .field = std::move(field), .message = std::move(message)};
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::optional<std::uint16_t> u16(const bytes& data, std::size_t offset)
{
    if (offset > data.size() || data.size() - offset < 2) return std::nullopt;
    return static_cast<std::uint16_t>(data[offset] | (data[offset + 1] << 8));
}

std::optional<std::uint32_t> u32(const bytes& data, std::size_t offset)
{
    if (offset > data.size() || data.size() - offset < 4) return std::nullopt;
    return static_cast<std::uint32_t>(data[offset]) | (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16) | (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

std::optional<std::string> cstring(const bytes& data, std::size_t offset)
{
    if (offset >= data.size()) return std::nullopt;
    std::string result;
    for (std::size_t i = offset; i < data.size() && result.size() < 260; ++i)
    {
        if (data[i] == 0) return result.empty() ? std::nullopt : std::optional(std::move(result));
        if (data[i] < 32 || data[i] > 126) return std::nullopt;
        result.push_back(static_cast<char>(data[i]));
    }
    return std::nullopt;
}

struct section
{
    std::uint32_t rva{}, virtual_size{}, raw_size{}, raw_offset{};
};

std::optional<std::size_t> resolve_rva(const bytes& data, std::uint32_t rva, const std::vector<section>& sections,
                                       std::size_t header_size)
{
    if (rva < header_size && rva < data.size()) return rva;
    for (const auto& item : sections)
    {
        const auto extent = std::max(item.virtual_size, item.raw_size);
        if (rva >= item.rva && std::uint64_t(rva) - item.rva < extent)
        {
            const auto displacement = std::uint64_t(rva) - item.rva;
            if (displacement >= item.raw_size) return std::nullopt;
            const auto offset = std::uint64_t(item.raw_offset) + displacement;
            if (offset < data.size()) return static_cast<std::size_t>(offset);
        }
    }
    return std::nullopt;
}

core::result<std::vector<std::string>, project_error> imports(const fs::path& file)
{
    using result = core::result<std::vector<std::string>, project_error>;
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    if (!input) return result::failure(error("dependencies", "Cannot read PE file: " + file.string()));
    const auto length = input.tellg();
    if (length < 0 || length > static_cast<std::streamoff>(512ull * 1024 * 1024))
        return result::failure(error("dependencies", "Invalid PE size: " + file.string()));
    bytes data(static_cast<std::size_t>(length));
    input.seekg(0);
    if (!data.empty()) input.read(reinterpret_cast<char*>(data.data()), length);
    if (!input || u16(data, 0) != 0x5a4d)
        return result::failure(error("dependencies", "Input is not a valid DOS/PE image: " + file.string()));

    const auto pe = u32(data, 0x3c);
    if (!pe || !u32(data, *pe) || *u32(data, *pe) != 0x00004550)
        return result::failure(error("dependencies", "Missing PE signature: " + file.string()));
    const auto count = u16(data, std::size_t(*pe) + 6);
    const auto optional_size = u16(data, std::size_t(*pe) + 20);
    const auto optional = std::size_t(*pe) + 24;
    const auto magic = u16(data, optional);
    if (!count || !optional_size || !magic || (*magic != 0x10b && *magic != 0x20b) || *count > 96)
        return result::failure(error("dependencies", "Invalid PE headers: " + file.string()));
    const auto directory = optional + (*magic == 0x20b ? 112 : 96);
    const auto sections_at = optional + *optional_size;
    if (directory + 16 > sections_at || sections_at > data.size() ||
        std::uint64_t(*count) * 40 > data.size() - sections_at)
        return result::failure(error("dependencies", "Truncated PE headers: " + file.string()));
    const auto image_headers = u32(data, optional + 60);
    const auto import_rva = u32(data, directory + 8);
    const auto import_size = u32(data, directory + 12);
    if (!image_headers || !import_rva || !import_size)
        return result::failure(error("dependencies", "Invalid PE import directory: " + file.string()));
    if (*import_rva == 0 && *import_size == 0) return result::success({});

    std::vector<section> sections;
    for (std::size_t i = 0; i < *count; ++i)
    {
        const auto offset = sections_at + i * 40;
        const auto virtual_size = u32(data, offset + 8);
        const auto rva = u32(data, offset + 12);
        const auto raw_size = u32(data, offset + 16);
        const auto raw_offset = u32(data, offset + 20);
        if (!virtual_size || !rva || !raw_size || !raw_offset)
            return result::failure(error("dependencies", "Invalid PE section table: " + file.string()));
        sections.push_back({*rva, *virtual_size, *raw_size, *raw_offset});
    }

    const auto first = resolve_rva(data, *import_rva, sections, *image_headers);
    if (!first || *import_size < 20)
        return result::failure(error("dependencies", "Invalid PE import RVA: " + file.string()));
    std::vector<std::string> found;
    const auto maximum = std::min<std::uint64_t>(*import_size / 20, 4096);
    bool terminated = false;
    for (std::size_t i = 0; i < maximum; ++i)
    {
        const auto entry = std::uint64_t(*first) + i * 20;
        if (entry > data.size() || data.size() - entry < 20) break;
        const auto name = u32(data, static_cast<std::size_t>(entry + 12));
        if (!name) break;
        if (*name == 0)
        {
            terminated = true;
            break;
        }
        const auto name_offset = resolve_rva(data, *name, sections, *image_headers);
        const auto value = name_offset ? cstring(data, *name_offset) : std::nullopt;
        if (!value || lower(fs::path(*value).filename().string()) != lower(*value) || !lower(*value).ends_with(".dll"))
            return result::failure(error("dependencies", "Unsafe PE import name: " + file.string()));
        found.push_back(*value);
    }
    if (!terminated) return result::failure(error("dependencies", "Unterminated PE import table: " + file.string()));
    return result::success(std::move(found));
}

bool system_dll(std::string_view name)
{
    if (name.starts_with("api-ms-win-") || name.starts_with("ext-ms-")) return true;
    constexpr std::array names = {
        "kernel32.dll", "kernelbase.dll", "ntdll.dll",     "user32.dll",         "gdi32.dll",    "advapi32.dll",
        "shell32.dll",  "ole32.dll",      "oleaut32.dll",  "combase.dll",        "comdlg32.dll", "shlwapi.dll",
        "ws2_32.dll",   "winmm.dll",      "imm32.dll",     "version.dll",        "setupapi.dll", "bcrypt.dll",
        "crypt32.dll",  "secur32.dll",    "rpcrt4.dll",    "iphlpapi.dll",       "dwmapi.dll",   "uxtheme.dll",
        "dxgi.dll",     "d3d12.dll",      "d3d11.dll",     "d3dcompiler_47.dll", "opengl32.dll", "vulkan-1.dll",
        "msvcrt.dll",   "ucrtbase.dll",   "msvcp_win.dll", "userenv.dll",        "psapi.dll",    "cfgmgr32.dll",
        "winhttp.dll",  "wininet.dll",    "dbghelp.dll",   "normaliz.dll",       "wldap32.dll",  "winspool.drv"};
    return std::find(names.begin(), names.end(), name) != names.end();
}

} // namespace

windows_dependency_result resolve_windows_dependencies(const fs::path& executable, const std::vector<fs::path>& plugins,
                                                       const std::vector<fs::path>& search_directories)
{
    const auto reject = [](std::string message)
    { return windows_dependency_result::failure(error("dependencies", std::move(message))); };
    std::vector<fs::path> roots{executable.parent_path()};
    roots.insert(roots.end(), search_directories.begin(), search_directories.end());
    std::map<std::string, fs::path> discovered;
    for (const auto& root : roots)
    {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) return reject("Missing DLL search directory: " + root.string());
        for (const auto& entry : fs::directory_iterator(root, ec))
        {
            if (ec) return reject("Cannot scan DLL search directory: " + root.string());
            if (!entry.is_regular_file(ec) || entry.is_symlink(ec) || ec) continue;
            const auto name = lower(entry.path().filename().string());
            if (name.ends_with(".dll")) discovered.try_emplace(name, entry.path());
        }
    }

    windows_dependency_closure closure;
    std::vector<fs::path> pending{executable};
    pending.insert(pending.end(), plugins.begin(), plugins.end());
    std::set<std::string> processed;
    std::set<std::string> destinations;
    for (const auto& plugin : plugins)
    {
        const auto name = lower(plugin.filename().string());
        if (!destinations.insert(name).second) return reject("Duplicate plugin/DLL filename: " + name);
        closure.files.push_back(
            {.source = plugin, .destination = "plugins/" + plugin.filename().string(), .kind = "plugin"});
        closure.dependencies.push_back({.destination = closure.files.back().destination, .kind = "plugin"});
    }

    for (std::size_t index = 0; index < pending.size(); ++index)
    {
        const auto file = pending[index];
        std::error_code ec;
        const auto physical = fs::canonical(file, ec);
        if (ec || !fs::is_regular_file(physical)) return reject("Missing native image: " + file.string());
        if (!processed.insert(lower(physical.generic_string())).second) continue;
        auto listed = imports(physical);
        if (!listed) return windows_dependency_result::failure(listed.error());
        for (const auto& import : listed.value())
        {
            const auto name = lower(import);
            if (system_dll(name) || destinations.contains(name)) continue;
            const auto found = discovered.find(name);
            if (found == discovered.end())
                return reject("Missing required imported DLL " + import + " from " + physical.filename().string());
            destinations.insert(name);
            closure.files.push_back({.source = found->second,
                                     .destination = "bin/" + found->second.filename().string(),
                                     .kind = "library"});
            closure.dependencies.push_back({.destination = closure.files.back().destination, .kind = "library"});
            pending.push_back(found->second);
        }
    }
    return windows_dependency_result::success(std::move(closure));
}

} // namespace arc::project

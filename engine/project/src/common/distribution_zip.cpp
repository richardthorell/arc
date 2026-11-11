#include <arc/project/distribution_zip.h>

#include <arc/assets/assets.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace arc::project
{
namespace
{
namespace fs = std::filesystem;

project_status fail(project_error_code code, std::string field, std::string message)
{
    return project_status::failure({.code = code, .field = std::move(field), .message = std::move(message)});
}

void put16(std::ostream& out, std::uint16_t value)
{
    for (unsigned i = 0; i < 2; ++i) out.put(static_cast<char>(value >> (8 * i)));
}

void put32(std::ostream& out, std::uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) out.put(static_cast<char>(value >> (8 * i)));
}

std::uint32_t crc32(const std::vector<char>& bytes)
{
    std::uint32_t crc = 0xffffffffu;
    for (const unsigned char c : bytes)
    {
        crc ^= c;
        for (unsigned i = 0; i < 8; ++i)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xedb88320u : 0u);
    }
    return crc ^ 0xffffffffu;
}

bool within(const fs::path& root, const fs::path& child)
{
    auto lhs = root.begin();
    auto rhs = child.begin();
    while (lhs != root.end() && rhs != child.end())
        if (*lhs++ != *rhs++) return false;
    return lhs == root.end();
}

struct zip_entry
{
    std::string name;
    std::vector<char> bytes;
    std::uint32_t checksum{};
    std::uint32_t offset{};
};

bool write_zip(std::ostream& stream, std::vector<zip_entry>& entries)
{
    for (auto& entry : entries)
    {
        const auto offset = stream.tellp();
        if (offset < 0 || static_cast<std::uint64_t>(offset) > UINT32_MAX ||
            entry.bytes.size() > UINT32_MAX || entry.name.size() > UINT16_MAX)
            return false;
        entry.offset = static_cast<std::uint32_t>(offset);
        const auto size = static_cast<std::uint32_t>(entry.bytes.size());
        put32(stream, 0x04034b50u);
        put16(stream, 20);
        put16(stream, 0);
        put16(stream, 0); // Store without compression.
        put16(stream, 0); // Fixed DOS time: 1980-01-01 00:00.
        put16(stream, 33);
        put32(stream, entry.checksum);
        put32(stream, size);
        put32(stream, size);
        put16(stream, static_cast<std::uint16_t>(entry.name.size()));
        put16(stream, 0);
        stream.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
        stream.write(entry.bytes.data(), static_cast<std::streamsize>(entry.bytes.size()));
        if (!stream) return false;
    }

    const auto directory_start = stream.tellp();
    if (directory_start < 0 || static_cast<std::uint64_t>(directory_start) > UINT32_MAX) return false;
    for (const auto& entry : entries)
    {
        const auto size = static_cast<std::uint32_t>(entry.bytes.size());
        put32(stream, 0x02014b50u);
        put16(stream, 20);
        put16(stream, 20);
        put16(stream, 0);
        put16(stream, 0);
        put16(stream, 0);
        put16(stream, 33);
        put32(stream, entry.checksum);
        put32(stream, size);
        put32(stream, size);
        put16(stream, static_cast<std::uint16_t>(entry.name.size()));
        put16(stream, 0);
        put16(stream, 0);
        put16(stream, 0);
        put16(stream, 0);
        put32(stream, 0);
        put32(stream, entry.offset);
        stream.write(entry.name.data(), static_cast<std::streamsize>(entry.name.size()));
        if (!stream) return false;
    }
    const auto end = stream.tellp();
    if (end < directory_start || static_cast<std::uint64_t>(end) > UINT32_MAX ||
        entries.size() > UINT16_MAX)
        return false;
    put32(stream, 0x06054b50u);
    put16(stream, 0);
    put16(stream, 0);
    put16(stream, static_cast<std::uint16_t>(entries.size()));
    put16(stream, static_cast<std::uint16_t>(entries.size()));
    put32(stream, static_cast<std::uint32_t>(end - directory_start));
    put32(stream, static_cast<std::uint32_t>(directory_start));
    put16(stream, 0);
    return stream.good();
}

} // namespace

project_status write_distribution_zip(const fs::path& distribution_root, const fs::path& zip_path)
{
    std::error_code ec;
    const auto root = fs::canonical(distribution_root, ec);
    if (ec || !fs::is_directory(root))
        return fail(project_error_code::not_found, "distributionRoot", "Distribution directory does not exist");
    if (zip_path.empty() || !zip_path.is_absolute() || fs::exists(zip_path, ec) || ec)
        return fail(project_error_code::invalid_descriptor, "zipPath", "ZIP output must be an unused absolute path");
    const auto target = zip_path.lexically_normal();
    if (within(root, target))
        return fail(project_error_code::unsafe_path, "zipPath", "ZIP output must be outside the distribution");

    const auto manifest_path = root / "arc-distribution.json";
    std::ifstream manifest_file(manifest_path, std::ios::binary);
    if (!manifest_file) return fail(project_error_code::not_found, "manifest", "Missing distribution manifest");
    const std::string manifest_text(std::istreambuf_iterator<char>{manifest_file}, {});
    const auto parsed = parse_distribution_manifest(manifest_text);
    if (!parsed) return project_status::failure(parsed.error());

    std::vector<zip_entry> entries;
    entries.reserve(parsed.value().files.size() + 1);
    for (const auto& file : parsed.value().files)
    {
        const auto path = root / fs::path(file.path);
        const auto status = fs::symlink_status(path, ec);
        if (ec || fs::is_symlink(status) || !fs::is_regular_file(status))
            return fail(project_error_code::unsafe_path, "files", "Missing or unsafe file: " + file.path);
        const auto physical = fs::canonical(path, ec);
        if (ec || !within(root, physical))
            return fail(project_error_code::unsafe_path, "files", "File escapes distribution root: " + file.path);
        if (fs::file_size(path, ec) != file.size || ec)
            return fail(project_error_code::corrupt_content, "files.size", "File size mismatch: " + file.path);
        const auto digest = assets::hash_file(path);
        if (!digest || assets::to_string(digest.value()) != file.sha256)
            return fail(project_error_code::corrupt_content, "files.sha256", "File hash mismatch: " + file.path);

        std::ifstream input(path, std::ios::binary);
        if (!input) return fail(project_error_code::io_failed, "files", "Cannot read file: " + file.path);
        std::vector<char> bytes(std::istreambuf_iterator<char>{input}, {});
        if (!input.eof() && input.fail())
            return fail(project_error_code::io_failed, "files", "Cannot read entire file: " + file.path);
        entries.push_back({.name = file.path, .bytes = std::move(bytes)});
    }
    // Reject unlisted payloads as the directory finalizer does: ZIP must carry
    // exactly the verified distribution, never incidental files or build output.
    for (fs::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec))
    {
        if (ec) return fail(project_error_code::io_failed, "distributionRoot", ec.message());
        const auto status = it->symlink_status(ec);
        if (ec || fs::is_symlink(status))
            return fail(project_error_code::unsafe_path, "files", "Distribution contains a symlink");
        if (fs::is_directory(status)) continue;
        if (!fs::is_regular_file(status))
            return fail(project_error_code::unsafe_path, "files", "Distribution contains unsupported entry");
        const auto relative = it->path().lexically_relative(root).generic_string();
        if (relative == "arc-distribution.json") continue;
        if (std::none_of(entries.begin(), entries.end(), [&](const zip_entry& e) { return e.name == relative; }))
            return fail(project_error_code::invalid_descriptor, "files", "Unlisted distribution file: " + relative);
    }
    entries.push_back({.name = "arc-distribution.json",
                       .bytes = std::vector<char>(manifest_text.begin(), manifest_text.end())});
    std::sort(entries.begin(), entries.end(),
              [](const zip_entry& a, const zip_entry& b) { return a.name < b.name; });
    for (auto& entry : entries) entry.checksum = crc32(entry.bytes);

    const auto pending = fs::path(target.string() + ".pending");
    if (fs::exists(pending, ec) || ec)
        return fail(project_error_code::invalid_descriptor, "zipPath", "ZIP pending output already exists");
    fs::create_directories(target.parent_path(), ec);
    if (ec) return fail(project_error_code::io_failed, "zipPath", ec.message());
    {
        std::ofstream stream(pending, std::ios::binary | std::ios::trunc);
        if (!stream || !write_zip(stream, entries) || !stream.flush())
        {
            stream.close();
            fs::remove(pending, ec);
            return fail(project_error_code::io_failed, "zipPath", "Cannot write ZIP32 archive");
        }
    }
    fs::rename(pending, target, ec);
    if (ec)
    {
        std::error_code ignored;
        fs::remove(pending, ignored);
        return fail(project_error_code::io_failed, "zipPath", "Cannot publish ZIP archive: " + ec.message());
    }
    return project_status::success();
}

} // namespace arc::project

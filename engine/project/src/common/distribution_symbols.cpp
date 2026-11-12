#include <arc/project/distribution_symbols.h>

#include <arc/assets/assets.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace arc::project
{
namespace
{
namespace fs = std::filesystem;
using json = nlohmann::json;

project_status fail(project_error_code code, std::string field, std::string message)
{
    return project_status::failure({.code = code, .field = std::move(field), .message = std::move(message)});
}

bool within(const fs::path& parent, const fs::path& child)
{
    auto lhs = parent.begin();
    auto rhs = child.begin();
    while (lhs != parent.end() && rhs != child.end())
    {
        if (*lhs++ != *rhs++) return false;
    }
    return lhs == parent.end();
}

std::string lowercase(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return text;
}

struct symbol_entry
{
    fs::path source;
    std::string filename;
    std::string binary;
    std::uint64_t size{};
    std::string digest;
};

} // namespace

project_status write_distribution_symbols(const fs::path& distribution_root, const fs::path& symbols_root,
                                          const std::vector<fs::path>& symbol_files)
{
    std::error_code ec;
    const auto distribution = fs::canonical(distribution_root, ec);
    if (ec || !fs::is_directory(distribution))
        return fail(project_error_code::not_found, "distributionRoot", "Shipping distribution is missing");
    if (symbols_root.empty() || !symbols_root.is_absolute() || symbol_files.empty())
        return fail(project_error_code::invalid_descriptor, "symbolsRoot",
                    "Symbols output must be an absolute path and include at least one PDB");

    const auto output = symbols_root.lexically_normal();
    const auto pending = fs::path(output.string() + ".pending");
    if (within(distribution, output) || within(output, distribution) || within(distribution, pending))
        return fail(project_error_code::unsafe_path, "symbolsRoot", "Symbols must be outside the runtime distribution");
    if (fs::exists(output, ec) || ec || fs::exists(pending, ec) || ec)
        return fail(project_error_code::destination_not_empty, "symbolsRoot",
                    "Symbols output or pending directory already exists");

    const auto manifest_path = distribution / "arc-distribution.json";
    const auto manifest_status = fs::symlink_status(manifest_path, ec);
    if (ec || !fs::is_regular_file(manifest_status) || fs::is_symlink(manifest_status))
        return fail(project_error_code::not_found, "manifest", "Shipping distribution manifest is missing");
    std::ifstream input(manifest_path, std::ios::binary);
    if (!input) return fail(project_error_code::io_failed, "manifest", "Cannot read distribution manifest");
    const std::string manifest_text(std::istreambuf_iterator<char>{input}, {});
    const auto parsed = parse_distribution_manifest(manifest_text);
    if (!parsed) return project_status::failure(parsed.error());
    if (parsed.value().target.platform != "windows" || parsed.value().target.configuration != "Shipping")
        return fail(project_error_code::unsupported_platform, "target", "PDB sidecar requires Windows Shipping");
    const auto manifest_hash = assets::hash_file(manifest_path);
    if (!manifest_hash) return fail(project_error_code::io_failed, "manifest", "Cannot hash distribution manifest");

    // Verify the runtime inventory as well as PDB selection: don't produce
    // a symbol artifact for an already damaged or symbol-contaminated package.
    std::map<std::string, std::string> binary_by_stem;
    for (const auto& file : parsed.value().files)
    {
        if (file.kind == "symbols")
            return fail(project_error_code::invalid_descriptor, "files.kind", "Symbols are present in Shipping");
        const auto path = distribution / fs::path(file.path);
        const auto status = fs::symlink_status(path, ec);
        if (ec || !fs::is_regular_file(status) || fs::is_symlink(status))
            return fail(project_error_code::unsafe_path, "files.path", "Missing or unsafe file: " + file.path);
        const auto physical = fs::canonical(path, ec);
        if (ec || !within(distribution, physical) || fs::file_size(physical, ec) != file.size || ec)
            return fail(project_error_code::corrupt_content, "files.size", "Runtime file is invalid: " + file.path);
        const auto hash = assets::hash_file(physical);
        if (!hash || assets::to_string(hash.value()) != file.sha256)
            return fail(project_error_code::corrupt_content, "files.sha256",
                        "Runtime file hash mismatch: " + file.path);
        if (file.kind == "executable" || file.kind == "library" || file.kind == "plugin")
        {
            const auto stem = lowercase(fs::path(file.path).stem().string());
            if (!binary_by_stem.emplace(stem, file.path).second)
                return fail(project_error_code::invalid_descriptor, "files.path",
                            "Ambiguous binary basename for PDB matching: " + stem);
        }
    }

    std::set<std::string> selected;
    std::vector<symbol_entry> entries;
    entries.reserve(symbol_files.size());
    for (const auto& source : symbol_files)
    {
        if (source.empty() || !source.is_absolute() || lowercase(source.extension().string()) != ".pdb")
            return fail(project_error_code::invalid_descriptor, "symbols", "Every symbol must be an absolute PDB path");
        const auto status = fs::symlink_status(source, ec);
        if (ec || !fs::is_regular_file(status) || fs::is_symlink(status))
            return fail(project_error_code::not_found, "symbols", "Missing or unsafe PDB: " + source.string());
        const auto canonical = fs::canonical(source, ec);
        if (ec || within(distribution, canonical) || within(output, canonical) || within(pending, canonical))
            return fail(project_error_code::unsafe_path, "symbols", "PDB source overlaps distribution or output");
        const auto stem = lowercase(source.stem().string());
        const auto binary = binary_by_stem.find(stem);
        if (binary == binary_by_stem.end())
            return fail(project_error_code::invalid_descriptor, "symbols",
                        "PDB does not match a packaged native binary: " + source.filename().string());
        const auto filename = lowercase(source.filename().string());
        if (!selected.insert(filename).second)
            return fail(project_error_code::invalid_descriptor, "symbols", "Duplicate PDB basename: " + filename);
        const auto hash = assets::hash_file(canonical);
        const auto size = fs::file_size(canonical, ec);
        if (ec || !hash) return fail(project_error_code::io_failed, "symbols", "Cannot hash selected PDB");
        entries.push_back({.source = canonical,
                           .filename = filename,
                           .binary = binary->second,
                           .size = size,
                           .digest = assets::to_string(hash.value())});
    }
    std::sort(entries.begin(), entries.end(),
              [](const symbol_entry& left, const symbol_entry& right) { return left.filename < right.filename; });

    fs::create_directories(output.parent_path(), ec);
    if (ec) return fail(project_error_code::io_failed, "symbolsRoot", ec.message());
    fs::create_directories(pending / "symbols", ec);
    if (ec) return fail(project_error_code::io_failed, "symbolsRoot", ec.message());
    const auto rollback = [&]
    {
        std::error_code ignored;
        fs::remove_all(pending, ignored);
    };

    json record = {{"format", "arc.distribution.symbols"},
                   {"formatVersion", 1},
                   {"projectGuid", parsed.value().project_guid},
                   {"engineVersion", parsed.value().engine_version},
                   {"distributionManifestSha256", assets::to_string(manifest_hash.value())},
                   {"files", json::array()}};
    for (const auto& entry : entries)
    {
        const auto destination = pending / "symbols" / entry.filename;
        fs::copy_file(entry.source, destination, fs::copy_options::none, ec);
        if (ec)
        {
            rollback();
            return fail(project_error_code::io_failed, "symbols", "Cannot copy PDB: " + ec.message());
        }
        const auto copied_hash = assets::hash_file(destination);
        if (!copied_hash || assets::to_string(copied_hash.value()) != entry.digest)
        {
            rollback();
            return fail(project_error_code::corrupt_content, "symbols", "Copied PDB hash mismatch");
        }
        record["files"].push_back({{"path", "symbols/" + entry.filename},
                                   {"binary", entry.binary},
                                   {"size", entry.size},
                                   {"sha256", entry.digest}});
    }

    {
        std::ofstream out(pending / "arc-symbols.json", std::ios::binary | std::ios::trunc);
        out << record.dump(2) << '\n';
        out.flush();
        if (!out)
        {
            out.close();
            rollback();
            return fail(project_error_code::io_failed, "symbols", "Cannot write symbol manifest");
        }
    }
    fs::rename(pending, output, ec);
    if (ec)
    {
        rollback();
        return fail(project_error_code::io_failed, "symbolsRoot", "Cannot publish symbols: " + ec.message());
    }
    return project_status::success();
}

} // namespace arc::project

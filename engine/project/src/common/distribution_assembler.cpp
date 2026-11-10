#include <arc/project/distribution_assembler.h>

#include <arc/assets/assets.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace arc::project
{
namespace
{
namespace fs = std::filesystem;

project_error failure(project_error_code code, std::string field, std::string message)
{
    return {.code = code, .field = std::move(field), .message = std::move(message)};
}

bool within(const fs::path& root, const fs::path& candidate)
{
    auto lhs = root.begin();
    auto rhs = candidate.begin();
    while (lhs != root.end() && rhs != candidate.end())
    {
        if (*lhs++ != *rhs++) return false;
    }
    return lhs == root.end();
}

bool portable_path(std::string_view value)
{
    if (value.empty() || value.front() == '/' || value.find('\\') != value.npos || value.find(':') != value.npos)
        return false;
    std::size_t offset{};
    do
    {
        const auto end = value.find('/', offset);
        const auto part = value.substr(offset, end == value.npos ? end : end - offset);
        if (part.empty() || part == "." || part == ".." || part.back() == '.' || part.back() == ' ') return false;
        for (const unsigned char ch : part)
            if (ch < 32 || ch == 127) return false;
        if (end == value.npos) return true;
        offset = end + 1;
    } while (offset < value.size());
    return false;
}

struct staging_guard
{
    fs::path path;
    ~staging_guard()
    {
        if (!path.empty())
        {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    }
};

} // namespace

distribution_assembly_result assemble_distribution(const distribution_assembly_request& request,
                                                   distribution_finalizer& finalizer)
{
    const auto reject = [](project_error_code code, std::string field, std::string message)
    { return distribution_assembly_result::failure(failure(code, std::move(field), std::move(message))); };

    if (!request.manifest.files.empty())
        return reject(project_error_code::invalid_descriptor, "files",
                      "Assembler requires metadata without a precomputed file inventory");
    if (request.files.empty())
        return reject(project_error_code::invalid_descriptor, "files", "No runtime files selected for assembly");
    if (request.output_root.empty() || !request.output_root.is_absolute())
        return reject(project_error_code::invalid_descriptor, "outputRoot", "Output path must be absolute");

    std::error_code ec;
    const auto output = request.output_root.lexically_normal();
    const auto stage = fs::path(output.string() + ".assemble");
    if (fs::exists(output, ec) || ec || fs::exists(stage, ec) || ec ||
        fs::exists(fs::path(output.string() + ".pending"), ec) || ec)
        return reject(project_error_code::invalid_descriptor, "outputRoot",
                      "Output or reserved staging directory already exists");

    std::set<std::string> destinations;
    for (const auto& file : request.files)
    {
        if (!portable_path(file.destination) || file.destination == "arc-distribution.json")
            return reject(project_error_code::unsafe_path, "files.path",
                          "Unsafe distribution-relative destination: " + file.destination);
        std::string key = file.destination;
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!destinations.insert(std::move(key)).second)
            return reject(project_error_code::invalid_descriptor, "files.path",
                          "Duplicate case-insensitive destination: " + file.destination);

        if (file.source.empty() || !file.source.is_absolute())
            return reject(project_error_code::invalid_descriptor, "files.source",
                          "Every selected input must have an absolute native path");
        const auto source_status = fs::symlink_status(file.source, ec);
        if (ec || !fs::is_regular_file(source_status) || fs::is_symlink(source_status))
            return reject(project_error_code::not_found, "files.source",
                          "Selected input must be a real regular file: " + file.source.string());
        const auto source = fs::canonical(file.source, ec);
        if (ec || within(output, source) || within(stage, source))
            return reject(project_error_code::unsafe_path, "files.source",
                          "Selected source is inaccessible or within distribution output: " + file.source.string());
    }

    fs::create_directories(stage, ec);
    if (ec) return reject(project_error_code::io_failed, "stagingRoot", ec.message());
    staging_guard cleanup{stage};

    auto manifest = request.manifest;
    manifest.files.reserve(request.files.size());
    for (const auto& file : request.files)
    {
        const auto staged = stage / fs::path(file.destination);
        fs::create_directories(staged.parent_path(), ec);
        if (!ec) fs::copy_file(file.source, staged, fs::copy_options::none, ec);
        if (ec)
            return reject(project_error_code::io_failed, "files",
                          "Cannot copy " + file.destination + ": " + ec.message());
        const auto digest = assets::hash_file(staged);
        const auto size = fs::file_size(staged, ec);
        if (ec || !digest)
            return reject(project_error_code::io_failed, "files", "Cannot hash staged file: " + file.destination);
        manifest.files.push_back(
            {.path = file.destination, .kind = file.kind, .size = size, .sha256 = assets::to_string(digest.value())});
    }

    distribution_finalize_request finalize{.workflow = request.workflow,
                                           .manifest = manifest,
                                           .staging_root = stage,
                                           .output_root = output,
                                           .dependencies = request.dependencies};
    const auto result = run_distribution_finalizer(finalize, finalizer);
    if (!result) return distribution_assembly_result::failure(result.error());
    return distribution_assembly_result::success(std::move(manifest));
}

} // namespace arc::project

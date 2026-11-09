#include <arc/project/windows_distribution_finalizer.h>

#include <arc/assets/assets.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

namespace arc::project
{
namespace
{
namespace fs = std::filesystem;

project_status fail(project_error_code code, std::string field, std::string message)
{
    return project_status::failure({.code = code, .field = std::move(field), .message = std::move(message)});
}

bool is_within(const fs::path& parent, const fs::path& child)
{
    auto a = parent.begin();
    auto b = child.begin();
    while (a != parent.end() && b != child.end())
    {
        if (*a != *b) return false;
        ++a;
        ++b;
    }
    return a == parent.end();
}

project_status verify_files(const distribution_finalize_request& request)
{
    std::error_code ec;
    const auto root = fs::canonical(request.staging_root, ec);
    if (ec || !fs::is_directory(root))
        return fail(project_error_code::not_found, "stagingRoot", "Staged distribution directory does not exist");
    if (fs::exists(root / "arc-distribution.json", ec))
        return fail(project_error_code::invalid_descriptor, "stagingRoot",
                    "Staged inventory must not contain a pre-existing distribution manifest");

    for (const auto& file : request.manifest.files)
    {
        const fs::path candidate = root / fs::path(file.path);
        const auto actual = fs::canonical(candidate, ec);
        if (ec || !is_within(root, actual) || !fs::is_regular_file(actual, ec) || ec)
            return fail(project_error_code::unsafe_path, "files.path",
                        "Missing, non-regular or escaping staged file: " + file.path);
        const auto size = fs::file_size(actual, ec);
        if (ec || size != file.size)
            return fail(project_error_code::invalid_descriptor, "files.size",
                        "Staged file size disagrees with inventory: " + file.path);
        const auto digest = assets::hash_file(actual);
        if (!digest || assets::to_string(digest.value()) != file.sha256)
            return fail(project_error_code::corrupt_content, "files.sha256",
                        "Staged file hash disagrees with inventory: " + file.path);
    }
    // The assembler must inventory every staged file. Undeclared content
    // could otherwise leak authoring artifacts or executable dependencies.
    for (const auto& entry : fs::recursive_directory_iterator(root, ec))
    {
        if (ec) return fail(project_error_code::io_failed, "stagingRoot", ec.message());
        // Never follow links (including junctions) in a publishable stage.
        // Otherwise an apparently harmless directory could conceal a
        // dependency on files outside the distribution.
        const auto status = entry.symlink_status(ec);
        if (ec || fs::is_symlink(status))
            return fail(project_error_code::unsafe_path, "stagingRoot",
                        "Staged distribution cannot contain symbolic links or junctions");
        if (fs::is_directory(status)) continue;
        if (!fs::is_regular_file(status))
            return fail(project_error_code::unsafe_path, "stagingRoot", "Staged directory contains a non-regular file");
        const auto relative = fs::relative(entry.path(), root, ec).generic_string();
        if (ec || std::none_of(request.manifest.files.begin(), request.manifest.files.end(),
                               [&](const distribution_file& file) { return file.path == relative; }))
            return fail(project_error_code::invalid_descriptor, "files",
                        "Staged file is missing from the distribution inventory: " + relative);
    }
    return project_status::success();
}

} // namespace

project_status windows_directory_finalizer::validate(const distribution_finalize_request& request) const
{
    if (request.manifest.target.platform != "windows" || request.manifest.target.configuration != "Shipping" ||
        request.manifest.entrypoint.kind != "executable" ||
        (request.manifest.target.renderer != "vulkan" && request.manifest.target.renderer != "direct3d12" &&
         request.manifest.target.renderer != "none"))
        return fail(project_error_code::unsupported_platform, "target",
                    "Windows directory finalizer requires a Shipping Windows executable target");

    std::error_code ec;
    const auto root = fs::canonical(request.staging_root, ec);
    if (ec) return fail(project_error_code::not_found, "stagingRoot", "Staging directory is missing");
    const auto output = fs::absolute(request.output_root, ec).lexically_normal();
    if (ec || is_within(root, output) || is_within(output, root))
        return fail(project_error_code::unsafe_path, "outputRoot",
                    "Distribution output and staging directory must not overlap");
    if (fs::exists(output, ec) || ec)
        return fail(project_error_code::invalid_descriptor, "outputRoot",
                    "Distribution output already exists; explicit replacement is not supported");

    return verify_files(request);
}

project_status windows_directory_finalizer::finalize(const distribution_finalize_request& request)
{
    // Defensive for direct callers as well as run_distribution_finalizer().
    if (const auto checked = validate(request); !checked) return checked;
    const auto serialized = serialize_distribution_manifest(request.manifest);
    if (!serialized) return project_status::failure(serialized.error());

    namespace fs = std::filesystem;
    std::error_code ec;
    const auto output = request.output_root.lexically_normal();
    const auto pending = fs::path(output.string() + ".pending");
    if (fs::exists(pending, ec) || ec)
        return fail(project_error_code::invalid_descriptor, "outputRoot",
                    "Pending distribution directory already exists");

    fs::create_directories(output.parent_path(), ec);
    if (ec) return fail(project_error_code::io_failed, "outputRoot", ec.message());
    fs::create_directories(pending, ec);
    if (ec) return fail(project_error_code::io_failed, "outputRoot", ec.message());
    const auto rollback = [&]
    {
        std::error_code ignored;
        fs::remove_all(pending, ignored);
    };

    for (const auto& file : request.manifest.files)
    {
        const auto target = pending / fs::path(file.path);
        fs::create_directories(target.parent_path(), ec);
        if (!ec) fs::copy_file(request.staging_root / fs::path(file.path), target, fs::copy_options::none, ec);
        if (ec)
        {
            rollback();
            return fail(project_error_code::io_failed, "files", "Cannot stage " + file.path + ": " + ec.message());
        }
    }
    // Reverify destination bytes: copying may fail midway or race mutations.
    // The manifest is only committed after all content is present.
    for (const auto& file : request.manifest.files)
    {
        const auto target = pending / fs::path(file.path);
        const auto digest = assets::hash_file(target);
        if (!digest || assets::to_string(digest.value()) != file.sha256)
        {
            rollback();
            return fail(project_error_code::corrupt_content, "files.sha256",
                        "Copied distribution file failed verification: " + file.path);
        }
    }
    {
        std::ofstream manifest(pending / "arc-distribution.json", std::ios::binary | std::ios::trunc);
        manifest << serialized.value();
        manifest.flush();
        if (!manifest.good())
        {
            rollback();
            return fail(project_error_code::io_failed, "manifest", "Cannot write distribution manifest");
        }
    }

    fs::rename(pending, output, ec);
    if (ec)
    {
        rollback();
        return fail(project_error_code::io_failed, "outputRoot",
                    "Cannot publish completed Windows distribution: " + ec.message());
    }
    return project_status::success();
}

} // namespace arc::project

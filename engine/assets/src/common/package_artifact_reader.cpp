#include <arc/assets/package_artifact_reader.h>

#include <arc/assets/runtime_content.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <utility>

namespace arc::assets
{
namespace
{

core::result<std::vector<std::byte>, asset_error> read_failure(asset_error_code code, asset_guid guid,
                                                               const std::filesystem::path& path, std::string message)
{
    return core::result<std::vector<std::byte>, asset_error>::failure(
        {.code = code, .guid = guid, .path = path, .message = std::move(message)});
}

} // namespace

package_artifact_reader::~package_artifact_reader()
{
    release_virtual_mount();
}

void package_artifact_reader::release_virtual_mount() noexcept
{
    if (virtual_files_ && virtual_mount_ != 0) static_cast<void>(virtual_files_->unmount(virtual_mount_));
    virtual_files_ = nullptr;
    virtual_mount_ = 0;
    virtual_provider_.reset();
    artifact_root_ = {};
}

asset_status package_artifact_reader::mount(const std::filesystem::path& manifest_path)
{
    release_virtual_mount();
    auto loaded = load_cook_manifest(manifest_path);
    if (!loaded) return asset_status::failure(std::move(loaded).error());

    auto staged = std::move(loaded).value();
    for (const auto& artifact : staged.artifacts)
    {
        if (artifact.name.empty() || artifact.stored_size == 0u || artifact.compressed)
        {
            return asset_status::failure(
                {.code = asset_error_code::invalid_metadata,
                 .guid = artifact.asset,
                 .path = manifest_path,
                 .message = artifact.compressed ? "Named package reader does not support compressed package records"
                                                : "Cooked artifact address or stored size is invalid"});
        }

        const auto package = manifest_path.parent_path() / artifact.chunk;
        std::error_code filesystem_error;
        const auto package_size = std::filesystem::file_size(package, filesystem_error);
        if (filesystem_error || artifact.offset > package_size || artifact.stored_size > package_size - artifact.offset)
        {
            return asset_status::failure({.code = asset_error_code::invalid_metadata,
                                          .guid = artifact.asset,
                                          .path = package,
                                          .message = "Cooked package artifact range is missing or invalid"});
        }
    }

    root_ = manifest_path.parent_path();
    manifest_ = std::move(staged);
    bytes_read_.store(0u, std::memory_order_relaxed);
    return asset_status::success();
}

asset_status package_artifact_reader::mount(io::virtual_file_system& files, const io::virtual_path& manifest_path,
                                            io::virtual_path artifact_root, std::int32_t priority)
{
    release_virtual_mount();
    auto provider = package_artifact_provider::create(files, manifest_path);
    if (!provider) return asset_status::failure(std::move(provider).error());
    auto mounted = files.mount({.root = artifact_root,
                                .priority = priority,
                                .provider = provider.value(),
                                .debug_name = "package artifact reader"});
    if (!mounted)
        return asset_status::failure({.code = asset_error_code::invalid_request,
                                      .message = "Could not mount package provider: " + mounted.error().message});
    root_.clear();
    manifest_ = provider.value()->manifest();
    virtual_files_ = &files;
    artifact_root_ = std::move(artifact_root);
    virtual_mount_ = mounted.value();
    virtual_provider_ = std::move(provider).value();
    bytes_read_.store(0u, std::memory_order_relaxed);
    return asset_status::success();
}

const cook_manifest_artifact* package_artifact_reader::find(const cooked_artifact_address& address) const noexcept
{
    if (!address.valid()) return nullptr;
    const auto found = std::find_if(manifest_.artifacts.begin(), manifest_.artifacts.end(),
                                    [&](const auto& artifact) {
                                        return artifact.asset == address.asset && artifact.schema == address.schema &&
                                               artifact.name == address.name;
                                    });
    return found == manifest_.artifacts.end() ? nullptr : &*found;
}

std::optional<cooked_artifact_location>
package_artifact_reader::locate(const cooked_artifact_address& address) const noexcept
{
    const auto* artifact = find(address);
    if (!artifact) return std::nullopt;
    if (virtual_files_)
    {
        auto logical = cooked_artifact_virtual_path(address, artifact_root_.string());
        if (!logical) return std::nullopt;
        auto resolved = virtual_files_->resolve(logical.value());
        if (!resolved) return std::nullopt;
        return cooked_artifact_location{.file = std::move(resolved).value(),
                                        .offset = 0,
                                        .size = artifact->stored_size};
    }
    return cooked_artifact_location{
        .path = root_ / artifact->chunk, .offset = artifact->offset, .size = artifact->stored_size};
}

core::result<std::vector<std::byte>, asset_error>
package_artifact_reader::read(const cooked_artifact_address& address) const
{
    const auto* artifact = find(address);
    if (!artifact)
        return read_failure(asset_error_code::not_found, address.asset, {},
                            "Named cooked artifact is not present in the mounted package");

    auto bytes = read_range(address, 0u, artifact->stored_size);
    if (!bytes) return bytes;
    if (hash_bytes(bytes.value()) != artifact->hash)
        return read_failure(asset_error_code::invalid_metadata, address.asset, root_ / artifact->chunk,
                            "Cooked package artifact failed content verification");
    return bytes;
}

core::result<std::vector<std::byte>, asset_error>
package_artifact_reader::read_range(const cooked_artifact_address& address, std::uint64_t relative_offset,
                                    std::uint64_t size) const
{
    const auto location = locate(address);
    if (!location)
        return read_failure(asset_error_code::not_found, address.asset, {},
                            "Named cooked artifact is not present in the mounted package");
    if (relative_offset > location->size || size > location->size - relative_offset ||
        size > std::numeric_limits<std::size_t>::max())
        return read_failure(asset_error_code::invalid_metadata, address.asset, location->path,
                            "Requested cooked artifact range is out of bounds");

    if (location->file.valid())
    {
        auto loaded = virtual_files_->read_range(location->file, location->offset + relative_offset,
                                                 static_cast<std::size_t>(size)).get();
        if (!loaded)
            return read_failure(loaded.error().code == io::file_error_code::stale_handle
                                    ? asset_error_code::invalid_request
                                    : asset_error_code::io_failed,
                                address.asset, {}, loaded.error().message);
        bytes_read_.fetch_add(size, std::memory_order_relaxed);
        return core::result<std::vector<std::byte>, asset_error>::success(std::move(loaded).value());
    }

    std::ifstream stream(location->path, std::ios::binary);
    if (!stream)
        return read_failure(asset_error_code::io_failed, address.asset, location->path,
                            "Could not open cooked package chunk");
    stream.seekg(static_cast<std::streamoff>(location->offset + relative_offset));
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    if (!bytes.empty()) stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream)
        return read_failure(asset_error_code::io_failed, address.asset, location->path,
                            "Could not read cooked artifact range");

    bytes_read_.fetch_add(size, std::memory_order_relaxed);
    return core::result<std::vector<std::byte>, asset_error>::success(std::move(bytes));
}

const cook_manifest& package_artifact_reader::manifest() const noexcept
{
    return manifest_;
}

std::uint64_t package_artifact_reader::bytes_read() const noexcept
{
    return bytes_read_.load(std::memory_order_relaxed);
}

void package_artifact_reader::reset_statistics() noexcept
{
    bytes_read_.store(0u, std::memory_order_relaxed);
}

} // namespace arc::assets

#include <arc/assets/cook.h>

#include <algorithm>
#include <fstream>
#include <map>

namespace arc::assets
{
namespace
{

constexpr std::uint64_t package_alignment = 4096;
constexpr std::array<char, 8> package_magic{ 'A', 'R', 'C', 'P', 'A', 'K', '1', '\0' };

std::uint64_t align_up(std::uint64_t value) noexcept
{
    return (value + package_alignment - 1) & ~(package_alignment - 1);
}

std::optional<std::vector<std::byte>> read_file(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return std::nullopt;
    const auto end = stream.tellg();
    if (end < 0)
        return std::nullopt;
    std::vector<std::byte> result(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!result.empty())
        stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size()));
    return stream ? std::optional(std::move(result)) : std::nullopt;
}

}

package_build_result build_asset_packages(
    cook_manifest manifest,
    derived_data_cache& cache,
    const std::filesystem::path& output)
{
    package_build_result result;
    std::error_code filesystem_error;
    std::filesystem::create_directories(output, filesystem_error);
    if (filesystem_error)
    {
        result.error = "Could not create package output directory";
        return result;
    }

    std::map<std::string, std::vector<std::size_t>> groups;
    for (std::size_t index = 0; index < manifest.artifacts.size(); ++index)
        groups[manifest.artifacts[index].chunk].push_back(index);

    for (auto& [group, indices] : groups)
    {
        std::sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) {
            const auto& left = manifest.artifacts[lhs];
            const auto& right = manifest.artifacts[rhs];
            if (left.asset != right.asset) return left.asset < right.asset;
            if (left.schema != right.schema) return left.schema < right.schema;
            return left.hash < right.hash;
        });
        std::vector<std::byte> package(
            reinterpret_cast<const std::byte*>(package_magic.data()),
            reinterpret_cast<const std::byte*>(package_magic.data() + package_magic.size()));
        for (const auto index : indices)
        {
            auto& artifact = manifest.artifacts[index];
            cache_error error;
            auto blob = cache.get_blob(artifact.hash, error);
            if (!blob)
            {
                result.error = "Could not read artifact " + to_string(artifact.hash) + ": " + error.message;
                return result;
            }
            package.resize(static_cast<std::size_t>(align_up(package.size())));
            artifact.offset = package.size();
            artifact.stored_size = blob->bytes.size();
            artifact.compressed = false;
            result.source_bytes += blob->bytes.size();
            package.insert(package.end(), blob->bytes.begin(), blob->bytes.end());
        }
        const auto package_hash = hash_bytes(package);
        const auto filename = group + "-" + to_string(package_hash) + ".arcpak";
        const auto destination = output / filename;
        const auto existing = read_file(destination);
        if (!existing || hash_bytes(*existing) != package_hash)
        {
            if (existing)
            {
                const auto quarantine = destination.string() + ".corrupt-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
                std::filesystem::rename(destination, quarantine, filesystem_error);
                if (filesystem_error)
                {
                    result.error = "Could not quarantine corrupt package chunk: " +
                        filesystem_error.message();
                    return result;
                }
            }
            const auto temporary = destination.string() + ".tmp";
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            if (!stream)
            {
                result.error = "Could not create package chunk";
                return result;
            }
            stream.write(reinterpret_cast<const char*>(package.data()),
                static_cast<std::streamsize>(package.size()));
            stream.close();
            std::filesystem::rename(temporary, destination, filesystem_error);
            if (filesystem_error)
            {
                result.error = "Could not publish package chunk: " + filesystem_error.message();
                return result;
            }
        }
        for (const auto index : indices)
            manifest.artifacts[index].chunk = filename;
        result.stored_bytes += package.size();
        result.chunks.push_back(destination);
    }

    result.manifest_path = output / (manifest.target.name + ".arccookmanifest");
    auto saved = save_cook_manifest(result.manifest_path, manifest);
    if (!saved)
    {
        result.error = saved.error().message;
        return result;
    }
    return result;
}

struct asset_package_mount::implementation
{
    std::filesystem::path root;
    cook_manifest manifest;
};

asset_package_mount::asset_package_mount()
    : implementation_(std::make_unique<implementation>())
{
}

asset_package_mount::~asset_package_mount() = default;
asset_package_mount::asset_package_mount(asset_package_mount&&) noexcept = default;
asset_package_mount& asset_package_mount::operator=(asset_package_mount&&) noexcept = default;

asset_status asset_package_mount::mount(const std::filesystem::path& manifest_path)
{
    auto loaded = load_cook_manifest(manifest_path);
    if (!loaded)
        return asset_status::failure(std::move(loaded).error());
    cook_manifest staged = std::move(loaded).value();
    for (const auto& artifact : staged.artifacts)
    {
        const auto package = manifest_path.parent_path() / artifact.chunk;
        std::error_code filesystem_error;
        if (!std::filesystem::exists(package, filesystem_error) ||
            artifact.stored_size == 0 || artifact.compressed)
        {
            return asset_status::failure({
                .code = asset_error_code::invalid_metadata,
                .guid = artifact.asset,
                .path = package,
                .message = artifact.compressed
                    ? "This runtime does not support compressed ARC package records"
                    : "Cook package chunk is missing or invalid"
            });
        }
    }
    implementation_->root = manifest_path.parent_path();
    implementation_->manifest = std::move(staged);
    return asset_status::success();
}

core::result<std::vector<std::byte>, asset_error> asset_package_mount::read(
    asset_guid asset,
    artifact_schema_id schema) const
{
    const auto found = std::find_if(
        implementation_->manifest.artifacts.begin(),
        implementation_->manifest.artifacts.end(),
        [&](const auto& value) { return value.asset == asset && value.schema == schema; });
    if (found == implementation_->manifest.artifacts.end())
    {
        return core::result<std::vector<std::byte>, asset_error>::failure({
            .code = asset_error_code::not_found,
            .guid = asset,
            .message = "Cooked artifact is not present in the mounted package"
        });
    }
    std::ifstream stream(implementation_->root / found->chunk, std::ios::binary);
    if (!stream)
    {
        return core::result<std::vector<std::byte>, asset_error>::failure({
            .code = asset_error_code::io_failed,
            .guid = asset,
            .path = implementation_->root / found->chunk,
            .message = "Could not open cooked package chunk"
        });
    }
    stream.seekg(static_cast<std::streamoff>(found->offset));
    std::vector<std::byte> bytes(static_cast<std::size_t>(found->stored_size));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream || hash_bytes(bytes) != found->hash)
    {
        return core::result<std::vector<std::byte>, asset_error>::failure({
            .code = asset_error_code::invalid_metadata,
            .guid = asset,
            .path = implementation_->root / found->chunk,
            .message = "Cooked package artifact failed content verification"
        });
    }
    return core::result<std::vector<std::byte>, asset_error>::success(std::move(bytes));
}

jobs::job_future<io::file_result<io::file_buffer>> asset_package_mount::read_async(
    asset_guid asset,
    artifact_schema_id schema,
    io::async_file_service& files,
    jobs::cancellation_token cancellation) const
{
    const auto found = std::find_if(
        implementation_->manifest.artifacts.begin(),
        implementation_->manifest.artifacts.end(),
        [&](const auto& value) { return value.asset == asset && value.schema == schema; });
    if (found == implementation_->manifest.artifacts.end())
    {
        return files.scheduler().submit_future({
            .name = "assets.package.missing",
            .affinity = jobs::job_affinity::io_thread,
            .cancellation = cancellation
        }, [] {
            return io::file_result<io::file_buffer>::failure({
                .code = io::file_error_code::not_found,
                .message = "Cooked artifact is not present in the mounted package"
            });
        });
    }

    const auto path = implementation_->root / found->chunk;
    const auto expected_hash = found->hash;
    auto range = files.read_range(
        path,
        found->offset,
        static_cast<std::size_t>(found->stored_size),
        cancellation);
    jobs::job_descriptor descriptor{
        .name = "assets.package.verify",
        .affinity = jobs::job_affinity::io_thread,
        .dependencies = { range.handle() },
        .cancellation = cancellation
    };
    return files.scheduler().submit_future(std::move(descriptor),
        [range = std::move(range), path, expected_hash]() mutable {
            auto result = range.get();
            if (!result)
                return result;
            if (hash_bytes(result.value()) != expected_hash)
            {
                return io::file_result<io::file_buffer>::failure({
                    .code = io::file_error_code::read_failed,
                    .path = path,
                    .message = "Cooked package artifact failed content verification"
                });
            }
            return result;
        });
}

const cook_manifest& asset_package_mount::manifest() const noexcept
{
    return implementation_->manifest;
}

} // namespace arc::assets

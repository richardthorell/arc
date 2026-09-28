#include <arc/assets/runtime_content.h>

#include <algorithm>
#include <charconv>
#include <limits>
#include <unordered_map>

namespace arc::assets
{
namespace
{

asset_error asset_failure(asset_error_code code, std::string message, asset_guid guid = {})
{
    return {.code = code, .guid = guid, .message = std::move(message)};
}

bool artifact_name_character(unsigned char value) noexcept
{
    return std::isalnum(value) || value == '-' || value == '_' || value == '.';
}

std::string encode_component(std::string_view value)
{
    constexpr char digits[] = "0123456789ABCDEF";
    std::string result;
    for (const auto character : value)
    {
        const auto byte = static_cast<unsigned char>(character);
        if (artifact_name_character(byte))
            result.push_back(character);
        else
        {
            result.push_back('%');
            result.push_back(digits[byte >> 4u]);
            result.push_back(digits[byte & 0x0fu]);
        }
    }
    return result;
}

int hex_value(char value) noexcept
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

std::optional<std::string> decode_component(std::string_view value)
{
    std::string result;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        if (value[index] != '%')
        {
            result.push_back(value[index]);
            continue;
        }
        if (index + 2 >= value.size()) return std::nullopt;
        const auto high = hex_value(value[index + 1]);
        const auto low = hex_value(value[index + 2]);
        if (high < 0 || low < 0) return std::nullopt;
        result.push_back(static_cast<char>((high << 4) | low));
        index += 2;
    }
    return result;
}

std::optional<artifact_schema_id> parse_schema(std::string_view value)
{
    if (value.size() != 32) return std::nullopt;
    artifact_schema_id result;
    const auto high = std::from_chars(value.data(), value.data() + 16, result.high, 16);
    const auto low = std::from_chars(value.data() + 16, value.data() + 32, result.low, 16);
    if (high.ec != std::errc{} || high.ptr != value.data() + 16 || low.ec != std::errc{} ||
        low.ptr != value.data() + 32 || !result.valid())
        return std::nullopt;
    return result;
}

std::uint64_t artifact_generation(const cook_manifest_artifact& artifact,
                                  const io::resolved_virtual_file& chunk) noexcept
{
    std::uint64_t hash_prefix{};
    for (std::size_t index = 0; index < sizeof(hash_prefix); ++index)
        hash_prefix |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(artifact.hash.bytes[index]))
                       << (index * 8u);
    auto result = chunk.content_generation() ^ hash_prefix ^ artifact.offset ^ artifact.stored_size;
    return result == 0 ? 1 : result;
}

io::file_error io_failure(io::file_error_code code, std::string_view logical_path, std::string message)
{
    return {.code = code, .message = std::move(message), .logical_path = std::string(logical_path)};
}

} // namespace

io::file_result<io::virtual_path> cooked_artifact_virtual_path(const cooked_artifact_address& address,
                                                               std::string_view mount)
{
    if (!address.valid())
        return io::file_result<io::virtual_path>::failure(
            io_failure(io::file_error_code::invalid_request, mount, "Cooked artifact address is invalid"));
    auto root = io::virtual_path::parse(mount);
    if (!root) return root;
    std::string relative(root.value().relative_path());
    if (!relative.empty()) relative.push_back('/');
    relative += to_string(address.asset);
    relative.push_back('/');
    relative += to_string(address.schema);
    relative.push_back('/');
    relative += encode_component(address.name);
    return io::virtual_path::from_parts(root.value().scheme(), root.value().authority(), relative);
}

std::optional<cooked_artifact_address> parse_cooked_artifact_virtual_path(const io::virtual_path& path)
{
    const auto relative = path.relative_path();
    const auto first = relative.find('/');
    const auto second = first == std::string_view::npos ? first : relative.find('/', first + 1);
    if (first == std::string_view::npos || second == std::string_view::npos ||
        relative.find('/', second + 1) != std::string_view::npos)
        return std::nullopt;
    const auto asset = parse_asset_guid(relative.substr(0, first));
    const auto schema = parse_schema(relative.substr(first + 1, second - first - 1));
    const auto name = decode_component(relative.substr(second + 1));
    if (!asset || !schema || !name || name->empty()) return std::nullopt;
    return cooked_artifact_address{.asset = *asset, .schema = *schema, .name = std::move(*name)};
}

struct package_artifact_provider::implementation
{
    struct record
    {
        cook_manifest_artifact artifact;
        io::resolved_virtual_file chunk;
        std::uint64_t generation{};
    };

    io::virtual_file_system* files{};
    cook_manifest manifest;
    std::unordered_map<std::string, record> records;
};

package_artifact_provider::package_artifact_provider(std::unique_ptr<implementation> implementation)
    : implementation_(std::move(implementation))
{
}

package_artifact_provider::~package_artifact_provider() = default;

package_artifact_provider::create_result package_artifact_provider::create(io::virtual_file_system& files,
                                                                           const io::virtual_path& manifest_file)
{
    auto resolved_manifest = files.resolve(manifest_file);
    if (!resolved_manifest)
        return create_result::failure(
            asset_failure(asset_error_code::not_found, "Cook manifest is unavailable through the VFS"));
    auto bytes = files.read_all(resolved_manifest.value()).get();
    if (!bytes)
        return create_result::failure(
            asset_failure(asset_error_code::io_failed, "Cook manifest could not be read through the VFS"));
    auto parsed = parse_cook_manifest(bytes.value(), manifest_file.string());
    if (!parsed) return create_result::failure(std::move(parsed).error());

    auto state = std::make_unique<implementation>();
    state->files = &files;
    state->manifest = std::move(parsed).value();
    const auto manifest_relative = manifest_file.relative_path();
    const auto separator = manifest_relative.find_last_of('/');
    const auto directory =
        separator == std::string_view::npos ? std::string_view{} : manifest_relative.substr(0, separator);
    for (const auto& artifact : state->manifest.artifacts)
    {
        if (artifact.name.empty() || artifact.stored_size == 0 || artifact.compressed)
            return create_result::failure(
                asset_failure(asset_error_code::invalid_metadata,
                              artifact.compressed ? "Package VFS provider does not support compressed package records"
                                                  : "Cooked package artifact record is incomplete",
                              artifact.asset));
        std::string chunk_relative(directory);
        if (!chunk_relative.empty()) chunk_relative.push_back('/');
        chunk_relative += artifact.chunk;
        auto chunk_path =
            io::virtual_path::from_parts(manifest_file.scheme(), manifest_file.authority(), chunk_relative);
        if (!chunk_path)
            return create_result::failure(asset_failure(asset_error_code::invalid_metadata,
                                                        "Cooked package chunk path is invalid", artifact.asset));
        auto chunk = files.resolve(chunk_path.value());
        if (!chunk || artifact.offset > chunk.value().size() ||
            artifact.stored_size > chunk.value().size() - artifact.offset)
            return create_result::failure(asset_failure(asset_error_code::invalid_metadata,
                                                        "Cooked package artifact range is missing or invalid",
                                                        artifact.asset));
        auto logical =
            cooked_artifact_virtual_path({.asset = artifact.asset, .schema = artifact.schema, .name = artifact.name});
        if (!logical)
            return create_result::failure(asset_failure(asset_error_code::invalid_metadata,
                                                        "Cooked artifact logical path is invalid", artifact.asset));
        const auto key = std::string(logical.value().relative_path());
        if (state->records.contains(key))
            return create_result::failure(asset_failure(asset_error_code::invalid_metadata,
                                                        "Cooked package contains a duplicate artifact address",
                                                        artifact.asset));
        const auto generation = artifact_generation(artifact, chunk.value());
        state->records.emplace(
            key,
            implementation::record{.artifact = artifact, .chunk = std::move(chunk.value()), .generation = generation});
    }
    return create_result::success(
        std::shared_ptr<package_artifact_provider>(new package_artifact_provider(std::move(state))));
}

io::provider_capabilities package_artifact_provider::capabilities() const noexcept
{
    return io::provider_capability::full_read | io::provider_capability::range_read |
           io::provider_capability::metadata | io::provider_capability::enumeration;
}

std::uint64_t package_artifact_provider::provider_generation() const noexcept
{
    return 1;
}

io::provider_lookup_result package_artifact_provider::resolve(std::string_view relative_path)
{
    const auto found = implementation_->records.find(std::string(relative_path));
    if (found == implementation_->records.end()) return {};
    return {.status = io::provider_lookup_status::found,
            .file = {.key = found->first,
                     .size = found->second.artifact.stored_size,
                     .content_generation = found->second.generation}};
}

jobs::job_future<io::file_result<io::file_buffer>>
package_artifact_provider::read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
                                      jobs::cancellation_token cancellation)
{
    const auto found = implementation_->records.find(file.key);
    if (found == implementation_->records.end() || found->second.generation != file.content_generation)
        return implementation_->files->read_range({}, 0, 0, cancellation);
    if (offset > found->second.artifact.stored_size || bytes > found->second.artifact.stored_size - offset)
        return implementation_->files->read_range(found->second.chunk, found->second.chunk.size() + 1u, bytes,
                                                  cancellation);
    return implementation_->files->read_range(found->second.chunk, found->second.artifact.offset + offset, bytes,
                                              cancellation);
}

io::file_result<std::vector<io::provider_directory_entry>>
package_artifact_provider::enumerate(std::string_view relative_prefix)
{
    std::vector<io::provider_directory_entry> result;
    for (const auto& [path, record] : implementation_->records)
    {
        if (!relative_prefix.empty() && !path.starts_with(relative_prefix)) continue;
        result.push_back(
            {.relative_path = path, .size = record.artifact.stored_size, .content_generation = record.generation});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.relative_path < rhs.relative_path; });
    return io::file_result<std::vector<io::provider_directory_entry>>::success(std::move(result));
}

const cook_manifest& package_artifact_provider::manifest() const noexcept
{
    return implementation_->manifest;
}

cooked_asset_catalog::cooked_asset_catalog(io::virtual_file_system& files, io::virtual_path artifact_root)
    : files_(&files), artifact_root_(std::move(artifact_root))
{
}

cooked_asset_catalog::~cooked_asset_catalog()
{
    for (const auto mount : mounts_)
        static_cast<void>(files_->unmount(mount));
}

asset_status cooked_asset_catalog::mount_package(const io::virtual_path& manifest_file, std::int32_t priority)
{
    auto provider = package_artifact_provider::create(*files_, manifest_file);
    if (!provider) return asset_status::failure(std::move(provider).error());
    return mount_provider(std::move(provider).value(), priority, "cooked package");
}

asset_status cooked_asset_catalog::mount_provider(std::shared_ptr<io::virtual_file_provider> provider,
                                                  std::int32_t priority, std::string debug_name)
{
    auto mounted = files_->mount({.root = artifact_root_,
                                  .priority = priority,
                                  .provider = std::move(provider),
                                  .debug_name = std::move(debug_name)});
    if (!mounted)
        return asset_status::failure(asset_failure(asset_error_code::invalid_request, mounted.error().message));
    mounts_.push_back(mounted.value());
    return asset_status::success();
}

io::file_result<io::resolved_virtual_file> cooked_asset_catalog::resolve(const cooked_artifact_address& address)
{
    ++telemetry_.resolves;
    auto path = cooked_artifact_virtual_path(address, artifact_root_.string());
    if (!path)
    {
        ++telemetry_.resolve_misses;
        return io::file_result<io::resolved_virtual_file>::failure(path.error());
    }
    auto result = files_->resolve(path.value());
    if (!result) ++telemetry_.resolve_misses;
    return result;
}

cooked_artifact_change_batch cooked_asset_catalog::poll_changes()
{
    static_cast<void>(files_->poll_changes());
    const auto changes = files_->events_since(last_vfs_sequence_);
    cooked_artifact_change_batch result;
    result.catalog_reset = changes.history_overflow;
    for (const auto& change : changes.events)
    {
        last_vfs_sequence_ = std::max(last_vfs_sequence_, change.sequence);
        if (change.path.scheme() != artifact_root_.scheme() || change.path.authority() != artifact_root_.authority())
            continue;
        if (change.kind == io::virtual_change_kind::reset || change.kind == io::virtual_change_kind::mount_changed)
        {
            result.catalog_reset = true;
            continue;
        }
        auto address = parse_cooked_artifact_virtual_path(change.path);
        if (!address) continue;
        auto kind = cooked_artifact_change_kind::updated;
        if (change.kind == io::virtual_change_kind::added) kind = cooked_artifact_change_kind::added;
        if (change.kind == io::virtual_change_kind::removed) kind = cooked_artifact_change_kind::removed;
        cooked_artifact_change_event event{.sequence = change.sequence,
                                           .kind = kind,
                                           .address = std::move(*address),
                                           .content_generation = change.content_generation};
        if (result.first_sequence == 0) result.first_sequence = event.sequence;
        result.last_sequence = event.sequence;
        result.events.push_back(std::move(event));
    }
    if (result.catalog_reset)
    {
        ++telemetry_.resets;
        cooked_artifact_change_event event{.sequence = changes.last_sequence,
                                           .kind = cooked_artifact_change_kind::catalog_reset};
        result.events.insert(result.events.begin(), std::move(event));
    }
    if (!result.events.empty()) ++telemetry_.change_batches;
    return result;
}

cooked_asset_catalog_telemetry cooked_asset_catalog::telemetry() const noexcept
{
    return telemetry_;
}

} // namespace arc::assets

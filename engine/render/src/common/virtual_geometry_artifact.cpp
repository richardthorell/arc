#include <arc/render/virtual_geometry_artifact.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <type_traits>

namespace arc::render
{
namespace
{

constexpr std::array<std::byte, 8> artifact_magic{static_cast<std::byte>('A'), static_cast<std::byte>('R'),
                                                  static_cast<std::byte>('C'), static_cast<std::byte>('V'),
                                                  static_cast<std::byte>('G'), static_cast<std::byte>('0'),
                                                  static_cast<std::byte>('0'), static_cast<std::byte>('3')};
constexpr std::uint32_t header_bytes = 48;

template <typename T, bool = std::is_enum_v<T>> struct stored_type_for
{
    using type = T;
};

template <typename T> struct stored_type_for<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <typename T> using stored_type_for_t = typename stored_type_for<T>::type;

std::uint64_t hash_bytes(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto value : bytes)
    {
        hash ^= std::to_integer<std::uint8_t>(value);
        hash *= 1099511628211ull;
    }
    return hash;
}

class byte_writer
{
public:
    template <typename T> void value(T input)
    {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>);
        using stored_type = stored_type_for_t<T>;
        stored_type stored = static_cast<stored_type>(input);
        if constexpr (std::is_floating_point_v<stored_type>)
        {
            using bits_type = std::conditional_t<sizeof(stored_type) == 4, std::uint32_t, std::uint64_t>;
            value(std::bit_cast<bits_type>(stored));
        }
        else
        {
            using unsigned_type = std::make_unsigned_t<stored_type>;
            auto bits = static_cast<unsigned_type>(stored);
            for (std::size_t index = 0; index < sizeof(stored_type); ++index)
                bytes_.push_back(static_cast<std::byte>((bits >> (index * 8u)) & 0xffu));
        }
    }

    void raw(std::span<const std::byte> bytes)
    {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    void string(std::string_view value)
    {
        this->value(static_cast<std::uint32_t>(value.size()));
        raw(std::as_bytes(std::span(value.data(), value.size())));
    }

    void vector3(const math::vector3f& value)
    {
        this->value(value[0]);
        this->value(value[1]);
        this->value(value[2]);
    }

    void align(std::uint32_t alignment)
    {
        const auto remainder = bytes_.size() % alignment;
        if (remainder != 0) bytes_.resize(bytes_.size() + alignment - remainder);
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return bytes_.size();
    }

    [[nodiscard]] std::vector<std::byte>& bytes() noexcept
    {
        return bytes_;
    }

private:
    std::vector<std::byte> bytes_;
};

class byte_reader
{
public:
    explicit byte_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <typename T> bool value(T& output)
    {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>);
        if constexpr (std::is_floating_point_v<T>)
        {
            using bits_type = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
            bits_type bits{};
            if (!value(bits)) return false;
            output = std::bit_cast<T>(bits);
            return true;
        }
        else
        {
            using stored_type = stored_type_for_t<T>;
            using unsigned_type = std::make_unsigned_t<stored_type>;
            if (remaining() < sizeof(stored_type)) return false;
            unsigned_type bits{};
            for (std::size_t index = 0; index < sizeof(stored_type); ++index)
                bits |= static_cast<unsigned_type>(std::to_integer<std::uint8_t>(bytes_[cursor_ + index]))
                        << (index * 8u);
            cursor_ += sizeof(stored_type);
            if constexpr (std::is_enum_v<T>)
                output = static_cast<T>(static_cast<stored_type>(bits));
            else
                output = static_cast<T>(bits);
            return true;
        }
    }

    bool raw(std::span<std::byte> output)
    {
        if (remaining() < output.size()) return false;
        std::memcpy(output.data(), bytes_.data() + cursor_, output.size());
        cursor_ += output.size();
        return true;
    }

    bool string(std::string& output)
    {
        std::uint32_t size{};
        if (!value(size) || remaining() < size) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_), size);
        cursor_ += size;
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept
    {
        return bytes_.size() - cursor_;
    }

private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

void write_cluster(byte_writer& writer, const virtual_mesh_cluster& cluster)
{
    writer.value(cluster.first_index);
    writer.value(cluster.index_count);
    writer.value(cluster.first_triangle);
    writer.value(cluster.triangle_count);
    writer.value(cluster.first_vertex);
    writer.value(cluster.vertex_count);
    writer.value(static_cast<std::uint64_t>(cluster.material_index));
    writer.vector3(cluster.bounds_min);
    writer.vector3(cluster.bounds_max);
    writer.vector3(cluster.sphere_center);
    writer.value(cluster.sphere_radius);
    writer.vector3(cluster.cone_axis);
    writer.value(cluster.cone_cutoff);
    writer.value(cluster.geometric_error);
    writer.value(cluster.hierarchy_node);
    writer.value(cluster.page_index);
    writer.value(cluster.page_byte_offset);
    writer.value(cluster.hierarchy_level);
    writer.value(cluster.flags);
}

void write_node(byte_writer& writer, const virtual_mesh_lod_node& node)
{
    writer.value(node.first_cluster);
    writer.value(node.cluster_count);
    writer.value(node.first_child);
    writer.value(node.child_count);
    writer.value(node.parent);
    writer.value(node.page_index);
    writer.value(node.error);
    writer.vector3(node.bounds_min);
    writer.vector3(node.bounds_max);
    writer.vector3(node.sphere_center);
    writer.value(node.sphere_radius);
    writer.vector3(node.cone_axis);
    writer.value(node.cone_cutoff);
    writer.value(node.level);
    writer.value(node.flags);
}

virtual_geometry_artifact_error failure(virtual_geometry_artifact_error_code code, std::string message)
{
    return {.code = code, .message = std::move(message)};
}

} // namespace

virtual_geometry_artifact_bytes_result
encode_virtual_geometry_artifact(std::span<const virtual_geometry_artifact_source> meshes,
                                 std::uint64_t conventional_artifact_hash)
{
    if (meshes.size() > std::numeric_limits<std::uint32_t>::max())
        return virtual_geometry_artifact_bytes_result::failure(
            failure(virtual_geometry_artifact_error_code::size_overflow, "too many virtual-geometry meshes"));
    for (const auto& mesh : meshes)
        if (!mesh.geometry)
            return virtual_geometry_artifact_bytes_result::failure(
                failure(virtual_geometry_artifact_error_code::invalid_data, "mesh geometry is null"));

    struct encoded_mesh
    {
        std::string_view name;
        std::uint64_t material_index{};
        byte_writer metadata;
        const virtual_mesh_data* geometry{};
        std::vector<std::uint64_t> page_offsets;
    };
    std::vector<encoded_mesh> encoded;
    encoded.reserve(meshes.size());
    for (const auto& source : meshes)
    {
        encoded_mesh mesh{.name = source.name, .material_index = source.material_index, .geometry = source.geometry};
        mesh.metadata.value(static_cast<std::uint32_t>(source.geometry->clusters.size()));
        for (const auto& cluster : source.geometry->clusters) write_cluster(mesh.metadata, cluster);
        mesh.metadata.value(static_cast<std::uint32_t>(source.geometry->lod_nodes.size()));
        for (const auto& node : source.geometry->lod_nodes) write_node(mesh.metadata, node);
        mesh.metadata.value(static_cast<std::uint32_t>(source.geometry->hierarchy_children.size()));
        for (const auto child : source.geometry->hierarchy_children) mesh.metadata.value(child);
        mesh.metadata.value(static_cast<std::uint32_t>(source.geometry->root_nodes.size()));
        for (const auto root : source.geometry->root_nodes) mesh.metadata.value(root);
        encoded.push_back(std::move(mesh));
    }

    byte_writer output;
    output.raw(artifact_magic);
    output.value(virtual_geometry_artifact_schema_version);
    output.value(header_bytes);
    output.value(conventional_artifact_hash);
    output.value(static_cast<std::uint32_t>(encoded.size()));
    output.value(std::uint32_t{});
    output.value(std::uint64_t{}); // table offset, patched below
    output.value(std::uint64_t{}); // artifact size, patched below

    const auto table_offset = output.size();
    std::size_t table_size{};
    for (const auto& mesh : encoded)
        table_size += sizeof(std::uint32_t) + mesh.name.size() + sizeof(std::uint64_t) * 3u + sizeof(std::uint32_t) * 2u +
                      mesh.geometry->pages.size() * (sizeof(std::uint64_t) * 2u + sizeof(std::uint32_t) * 3u);
    output.bytes().resize(output.size() + table_size);

    std::vector<std::uint64_t> metadata_offsets;
    metadata_offsets.reserve(encoded.size());
    for (auto& mesh : encoded)
    {
        metadata_offsets.push_back(output.size());
        output.raw(mesh.metadata.bytes());
    }
    for (auto& mesh : encoded)
    {
        mesh.page_offsets.reserve(mesh.geometry->pages.size());
        for (const auto& page : mesh.geometry->pages)
        {
            output.align(virtual_geometry_artifact_page_alignment);
            mesh.page_offsets.push_back(output.size());
            const auto end = static_cast<std::uint64_t>(page.compressed_offset) + page.compressed_size;
            if (end > mesh.geometry->page_payload.size())
                return virtual_geometry_artifact_bytes_result::failure(
                    failure(virtual_geometry_artifact_error_code::out_of_bounds, "page payload range is invalid"));
            output.raw(std::span(mesh.geometry->page_payload).subspan(page.compressed_offset, page.compressed_size));
        }
    }

    byte_writer table;
    for (std::size_t mesh_index = 0; mesh_index < encoded.size(); ++mesh_index)
    {
        const auto& mesh = encoded[mesh_index];
        table.string(mesh.name);
        table.value(mesh.material_index);
        table.value(metadata_offsets[mesh_index]);
        table.value(static_cast<std::uint64_t>(mesh.metadata.size()));
        table.value(static_cast<std::uint32_t>(mesh.geometry->pages.size()));
        table.value(std::uint32_t{});
        for (std::size_t page_index = 0; page_index < mesh.geometry->pages.size(); ++page_index)
        {
            const auto& page = mesh.geometry->pages[page_index];
            table.value(mesh.page_offsets[page_index]);
            table.value(page.compressed_size);
            table.value(page.uncompressed_size);
            table.value(page.content_hash);
            table.value(static_cast<std::uint32_t>(page.root ? 1u : 0u));
        }
    }
    if (table.size() != table_size)
        return virtual_geometry_artifact_bytes_result::failure(
            failure(virtual_geometry_artifact_error_code::invalid_data, "virtual-geometry table size mismatch"));
    std::copy(table.bytes().begin(), table.bytes().end(), output.bytes().begin() + table_offset);

    auto patch_u64 = [&](std::size_t offset, std::uint64_t value)
    {
        for (std::size_t index = 0; index < sizeof(value); ++index)
            output.bytes()[offset + index] = static_cast<std::byte>((value >> (index * 8u)) & 0xffu);
    };
    patch_u64(32, table_offset);
    patch_u64(40, output.size());
    return virtual_geometry_artifact_bytes_result::success(std::move(output.bytes()));
}

virtual_geometry_artifact_index_result inspect_virtual_geometry_artifact(std::span<const std::byte> bytes)
{
    if (bytes.size() < header_bytes || !std::equal(artifact_magic.begin(), artifact_magic.end(), bytes.begin()))
        return virtual_geometry_artifact_index_result::failure(
            failure(virtual_geometry_artifact_error_code::invalid_data, "invalid virtual-geometry artifact magic"));
    byte_reader reader(bytes.subspan(artifact_magic.size()));
    virtual_geometry_artifact_index result;
    std::uint32_t declared_header{};
    std::uint32_t mesh_count{};
    std::uint32_t reserved{};
    std::uint64_t table_offset{};
    if (!reader.value(result.schema_version) || !reader.value(declared_header) ||
        !reader.value(result.conventional_artifact_hash) || !reader.value(mesh_count) || !reader.value(reserved) ||
        !reader.value(table_offset) || !reader.value(result.artifact_size))
        return virtual_geometry_artifact_index_result::failure(
            failure(virtual_geometry_artifact_error_code::invalid_data, "truncated virtual-geometry header"));
    if (result.schema_version != virtual_geometry_artifact_schema_version || declared_header != header_bytes)
        return virtual_geometry_artifact_index_result::failure(
            failure(virtual_geometry_artifact_error_code::unsupported_version,
                    "unsupported virtual-geometry artifact schema"));
    if (result.artifact_size != bytes.size() || table_offset < header_bytes || table_offset >= bytes.size())
        return virtual_geometry_artifact_index_result::failure(
            failure(virtual_geometry_artifact_error_code::out_of_bounds, "invalid artifact size or table offset"));

    byte_reader table(bytes.subspan(static_cast<std::size_t>(table_offset)));
    result.meshes.reserve(mesh_count);
    for (std::uint32_t mesh_index = 0; mesh_index < mesh_count; ++mesh_index)
    {
        virtual_geometry_artifact_mesh_index mesh;
        std::uint32_t page_count{};
        if (!table.string(mesh.name) || !table.value(mesh.material_index) || !table.value(mesh.metadata_offset) ||
            !table.value(mesh.metadata_size) || !table.value(page_count) || !table.value(reserved))
            return virtual_geometry_artifact_index_result::failure(
                failure(virtual_geometry_artifact_error_code::invalid_data, "truncated virtual-geometry mesh table"));
        if (mesh.metadata_offset > bytes.size() || mesh.metadata_size > bytes.size() - mesh.metadata_offset)
            return virtual_geometry_artifact_index_result::failure(
                failure(virtual_geometry_artifact_error_code::out_of_bounds, "mesh metadata range is invalid"));
        mesh.pages.reserve(page_count);
        for (std::uint32_t page_index = 0; page_index < page_count; ++page_index)
        {
            virtual_geometry_artifact_page_range page;
            std::uint32_t root{};
            if (!table.value(page.offset) || !table.value(page.stored_size) || !table.value(page.decoded_size) ||
                !table.value(page.content_hash) || !table.value(root))
                return virtual_geometry_artifact_index_result::failure(
                    failure(virtual_geometry_artifact_error_code::invalid_data, "truncated virtual-geometry page table"));
            page.root = root != 0;
            if (page.offset % virtual_geometry_artifact_page_alignment != 0 || page.offset > bytes.size() ||
                page.stored_size > bytes.size() - page.offset)
                return virtual_geometry_artifact_index_result::failure(
                    failure(virtual_geometry_artifact_error_code::out_of_bounds, "virtual-geometry page range is invalid"));
            mesh.pages.push_back(page);
        }
        result.meshes.push_back(std::move(mesh));
    }
    return virtual_geometry_artifact_index_result::success(std::move(result));
}

virtual_geometry_artifact_bytes_result
read_virtual_geometry_artifact_page(std::span<const std::byte> bytes,
                                    const virtual_geometry_artifact_index& index, std::uint32_t mesh_index,
                                    std::uint32_t page_index)
{
    if (mesh_index >= index.meshes.size() || page_index >= index.meshes[mesh_index].pages.size())
        return virtual_geometry_artifact_bytes_result::failure(
            failure(virtual_geometry_artifact_error_code::out_of_bounds,
                    "virtual-geometry page index is out of range"));
    const auto& page = index.meshes[mesh_index].pages[page_index];
    if (page.offset > bytes.size() || page.stored_size > bytes.size() - page.offset)
        return virtual_geometry_artifact_bytes_result::failure(
            failure(virtual_geometry_artifact_error_code::out_of_bounds,
                    "virtual-geometry page bytes are unavailable"));
    const auto payload = bytes.subspan(static_cast<std::size_t>(page.offset), page.stored_size);
    if (hash_bytes(payload) != page.content_hash)
        return virtual_geometry_artifact_bytes_result::failure(
            failure(virtual_geometry_artifact_error_code::integrity_failure,
                    "virtual-geometry page content hash mismatch"));
    return virtual_geometry_artifact_bytes_result::success(std::vector<std::byte>(payload.begin(), payload.end()));
}

} // namespace arc::render

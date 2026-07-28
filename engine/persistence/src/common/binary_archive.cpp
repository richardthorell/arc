#include <arc/persistence/persistence.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace arc::persistence
{
namespace
{

constexpr std::array<char, 8> binary_magic{ 'A', 'R', 'C', 'P', 'B', 'I', 'N', '1' };
constexpr std::uint32_t binary_format_version = 1;

const ecs::component_field_descriptor* find_field(
    const ecs::component_descriptor& component,
    ecs::component_field_id id) noexcept
{
    const auto iterator = std::find_if(
        component.fields.begin(), component.fields.end(),
        [id](const ecs::component_field_descriptor& field)
        {
            return field.id == id;
        });
    return iterator == component.fields.end() ? nullptr : &*iterator;
}

template <class Integer>
void write_integer(std::vector<std::byte>& output, Integer value)
{
    using unsigned_type = std::make_unsigned_t<Integer>;
    const auto converted = static_cast<unsigned_type>(value);
    for (std::size_t index = 0; index < sizeof(Integer); ++index)
        output.push_back(static_cast<std::byte>((converted >> (index * 8u)) & 0xffu));
}

void write_string(std::vector<std::byte>& output, std::string_view value)
{
    write_integer(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(),
        reinterpret_cast<const std::byte*>(value.data()),
        reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

void write_guid(std::vector<std::byte>& output, ecs::entity_guid value)
{
    write_integer(output, value.high);
    write_integer(output, value.low);
}

void write_component_id(std::vector<std::byte>& output, ecs::component_type_id value)
{
    write_integer(output, value.high);
    write_integer(output, value.low);
}

void write_asset_guid(std::vector<std::byte>& output, assets::asset_guid value)
{
    write_integer(output, value.high);
    write_integer(output, value.low);
}

void write_asset_type(std::vector<std::byte>& output, assets::asset_type_id value)
{
    write_integer(output, value.high);
    write_integer(output, value.low);
}

void write_value_payload(std::vector<std::byte>& output, const archive_value& value);

void write_embedded_value(std::vector<std::byte>& output, const archive_value& value)
{
    output.push_back(static_cast<std::byte>(value.kind));
    std::vector<std::byte> payload;
    write_value_payload(payload, value);
    write_integer(output, static_cast<std::uint64_t>(payload.size()));
    output.insert(output.end(), payload.begin(), payload.end());
}

void write_value_payload(std::vector<std::byte>& output, const archive_value& value)
{
    switch (value.kind)
    {
    case archive_value_kind::null: break;
    case archive_value_kind::boolean:
        output.push_back(value.boolean ? std::byte{ 1 } : std::byte{});
        break;
    case archive_value_kind::signed_integer:
        write_integer(output, value.signed_integer);
        break;
    case archive_value_kind::unsigned_integer:
        write_integer(output, value.unsigned_integer);
        break;
    case archive_value_kind::floating_point:
        write_integer(output, std::bit_cast<std::uint64_t>(value.floating_point));
        break;
    case archive_value_kind::string:
        write_string(output, value.string);
        break;
    case archive_value_kind::bytes:
        write_integer(output, static_cast<std::uint64_t>(value.bytes.size()));
        output.insert(output.end(), value.bytes.begin(), value.bytes.end());
        break;
    case archive_value_kind::array:
        write_integer(output, static_cast<std::uint64_t>(value.array.size()));
        for (const auto& item : value.array)
            write_embedded_value(output, item);
        break;
    case archive_value_kind::object:
        write_integer(output, static_cast<std::uint64_t>(value.object.size()));
        for (const auto& [name, item] : value.object)
        {
            write_string(output, name);
            write_embedded_value(output, item);
        }
        break;
    }
}

class binary_reader
{
public:
    explicit binary_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class Integer>
    bool integer(Integer& value)
    {
        if (remaining() < sizeof(Integer)) return false;
        using unsigned_type = std::make_unsigned_t<Integer>;
        unsigned_type result{};
        for (std::size_t index = 0; index < sizeof(Integer); ++index)
            result |= static_cast<unsigned_type>(
                std::to_integer<std::uint8_t>(bytes_[offset_ + index])) << (index * 8u);
        value = static_cast<Integer>(result);
        offset_ += sizeof(Integer);
        return true;
    }

    bool byte(std::uint8_t& value)
    {
        if (remaining() < 1) return false;
        value = std::to_integer<std::uint8_t>(bytes_[offset_++]);
        return true;
    }

    bool string(std::string& value, std::size_t maximum)
    {
        std::uint64_t size{};
        if (!integer(size) || size > maximum || size > remaining()) return false;
        value.assign(
            reinterpret_cast<const char*>(bytes_.data() + offset_),
            static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    bool guid(ecs::entity_guid& value)
    {
        return integer(value.high) && integer(value.low);
    }

    bool component_id(ecs::component_type_id& value)
    {
        return integer(value.high) && integer(value.low);
    }

    bool asset_guid(assets::asset_guid& value)
    {
        return integer(value.high) && integer(value.low);
    }

    bool asset_type(assets::asset_type_id& value)
    {
        return integer(value.high) && integer(value.low);
    }

    std::optional<std::span<const std::byte>> take(std::size_t size)
    {
        if (size > remaining()) return std::nullopt;
        const auto result = bytes_.subspan(offset_, size);
        offset_ += size;
        return result;
    }

    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
    bool complete() const noexcept { return offset_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t offset_{};
};

bool read_value_payload(
    binary_reader& reader,
    archive_value& value,
    std::size_t depth,
    const archive_limits& limits);

bool read_embedded_value(
    binary_reader& reader,
    archive_value& value,
    std::size_t depth,
    const archive_limits& limits)
{
    std::uint8_t kind{};
    std::uint64_t size{};
    if (!reader.byte(kind) || kind > static_cast<std::uint8_t>(archive_value_kind::object) ||
        !reader.integer(size) || size > reader.remaining())
        return false;
    value.kind = static_cast<archive_value_kind>(kind);
    const auto bytes = reader.take(static_cast<std::size_t>(size));
    if (!bytes) return false;
    binary_reader payload(*bytes);
    return read_value_payload(payload, value, depth, limits) && payload.complete();
}

bool read_value_payload(
    binary_reader& reader,
    archive_value& value,
    std::size_t depth,
    const archive_limits& limits)
{
    if (depth > limits.maximum_nesting) return false;
    switch (value.kind)
    {
    case archive_value_kind::null: return true;
    case archive_value_kind::boolean:
    {
        std::uint8_t result{};
        if (!reader.byte(result) || result > 1) return false;
        value.boolean = result != 0;
        return true;
    }
    case archive_value_kind::signed_integer:
        return reader.integer(value.signed_integer);
    case archive_value_kind::unsigned_integer:
        return reader.integer(value.unsigned_integer);
    case archive_value_kind::floating_point:
    {
        std::uint64_t bits{};
        if (!reader.integer(bits)) return false;
        value.floating_point = std::bit_cast<double>(bits);
        return std::isfinite(value.floating_point);
    }
    case archive_value_kind::string:
        return reader.string(value.string, limits.maximum_document_bytes);
    case archive_value_kind::bytes:
    {
        std::uint64_t size{};
        if (!reader.integer(size) || size > reader.remaining()) return false;
        const auto bytes = reader.take(static_cast<std::size_t>(size));
        if (!bytes) return false;
        value.bytes.assign(bytes->begin(), bytes->end());
        return true;
    }
    case archive_value_kind::array:
    {
        std::uint64_t count{};
        if (!reader.integer(count) || count > limits.maximum_fields_per_component * 16ull)
            return false;
        value.array.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index)
        {
            archive_value child;
            if (!read_embedded_value(reader, child, depth + 1, limits)) return false;
            value.array.push_back(std::move(child));
        }
        return true;
    }
    case archive_value_kind::object:
    {
        std::uint64_t count{};
        if (!reader.integer(count) || count > limits.maximum_fields_per_component * 16ull)
            return false;
        value.object.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index)
        {
            std::string name;
            archive_value child;
            if (!reader.string(name, 1024u * 1024u) ||
                !read_embedded_value(reader, child, depth + 1, limits))
                return false;
            value.object.emplace_back(std::move(name), std::move(child));
        }
        return true;
    }
    }
    return false;
}

void sort_document(archive_document& document)
{
    std::sort(document.dependencies.begin(), document.dependencies.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.reference.guid != rhs.reference.guid) return lhs.reference.guid < rhs.reference.guid;
        if (lhs.owner_entity != rhs.owner_entity)
            return std::tie(lhs.owner_entity.high, lhs.owner_entity.low) <
                std::tie(rhs.owner_entity.high, rhs.owner_entity.low);
        if (lhs.owner_component != rhs.owner_component) return lhs.owner_component < rhs.owner_component;
        return lhs.owner_field < rhs.owner_field;
    });
    std::sort(document.entities.begin(), document.entities.end(),
        [](const auto& lhs, const auto& rhs)
        {
            return std::tie(lhs.id.high, lhs.id.low) < std::tie(rhs.id.high, rhs.id.low);
        });
    for (auto& entity : document.entities)
    {
        std::sort(entity.components.begin(), entity.components.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.type < rhs.type; });
        for (auto& component : entity.components)
            std::sort(component.fields.begin(), component.fields.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.id != rhs.id ? lhs.id < rhs.id : lhs.name < rhs.name;
                });
    }
}

}

std::vector<std::byte> write_tagged_binary(
    const archive_document& source,
    std::string_view target_identity,
    std::string& error)
{
    if (!source.id.valid())
    {
        error = "persistence document has no valid identity";
        return {};
    }
    archive_document document = source;
    sort_document(document);
    std::vector<std::byte> payload;
    write_string(payload, target_identity);
    write_integer(payload, document.format_version);
    write_guid(payload, document.id);
    write_guid(payload, document.root);
    write_string(payload, document.name);

    write_integer(payload, static_cast<std::uint64_t>(document.dependencies.size()));
    for (const auto& dependency : document.dependencies)
    {
        write_asset_guid(payload, dependency.reference.guid);
        write_asset_type(payload, dependency.reference.expected_type);
        write_string(payload, dependency.reference.path_hint);
        write_guid(payload, dependency.owner_entity);
        write_component_id(payload, dependency.owner_component);
        write_integer(payload, dependency.owner_field);
        payload.push_back(dependency.required ? std::byte{ 1 } : std::byte{});
    }

    write_integer(payload, static_cast<std::uint64_t>(document.entities.size()));
    for (const auto& entity : document.entities)
    {
        write_guid(payload, entity.id);
        write_guid(payload, entity.parent);
        write_integer(payload, entity.sibling_order);
        write_guid(payload, entity.region);
        write_integer(payload, static_cast<std::uint64_t>(entity.components.size()));
        for (const auto& component : entity.components)
        {
            std::vector<std::byte> component_payload;
            write_component_id(component_payload, component.type);
            write_integer(component_payload, component.schema_version);
            write_string(component_payload, component.name);
            component_payload.push_back(component.known ? std::byte{ 1 } : std::byte{});
            write_integer(component_payload, static_cast<std::uint64_t>(component.fields.size()));
            for (const auto& field : component.fields)
            {
                std::vector<std::byte> field_payload;
                write_string(field_payload, field.name);
                field_payload.push_back(field.known ? std::byte{ 1 } : std::byte{});
                write_value_payload(field_payload, field.value);
                write_integer(component_payload, field.id);
                component_payload.push_back(static_cast<std::byte>(field.value.kind));
                write_integer(component_payload, static_cast<std::uint64_t>(field_payload.size()));
                component_payload.insert(
                    component_payload.end(), field_payload.begin(), field_payload.end());
            }
            write_integer(payload, static_cast<std::uint64_t>(component_payload.size()));
            write_integer(payload, crc32c(component_payload));
            payload.insert(payload.end(), component_payload.begin(), component_payload.end());
        }
        write_embedded_value(payload, entity.extensions);
    }
    write_embedded_value(payload, document.extensions);

    const auto payload_hash = assets::hash_bytes(payload);
    std::vector<std::byte> output(
        reinterpret_cast<const std::byte*>(binary_magic.data()),
        reinterpret_cast<const std::byte*>(binary_magic.data() + binary_magic.size()));
    write_integer(output, binary_format_version);
    output.push_back(static_cast<std::byte>(document.kind));
    output.insert(output.end(), 3, std::byte{});
    output.insert(output.end(), payload_hash.bytes.begin(), payload_hash.bytes.end());
    write_integer(output, static_cast<std::uint64_t>(payload.size()));
    output.insert(output.end(), payload.begin(), payload.end());
    return output;
}

archive_result read_tagged_binary(
    std::span<const std::byte> bytes,
    const component_persistence_registry& components,
    const schema_migration_registry* migrations,
    archive_limits limits)
{
    archive_result result;
    constexpr std::size_t header_size = 8u + 4u + 4u + 32u + 8u;
    if (bytes.size() < header_size || bytes.size() > limits.maximum_document_bytes ||
        std::memcmp(bytes.data(), binary_magic.data(), binary_magic.size()) != 0)
    {
        result.error = "tagged archive header is invalid";
        return result;
    }
    binary_reader header(bytes.subspan(8));
    std::uint32_t version{};
    std::uint8_t kind{};
    if (!header.integer(version) || version != binary_format_version || !header.byte(kind) ||
        kind > static_cast<std::uint8_t>(document_kind::prefab) || !header.take(3))
    {
        result.error = "tagged archive version or kind is unsupported";
        return result;
    }
    const auto expected_hash = header.take(32);
    std::uint64_t payload_size{};
    if (!expected_hash || !header.integer(payload_size) ||
        payload_size != header.remaining())
    {
        result.error = "tagged archive payload size is invalid";
        return result;
    }
    const auto payload = header.take(static_cast<std::size_t>(payload_size));
    if (!payload || !header.complete())
    {
        result.error = "tagged archive payload is truncated";
        return result;
    }
    const auto actual_hash = assets::hash_bytes(*payload);
    if (!std::equal(actual_hash.bytes.begin(), actual_hash.bytes.end(), expected_hash->begin()))
    {
        result.error = "tagged archive payload failed SHA-256 verification";
        return result;
    }
    result.integrity_verified = true;
    result.document.kind = static_cast<document_kind>(kind);
    binary_reader reader(*payload);
    if (!reader.string(result.target_identity, 4096) ||
        !reader.integer(result.document.format_version) ||
        !reader.guid(result.document.id) ||
        !reader.guid(result.document.root) ||
        !reader.string(result.document.name, 1024u * 1024u))
    {
        result.error = "tagged archive metadata is malformed";
        return result;
    }
    const auto supported_document_version =
        result.document.kind == document_kind::scene
        ? archive_document::current_scene_version
        : archive_document::current_prefab_version;
    if (result.document.format_version == 0u ||
        result.document.format_version > supported_document_version)
    {
        result.error = "tagged archive document version is unsupported";
        return result;
    }
    std::uint64_t dependency_count{};
    if (!reader.integer(dependency_count) || dependency_count > limits.maximum_fields_per_component * 16ull)
    {
        result.error = "tagged archive dependency count is invalid";
        return result;
    }
    for (std::uint64_t index = 0; index < dependency_count; ++index)
    {
        dependency_manifest_entry dependency;
        std::uint8_t required{};
        if (!reader.asset_guid(dependency.reference.guid) ||
            !reader.asset_type(dependency.reference.expected_type) ||
            !reader.string(dependency.reference.path_hint, limits.maximum_document_bytes) ||
            !reader.guid(dependency.owner_entity) ||
            !reader.component_id(dependency.owner_component) ||
            !reader.integer(dependency.owner_field) ||
            !reader.byte(required) || required > 1)
        {
            result.error = "tagged archive dependency record is malformed";
            return result;
        }
        dependency.required = required != 0;
        result.document.dependencies.push_back(std::move(dependency));
    }

    std::uint64_t entity_count{};
    if (!reader.integer(entity_count) || entity_count > limits.maximum_entities)
    {
        result.error = "tagged archive entity count is invalid";
        return result;
    }
    std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> entity_ids;
    for (std::uint64_t entity_index = 0; entity_index < entity_count; ++entity_index)
    {
        archive_entity_record entity;
        std::uint64_t component_count{};
        if (!reader.guid(entity.id) || !entity.id.valid() ||
            !entity_ids.insert(entity.id).second ||
            !reader.guid(entity.parent) ||
            !reader.integer(entity.sibling_order) ||
            !reader.guid(entity.region) ||
            !reader.integer(component_count) ||
            component_count > limits.maximum_components_per_entity)
        {
            result.error = "tagged archive entity record is malformed";
            return result;
        }
        for (std::uint64_t component_index = 0; component_index < component_count; ++component_index)
        {
            std::uint64_t component_size{};
            std::uint32_t expected_crc{};
            if (!reader.integer(component_size) || !reader.integer(expected_crc) ||
                component_size > reader.remaining())
            {
                result.error = "tagged component range is invalid";
                return result;
            }
            const auto component_bytes = reader.take(static_cast<std::size_t>(component_size));
            if (!component_bytes || crc32c(*component_bytes) != expected_crc)
            {
                result.error = "tagged component failed CRC32C verification";
                return result;
            }
            binary_reader component_reader(*component_bytes);
            archive_component_record component;
            std::uint8_t known{};
            std::uint64_t field_count{};
            if (!component_reader.component_id(component.type) ||
                !component_reader.integer(component.schema_version) ||
                !component_reader.string(component.name, 1024u * 1024u) ||
                !component_reader.byte(known) ||
                !component_reader.integer(field_count) ||
                field_count > limits.maximum_fields_per_component)
            {
                result.error = "tagged component record is malformed";
                return result;
            }
            const auto* registered = components.find(component.type);
            component.known = registered != nullptr && known != 0;
            for (std::uint64_t field_index = 0; field_index < field_count; ++field_index)
            {
                archive_field_record field;
                std::uint8_t kind_value{};
                std::uint64_t field_size{};
                if (!component_reader.integer(field.id) ||
                    !component_reader.byte(kind_value) ||
                    kind_value > static_cast<std::uint8_t>(archive_value_kind::object) ||
                    !component_reader.integer(field_size) ||
                    field_size > component_reader.remaining())
                {
                    result.error = "tagged field range is invalid";
                    return result;
                }
                const auto field_bytes = component_reader.take(static_cast<std::size_t>(field_size));
                binary_reader field_reader(*field_bytes);
                std::uint8_t field_known{};
                if (!field_reader.string(field.name, 1024u * 1024u) ||
                    !field_reader.byte(field_known))
                {
                    result.error = "tagged field metadata is malformed";
                    return result;
                }
                field.value.kind = static_cast<archive_value_kind>(kind_value);
                if (!read_value_payload(field_reader, field.value, 0, limits) ||
                    !field_reader.complete())
                {
                    result.error = "tagged field payload is malformed";
                    return result;
                }
                field.known = registered && find_field(*registered->component, field.id) &&
                    field_known != 0;
                component.fields.push_back(std::move(field));
            }
            if (!component_reader.complete())
            {
                result.error = "tagged component contains trailing bytes";
                return result;
            }
            if (registered && migrations &&
                component.schema_version < registered->component->schema_version)
            {
                if (!migrations->migrate(
                    component, registered->component->schema_version, result.error))
                    return result;
                result.migrated = true;
            }
            entity.components.push_back(std::move(component));
        }
        if (!read_embedded_value(reader, entity.extensions, 0, limits))
        {
            result.error = "tagged entity extension is malformed";
            return result;
        }
        result.document.entities.push_back(std::move(entity));
    }
    if (!read_embedded_value(reader, result.document.extensions, 0, limits) ||
        !reader.complete())
    {
        result.error = "tagged archive extension or trailing data is malformed";
        return result;
    }
    std::unordered_map<ecs::entity_guid, ecs::entity_guid, ecs::entity_guid_hash> parents;
    parents.reserve(result.document.entities.size());
    for (const auto& entity : result.document.entities)
        parents.emplace(entity.id, entity.parent);
    for (const auto& entity : result.document.entities)
    {
        if (entity.parent.valid() && !entity_ids.contains(entity.parent))
        {
            result.error = "tagged archive parent reference is unresolved";
            return result;
        }
        std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> ancestors;
        auto current = entity.id;
        while (current.valid())
        {
            if (!ancestors.insert(current).second)
            {
                result.error = "tagged archive hierarchy contains a cycle";
                return result;
            }
            const auto found = parents.find(current);
            current = found != parents.end() ? found->second : ecs::entity_guid{};
        }
    }
    if (result.document.kind == document_kind::prefab &&
        !entity_ids.contains(result.document.root))
    {
        result.error = "tagged prefab root is unresolved";
        return result;
    }
    for (const auto& dependency : result.document.dependencies)
    {
        if (dependency.owner_entity.valid() &&
            !entity_ids.contains(dependency.owner_entity))
        {
            result.error = "tagged dependency owner is unresolved";
            return result;
        }
        if (dependency.required && !dependency.reference.guid.valid() &&
            dependency.reference.path_hint.empty())
        {
            result.error = "tagged required dependency has no identity";
            return result;
        }
    }
    const auto target_version = result.document.kind == document_kind::scene
        ? archive_document::current_scene_version : archive_document::current_prefab_version;
    if (migrations && result.document.format_version < target_version)
    {
        if (!migrations->migrate(result.document, target_version, result.error))
            return result;
        result.migrated = true;
    }
    return result;
}

} // namespace arc::persistence

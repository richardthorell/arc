#include <arc/persistence/persistence.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <tuple>
#include <unordered_set>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace arc::persistence
{
namespace
{

using json = nlohmann::json;

constexpr std::string_view scene_format = "arc.scene";
constexpr std::string_view prefab_format = "arc.prefab";

std::uint64_t unknown_field_id(std::string_view name) noexcept
{
    std::uint64_t result = 14695981039346656037ull;
    for (const char value : name)
    {
        result ^= static_cast<std::uint8_t>(value);
        result *= 1099511628211ull;
    }
    return result | (std::uint64_t{ 1 } << 63u);
}

std::string field_id_string(ecs::component_field_id value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(16, '0');
    for (std::size_t index = 0; index < 16; ++index)
        result[15 - index] = digits[(value >> (index * 4u)) & 0xfu];
    return result;
}

std::optional<ecs::component_field_id> parse_field_id(std::string_view value)
{
    if (value.size() != 16)
        return std::nullopt;
    ecs::component_field_id result{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), result, 16);
    return error == std::errc{} && end == value.data() + value.size()
        ? std::optional(result) : std::nullopt;
}

const ecs::component_field_descriptor* find_field(
    const ecs::component_descriptor& descriptor,
    std::string_view name) noexcept
{
    const auto found = std::find_if(descriptor.fields.begin(), descriptor.fields.end(),
        [&](const auto& field) { return field.name == name; });
    return found == descriptor.fields.end() ? nullptr : &*found;
}

const ecs::component_field_descriptor* find_field(
    const ecs::component_descriptor& descriptor,
    ecs::component_field_id id) noexcept
{
    const auto found = std::find_if(descriptor.fields.begin(), descriptor.fields.end(),
        [&](const auto& field) { return field.id == id; });
    return found == descriptor.fields.end() ? nullptr : &*found;
}

std::uint64_t read_unsigned(const std::byte* data, std::size_t size) noexcept
{
    std::uint64_t result{};
    std::memcpy(&result, data, std::min(size, sizeof(result)));
    return result;
}

std::int64_t read_signed(const std::byte* data, std::size_t size) noexcept
{
    switch (size)
    {
    case 1: { std::int8_t value{}; std::memcpy(&value, data, 1); return value; }
    case 2: { std::int16_t value{}; std::memcpy(&value, data, 2); return value; }
    case 4: { std::int32_t value{}; std::memcpy(&value, data, 4); return value; }
    case 8: { std::int64_t value{}; std::memcpy(&value, data, 8); return value; }
    default: return {};
    }
}

std::size_t floating_element_count(ecs::reflected_field_kind kind) noexcept
{
    switch (kind)
    {
    case ecs::reflected_field_kind::vector2: return 2;
    case ecs::reflected_field_kind::vector3: return 3;
    case ecs::reflected_field_kind::vector4:
    case ecs::reflected_field_kind::quaternion: return 4;
    case ecs::reflected_field_kind::matrix: return 16;
    default: return 0;
    }
}

bool encode_reflected_field(
    const ecs::component_field_descriptor& field,
    const void* component,
    archive_value& output)
{
    if (field.offset == ecs::component_field_descriptor::invalid_offset ||
        field.value_size == 0)
        return false;
    const auto* data = reinterpret_cast<const std::byte*>(component) + field.offset;
    switch (field.kind)
    {
    case ecs::reflected_field_kind::boolean:
        output.kind = archive_value_kind::boolean;
        output.boolean = read_unsigned(data, field.value_size) != 0;
        return true;
    case ecs::reflected_field_kind::signed_integer:
        if (field.value_size > sizeof(std::int64_t)) return false;
        output.kind = archive_value_kind::signed_integer;
        output.signed_integer = read_signed(data, field.value_size);
        return true;
    case ecs::reflected_field_kind::unsigned_integer:
    case ecs::reflected_field_kind::enumeration:
        if (field.value_size > sizeof(std::uint64_t)) return false;
        output.kind = archive_value_kind::unsigned_integer;
        output.unsigned_integer = read_unsigned(data, field.value_size);
        return true;
    case ecs::reflected_field_kind::floating_point:
        output.kind = archive_value_kind::floating_point;
        if (field.value_size == sizeof(float))
        {
            float value{};
            std::memcpy(&value, data, sizeof(value));
            output.floating_point = value;
        }
        else if (field.value_size == sizeof(double))
            std::memcpy(&output.floating_point, data, sizeof(double));
        else
            return false;
        return std::isfinite(output.floating_point);
    case ecs::reflected_field_kind::string:
        output.kind = archive_value_kind::string;
        output.string = *reinterpret_cast<const std::string*>(data);
        return true;
    default:
    {
        const auto count = floating_element_count(field.kind);
        if (count == 0 || field.value_size < count * sizeof(float))
            return false;
        output.kind = archive_value_kind::array;
        output.array.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
        {
            float value{};
            std::memcpy(&value, data + index * sizeof(float), sizeof(value));
            if (!std::isfinite(value)) return false;
            archive_value item;
            item.kind = archive_value_kind::floating_point;
            item.floating_point = value;
            output.array.push_back(std::move(item));
        }
        return true;
    }
    }
}

bool write_integer(
    std::byte* destination,
    std::size_t size,
    std::uint64_t value) noexcept
{
    if (size == 0 || size > sizeof(value))
        return false;
    if (size < sizeof(value) && value >= (std::uint64_t{ 1 } << (size * 8u)))
        return false;
    std::memcpy(destination, &value, size);
    return true;
}

bool write_signed_integer(
    std::byte* destination,
    std::size_t size,
    std::int64_t value) noexcept
{
    if (size == 0 || size > sizeof(value))
        return false;
    if (size < sizeof(value))
    {
        const auto bits = size * 8u;
        const auto minimum = -(std::int64_t{ 1 } << (bits - 1u));
        const auto maximum = (std::int64_t{ 1 } << (bits - 1u)) - 1;
        if (value < minimum || value > maximum)
            return false;
    }
    std::memcpy(destination, &value, size);
    return true;
}

bool decode_reflected_field(
    const ecs::component_field_descriptor& field,
    const archive_value& input,
    void* component)
{
    if (field.offset == ecs::component_field_descriptor::invalid_offset ||
        field.value_size == 0)
        return false;
    auto* data = reinterpret_cast<std::byte*>(component) + field.offset;
    switch (field.kind)
    {
    case ecs::reflected_field_kind::boolean:
        return input.kind == archive_value_kind::boolean &&
            write_integer(data, field.value_size, input.boolean ? 1u : 0u);
    case ecs::reflected_field_kind::signed_integer:
        return input.kind == archive_value_kind::signed_integer &&
            write_signed_integer(data, field.value_size, input.signed_integer);
    case ecs::reflected_field_kind::unsigned_integer:
    case ecs::reflected_field_kind::enumeration:
        return input.kind == archive_value_kind::unsigned_integer &&
            write_integer(data, field.value_size, input.unsigned_integer);
    case ecs::reflected_field_kind::floating_point:
        if (input.kind != archive_value_kind::floating_point ||
            !std::isfinite(input.floating_point))
            return false;
        if (field.value_size == sizeof(float))
        {
            const auto value = static_cast<float>(input.floating_point);
            std::memcpy(data, &value, sizeof(value));
            return true;
        }
        if (field.value_size == sizeof(double))
        {
            std::memcpy(data, &input.floating_point, sizeof(double));
            return true;
        }
        return false;
    case ecs::reflected_field_kind::string:
        if (input.kind != archive_value_kind::string) return false;
        *reinterpret_cast<std::string*>(data) = input.string;
        return true;
    default:
    {
        const auto count = floating_element_count(field.kind);
        if (count == 0 || input.kind != archive_value_kind::array ||
            input.array.size() != count ||
            field.value_size < count * sizeof(float))
            return false;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto& item = input.array[index];
            if (item.kind != archive_value_kind::floating_point ||
                !std::isfinite(item.floating_point))
                return false;
            const auto value = static_cast<float>(item.floating_point);
            std::memcpy(data + index * sizeof(float), &value, sizeof(value));
        }
        return true;
    }
    }
}

bool value_from_json(
    const json& source,
    archive_value& output,
    std::size_t depth,
    const archive_limits& limits)
{
    if (depth > limits.maximum_nesting)
        return false;
    if (source.is_null())
    {
        output.kind = archive_value_kind::null;
        return true;
    }
    if (source.is_boolean())
    {
        output.kind = archive_value_kind::boolean;
        output.boolean = source.get<bool>();
        return true;
    }
    if (source.is_number_unsigned())
    {
        output.kind = archive_value_kind::unsigned_integer;
        output.unsigned_integer = source.get<std::uint64_t>();
        return true;
    }
    if (source.is_number_integer())
    {
        output.kind = archive_value_kind::signed_integer;
        output.signed_integer = source.get<std::int64_t>();
        return true;
    }
    if (source.is_number_float())
    {
        output.kind = archive_value_kind::floating_point;
        output.floating_point = source.get<double>();
        return std::isfinite(output.floating_point);
    }
    if (source.is_string())
    {
        output.kind = archive_value_kind::string;
        output.string = source.get<std::string>();
        return true;
    }
    if (source.is_array())
    {
        output.kind = archive_value_kind::array;
        output.array.reserve(source.size());
        for (const auto& item : source)
        {
            archive_value value;
            if (!value_from_json(item, value, depth + 1, limits))
                return false;
            output.array.push_back(std::move(value));
        }
        return true;
    }
    if (source.is_object())
    {
        output.kind = archive_value_kind::object;
        output.object.reserve(source.size());
        for (const auto& [name, item] : source.items())
        {
            archive_value value;
            if (!value_from_json(item, value, depth + 1, limits))
                return false;
            output.object.emplace_back(name, std::move(value));
        }
        return true;
    }
    return false;
}

json value_to_json(const archive_value& value)
{
    switch (value.kind)
    {
    case archive_value_kind::null: return nullptr;
    case archive_value_kind::boolean: return value.boolean;
    case archive_value_kind::signed_integer: return value.signed_integer;
    case archive_value_kind::unsigned_integer: return value.unsigned_integer;
    case archive_value_kind::floating_point: return value.floating_point;
    case archive_value_kind::string: return value.string;
    case archive_value_kind::bytes:
    {
        std::string result;
        constexpr char digits[] = "0123456789abcdef";
        result.reserve(value.bytes.size() * 2u);
        for (const auto byte : value.bytes)
        {
            const auto number = std::to_integer<std::uint8_t>(byte);
            result.push_back(digits[number >> 4u]);
            result.push_back(digits[number & 0xfu]);
        }
        return json{ { "$bytes", std::move(result) } };
    }
    case archive_value_kind::array:
    {
        json result = json::array();
        for (const auto& item : value.array)
            result.push_back(value_to_json(item));
        return result;
    }
    case archive_value_kind::object:
    {
        json result = json::object();
        for (const auto& [name, item] : value.object)
            result[name] = value_to_json(item);
        return result;
    }
    }
    return nullptr;
}

json dependency_json(const dependency_manifest_entry& dependency)
{
    return {
        { "guid", dependency.reference.guid.valid()
            ? assets::to_string(dependency.reference.guid) : std::string{} },
        { "expectedType", dependency.reference.expected_type.valid()
            ? assets::to_string(dependency.reference.expected_type) : std::string{} },
        { "pathHint", dependency.reference.path_hint },
        { "ownerEntity", dependency.owner_entity.valid()
            ? ecs::to_string(dependency.owner_entity) : std::string{} },
        { "ownerComponent", dependency.owner_component.valid()
            ? ecs::to_string(dependency.owner_component) : std::string{} },
        { "ownerField", dependency.owner_field != 0
            ? field_id_string(dependency.owner_field) : std::string{} },
        { "required", dependency.required }
    };
}

std::optional<dependency_manifest_entry> dependency_from_json(const json& value)
{
    if (!value.is_object())
        return std::nullopt;
    dependency_manifest_entry result;
    const auto guid_text = value.value("guid", "");
    const auto type_text = value.value("expectedType", "");
    const auto entity_text = value.value("ownerEntity", "");
    const auto component_text = value.value("ownerComponent", "");
    const auto field_text = value.value("ownerField", "");
    if (!guid_text.empty())
    {
        const auto guid = assets::parse_asset_guid(guid_text);
        if (!guid) return std::nullopt;
        result.reference.guid = *guid;
    }
    if (!type_text.empty())
    {
        const auto type = assets::parse_asset_type_id(type_text);
        if (!type) return std::nullopt;
        result.reference.expected_type = *type;
    }
    if (!entity_text.empty())
    {
        const auto entity = ecs::parse_entity_guid(entity_text);
        if (!entity) return std::nullopt;
        result.owner_entity = *entity;
    }
    if (!component_text.empty())
    {
        const auto component = ecs::parse_component_type_id(component_text);
        if (!component) return std::nullopt;
        result.owner_component = *component;
    }
    if (!field_text.empty())
    {
        const auto field = parse_field_id(field_text);
        if (!field) return std::nullopt;
        result.owner_field = *field;
    }
    if (!value.contains("pathHint") || !value["pathHint"].is_string() ||
        (value.contains("required") && !value["required"].is_boolean()))
        return std::nullopt;
    result.reference.path_hint = value["pathHint"].get<std::string>();
    if (!result.reference.path_hint.empty())
    {
        const std::filesystem::path hint(result.reference.path_hint);
        if (hint.is_absolute() ||
            result.reference.path_hint.find('\\') != std::string::npos)
            return std::nullopt;
        for (const auto& part : hint)
            if (part == "..")
                return std::nullopt;
    }
    result.required = value.value("required", true);
    return result;
}

json document_payload_json(const archive_document& document)
{
    json result{
        { "format", document.kind == document_kind::scene ? scene_format : prefab_format },
        { "formatVersion", document.format_version },
        { document.kind == document_kind::scene ? "scene" : "prefab", {
            { "id", ecs::to_string(document.id) },
            { "name", document.name }
        } },
        { "entities", json::array() },
        { "dependencies", json::array() }
    };
    if (document.kind == document_kind::prefab)
        result["prefab"]["root"] = ecs::to_string(document.root);
    for (const auto& dependency : document.dependencies)
        result["dependencies"].push_back(dependency_json(dependency));
    for (const auto& entity : document.entities)
    {
        json record{
            { "id", ecs::to_string(entity.id) },
            { "parent", entity.parent.valid() ? json(ecs::to_string(entity.parent)) : json(nullptr) },
            { "order", entity.sibling_order },
            { "components", json::object() }
        };
        if (entity.region.valid())
            record["region"] = ecs::to_string(entity.region);
        for (const auto& component : entity.components)
        {
            json value{
                { "typeId", ecs::to_string(component.type) },
                { "version", component.schema_version }
            };
            for (const auto& field : component.fields)
                value[field.name] = value_to_json(field.value);
            record["components"][component.name] = std::move(value);
        }
        if (entity.extensions.kind == archive_value_kind::object)
            for (const auto& [name, value] : entity.extensions.object)
                record[name] = value_to_json(value);
        result["entities"].push_back(std::move(record));
    }
    if (document.extensions.kind == archive_value_kind::object)
        for (const auto& [name, value] : document.extensions.object)
            result[name] = value_to_json(value);
    return result;
}

bool guid_less(ecs::entity_guid lhs, ecs::entity_guid rhs) noexcept
{
    return std::tie(lhs.high, lhs.low) < std::tie(rhs.high, rhs.low);
}

archive_document canonical_document(archive_document document)
{
    std::sort(document.dependencies.begin(), document.dependencies.end(),
        [](const auto& lhs, const auto& rhs)
        {
            if (lhs.reference.guid != rhs.reference.guid)
                return lhs.reference.guid < rhs.reference.guid;
            if (lhs.owner_entity != rhs.owner_entity)
                return guid_less(lhs.owner_entity, rhs.owner_entity);
            if (lhs.owner_component != rhs.owner_component)
                return lhs.owner_component < rhs.owner_component;
            return lhs.owner_field < rhs.owner_field;
        });
    std::sort(document.entities.begin(), document.entities.end(),
        [](const auto& lhs, const auto& rhs) { return guid_less(lhs.id, rhs.id); });
    for (auto& entity : document.entities)
    {
        std::sort(entity.components.begin(), entity.components.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.type < rhs.type; });
        for (auto& component : entity.components)
            std::sort(component.fields.begin(), component.fields.end(),
                [](const auto& lhs, const auto& rhs)
                {
                    return lhs.id != rhs.id ? lhs.id < rhs.id : lhs.name < rhs.name;
                });
    }
    return document;
}

assets::asset_hash canonical_payload_hash(json document)
{
    document.erase("integrity");
    const auto canonical = document.dump();
    return assets::hash_bytes(std::as_bytes(std::span(canonical.data(), canonical.size())));
}

bool atomic_replace(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination,
    std::string& error)
{
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    {
        error = "atomic replacement failed with Win32 error " + std::to_string(GetLastError());
        return false;
    }
#else
    std::error_code move_error;
    std::filesystem::rename(temporary, destination, move_error);
    if (move_error)
    {
        error = "atomic replacement failed: " + move_error.message();
        return false;
    }
#endif
    return true;
}

bool flush_file_to_storage(const std::filesystem::path& path, std::string& error)
{
#if defined(_WIN32)
    const HANDLE handle = CreateFileW(
        path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        error = "could not open temporary document for durable flush";
        return false;
    }
    const bool flushed = FlushFileBuffers(handle) != FALSE;
    CloseHandle(handle);
    if (!flushed)
    {
        error = "could not durably flush temporary document";
        return false;
    }
    return true;
#else
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0)
    {
        error = "could not open temporary document for durable flush";
        return false;
    }
    const bool flushed = ::fsync(descriptor) == 0;
    ::close(descriptor);
    if (!flushed)
    {
        error = "could not durably flush temporary document";
        return false;
    }
    return true;
#endif
}

std::optional<std::string> read_text(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::nullopt;
    const auto end = stream.tellg();
    if (end < 0) return std::nullopt;
    std::string result(static_cast<std::size_t>(end), '\0');
    stream.seekg(0);
    if (!result.empty())
        stream.read(result.data(), static_cast<std::streamsize>(result.size()));
    return stream ? std::optional(std::move(result)) : std::nullopt;
}

std::filesystem::path backup_path(const std::filesystem::path& path, std::size_t generation)
{
    return std::filesystem::path(path.string() + ".bak" + std::to_string(generation));
}

}

bool component_persistence_registry::register_component(component_persistence_descriptor descriptor)
{
    if (frozen_ || !descriptor.component || !descriptor.component->id.valid() ||
        std::any_of(descriptors_.begin(), descriptors_.end(), [&](const auto& existing) {
            return existing.component->id == descriptor.component->id ||
                existing.component->canonical_name == descriptor.component->canonical_name ||
                existing.component->display_name == descriptor.component->display_name;
        }))
        return false;
    descriptors_.push_back(std::move(descriptor));
    return true;
}

bool component_persistence_registry::freeze()
{
    if (frozen_) return true;
    std::sort(descriptors_.begin(), descriptors_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.component->id < rhs.component->id;
    });
    frozen_ = true;
    return true;
}

bool component_persistence_registry::frozen() const noexcept { return frozen_; }

const component_persistence_descriptor* component_persistence_registry::find(
    ecs::component_type_id type) const noexcept
{
    const auto found = std::find_if(descriptors_.begin(), descriptors_.end(),
        [&](const auto& descriptor) { return descriptor.component->id == type; });
    return found == descriptors_.end() ? nullptr : &*found;
}

const component_persistence_descriptor* component_persistence_registry::find(
    std::string_view name) const noexcept
{
    const auto found = std::find_if(descriptors_.begin(), descriptors_.end(), [&](const auto& descriptor) {
        return descriptor.component->canonical_name == name ||
            descriptor.component->display_name == name ||
            std::find(descriptor.legacy_names.begin(), descriptor.legacy_names.end(), name) !=
                descriptor.legacy_names.end();
    });
    return found == descriptors_.end() ? nullptr : &*found;
}

bool component_persistence_registry::encode(
    ecs::component_type_id type,
    const void* component,
    archive_component_record& output,
    std::string& error) const
{
    const auto* descriptor = find(type);
    if (!descriptor || !component)
    {
        error = "component has no registered persistence encoder";
        return false;
    }
    output.type = descriptor->component->id;
    output.name = std::string(descriptor->component->display_name);
    output.schema_version = descriptor->component->schema_version;
    output.known = true;
    if (descriptor->encode)
        return descriptor->encode(component, output, error);
    if (descriptor->component->custom_serialization)
    {
        error = "custom component has no registered persistence encoder";
        return false;
    }
    output.fields.clear();
    for (const auto& field : descriptor->component->fields)
    {
        if (!ecs::has_flag(field.flags, ecs::reflected_field_flags::serialized))
            continue;
        archive_value value;
        if (!encode_reflected_field(field, component, value))
        {
            error = "reflected field cannot be encoded: " + std::string(field.name);
            return false;
        }
        output.fields.push_back({
            field.id, std::string(field.name), std::move(value), true
        });
    }
    return true;
}

bool component_persistence_registry::decode(
    ecs::component_type_id type,
    void* component,
    const archive_component_record& input,
    std::string& error) const
{
    const auto* descriptor = find(type);
    if (!descriptor || !component)
    {
        error = "component has no registered persistence decoder";
        return false;
    }
    if (input.type != descriptor->component->id)
    {
        error = "component persistence decoder received the wrong stable type ID";
        return false;
    }
    if (descriptor->decode)
        return descriptor->decode(component, input, error);
    if (descriptor->component->custom_serialization)
    {
        error = "custom component has no registered persistence decoder";
        return false;
    }
    for (const auto& field : input.fields)
    {
        const auto* metadata = find_field(*descriptor->component, field.id);
        if (!metadata || !ecs::has_flag(
                metadata->flags, ecs::reflected_field_flags::serialized))
            continue;
        if (!decode_reflected_field(*metadata, field.value, component))
        {
            error = "reflected field cannot be decoded: " + field.name;
            return false;
        }
    }
    return true;
}

bool schema_migration_registry::register_component(
    ecs::component_type_id type,
    std::uint32_t from_version,
    std::uint32_t to_version,
    component_migration migration)
{
    if (frozen_ || !type.valid() || to_version != from_version + 1u || !migration)
        return false;
    if (std::any_of(component_edges_.begin(), component_edges_.end(), [&](const auto& edge) {
        return edge.type == type && edge.from == from_version;
    }))
        return false;
    component_edges_.push_back({ type, from_version, to_version, std::move(migration) });
    return true;
}

bool schema_migration_registry::register_document(
    document_kind kind,
    std::uint32_t from_version,
    std::uint32_t to_version,
    document_migration migration)
{
    if (frozen_ || to_version != from_version + 1u || !migration)
        return false;
    if (std::any_of(document_edges_.begin(), document_edges_.end(), [&](const auto& edge) {
        return edge.kind == kind && edge.from == from_version;
    }))
        return false;
    document_edges_.push_back({ kind, from_version, to_version, std::move(migration) });
    return true;
}

bool schema_migration_registry::freeze(std::string& error)
{
    if (frozen_) return true;
    std::sort(component_edges_.begin(), component_edges_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.type != rhs.type ? lhs.type < rhs.type : lhs.from < rhs.from;
    });
    std::sort(document_edges_.begin(), document_edges_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.kind != rhs.kind ? lhs.kind < rhs.kind : lhs.from < rhs.from;
    });
    for (std::size_t index = 0; index < component_edges_.size(); ++index)
    {
        const auto& edge = component_edges_[index];
        if (edge.to != edge.from + 1u ||
            (index > 0 && component_edges_[index - 1].type == edge.type &&
                component_edges_[index - 1].to != edge.from))
        {
            error = "component migration graph contains a gap or non-consecutive edge";
            return false;
        }
    }
    for (std::size_t index = 0; index < document_edges_.size(); ++index)
    {
        const auto& edge = document_edges_[index];
        if (edge.to != edge.from + 1u ||
            (index > 0 && document_edges_[index - 1].kind == edge.kind &&
                document_edges_[index - 1].to != edge.from))
        {
            error = "document migration graph contains a gap or non-consecutive edge";
            return false;
        }
    }
    frozen_ = true;
    return true;
}

bool schema_migration_registry::migrate(
    archive_component_record& component,
    std::uint32_t target_version,
    std::string& error) const
{
    if (!frozen_)
    {
        error = "component migration registry is not frozen";
        return false;
    }
    if (component.schema_version > target_version)
    {
        error = "component downgrade is not supported";
        return false;
    }
    while (component.schema_version < target_version)
    {
        const auto found = std::find_if(component_edges_.begin(), component_edges_.end(),
            [&](const auto& edge) {
                return edge.type == component.type && edge.from == component.schema_version;
            });
        if (found == component_edges_.end())
        {
            error = "component migration path has a gap";
            return false;
        }
        if (!found->function(component, error))
            return false;
        component.schema_version = found->to;
    }
    return true;
}

bool schema_migration_registry::migrate(
    archive_document& document,
    std::uint32_t target_version,
    std::string& error) const
{
    if (!frozen_)
    {
        error = "document migration registry is not frozen";
        return false;
    }
    if (document.format_version > target_version)
    {
        error = "document downgrade is not supported";
        return false;
    }
    while (document.format_version < target_version)
    {
        const auto found = std::find_if(document_edges_.begin(), document_edges_.end(),
            [&](const auto& edge) {
                return edge.kind == document.kind && edge.from == document.format_version;
            });
        if (found == document_edges_.end())
        {
            error = "document migration path has a gap";
            return false;
        }
        if (!found->function(document, error))
            return false;
        document.format_version = found->to;
    }
    return true;
}

json_seal_result seal_json_document(std::string_view unsealed_text, bool pretty)
{
    json_seal_result result;
    auto document = json::parse(unsealed_text, nullptr, false);
    if (!document.is_object())
    {
        result.error = "document JSON is malformed";
        return result;
    }
    const auto format = document.value("format", "");
    const std::string_view metadata = format == scene_format ? "scene" :
        (format == prefab_format ? "prefab" : "");
    if (metadata.empty() || !document.contains("formatVersion") ||
        !document["formatVersion"].is_number_unsigned() ||
        !document.contains(metadata) || !document[metadata].is_object() ||
        !document.contains("entities") || !document["entities"].is_array())
    {
        result.error = "document JSON does not satisfy the persistence envelope";
        return result;
    }
    document.erase("integrity");
    result.payload_hash = canonical_payload_hash(document);
    document["integrity"] = {
        { "algorithm", "sha256" },
        { "payloadHash", assets::to_string(result.payload_hash) }
    };
    result.text = document.dump(pretty ? 2 : -1) + '\n';
    return result;
}

bool verify_json_document(
    std::string_view text,
    assets::asset_hash* payload_hash,
    std::string& error)
{
    auto document = json::parse(text, nullptr, false);
    if (!document.is_object())
    {
        error = "document JSON is malformed";
        return false;
    }
    const auto format = document.value("format", "");
    const std::string_view metadata = format == scene_format ? "scene" :
        (format == prefab_format ? "prefab" : "");
    if (metadata.empty() || !document.contains("formatVersion") ||
        !document["formatVersion"].is_number_unsigned() ||
        !document.contains(metadata) || !document[metadata].is_object() ||
        !document.contains("entities") || !document["entities"].is_array())
    {
        error = "document JSON does not satisfy the persistence envelope";
        return false;
    }
    const auto hash = canonical_payload_hash(document);
    if (payload_hash) *payload_hash = hash;
    const auto integrity = document.find("integrity");
    if (integrity == document.end())
        return true;
    if (!integrity->is_object() || integrity->value("algorithm", "") != "sha256")
    {
        error = "document integrity metadata is malformed";
        return false;
    }
    const auto expected = assets::parse_asset_hash(integrity->value("payloadHash", ""));
    if (!expected || *expected != hash)
    {
        error = "document payload failed SHA-256 integrity verification";
        return false;
    }
    return true;
}

std::string write_reflected_json(
    const archive_document& document,
    bool pretty,
    std::string& error)
{
    if (!document.id.valid())
    {
        error = "persistence document has no valid identity";
        return {};
    }
    auto payload = document_payload_json(canonical_document(document)).dump();
    auto sealed = seal_json_document(payload, pretty);
    if (!sealed.succeeded())
    {
        error = std::move(sealed.error);
        return {};
    }
    return std::move(sealed.text);
}

archive_result read_reflected_json(
    std::string_view text,
    const component_persistence_registry& components,
    const schema_migration_registry* migrations,
    archive_limits limits)
{
    archive_result result;
    if (text.size() > limits.maximum_document_bytes)
    {
        result.error = "document exceeds the configured archive size limit";
        return result;
    }
    assets::asset_hash payload_hash;
    if (!verify_json_document(text, &payload_hash, result.error))
        return result;
    const auto source = json::parse(text, nullptr, false);
    result.integrity_verified = source.contains("integrity");
    if (!result.integrity_verified)
        result.diagnostics.push_back({ "persistence.integrity", "Legacy document has no integrity record" });
    const auto format = source.value("format", "");
    result.document.kind = format == prefab_format ? document_kind::prefab : document_kind::scene;
    if (format != scene_format && format != prefab_format)
    {
        result.error = "document format is unsupported";
        return result;
    }
    result.document.format_version = source.value("formatVersion", 0u);
    const auto supported_version = result.document.kind == document_kind::scene
        ? archive_document::current_scene_version : archive_document::current_prefab_version;
    if (result.document.format_version == 0u ||
        result.document.format_version > supported_version)
    {
        result.error = "document version is unsupported";
        return result;
    }
    const auto metadata_name = result.document.kind == document_kind::scene ? "scene" : "prefab";
    if (!source.contains(metadata_name) || !source[metadata_name].is_object() ||
        !source.contains("entities") || !source["entities"].is_array() ||
        source["entities"].size() > limits.maximum_entities)
    {
        result.error = "document metadata or entity array is malformed";
        return result;
    }
    const auto& metadata = source[metadata_name];
    const auto id = ecs::parse_entity_guid(metadata.value("id", ""));
    if (!id)
    {
        result.error = "document identity is invalid";
        return result;
    }
    result.document.id = *id;
    result.document.name = metadata.value("name", "");
    if (result.document.kind == document_kind::prefab)
    {
        const auto root = ecs::parse_entity_guid(metadata.value("root", ""));
        if (!root)
        {
            result.error = "prefab root identity is invalid";
            return result;
        }
        result.document.root = *root;
    }
    std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> entity_ids;
    std::uint32_t sequence{};
    for (const auto& source_entity : source["entities"])
    {
        if (!source_entity.is_object() || !source_entity.contains("components") ||
            !source_entity["components"].is_object())
        {
            result.error = "entity record is malformed";
            return result;
        }
        archive_entity_record entity;
        const auto entity_id = ecs::parse_entity_guid(source_entity.value("id", ""));
        if (!entity_id || !entity_ids.insert(*entity_id).second)
        {
            result.error = "entity identity is invalid or duplicated";
            return result;
        }
        entity.id = *entity_id;
        if (source_entity.contains("parent") && !source_entity["parent"].is_null())
        {
            if (!source_entity["parent"].is_string())
            {
                result.error = "entity parent identity is malformed";
                return result;
            }
            const auto parent = ecs::parse_entity_guid(source_entity["parent"].get<std::string>());
            if (!parent)
            {
                result.error = "entity parent identity is invalid";
                return result;
            }
            entity.parent = *parent;
        }
        entity.sibling_order = source_entity.value("order", sequence++);
        if (source_entity.contains("region"))
        {
            const auto region = ecs::parse_entity_guid(source_entity.value("region", ""));
            if (!region)
            {
                result.error = "entity region identity is invalid";
                return result;
            }
            entity.region = *region;
        }
        if (source_entity["components"].size() > limits.maximum_components_per_entity)
        {
            result.error = "entity exceeds the configured component limit";
            return result;
        }
        for (const auto& [component_name, source_component] : source_entity["components"].items())
        {
            if (!source_component.is_object())
            {
                result.error = "component record is malformed";
                return result;
            }
            archive_component_record component;
            component.name = component_name;
            component.schema_version = source_component.value("version", 0u);
            const auto type_text = source_component.value("typeId", "");
            const auto explicit_type = ecs::parse_component_type_id(type_text);
            const auto* registered = explicit_type ? components.find(*explicit_type) : components.find(component_name);
            if (explicit_type) component.type = *explicit_type;
            else if (registered) component.type = registered->component->id;
            else component.type = ecs::detail::fallback_type_id(component_name);
            component.known = registered != nullptr;
            if (source_component.size() > limits.maximum_fields_per_component + 2u)
            {
                result.error = "component exceeds the configured field limit";
                return result;
            }
            for (const auto& [field_name, source_field] : source_component.items())
            {
                if (field_name == "version" || field_name == "typeId")
                    continue;
                archive_field_record field;
                field.name = field_name;
                const auto* descriptor = registered
                    ? find_field(*registered->component, field_name) : nullptr;
                field.id = descriptor ? descriptor->id : unknown_field_id(field_name);
                field.known = descriptor != nullptr;
                if (!value_from_json(source_field, field.value, 0, limits))
                {
                    result.error = "component field value is invalid or too deeply nested";
                    return result;
                }
                component.fields.push_back(std::move(field));
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
        for (const auto& [name, value] : source_entity.items())
        {
            if (name == "id" || name == "parent" || name == "order" ||
                name == "region" || name == "components")
                continue;
            if (entity.extensions.kind == archive_value_kind::null)
                entity.extensions.kind = archive_value_kind::object;
            archive_value converted;
            if (!value_from_json(value, converted, 0, limits))
            {
                result.error = "entity extension is invalid";
                return result;
            }
            entity.extensions.object.emplace_back(name, std::move(converted));
        }
        result.document.entities.push_back(std::move(entity));
    }
    std::unordered_map<ecs::entity_guid, ecs::entity_guid, ecs::entity_guid_hash> parents;
    parents.reserve(result.document.entities.size());
    for (const auto& entity : result.document.entities)
        parents.emplace(entity.id, entity.parent);
    for (const auto& entity : result.document.entities)
    {
        if (entity.parent.valid() && !entity_ids.contains(entity.parent))
        {
            result.error = "entity parent reference is unresolved";
            return result;
        }
        std::unordered_set<ecs::entity_guid, ecs::entity_guid_hash> ancestors;
        auto current = entity.id;
        while (current.valid())
        {
            if (!ancestors.insert(current).second)
            {
                result.error = "entity hierarchy contains a cycle";
                return result;
            }
            const auto found = parents.find(current);
            current = found != parents.end() ? found->second : ecs::entity_guid{};
        }
    }
    if (result.document.kind == document_kind::prefab &&
        !entity_ids.contains(result.document.root))
    {
        result.error = "prefab root does not reference a document entity";
        return result;
    }
    if (source.contains("dependencies"))
    {
        if (!source["dependencies"].is_array())
        {
            result.error = "dependency manifest is malformed";
            return result;
        }
        for (const auto& dependency : source["dependencies"])
        {
            auto parsed = dependency_from_json(dependency);
            if (!parsed)
            {
                result.error = "dependency manifest entry is malformed";
                return result;
            }
            result.document.dependencies.push_back(std::move(*parsed));
        }
        for (const auto& dependency : result.document.dependencies)
        {
            if (dependency.owner_entity.valid() &&
                !entity_ids.contains(dependency.owner_entity))
            {
                result.error = "dependency manifest owner entity is unresolved";
                return result;
            }
            if (dependency.required && !dependency.reference.guid.valid() &&
                dependency.reference.path_hint.empty())
            {
                result.error = "required dependency has neither GUID nor path hint";
                return result;
            }
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

document_store::document_store(std::size_t backup_generations)
    : backup_generations_(backup_generations)
{
}

document_save_result document_store::save_json(
    const std::filesystem::path& path,
    std::string_view unsealed_text,
    bool pretty) const
{
    document_save_result result;
    if (path.empty())
    {
        result.error = "document path is empty";
        return result;
    }
    const auto sealed = seal_json_document(unsealed_text, pretty);
    if (!sealed.succeeded())
    {
        result.error = sealed.error;
        return result;
    }
    std::error_code filesystem_error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), filesystem_error);
    if (filesystem_error)
    {
        result.error = "could not create document directory: " + filesystem_error.message();
        return result;
    }
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            result.error = "could not create temporary document";
            return result;
        }
        stream.write(sealed.text.data(), static_cast<std::streamsize>(sealed.text.size()));
        stream.flush();
        if (!stream)
        {
            result.error = "could not flush temporary document";
            return result;
        }
    }
    if (!flush_file_to_storage(temporary, result.error))
    {
        std::filesystem::remove(temporary, filesystem_error);
        return result;
    }
    const auto temporary_text = read_text(temporary);
    assets::asset_hash verified_hash;
    if (!temporary_text ||
        !verify_json_document(*temporary_text, &verified_hash, result.error) ||
        verified_hash != sealed.payload_hash)
    {
        std::filesystem::remove(temporary, filesystem_error);
        if (result.error.empty()) result.error = "temporary document verification failed";
        return result;
    }

    if (std::filesystem::exists(path, filesystem_error))
    {
        const auto current = read_text(path);
        std::string verification_error;
        if (current && verify_json_document(*current, nullptr, verification_error))
        {
            if (backup_generations_ > 0)
            {
                std::filesystem::remove(backup_path(path, backup_generations_), filesystem_error);
                for (std::size_t generation = backup_generations_; generation > 1; --generation)
                {
                    const auto previous = backup_path(path, generation - 1);
                    if (std::filesystem::exists(previous, filesystem_error))
                    {
                        std::filesystem::rename(previous, backup_path(path, generation), filesystem_error);
                        if (filesystem_error)
                        {
                            std::filesystem::remove(temporary, filesystem_error);
                            result.error = "could not rotate document backups";
                            return result;
                        }
                    }
                }
                std::filesystem::copy_file(
                    path, backup_path(path, 1),
                    std::filesystem::copy_options::overwrite_existing,
                    filesystem_error);
                if (filesystem_error)
                {
                    std::filesystem::remove(temporary, filesystem_error);
                    result.error = "could not create document backup";
                    return result;
                }
            }
        }
    }
    if (!atomic_replace(temporary, path, result.error))
    {
        std::filesystem::remove(temporary, filesystem_error);
        return result;
    }
    result.succeeded = true;
    result.payload_hash = sealed.payload_hash;
    return result;
}

document_load_result document_store::load_json(const std::filesystem::path& path) const
{
    document_load_result result;
    for (std::size_t candidate = 0; candidate <= backup_generations_; ++candidate)
    {
        const auto source = candidate == 0 ? path : backup_path(path, candidate);
        const auto text = read_text(source);
        if (!text) continue;
        std::string verification_error;
        assets::asset_hash hash;
        if (!verify_json_document(*text, &hash, verification_error))
        {
            result.diagnostics.push_back({
                "persistence.corruption",
                source.generic_string() + ": " + verification_error
            });
            continue;
        }
        const auto parsed = json::parse(*text, nullptr, false);
        if (!parsed.is_object())
            continue;
        result.succeeded = true;
        result.recovered = candidate != 0;
        result.integrity_verified = parsed.contains("integrity");
        result.source_path = source;
        result.text = *text;
        if (!result.integrity_verified)
            result.diagnostics.push_back({
                "persistence.integrity",
                "Document has no integrity metadata and will be sealed on its next save"
            });
        if (result.recovered)
            result.diagnostics.push_back({
                "persistence.recovery",
                "Recovered the document from " + source.filename().generic_string()
            });
        return result;
    }
    result.error = "no valid primary document or backup could be loaded";
    return result;
}

std::uint32_t crc32c(std::span<const std::byte> bytes) noexcept
{
    std::uint32_t result = 0xffffffffu;
    for (const auto byte : bytes)
    {
        result ^= std::to_integer<std::uint8_t>(byte);
        for (std::uint32_t bit = 0; bit < 8; ++bit)
            result = (result >> 1u) ^ (0x82f63b78u & (0u - (result & 1u)));
    }
    return ~result;
}

} // namespace arc::persistence

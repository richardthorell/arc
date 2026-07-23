#pragma once

#include <arc/ecs/identity.h>
#include <arc/ecs/reflection.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace arc::ecs
{

/** Format-neutral component archive used by JSON, binary, prefab, and save adapters. */
class component_archive_writer
{
public:
    virtual ~component_archive_writer() = default;
    virtual bool begin_component(const component_descriptor& descriptor) = 0;
    virtual bool write_boolean(component_field_id field, bool value) = 0;
    virtual bool write_signed(component_field_id field, std::int64_t value) = 0;
    virtual bool write_unsigned(component_field_id field, std::uint64_t value) = 0;
    virtual bool write_floating(component_field_id field, double value) = 0;
    virtual bool write_string(component_field_id field, std::string_view value) = 0;
    virtual bool write_bytes(component_field_id field, std::span<const std::byte> value) = 0;
    virtual bool write_entity(component_field_id field, entity_guid value) = 0;
    virtual bool end_component() = 0;
};

class component_archive_reader
{
public:
    virtual ~component_archive_reader() = default;
    virtual std::uint32_t schema_version() const noexcept = 0;
    virtual bool read_boolean(component_field_id field, bool& value) const = 0;
    virtual bool read_signed(component_field_id field, std::int64_t& value) const = 0;
    virtual bool read_unsigned(component_field_id field, std::uint64_t& value) const = 0;
    virtual bool read_floating(component_field_id field, double& value) const = 0;
    virtual bool read_string(component_field_id field, std::string_view& value) const = 0;
    virtual bool read_bytes(component_field_id field, std::span<const std::byte>& value) const = 0;
    virtual bool read_entity(component_field_id field, entity_guid& value) const = 0;
};

struct component_archive_hooks
{
    using write_function = bool (*)(const void*, component_archive_writer&);
    using read_function = bool (*)(void*, const component_archive_reader&);
    using migrate_function = bool (*)(std::uint32_t, component_archive_reader&, component_archive_writer&);

    write_function write{};
    read_function read{};
    migrate_function migrate{};
};

} // namespace arc::ecs

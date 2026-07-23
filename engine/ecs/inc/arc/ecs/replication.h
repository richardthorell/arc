#pragma once

#include <arc/ecs/world.h>

#include <span>

namespace arc::ecs
{

struct replication_change
{
    entity value{};
    component_type_id component{};
    change_revision revision{};
    std::uint64_t fields{};
    bool removed{};
};

class replication_sink
{
public:
    virtual ~replication_sink() = default;
    virtual bool begin_entity(entity_guid guid) = 0;
    virtual bool write_component(const replication_change& change, const void* component) = 0;
    virtual bool end_entity() = 0;
};

struct replication_hooks
{
    using encode_function = bool (*)(const void*, std::uint64_t fields, replication_sink&);
    using decode_function = bool (*)(void*, std::span<const std::byte>);
    encode_function encode{};
    decode_function decode{};
};

} // namespace arc::ecs

#pragma once

#include <cstdint>
#include <string_view>

namespace arc::network
{

using connection_id = std::uint64_t;
using replication_id = std::uint64_t;
inline constexpr connection_id invalid_connection_id = 0;
inline constexpr replication_id invalid_replication_id = 0;

enum class authority : std::uint8_t
{
    server,
    owning_client,
    shared
};

enum class delivery : std::uint8_t
{
    unreliable,
    reliable,
    reliable_ordered
};

struct transport_endpoint
{
    std::string_view address{};
    std::uint16_t port = 0;
};

struct replication_schema
{
    std::uint64_t type_id = 0;
    std::uint32_t version = 0;
    authority owner = authority::server;
};

struct replication_object
{
    replication_id id = invalid_replication_id;
    std::uint64_t type_id = 0;
    std::uint32_t schema_version = 0;
    connection_id owning_connection = invalid_connection_id;
};

struct state_update
{
    replication_id object = invalid_replication_id;
    std::uint32_t schema_version = 0;
    std::uint64_t sequence = 0;
};

struct event_header
{
    replication_id object = invalid_replication_id;
    std::uint64_t event_type = 0;
    std::uint32_t schema_version = 0;
    delivery mode = delivery::reliable_ordered;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_endpoint,
    invalid_schema,
    invalid_object,
    schema_mismatch,
    invalid_event
};

[[nodiscard]] validation_error validate(const transport_endpoint& endpoint) noexcept;
[[nodiscard]] validation_error validate(const replication_schema& schema) noexcept;
[[nodiscard]] validation_error validate(const replication_object& object, const replication_schema& schema) noexcept;
[[nodiscard]] validation_error validate(const state_update& update, const replication_schema& schema) noexcept;
[[nodiscard]] validation_error validate(const event_header& event, const replication_schema& schema) noexcept;

} // namespace arc::network

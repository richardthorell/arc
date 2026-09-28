#include <arc/network/network.h>

namespace arc::network
{

validation_error validate(const transport_endpoint& endpoint) noexcept
{
    if (endpoint.address.empty() || endpoint.port == 0) return validation_error::invalid_endpoint;
    return validation_error::none;
}

validation_error validate(const replication_schema& schema) noexcept
{
    if (schema.type_id == 0 || schema.version == 0) return validation_error::invalid_schema;
    return validation_error::none;
}

validation_error validate(const replication_object& object, const replication_schema& schema) noexcept
{
    if (validate(schema) != validation_error::none) return validation_error::invalid_schema;
    if (object.id == invalid_replication_id || object.type_id == 0) return validation_error::invalid_object;
    if (object.type_id != schema.type_id || object.schema_version != schema.version)
        return validation_error::schema_mismatch;
    if (schema.owner == authority::owning_client && object.owning_connection == invalid_connection_id)
        return validation_error::invalid_object;
    return validation_error::none;
}

validation_error validate(const state_update& update, const replication_schema& schema) noexcept
{
    if (validate(schema) != validation_error::none) return validation_error::invalid_schema;
    if (update.object == invalid_replication_id) return validation_error::invalid_object;
    if (update.schema_version != schema.version) return validation_error::schema_mismatch;
    return validation_error::none;
}

validation_error validate(const event_header& event, const replication_schema& schema) noexcept
{
    if (validate(schema) != validation_error::none) return validation_error::invalid_schema;
    if (event.object == invalid_replication_id || event.event_type == 0) return validation_error::invalid_event;
    if (event.schema_version != schema.version) return validation_error::schema_mismatch;
    return validation_error::none;
}

} // namespace arc::network

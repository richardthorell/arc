#include <arc/network/network.h>

int main()
{
    using namespace arc::network;

    transport_endpoint endpoint{"127.0.0.1", 7777};
    if (validate(endpoint) != validation_error::none) return 1;
    endpoint.port = 0;
    if (validate(endpoint) != validation_error::invalid_endpoint) return 2;

    replication_schema schema{42, 1, authority::server};
    if (validate(schema) != validation_error::none) return 3;

    replication_object object{1, 42, 1, invalid_connection_id};
    if (validate(object, schema) != validation_error::none) return 4;
    object.schema_version = 2;
    if (validate(object, schema) != validation_error::schema_mismatch) return 5;

    object.schema_version = 1;
    schema.owner = authority::owning_client;
    if (validate(object, schema) != validation_error::invalid_object) return 6;
    object.owning_connection = 7;
    if (validate(object, schema) != validation_error::none) return 7;

    state_update update{1, 1, 12};
    if (validate(update, schema) != validation_error::none) return 8;
    update.schema_version = 2;
    if (validate(update, schema) != validation_error::schema_mismatch) return 9;

    event_header event{1, 9, 1, delivery::unreliable};
    if (validate(event, schema) != validation_error::none) return 10;
    event.event_type = 0;
    if (validate(event, schema) != validation_error::invalid_event) return 11;

    return 0;
}

#include <arc/audio/audio.h>

int main()
{
    using namespace arc::audio;

    if (validate(stream_format{}) != validation_error::none)
    {
        return 1;
    }

    stream_format invalid_format{};
    invalid_format.channel_count = 0;
    if (validate(invalid_format) != validation_error::invalid_stream_format)
    {
        return 2;
    }

    if (validate(source_definition{}) != validation_error::none)
    {
        return 3;
    }

    source_definition invalid_source{};
    invalid_source.pitch = 0.0F;
    if (validate(invalid_source) != validation_error::invalid_source)
    {
        return 4;
    }

    source_definition spatial_source{};
    spatial_source.spatialized = true;
    spatial_source.min_distance = 10.0F;
    spatial_source.max_distance = 5.0F;
    if (validate(spatial_source) != validation_error::invalid_source)
    {
        return 5;
    }

    if (validate(listener_state{}) != validation_error::none)
    {
        return 6;
    }

    listener_state invalid_listener{};
    invalid_listener.forward = {0.0F, 1.0F, 0.0F};
    invalid_listener.up = {0.0F, 1.0F, 0.0F};
    if (validate(invalid_listener) != validation_error::invalid_listener)
    {
        return 7;
    }

    bus_definition master{};
    if (validate(master) != validation_error::none)
    {
        return 8;
    }

    bus_definition child{};
    child.id = 2;
    child.parent = master_bus_id;
    child.name = "Music";
    if (validate(child) != validation_error::none)
    {
        return 9;
    }

    child.parent = child.id;
    if (validate(child) != validation_error::invalid_bus)
    {
        return 10;
    }

    return 0;
}

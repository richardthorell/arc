#include <arc/audio/audio.h>

#include <cassert>
#include <limits>

int main()
{
    using namespace arc::audio;

    assert(validate(stream_format{}) == validation_error::none);
    stream_format invalid_format{};
    invalid_format.channel_count = 0;
    assert(validate(invalid_format) == validation_error::invalid_stream_format);

    assert(validate(source_definition{}) == validation_error::none);
    source_definition invalid_source{};
    invalid_source.pitch = 0.0F;
    assert(validate(invalid_source) == validation_error::invalid_source);

    source_definition spatial_source{};
    spatial_source.spatialized = true;
    spatial_source.min_distance = 10.0F;
    spatial_source.max_distance = 5.0F;
    assert(validate(spatial_source) == validation_error::invalid_source);

    assert(validate(listener_state{}) == validation_error::none);
    listener_state invalid_listener{};
    invalid_listener.forward = {0.0F, 1.0F, 0.0F};
    invalid_listener.up = {0.0F, 1.0F, 0.0F};
    assert(validate(invalid_listener) == validation_error::invalid_listener);

    bus_definition master{};
    assert(validate(master) == validation_error::none);
    bus_definition child{};
    child.id = 2;
    child.parent = master_bus_id;
    child.name = "Music";
    assert(validate(child) == validation_error::none);
    child.parent = child.id;
    assert(validate(child) == validation_error::invalid_bus);

    return 0;
}

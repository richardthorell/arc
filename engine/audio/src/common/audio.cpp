#include <arc/audio/audio.h>

#include <cmath>

namespace arc::audio
{
namespace
{

bool finite_vector(const math::vector<float, 3>& value) noexcept
{
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

float length_squared(const math::vector<float, 3>& value) noexcept
{
    return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
}

} // namespace

validation_error validate(const stream_format& format) noexcept
{
    if (format.sample_rate < 8000 || format.sample_rate > 384000 || format.channel_count == 0 ||
        format.channel_count > 32)
        return validation_error::invalid_stream_format;
    return validation_error::none;
}

validation_error validate(const source_definition& definition) noexcept
{
    if (definition.bus == 0 || !std::isfinite(definition.gain) || definition.gain < 0.0F ||
        !std::isfinite(definition.pitch) || definition.pitch <= 0.0F)
        return validation_error::invalid_source;

    if (definition.spatialized &&
        (!std::isfinite(definition.min_distance) || !std::isfinite(definition.max_distance) ||
         definition.min_distance < 0.0F || definition.max_distance <= definition.min_distance))
        return validation_error::invalid_source;

    return validation_error::none;
}

validation_error validate(const listener_state& listener) noexcept
{
    if (!finite_vector(listener.position) || !finite_vector(listener.forward) || !finite_vector(listener.up) ||
        !finite_vector(listener.velocity) || length_squared(listener.forward) <= 0.0F ||
        length_squared(listener.up) <= 0.0F)
        return validation_error::invalid_listener;

    const float alignment = listener.forward[0] * listener.up[0] + listener.forward[1] * listener.up[1] +
                            listener.forward[2] * listener.up[2];
    const float denominator = std::sqrt(length_squared(listener.forward) * length_squared(listener.up));
    if (!std::isfinite(denominator) || denominator <= 0.0F || std::abs(alignment / denominator) > 0.999F)
        return validation_error::invalid_listener;

    return validation_error::none;
}

validation_error validate(const bus_definition& bus) noexcept
{
    if (bus.id == 0 || bus.name.empty() || !std::isfinite(bus.gain) || bus.gain < 0.0F || bus.parent == bus.id)
        return validation_error::invalid_bus;
    if (bus.id == master_bus_id && bus.parent != 0) return validation_error::invalid_bus;
    return validation_error::none;
}

} // namespace arc::audio

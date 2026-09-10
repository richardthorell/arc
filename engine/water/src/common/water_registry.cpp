#include <arc/water/water_registry.h>

#include <algorithm>
#include <cmath>

namespace arc::water
{

bool water_body_bounds::valid() const noexcept
{
    return !bounded || (std::isfinite(minimum_x) && std::isfinite(minimum_z) && std::isfinite(maximum_x) &&
                        std::isfinite(maximum_z) && minimum_x <= maximum_x && minimum_z <= maximum_z);
}

bool water_body_bounds::contains(float world_x, float world_z) const noexcept
{
    return !bounded ||
           (valid() && world_x >= minimum_x && world_x <= maximum_x && world_z >= minimum_z && world_z <= maximum_z);
}

float water_body_bounds::area() const noexcept
{
    return bounded && valid() ? (maximum_x - minimum_x) * (maximum_z - minimum_z)
                              : std::numeric_limits<float>::infinity();
}

water_body_handle water_registry::add(water_body_descriptor descriptor)
{
    const auto available =
        std::find_if(slots_.begin(), slots_.end(), [](const slot& value) { return !value.occupied; });
    if (available != slots_.end())
    {
        available->descriptor = std::move(descriptor);
        available->occupied = true;
        ++size_;
        return {static_cast<std::uint32_t>(std::distance(slots_.begin(), available)), available->generation};
    }

    slots_.push_back({std::move(descriptor), 1, true});
    ++size_;
    return {static_cast<std::uint32_t>(slots_.size() - 1), 1};
}

bool water_registry::update(water_body_handle handle, water_body_descriptor descriptor)
{
    if (!alive(handle)) return false;
    slots_[handle.index].descriptor = std::move(descriptor);
    return true;
}

bool water_registry::remove(water_body_handle handle) noexcept
{
    if (!alive(handle)) return false;
    auto& value = slots_[handle.index];
    value.occupied = false;
    value.descriptor = {};
    ++value.generation;
    if (value.generation == 0) value.generation = 1;
    --size_;
    return true;
}

bool water_registry::alive(water_body_handle handle) const noexcept
{
    return handle.valid() && handle.index < slots_.size() && slots_[handle.index].occupied &&
           slots_[handle.index].generation == handle.generation;
}

const water_body_descriptor* water_registry::get(water_body_handle handle) const noexcept
{
    return alive(handle) ? &slots_[handle.index].descriptor : nullptr;
}

std::size_t water_registry::size() const noexcept
{
    return size_;
}

std::optional<water_body_handle> water_registry::resolve(float world_x, float world_z) const noexcept
{
    std::optional<water_body_handle> best;
    for (std::uint32_t index = 0; index < slots_.size(); ++index)
    {
        const auto& candidate = slots_[index];
        if (!candidate.occupied || !candidate.descriptor.enabled || !candidate.descriptor.queries_enabled ||
            !candidate.descriptor.bounds.contains(world_x, world_z))
            continue;

        if (!best)
        {
            best = water_body_handle{index, candidate.generation};
            continue;
        }

        const auto& current = slots_[best->index].descriptor;
        const auto& proposed = candidate.descriptor;
        const bool better_priority = proposed.priority > current.priority;
        const bool equal_priority = proposed.priority == current.priority;
        const bool better_ownership = proposed.bounds.bounded != current.bounds.bounded
                                          ? proposed.bounds.bounded
                                          : proposed.bounds.area() < current.bounds.area();
        if (better_priority || (equal_priority && better_ownership))
            best = water_body_handle{index, candidate.generation};
    }
    return best;
}

std::optional<water_surface_sample> water_registry::sample(float world_x, float world_z) const noexcept
{
    const auto handle = resolve(world_x, world_z);
    if (!handle) return std::nullopt;
    return water_surface_sample{.body = *handle, .height = slots_[handle->index].descriptor.water_level};
}

} // namespace arc::water

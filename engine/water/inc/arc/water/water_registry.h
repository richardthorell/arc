#pragma once

#include <arc/water/water_types.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace arc::water
{

/** @brief Stable runtime registry and deterministic query router for active Water bodies. */
class water_registry
{
public:
    [[nodiscard]] water_body_handle add(water_body_descriptor descriptor);
    bool update(water_body_handle handle, water_body_descriptor descriptor);
    bool remove(water_body_handle handle) noexcept;

    [[nodiscard]] bool alive(water_body_handle handle) const noexcept;
    [[nodiscard]] const water_body_descriptor* get(water_body_handle handle) const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;

    /** @brief Resolve priority first, then smallest bounded body, then Ocean fallback. */
    [[nodiscard]] std::optional<water_body_handle> resolve(float world_x, float world_z) const noexcept;

    /** @brief W0 flat-surface query contract. Later simulation can replace values without changing callers. */
    [[nodiscard]] std::optional<water_surface_sample> sample(float world_x, float world_z) const noexcept;

private:
    struct slot
    {
        water_body_descriptor descriptor;
        std::uint32_t generation{1};
        bool occupied{};
    };

    std::vector<slot> slots_;
    std::size_t size_{};
};

} // namespace arc::water

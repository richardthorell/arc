#pragma once

#include <arc/water/water_registry.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>

namespace arc::water
{

/** @brief One gameplay query. An invalid body handle requests registry routing. */
struct water_query
{
    math::vector3f position{};
    water_body_handle body{};
};

/** @brief CPU spectral and foam-history budget for one authored quality level. */
struct water_query_profile
{
    std::uint32_t maximum_frequency_index{20u};
    std::uint32_t foam_history_samples{3u};
    float foam_history_interval_seconds{0.25f};
    std::uint32_t maximum_debug_samples{64u};

    friend constexpr bool operator==(const water_query_profile&, const water_query_profile&) noexcept = default;
};

struct water_query_statistics
{
    std::uint64_t query_count{};
    std::uint64_t batch_count{};
    std::uint64_t cache_rebuild_count{};
    std::uint64_t spectral_mode_evaluation_count{};
    std::uint32_t cached_body_count{};
    std::uint32_t last_batch_query_count{};
    double last_batch_milliseconds{};
};

struct water_query_debug_sample
{
    water_query query;
    std::optional<water_surface_sample> result;
};

/** @brief Resolve the bounded CPU-query budget for an authored quality level. */
[[nodiscard]] water_query_profile query_profile(water_quality quality) noexcept;

/**
 * @brief CPU-queryable Water surface derived from the same deterministic spectra as rendering.
 *
 * Initial spectra are cached per stable body/settings signature. Routine samples evaluate a compact set of spectral
 * modes directly and never read back renderer resources.
 */
class water_query_system
{
public:
    water_query_system();
    ~water_query_system();

    water_query_system(water_query_system&&) noexcept;
    water_query_system& operator=(water_query_system&&) noexcept;
    water_query_system(const water_query_system&) = delete;
    water_query_system& operator=(const water_query_system&) = delete;

    /** @brief Build or refresh the cached CPU spectrum for a body before latency-sensitive queries. */
    bool prepare(const water_registry& registry, water_body_handle body);

    [[nodiscard]] std::optional<water_surface_sample>
    sample(const water_registry& registry, const math::vector3f& world_position, float simulation_time_seconds);

    [[nodiscard]] std::optional<water_surface_sample> sample(const water_registry& registry, water_body_handle body,
                                                             const math::vector3f& world_position,
                                                             float simulation_time_seconds);

    /** @brief Sample a batch without per-query allocation. Unresolved entries receive an invalid body handle. */
    std::size_t sample(const water_registry& registry, std::span<const water_query> queries,
                       std::span<water_surface_sample> results, float simulation_time_seconds);

    [[nodiscard]] const water_query_statistics& statistics() const noexcept;
    void reset_statistics() noexcept;
    [[nodiscard]] std::span<const water_query_debug_sample> debug_samples() const noexcept;
    void clear() noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> state_;
};

} // namespace arc::water

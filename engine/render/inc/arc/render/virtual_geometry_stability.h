#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace arc::render
{

/** @brief Reasons previous-frame occlusion data must not reject geometry this frame. */
enum class virtual_geometry_history_invalidation : std::uint32_t
{
    none = 0u,
    camera_cut = 1u << 0u,
    teleport = 1u << 1u,
    viewport_resize = 1u << 2u,
    projection_change = 1u << 3u,
    newly_visible_instance = 1u << 4u,
    world_reset = 1u << 5u
};

[[nodiscard]] constexpr virtual_geometry_history_invalidation
operator|(virtual_geometry_history_invalidation lhs, virtual_geometry_history_invalidation rhs) noexcept
{
    return static_cast<virtual_geometry_history_invalidation>(static_cast<std::uint32_t>(lhs) |
                                                              static_cast<std::uint32_t>(rhs));
}

constexpr virtual_geometry_history_invalidation& operator|=(virtual_geometry_history_invalidation& lhs,
                                                            virtual_geometry_history_invalidation rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

[[nodiscard]] constexpr bool contains(virtual_geometry_history_invalidation value,
                                      virtual_geometry_history_invalidation requested) noexcept
{
    return (static_cast<std::uint32_t>(value) & static_cast<std::uint32_t>(requested)) != 0u;
}

/** @brief Which depth history a virtual-geometry traversal phase may consume. */
enum class virtual_geometry_traversal_phase : std::uint8_t
{
    single_phase,
    previous_hzb,
    current_hzb
};

/** @brief Backend-neutral policy for temporal HZB use and projected-error hysteresis. */
struct virtual_geometry_traversal_stability
{
    float refine_hysteresis{0.10f};
    float coarsen_hysteresis{0.10f};
};

/** @brief Generation-safe identity for one hierarchy refinement decision. */
struct virtual_geometry_refinement_key
{
    std::uint32_t instance_index{};
    std::uint32_t instance_generation{};
    std::uint32_t resource_generation{};
    std::uint32_t hierarchy_node{};

    bool operator==(const virtual_geometry_refinement_key&) const noexcept = default;
};

/** @brief Hash mirrored by GPU refinement-history lookup. */
[[nodiscard]] std::uint32_t virtual_geometry_refinement_key_hash(virtual_geometry_refinement_key key) noexcept;

/** @brief Bounded double-buffered CPU reference for temporal refinement history. */
class virtual_geometry_refinement_history
{
public:
    explicit virtual_geometry_refinement_history(std::uint32_t capacity = 1024u);

    void begin_frame();
    [[nodiscard]] bool refined_last_frame(virtual_geometry_refinement_key key) const noexcept;
    [[nodiscard]] bool record(virtual_geometry_refinement_key key) noexcept;
    [[nodiscard]] bool previous_overflowed() const noexcept;
    [[nodiscard]] bool current_overflowed() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;

private:
    struct slot
    {
        virtual_geometry_refinement_key key{};
        bool occupied{};
    };

    std::array<std::vector<slot>, 2> generations_;
    std::array<bool, 2> overflowed_{};
    std::uint32_t current_generation_{1u};
    bool frame_started_{};
};

/** @brief Returns true when previous-frame HZB rejection is safe for the current view/instance. */
[[nodiscard]] bool virtual_geometry_history_valid(virtual_geometry_history_invalidation invalidation) noexcept;

/**
 * @brief Stable projected-error refinement decision shared by CPU reference and GPU implementations.
 * @param projected_error Current projected geometric error in pixels.
 * @param threshold Nominal refinement threshold in pixels.
 * @param refined_last_frame Whether this node was refined in the previous valid-history frame.
 * @param policy Hysteresis widths expressed as fractions of the nominal threshold.
 */
[[nodiscard]] bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                                  virtual_geometry_traversal_stability policy = {}) noexcept;

/** @brief Use hysteresis when history is valid, otherwise use the nominal threshold. */
[[nodiscard]] bool should_refine_virtual_geometry(float projected_error, float threshold, bool refined_last_frame,
                                                  bool history_valid,
                                                  virtual_geometry_traversal_stability policy = {}) noexcept;

} // namespace arc::render

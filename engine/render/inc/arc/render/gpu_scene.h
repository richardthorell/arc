#pragma once

#include <arc/geometric/box.h>
#include <arc/math/matrix.h>
#include <arc/math/vector.h>
#include <arc/render/handles.h>
#include <arc/render/virtual_mesh.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace arc::render
{

struct render_world_packet;

/** @brief Geometry representation referenced by one persistent GPU Scene instance. */
enum class gpu_scene_geometry_kind : std::uint8_t
{
    /** Ordinary indexed mesh. */
    mesh,
    /** One cluster from a virtual-mesh resource. */
    virtual_mesh_cluster
};

/** @brief Stable generational index into renderer-owned GPU Scene storage. */
struct gpu_scene_instance_handle
{
    /** Stable slot index, or the invalid resource index. */
    std::uint32_t index{resource_handle::invalid_index};
    /** Generation incremented whenever the slot is recycled. */
    std::uint32_t generation{};

    /** @return `true` when the handle names a slot. */
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != resource_handle::invalid_index;
    }

    friend constexpr bool operator==(gpu_scene_instance_handle, gpu_scene_instance_handle) noexcept = default;
};

/** @brief Persistent, backend-neutral representation of one renderable scene instance. */
struct gpu_scene_instance
{
    /** Current local-to-world transform. */
    math::matrix4f model{math::identity<float, 4>()};
    /** Previous-frame local-to-world transform used for motion. */
    math::matrix4f previous_model{math::identity<float, 4>()};
    /** Conservative world-space visibility bounds. */
    geometric::box3f world_bounds{};
    /** Ordinary mesh reference when @ref geometry_kind is `mesh`. */
    mesh_handle mesh{};
    /** Virtual mesh reference when @ref geometry_kind is `virtual_mesh_cluster`. */
    virtual_mesh_handle virtual_mesh{};
    /** Material referenced by the instance. */
    material_handle material{};
    /** Stable per-frame ObjectID used by editor picking. */
    render_object_id object_id{};
    /** Mesh subresource or virtual cluster index. */
    std::uint32_t submesh_or_cluster{};
    /** View layer visibility mask. */
    std::uint32_t render_layer_mask{1u};
    /** Backend-neutral packed visibility and rendering flags. */
    std::uint32_t flags{};
    /** Maximum camera distance in metres; zero disables distance culling. */
    float maximum_draw_distance{};
    /** Authored multiplier applied to geometry error thresholds. */
    float geometry_error_scale{1.0f};
    /** Monotonic instance content revision. */
    std::uint64_t revision{};
    /** Geometry representation selected for this slot. */
    gpu_scene_geometry_kind geometry_kind{gpu_scene_geometry_kind::mesh};
};

/** @brief Dirty fields carried by an incremental GPU Scene update. */
enum class gpu_scene_dirty : std::uint32_t
{
    none = 0,
    transform = 1u << 0u,
    bounds = 1u << 1u,
    geometry = 1u << 2u,
    material = 1u << 3u,
    flags = 1u << 4u,
    all = 0xffffffffu
};

[[nodiscard]] constexpr gpu_scene_dirty operator|(gpu_scene_dirty lhs, gpu_scene_dirty rhs) noexcept
{
    return static_cast<gpu_scene_dirty>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

/** @brief Incremental operation applied to backend GPU Scene buffers. */
enum class gpu_scene_update_kind : std::uint8_t
{
    reset,
    upsert,
    destroy
};

/** @brief One incremental persistent GPU Scene mutation. */
struct gpu_scene_update
{
    gpu_scene_update_kind kind{gpu_scene_update_kind::upsert};
    gpu_scene_instance_handle handle{};
    gpu_scene_dirty dirty{gpu_scene_dirty::all};
    gpu_scene_instance instance{};
};

/** @brief Complete set of GPU Scene mutations generated for one submitted frame. */
struct gpu_scene_update_batch
{
    std::uint64_t frame_index{};
    std::uint64_t world_id{};
    std::uint64_t world_epoch{};
    std::uint32_t active_instance_count{};
    std::uint32_t capacity{};
    std::vector<gpu_scene_update> updates;
};

/** @brief CPU-side authority that assigns stable slots and emits dirty GPU Scene ranges. */
class gpu_scene
{
public:
    /**
     * @brief Reconcile one extracted world packet with persistent scene slots.
     * @param packet Mutable packet that receives stable instance handles.
     * @param frame_index Monotonic renderer frame index.
     * @return Incremental backend update batch for the frame.
     */
    [[nodiscard]] gpu_scene_update_batch synchronize(render_world_packet& packet, std::uint64_t frame_index);
    /** @brief Release all persistent slots and world epochs. */
    void reset();

    /** @return Borrowed instance data, invalidated by the next synchronization or reset. */
    [[nodiscard]] const gpu_scene_instance* find(gpu_scene_instance_handle handle) const noexcept;
    /** @return Number of currently live instance slots. */
    [[nodiscard]] std::uint32_t active_instance_count() const noexcept;
    /** @return Total allocated stable-slot capacity. */
    [[nodiscard]] std::uint32_t capacity() const noexcept;

private:
    struct instance_key
    {
        std::uint64_t world_id{};
        render_object_id object_id{};
        gpu_scene_geometry_kind geometry_kind{gpu_scene_geometry_kind::mesh};
        std::uint32_t submesh_or_cluster{};

        friend bool operator==(const instance_key&, const instance_key&) noexcept = default;
    };

    struct instance_key_hash
    {
        std::size_t operator()(const instance_key& value) const noexcept;
    };

    struct slot
    {
        gpu_scene_instance instance{};
        instance_key key{};
        std::uint64_t last_seen_frame{};
        std::uint64_t world_epoch{};
        std::uint32_t generation{1};
        bool alive{};
    };

    [[nodiscard]] gpu_scene_instance_handle allocate_slot();
    void destroy_slot(std::uint32_t index, gpu_scene_update_batch& batch);

    std::vector<slot> slots_;
    std::vector<std::uint32_t> free_slots_;
    std::unordered_map<instance_key, std::uint32_t, instance_key_hash> lookup_;
    std::unordered_map<std::uint64_t, std::uint64_t> world_epochs_;
    std::uint32_t active_instance_count_{};
};

} // namespace arc::render

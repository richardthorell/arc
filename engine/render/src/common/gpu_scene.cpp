#include <arc/render/gpu_scene.h>

#include <arc/render/render_world.h>

#include <algorithm>
#include <bit>

namespace arc::render
{
namespace
{

template <class T> void hash_combine(std::size_t& seed, const T& value) noexcept
{
    seed ^= std::hash<T>{}(value) + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
}

bool matrices_equal(const math::matrix4f& lhs, const math::matrix4f& rhs) noexcept
{
    for (std::size_t row = 0; row < 4; ++row)
    {
        for (std::size_t column = 0; column < 4; ++column)
        {
            if (std::bit_cast<std::uint32_t>(lhs(row, column)) != std::bit_cast<std::uint32_t>(rhs(row, column)))
                return false;
        }
    }
    return true;
}

bool bounds_equal(const geometric::box3f& lhs, const geometric::box3f& rhs) noexcept
{
    for (std::size_t component = 0; component < 3; ++component)
    {
        if (std::bit_cast<std::uint32_t>(lhs.min[component]) != std::bit_cast<std::uint32_t>(rhs.min[component]) ||
            std::bit_cast<std::uint32_t>(lhs.max[component]) != std::bit_cast<std::uint32_t>(rhs.max[component]))
            return false;
    }
    return true;
}

gpu_scene_dirty changed_fields(const gpu_scene_instance& previous, const gpu_scene_instance& current) noexcept
{
    auto dirty = gpu_scene_dirty::none;
    if (!matrices_equal(previous.model, current.model) ||
        !matrices_equal(previous.previous_model, current.previous_model))
        dirty = dirty | gpu_scene_dirty::transform;
    if (!bounds_equal(previous.world_bounds, current.world_bounds) ||
        previous.maximum_draw_distance != current.maximum_draw_distance ||
        previous.geometry_error_scale != current.geometry_error_scale)
        dirty = dirty | gpu_scene_dirty::bounds;
    if (previous.mesh != current.mesh || previous.virtual_mesh != current.virtual_mesh ||
        previous.terrain != current.terrain || previous.skin_palette != current.skin_palette ||
        previous.skin_joint_count != current.skin_joint_count ||
        previous.submesh_or_cluster != current.submesh_or_cluster || previous.geometry_kind != current.geometry_kind)
        dirty = dirty | gpu_scene_dirty::geometry;
    if (previous.material != current.material) dirty = dirty | gpu_scene_dirty::material;
    if (previous.flags != current.flags || previous.render_layer_mask != current.render_layer_mask ||
        previous.object_id != current.object_id)
        dirty = dirty | gpu_scene_dirty::flags;
    return dirty;
}

gpu_scene_instance_flag instance_flags(bool visible, bool selected, bool transparent, bool casts_shadows,
                                       bool receives_shadows) noexcept
{
    auto result = gpu_scene_instance_flag::none;
    if (visible) result = result | gpu_scene_instance_flag::visible;
    if (selected) result = result | gpu_scene_instance_flag::selected;
    if (transparent) result = result | gpu_scene_instance_flag::transparent;
    if (casts_shadows) result = result | gpu_scene_instance_flag::casts_shadows;
    if (receives_shadows) result = result | gpu_scene_instance_flag::receives_shadows;
    return result;
}

} // namespace

std::size_t gpu_scene::instance_key_hash::operator()(const instance_key& value) const noexcept
{
    std::size_t seed{};
    hash_combine(seed, value.world_id);
    hash_combine(seed, value.object_id.index);
    hash_combine(seed, value.object_id.generation);
    hash_combine(seed, static_cast<std::uint8_t>(value.geometry_kind));
    hash_combine(seed, value.submesh_or_cluster);
    return seed;
}

gpu_scene_instance_handle gpu_scene::allocate_slot()
{
    std::uint32_t index{};
    if (!free_slots_.empty())
    {
        index = free_slots_.back();
        free_slots_.pop_back();
    }
    else
    {
        index = static_cast<std::uint32_t>(slots_.size());
        slots_.push_back({});
    }
    auto& target = slots_[index];
    target.alive = true;
    ++active_instance_count_;
    return {.index = index, .generation = target.generation};
}

void gpu_scene::destroy_slot(std::uint32_t index, std::uint64_t frame_index, gpu_scene_update_batch& batch)
{
    auto& target = slots_[index];
    if (!target.alive) return;
    batch.updates.push_back({.kind = gpu_scene_update_kind::destroy,
                             .handle = {.index = index, .generation = target.generation},
                             .dirty = gpu_scene_dirty::all});
    lookup_.erase(target.key);
    target.alive = false;
    ++target.generation;
    if (target.generation == 0) target.generation = 1;
    retired_slots_.push_back(
        {.index = index, .reuse_after_frame = frame_index + default_gpu_table_slot_reuse_delay_frames});
    --active_instance_count_;
}

void gpu_scene::release_retired_slots(std::uint64_t frame_index)
{
    const auto still_retired =
        std::stable_partition(retired_slots_.begin(), retired_slots_.end(),
                              [frame_index](const retired_slot& slot) { return slot.reuse_after_frame > frame_index; });
    for (auto iterator = still_retired; iterator != retired_slots_.end(); ++iterator)
        free_slots_.push_back(iterator->index);
    retired_slots_.erase(still_retired, retired_slots_.end());
}

gpu_scene_update_batch gpu_scene::synchronize(render_world_packet& packet, std::uint64_t frame_index)
{
    gpu_scene_update_batch batch{
        .frame_index = frame_index, .world_id = packet.gpu_scene_world_id, .world_epoch = packet.world_epoch};
    release_retired_slots(frame_index);
    const auto previous_epoch = world_epochs_.find(packet.gpu_scene_world_id);
    if (previous_epoch == world_epochs_.end() || previous_epoch->second != packet.world_epoch)
    {
        batch.updates.push_back({.kind = gpu_scene_update_kind::reset});
        for (std::uint32_t index = 0; index < slots_.size(); ++index)
        {
            if (slots_[index].alive && slots_[index].key.world_id == packet.gpu_scene_world_id)
                destroy_slot(index, frame_index, batch);
        }
        world_epochs_[packet.gpu_scene_world_id] = packet.world_epoch;
    }

    const auto upsert = [&](const instance_key& key, gpu_scene_instance instance)
    {
        const auto found = lookup_.find(key);
        const bool is_new = found == lookup_.end();
        gpu_scene_instance_handle handle{};
        gpu_scene_dirty dirty = gpu_scene_dirty::all;
        if (is_new)
        {
            handle = allocate_slot();
            lookup_.emplace(key, handle.index);
        }
        else
        {
            auto& existing = slots_[found->second];
            handle = {.index = found->second, .generation = existing.generation};
            const bool current_transform_changed = !matrices_equal(existing.instance.model, instance.model);
            instance.previous_model =
                existing.last_seen_frame == frame_index ? existing.instance.previous_model : existing.instance.model;
            dirty = changed_fields(existing.instance, instance);
            const auto spatial_dirty =
                static_cast<std::uint32_t>(dirty) & (static_cast<std::uint32_t>(gpu_scene_dirty::bounds) |
                                                     static_cast<std::uint32_t>(gpu_scene_dirty::geometry));
            if (current_transform_changed || spatial_dirty != 0u ||
                (existing.last_seen_frame == frame_index &&
                 contains(existing.instance.flags, gpu_scene_instance_flag::recently_changed)))
                instance.flags = instance.flags | gpu_scene_instance_flag::recently_changed;
            dirty = changed_fields(existing.instance, instance);
        }

        if (is_new)
        {
            instance.previous_model = instance.model;
            instance.flags = instance.flags | gpu_scene_instance_flag::recently_changed;
        }

        auto& target = slots_[handle.index];
        target.key = key;
        target.world_epoch = packet.world_epoch;
        target.last_seen_frame = frame_index;
        if (dirty != gpu_scene_dirty::none)
        {
            instance.revision = target.instance.revision + 1;
            target.instance = instance;
            batch.updates.push_back(
                {.kind = gpu_scene_update_kind::upsert, .handle = handle, .dirty = dirty, .instance = instance});
        }
    };

    for (auto& item : packet.items)
    {
        const auto geometry_kind =
            item.skin_matrices.valid() ? gpu_scene_geometry_kind::skinned_mesh : gpu_scene_geometry_kind::mesh;
        const instance_key key{.world_id = packet.gpu_scene_world_id,
                               .object_id = item.object_id,
                               .geometry_kind = geometry_kind,
                               .submesh_or_cluster = item.submesh};
        upsert(key, {.model = item.model,
                     .previous_model = item.previous_model,
                     .world_bounds = item.world_bounds,
                     .mesh = item.mesh,
                     .material = item.material,
                     .skin_palette = item.skin_matrices,
                     .skin_joint_count = item.skin_joint_count,
                     .object_id = item.object_id,
                     .submesh_or_cluster = item.submesh,
                     .render_layer_mask = item.render_layer_mask,
                     .flags = instance_flags(item.visible, item.selected, item.transparent, item.casts_shadows,
                                             item.receives_shadows),
                     .maximum_draw_distance = item.maximum_draw_distance,
                     .geometry_error_scale = item.geometry_error_scale,
                     .geometry_kind = geometry_kind});
        item.gpu_scene_instance = {.index = lookup_.at(key), .generation = slots_[lookup_.at(key)].generation};
    }

    for (auto& item : packet.terrains)
    {
        const instance_key key{.world_id = packet.gpu_scene_world_id,
                               .object_id = item.object_id,
                               .geometry_kind = gpu_scene_geometry_kind::terrain,
                               .submesh_or_cluster = 0u};
        upsert(key, {.model = item.model,
                     .previous_model = item.previous_model,
                     .world_bounds = item.world_bounds,
                     .terrain = item.terrain,
                     .material = item.material,
                     .object_id = item.object_id,
                     .render_layer_mask = item.render_layer_mask,
                     .flags = instance_flags(true, item.selected, false, item.cast_shadows, item.receive_shadows),
                     .geometry_error_scale = 1.0f,
                     .geometry_kind = gpu_scene_geometry_kind::terrain});
        item.gpu_scene_instance = {.index = lookup_.at(key), .generation = slots_[lookup_.at(key)].generation};
    }
    for (auto& item : packet.virtual_items)
    {
        const instance_key key{.world_id = packet.gpu_scene_world_id,
                               .object_id = item.object_id,
                               .geometry_kind = gpu_scene_geometry_kind::virtual_mesh,
                               .submesh_or_cluster = item.root_node};
        upsert(key,
               {.model = item.model,
                .previous_model = item.previous_model,
                .world_bounds = item.world_bounds,
                .virtual_mesh = item.mesh,
                .material = item.material,
                .object_id = item.object_id,
                .submesh_or_cluster = item.root_node,
                .render_layer_mask = item.render_layer_mask,
                .flags = instance_flags(item.visible, item.selected, false, item.casts_shadows, item.receives_shadows),
                .maximum_draw_distance = item.maximum_draw_distance,
                .geometry_error_scale = item.geometry_error_scale,
                .geometry_kind = gpu_scene_geometry_kind::virtual_mesh});
        item.gpu_scene_instance = {.index = lookup_.at(key), .generation = slots_[lookup_.at(key)].generation};
    }

    for (std::uint32_t index = 0; index < slots_.size(); ++index)
    {
        const auto& candidate = slots_[index];
        if (candidate.alive && candidate.key.world_id == packet.gpu_scene_world_id &&
            candidate.last_seen_frame != frame_index)
            destroy_slot(index, frame_index, batch);
    }

    batch.active_instance_count = active_instance_count_;
    batch.capacity = static_cast<std::uint32_t>(slots_.size());
    std::vector<std::uint32_t> dirty_slots;
    dirty_slots.reserve(batch.updates.size());
    for (const auto& update : batch.updates)
        if (update.kind != gpu_scene_update_kind::reset && update.handle.valid())
            dirty_slots.push_back(update.handle.index);
    batch.dirty_ranges = coalesce_gpu_table_dirty_ranges(dirty_slots);
    return batch;
}

void gpu_scene::reset()
{
    slots_.clear();
    free_slots_.clear();
    retired_slots_.clear();
    lookup_.clear();
    world_epochs_.clear();
    active_instance_count_ = 0;
}

const gpu_scene_instance* gpu_scene::find(gpu_scene_instance_handle handle) const noexcept
{
    if (!handle.valid() || handle.index >= slots_.size()) return nullptr;
    const auto& candidate = slots_[handle.index];
    return candidate.alive && candidate.generation == handle.generation ? &candidate.instance : nullptr;
}

std::uint32_t gpu_scene::active_instance_count() const noexcept
{
    return active_instance_count_;
}

std::uint32_t gpu_scene::capacity() const noexcept
{
    return static_cast<std::uint32_t>(slots_.size());
}

} // namespace arc::render

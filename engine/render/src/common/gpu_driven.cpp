#include <arc/render/gpu_driven.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace arc::render
{
namespace
{

std::uint64_t resource_key(resource_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) noexcept
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

} // namespace

gpu_resource_tables::gpu_resource_tables()
{
    tables_[0] = {.kind = gpu_resource_table_kind::geometry,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_geometry_table_record))};
    tables_[1] = {.kind = gpu_resource_table_kind::material,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_material_table_record))};
    tables_[2] = {.kind = gpu_resource_table_kind::texture,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_texture_table_record))};
    tables_[3] = {.kind = gpu_resource_table_kind::sampler,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_sampler_table_record))};
    tables_[4] = {.kind = gpu_resource_table_kind::skin_palette,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_skin_palette_table_record))};
    tables_[5] = {.kind = gpu_resource_table_kind::instance};
    tables_[6] = {.kind = gpu_resource_table_kind::visible_draw,
                  .stride = static_cast<std::uint32_t>(sizeof(gpu_draw_record))};
}

std::size_t gpu_resource_tables::table_offset(gpu_resource_table_kind table) noexcept
{
    return static_cast<std::size_t>(table);
}

gpu_table_update_batch gpu_resource_tables::publish_record(gpu_resource_table_kind table, resource_handle handle,
                                                           std::span<const std::byte> record, std::uint64_t frame_index)
{
    gpu_table_update_batch batch{.table = table,
                                 .reuse_after_frame = frame_index + default_gpu_table_slot_reuse_delay_frames};
    const auto offset = table_offset(table);
    if (!handle.valid() || handle.generation == 0u || offset >= tables_.size()) return batch;
    auto& state = tables_[offset];
    if (state.stride == 0u || record.size() != state.stride) return batch;

    if (handle.index >= state.slots.size())
    {
        const auto required = std::max(handle.index + 1u, 16u);
        state.slots.resize(std::bit_ceil(required));
        if (++state.generation == 0u) state.generation = 1u;
    }

    auto& slot = state.slots[handle.index];
    if (!slot.live)
    {
        ++state.live_entries;
        if (slot.generation != 0u && state.tombstones != 0u) --state.tombstones;
    }
    slot.live = true;
    slot.generation = handle.generation;

    batch.table_generation = state.generation;
    batch.capacity = static_cast<std::uint32_t>(state.slots.size());
    batch.element_stride = state.stride;
    batch.payload.assign(record.begin(), record.end());
    std::memcpy(batch.payload.data(), &handle.generation, sizeof(handle.generation));
    batch.updates.push_back({.table = table,
                             .kind = gpu_table_update_kind::upsert,
                             .slot = handle.index,
                             .generation = handle.generation,
                             .payload_size = state.stride});
    batch.dirty_ranges.push_back({.first = handle.index, .count = 1u});
    state.sparse_upload_bytes += state.stride;
    return batch;
}

gpu_resource_tables::heap_range gpu_resource_tables::allocate_heap_range(std::vector<std::byte>& heap,
                                                                         std::vector<heap_range>& free_ranges,
                                                                         std::uint64_t size, std::uint64_t alignment)
{
    if (size == 0u || alignment == 0u || (alignment & (alignment - 1u)) != 0u) return {};
    for (auto iterator = free_ranges.begin(); iterator != free_ranges.end(); ++iterator)
    {
        const auto aligned = align_up(iterator->offset, alignment);
        const auto prefix = aligned - iterator->offset;
        if (prefix > iterator->size || size > iterator->size - prefix) continue;
        const auto suffix_offset = aligned + size;
        const auto suffix = iterator->offset + iterator->size - suffix_offset;
        const auto original_offset = iterator->offset;
        free_ranges.erase(iterator);
        if (prefix != 0u) free_ranges.push_back({.offset = original_offset, .size = prefix});
        if (suffix != 0u) free_ranges.push_back({.offset = suffix_offset, .size = suffix});
        return {.offset = aligned, .size = size};
    }

    const auto offset = align_up(heap.size(), alignment);
    if (offset > std::numeric_limits<std::size_t>::max() || size > std::numeric_limits<std::size_t>::max() - offset)
        return {};
    heap.resize(static_cast<std::size_t>(offset + size));
    return {.offset = offset, .size = size};
}

void gpu_resource_tables::release_heap_range(std::vector<heap_range>& free_ranges, heap_range range)
{
    if (range.size == 0u) return;
    free_ranges.push_back(range);
    std::sort(free_ranges.begin(), free_ranges.end(),
              [](const heap_range& lhs, const heap_range& rhs) { return lhs.offset < rhs.offset; });
    std::vector<heap_range> merged;
    merged.reserve(free_ranges.size());
    for (const auto candidate : free_ranges)
    {
        if (!merged.empty() && merged.back().offset + merged.back().size >= candidate.offset)
            merged.back().size =
                std::max(merged.back().offset + merged.back().size, candidate.offset + candidate.size) -
                merged.back().offset;
        else
            merged.push_back(candidate);
    }
    free_ranges = std::move(merged);
}

gpu_table_update_batch gpu_resource_tables::publish_geometry(resource_handle handle,
                                                             std::span<const std::byte> vertices,
                                                             std::uint32_t vertex_stride,
                                                             std::span<const std::byte> indices,
                                                             std::uint32_t index_stride, std::uint64_t frame_index)
{
    if (!handle.valid() || vertex_stride == 0u || index_stride == 0u || vertices.empty() || indices.empty() ||
        vertices.size() % vertex_stride != 0u || indices.size() % index_stride != 0u)
        return {.table = gpu_resource_table_kind::geometry};

    const auto key = resource_key(handle);
    auto found = geometry_allocations_.find(key);
    if (found != geometry_allocations_.end() &&
        (found->second.vertices.size != vertices.size() || found->second.indices.size != indices.size()))
    {
        release_heap_range(free_vertex_ranges_, found->second.vertices);
        release_heap_range(free_index_ranges_, found->second.indices);
        live_vertex_bytes_ -= found->second.vertices.size;
        live_index_bytes_ -= found->second.indices.size;
        geometry_allocations_.erase(found);
        found = geometry_allocations_.end();
    }
    if (found == geometry_allocations_.end())
    {
        const auto vertex_range = allocate_heap_range(vertex_heap_, free_vertex_ranges_, vertices.size(), 16u);
        const auto index_range = allocate_heap_range(index_heap_, free_index_ranges_, indices.size(), 4u);
        if (vertex_range.size == 0u || index_range.size == 0u)
        {
            release_heap_range(free_vertex_ranges_, vertex_range);
            release_heap_range(free_index_ranges_, index_range);
            return {.table = gpu_resource_table_kind::geometry};
        }
        found = geometry_allocations_.emplace(key, geometry_allocation{vertex_range, index_range}).first;
        live_vertex_bytes_ += vertex_range.size;
        live_index_bytes_ += index_range.size;
    }

    const auto allocation = found->second;
    std::copy(vertices.begin(), vertices.end(),
              vertex_heap_.begin() + static_cast<std::ptrdiff_t>(allocation.vertices.offset));
    std::copy(indices.begin(), indices.end(),
              index_heap_.begin() + static_cast<std::ptrdiff_t>(allocation.indices.offset));

    gpu_geometry_table_record record{.generation = handle.generation,
                                     .vertex_offset = allocation.vertices.offset,
                                     .index_offset = allocation.indices.offset,
                                     .vertex_count = static_cast<std::uint32_t>(vertices.size() / vertex_stride),
                                     .index_count = static_cast<std::uint32_t>(indices.size() / index_stride),
                                     .vertex_stride = vertex_stride,
                                     .index_stride = index_stride};
    auto batch =
        publish_record(gpu_resource_table_kind::geometry, handle, std::as_bytes(std::span{&record, 1}), frame_index);
    if (batch.updates.empty()) return batch;
    if (++geometry_heap_generation_ == 0u) geometry_heap_generation_ = 1u;
    batch.geometry_heap_generation = geometry_heap_generation_;
    batch.vertex_heap_capacity = vertex_heap_.size();
    batch.index_heap_capacity = index_heap_.size();
    batch.heap_payload.reserve(vertices.size() + indices.size());
    batch.heap_updates.push_back({.destination_offset = allocation.vertices.offset,
                                  .payload_size = static_cast<std::uint32_t>(vertices.size())});
    batch.heap_payload.insert(batch.heap_payload.end(), vertices.begin(), vertices.end());
    batch.heap_updates.push_back({.index_heap = true,
                                  .destination_offset = allocation.indices.offset,
                                  .payload_offset = static_cast<std::uint32_t>(vertices.size()),
                                  .payload_size = static_cast<std::uint32_t>(indices.size())});
    batch.heap_payload.insert(batch.heap_payload.end(), indices.begin(), indices.end());
    return batch;
}

gpu_table_update_batch gpu_resource_tables::publish_material(resource_handle handle,
                                                             const gpu_material_table_record& record,
                                                             std::uint64_t frame_index)
{
    return publish_record(gpu_resource_table_kind::material, handle, std::as_bytes(std::span{&record, 1}), frame_index);
}

gpu_table_update_batch gpu_resource_tables::publish_texture(resource_handle handle,
                                                            const gpu_texture_table_record& record,
                                                            std::uint64_t frame_index)
{
    return publish_record(gpu_resource_table_kind::texture, handle, std::as_bytes(std::span{&record, 1}), frame_index);
}

gpu_table_update_batch gpu_resource_tables::publish_sampler(resource_handle handle,
                                                            const gpu_sampler_table_record& record,
                                                            std::uint64_t frame_index)
{
    return publish_record(gpu_resource_table_kind::sampler, handle, std::as_bytes(std::span{&record, 1}), frame_index);
}

gpu_table_update_batch gpu_resource_tables::publish_skin_palette(resource_handle handle,
                                                                 const gpu_skin_palette_table_record& record,
                                                                 std::uint64_t frame_index)
{
    return publish_record(gpu_resource_table_kind::skin_palette, handle, std::as_bytes(std::span{&record, 1}),
                          frame_index);
}

gpu_table_update_batch gpu_resource_tables::tombstone(gpu_resource_table_kind table, resource_handle handle,
                                                      std::uint64_t frame_index)
{
    gpu_table_update_batch batch{.table = table,
                                 .reuse_after_frame = frame_index + default_gpu_table_slot_reuse_delay_frames};
    const auto offset = table_offset(table);
    if (!handle.valid() || offset >= tables_.size()) return batch;
    auto& state = tables_[offset];
    batch.table_generation = state.generation;
    batch.capacity = static_cast<std::uint32_t>(state.slots.size());
    batch.element_stride = state.stride;
    if (handle.index >= state.slots.size()) return batch;
    auto& slot = state.slots[handle.index];
    if (!slot.live || slot.generation != handle.generation) return batch;
    slot.live = false;
    --state.live_entries;
    ++state.tombstones;
    state.sparse_upload_bytes += state.stride;
    batch.updates.push_back({.table = table,
                             .kind = gpu_table_update_kind::tombstone,
                             .slot = handle.index,
                             .generation = handle.generation});
    batch.dirty_ranges.push_back({.first = handle.index, .count = 1u});

    if (table == gpu_resource_table_kind::geometry)
    {
        const auto found = geometry_allocations_.find(resource_key(handle));
        if (found != geometry_allocations_.end())
        {
            release_heap_range(free_vertex_ranges_, found->second.vertices);
            release_heap_range(free_index_ranges_, found->second.indices);
            live_vertex_bytes_ -= found->second.vertices.size;
            live_index_bytes_ -= found->second.indices.size;
            geometry_allocations_.erase(found);
        }
        batch.geometry_heap_generation = geometry_heap_generation_;
        batch.vertex_heap_capacity = vertex_heap_.size();
        batch.index_heap_capacity = index_heap_.size();
    }
    return batch;
}

std::optional<gpu_resource_table_reference> gpu_resource_tables::find(gpu_resource_table_kind table,
                                                                      resource_handle handle) const noexcept
{
    const auto offset = table_offset(table);
    if (!handle.valid() || offset >= tables_.size()) return std::nullopt;
    const auto& state = tables_[offset];
    if (handle.index >= state.slots.size()) return std::nullopt;
    const auto& slot = state.slots[handle.index];
    if (!slot.live || slot.generation != handle.generation) return std::nullopt;
    return gpu_resource_table_reference{.index = handle.index, .generation = slot.generation};
}

gpu_resource_table_snapshot gpu_resource_tables::snapshot(gpu_resource_table_kind table) const noexcept
{
    const auto offset = table_offset(table);
    if (offset >= tables_.size()) return {.table = table};
    const auto& state = tables_[offset];
    return {.table = table,
            .table_generation = state.generation,
            .capacity = static_cast<std::uint32_t>(state.slots.size()),
            .live_entries = state.live_entries,
            .tombstones = state.tombstones,
            .element_stride = state.stride,
            .sparse_upload_bytes = state.sparse_upload_bytes};
}

gpu_geometry_heap_snapshot gpu_resource_tables::geometry_heap_snapshot() const noexcept
{
    return {.generation = geometry_heap_generation_,
            .vertex_bytes = vertex_heap_.size(),
            .index_bytes = index_heap_.size(),
            .live_vertex_bytes = live_vertex_bytes_,
            .live_index_bytes = live_index_bytes_,
            .live_allocations = static_cast<std::uint32_t>(geometry_allocations_.size())};
}

void gpu_resource_tables::reset()
{
    *this = gpu_resource_tables{};
}

std::vector<gpu_table_dirty_range> coalesce_gpu_table_dirty_ranges(std::span<const std::uint32_t> indices)
{
    if (indices.empty()) return {};

    std::vector<std::uint32_t> sorted(indices.begin(), indices.end());
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

    std::vector<gpu_table_dirty_range> ranges;
    ranges.reserve(sorted.size());
    std::uint32_t first = sorted.front();
    std::uint32_t previous = first;
    for (std::size_t index = 1; index < sorted.size(); ++index)
    {
        const auto current = sorted[index];
        if (current != previous + 1u)
        {
            ranges.push_back({.first = first, .count = previous - first + 1u});
            first = current;
        }
        previous = current;
    }
    ranges.push_back({.first = first, .count = previous - first + 1u});
    return ranges;
}

gpu_draw_compaction_result compact_gpu_draw_records(std::span<const gpu_draw_record> records,
                                                    std::uint32_t pipeline_bin_capacity,
                                                    std::uint32_t maximum_visible_draws)
{
    gpu_draw_compaction_result result;
    result.statistics.candidates =
        static_cast<std::uint32_t>(std::min<std::size_t>(records.size(), std::numeric_limits<std::uint32_t>::max()));
    result.bin_offsets.resize(pipeline_bin_capacity);
    result.bin_counts.resize(pipeline_bin_capacity);

    for (const auto& record : records)
    {
        if (record.pipeline_bin >= pipeline_bin_capacity)
        {
            result.overflow_draws.push_back(record);
            continue;
        }
        ++result.bin_counts[record.pipeline_bin];
    }

    std::uint32_t running_offset{};
    for (std::uint32_t bin = 0; bin < pipeline_bin_capacity; ++bin)
    {
        result.bin_offsets[bin] = running_offset;
        if (result.bin_counts[bin] != 0u) ++result.statistics.active_bins;
        const auto available = maximum_visible_draws - std::min(maximum_visible_draws, running_offset);
        const auto retained = std::min(result.bin_counts[bin], available);
        running_offset += retained;
    }

    result.visible_draws.resize(running_offset);
    auto write_offsets = result.bin_offsets;
    auto retained_counts = std::vector<std::uint32_t>(pipeline_bin_capacity);
    for (const auto& record : records)
    {
        if (record.pipeline_bin >= pipeline_bin_capacity) continue;
        const auto bin = record.pipeline_bin;
        const auto bin_limit = bin + 1u < pipeline_bin_capacity ? result.bin_offsets[bin + 1u] : running_offset;
        if (write_offsets[bin] >= bin_limit)
        {
            result.overflow_draws.push_back(record);
            continue;
        }
        result.visible_draws[write_offsets[bin]++] = record;
        ++retained_counts[bin];
    }
    result.bin_counts = std::move(retained_counts);
    result.statistics.visible = static_cast<std::uint32_t>(result.visible_draws.size());
    result.statistics.indirect_commands = result.statistics.visible;
    result.statistics.overflow_records = static_cast<std::uint32_t>(result.overflow_draws.size());
    result.statistics.cpu_submissions = result.statistics.overflow_records;
    return result;
}

std::uint64_t make_gpu_transparent_sort_key(float normalized_depth, std::uint16_t pipeline_bin,
                                            std::uint32_t stable_instance_index) noexcept
{
    const auto finite_depth = std::isfinite(normalized_depth) ? normalized_depth : 0.0f;
    const auto depth = static_cast<std::uint32_t>(std::clamp(finite_depth, 0.0f, 1.0f) * 16'777'215.0f);
    const auto descending_depth = 0x00ffffffu - depth;
    return (static_cast<std::uint64_t>(pipeline_bin) << 48u) | (static_cast<std::uint64_t>(descending_depth) << 24u) |
           static_cast<std::uint64_t>(stable_instance_index & 0x00ffffffu);
}

std::vector<gpu_draw_record> sort_gpu_transparent_records(std::span<const gpu_draw_record> records)
{
    std::vector<gpu_draw_record> result(records.begin(), records.end());
    std::stable_sort(result.begin(), result.end(), [](const gpu_draw_record& lhs, const gpu_draw_record& rhs)
                     {
                         if (lhs.pipeline_bin != rhs.pipeline_bin) return lhs.pipeline_bin < rhs.pipeline_bin;
                         if (lhs.sort_key != rhs.sort_key) return lhs.sort_key < rhs.sort_key;
                         return lhs.instance_index < rhs.instance_index;
                     });
    return result;
}

} // namespace arc::render

#include <arc/render/texture_streaming.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace arc::render
{
std::optional<std::uint32_t>
resolve_virtual_texture_page(std::span<const virtual_texture_page_table_entry> pages, std::uint32_t requested_page,
                             std::uint32_t generation) noexcept
{
    for (std::uint32_t remaining = 0; remaining <= pages.size() && requested_page < pages.size(); ++remaining)
    {
        const auto& page = pages[requested_page];
        if (page.generation == generation && virtual_texture_page_resident(page)) return requested_page;
        if (page.parent_page == resource_handle::invalid_index || page.parent_page == requested_page) break;
        requested_page = page.parent_page;
    }
    return std::nullopt;
}

texture_streaming_mode resolve_texture_streaming_mode(texture_streaming_mode authored,
                                                      texture_streaming_capabilities capabilities) noexcept
{
    if (authored == texture_streaming_mode::resident) return texture_streaming_mode::resident;
    if (authored == texture_streaming_mode::streamed_mips)
        return capabilities.mip_streaming ? texture_streaming_mode::streamed_mips : texture_streaming_mode::resident;
    if (capabilities.virtual_textures) return texture_streaming_mode::virtual_tiles;
    return capabilities.mip_streaming ? texture_streaming_mode::streamed_mips : texture_streaming_mode::resident;
}

std::uint32_t texture_requested_mip(std::uint32_t width, std::uint32_t height, std::uint32_t mip_count,
                                    float projected_texel_extent, float lod_bias) noexcept
{
    if (mip_count == 0) return 0;
    const auto maximum = static_cast<float>(std::max(width, height));
    if (!(projected_texel_extent > 0.0f) || !(maximum > 0.0f)) return mip_count - 1;
    const auto lod = std::log2(maximum / projected_texel_extent) + lod_bias;
    return static_cast<std::uint32_t>(
        std::clamp(std::floor(std::max(0.0f, lod)), 0.0f, static_cast<float>(mip_count - 1)));
}

namespace
{

std::uint64_t resource_key(texture_handle handle) noexcept
{
    return (static_cast<std::uint64_t>(handle.generation) << 32u) | handle.index;
}

std::uint64_t tile_key(std::uint32_t mip, std::uint32_t x, std::uint32_t y) noexcept
{
    return (static_cast<std::uint64_t>(mip) << 48u) | (static_cast<std::uint64_t>(y) << 24u) | x;
}

texture_handle handle_from_resource_key(std::uint64_t key) noexcept
{
    return {.index = static_cast<std::uint32_t>(key), .generation = static_cast<std::uint32_t>(key >> 32u)};
}

float mip_priority(std::uint32_t mip_count, std::uint32_t mip, float coverage, bool pinned) noexcept
{
    return (pinned ? 1'000'000.0f : 0.0f) + static_cast<float>(mip_count - std::min(mip, mip_count)) * 4096.0f +
           std::max(0.0f, coverage) * 1024.0f;
}

float tile_priority(std::uint32_t mip_count, std::uint32_t mip, float coverage) noexcept
{
    return static_cast<float>(mip_count - std::min(mip, mip_count)) * 4096.0f + std::max(0.0f, coverage) * 1024.0f;
}

} // namespace

struct texture_residency_manager::implementation
{
    struct subresource_entry
    {
        texture_subresource_kind kind{texture_subresource_kind::mip};
        std::uint32_t mip{};
        std::uint32_t x{};
        std::uint32_t y{};
        std::uint64_t byte_offset{};
        std::uint32_t byte_size{};
        std::uint32_t decoded_size{};
        std::uint64_t content_hash{};
        texture_residency_state state{texture_residency_state::nonresident};
        std::uint64_t last_used_frame{};
        std::uint64_t retry_frame{};
        std::uint32_t failures{};
        std::uint32_t gpu_bytes{};
        std::uint32_t cpu_bytes{};
        float priority{};
        bool pinned{};
    };

    struct resource_entry
    {
        resource_entry(texture_handle resource, const streamed_texture_descriptor& streamed_descriptor)
            : handle(resource), descriptor(streamed_descriptor)
        {
        }

        texture_handle handle{};
        streamed_texture_descriptor descriptor;
        std::vector<subresource_entry> mips;
        std::vector<subresource_entry> tiles;
        std::unordered_map<std::uint64_t, std::uint32_t> tile_lookup;
        texture_streaming_mode authored_mode{texture_streaming_mode::resident};
        std::optional<std::uint32_t> forced_mip;
        std::uint32_t requested_mip{};
    };

    texture_residency_config config{};
    texture_streaming_capabilities capabilities{};
    std::unordered_map<std::uint64_t, resource_entry> resources;
    std::vector<texture_stream_eviction> evictions;
    std::uint64_t frame_index{};
    std::uint64_t gpu_bytes{};
    std::uint64_t cpu_bytes{};
    std::uint64_t uploaded_bytes{};
    std::uint32_t eviction_count{};
    std::uint32_t deduplicated_requests{};
    std::uint32_t stale_requests{};
    std::uint32_t feedback_overflow{};
    std::uint32_t parent_fallbacks{};

    resource_entry* find_resource(texture_handle handle, std::uint32_t generation) noexcept
    {
        const auto found = resources.find(resource_key(handle));
        if (found == resources.end() || found->second.descriptor.content_generation != generation) return nullptr;
        return &found->second;
    }

    const resource_entry* find_resource(texture_handle handle, std::uint32_t generation) const noexcept
    {
        const auto found = resources.find(resource_key(handle));
        if (found == resources.end() || found->second.descriptor.content_generation != generation) return nullptr;
        return &found->second;
    }

    subresource_entry* find(texture_handle handle, std::uint32_t generation, texture_subresource_kind kind,
                            std::uint32_t mip, std::uint32_t x, std::uint32_t y) noexcept
    {
        auto* resource = find_resource(handle, generation);
        if (!resource) return nullptr;
        if (kind == texture_subresource_kind::mip) return mip < resource->mips.size() ? &resource->mips[mip] : nullptr;
        const auto found = resource->tile_lookup.find(tile_key(mip, x, y));
        return found == resource->tile_lookup.end() ? nullptr : &resource->tiles[found->second];
    }

    const subresource_entry* find(texture_handle handle, std::uint32_t generation, texture_subresource_kind kind,
                                  std::uint32_t mip, std::uint32_t x, std::uint32_t y) const noexcept
    {
        const auto* resource = find_resource(handle, generation);
        if (!resource) return nullptr;
        if (kind == texture_subresource_kind::mip) return mip < resource->mips.size() ? &resource->mips[mip] : nullptr;
        const auto found = resource->tile_lookup.find(tile_key(mip, x, y));
        return found == resource->tile_lookup.end() ? nullptr : &resource->tiles[found->second];
    }

    void demand(subresource_entry& entry, float priority)
    {
        entry.last_used_frame = frame_index;
        entry.priority = std::max(entry.priority, priority);
        if (entry.state == texture_residency_state::resident) return;
        if (entry.state == texture_residency_state::requested || entry.state == texture_residency_state::loading ||
            entry.state == texture_residency_state::uploading)
        {
            ++deduplicated_requests;
            return;
        }
        if (entry.state == texture_residency_state::failed && frame_index < entry.retry_frame) return;
        entry.state = texture_residency_state::requested;
    }

    void evict(resource_entry& resource, subresource_entry& entry)
    {
        gpu_bytes -= std::min<std::uint64_t>(gpu_bytes, entry.gpu_bytes);
        cpu_bytes -= std::min<std::uint64_t>(cpu_bytes, entry.cpu_bytes);
        entry.gpu_bytes = 0;
        entry.cpu_bytes = 0;
        entry.priority = 0.0f;
        entry.state = texture_residency_state::nonresident;
        evictions.push_back({.resource = resource.handle,
                             .content_generation = resource.descriptor.content_generation,
                             .kind = entry.kind,
                             .mip = entry.mip,
                             .x = entry.x,
                             .y = entry.y});
        ++eviction_count;
    }

    void trim()
    {
        while (gpu_bytes > config.gpu_budget_bytes || cpu_bytes > config.cpu_cache_budget_bytes)
        {
            resource_entry* victim_resource{};
            subresource_entry* victim{};
            for (auto& [_, resource] : resources)
            {
                subresource_entry* finest_mip{};
                for (auto& mip : resource.mips)
                    if (!mip.pinned && mip.state == texture_residency_state::resident)
                    {
                        finest_mip = &mip;
                        break;
                    }
                const auto consider = [&](subresource_entry& candidate)
                {
                    if (candidate.pinned || candidate.state != texture_residency_state::resident ||
                        frame_index - std::min(frame_index, candidate.last_used_frame) <= config.protected_frame_count)
                        return;
                    if (!victim || candidate.last_used_frame < victim->last_used_frame ||
                        (candidate.last_used_frame == victim->last_used_frame && candidate.priority < victim->priority))
                    {
                        victim_resource = &resource;
                        victim = &candidate;
                    }
                };
                if (finest_mip) consider(*finest_mip);
                for (auto& tile : resource.tiles)
                    consider(tile);
            }
            if (!victim || !victim_resource) break;
            evict(*victim_resource, *victim);
        }
    }
};

texture_residency_manager::texture_residency_manager(texture_residency_config config,
                                                     texture_streaming_capabilities capabilities)
    : implementation_(std::make_unique<implementation>())
{
    implementation_->capabilities = capabilities;
    configure(config);
}

texture_residency_manager::~texture_residency_manager() = default;
texture_residency_manager::texture_residency_manager(texture_residency_manager&&) noexcept = default;
texture_residency_manager& texture_residency_manager::operator=(texture_residency_manager&&) noexcept = default;

void texture_residency_manager::configure(texture_residency_config config)
{
    config.maximum_requests_per_frame = std::max(1u, config.maximum_requests_per_frame);
    config.upload_budget_per_frame = std::max<std::uint64_t>(1u, config.upload_budget_per_frame);
    implementation_->config = config;
    implementation_->trim();
}

void texture_residency_manager::set_capabilities(texture_streaming_capabilities capabilities)
{
    implementation_->capabilities = capabilities;
}

void texture_residency_manager::register_resource(texture_handle resource,
                                                  const streamed_texture_descriptor& descriptor)
{
    unregister_resource(resource);
    implementation::resource_entry entry(resource, descriptor);
    entry.authored_mode = descriptor.mode;
    entry.descriptor.mode = resolve_texture_streaming_mode(descriptor.mode, implementation_->capabilities);
    entry.requested_mip = descriptor.artifact.tail_first_mip;
    entry.mips.reserve(descriptor.artifact.mips.size());
    for (std::uint32_t mip = 0; mip < descriptor.artifact.mips.size(); ++mip)
    {
        const auto& range = descriptor.artifact.mips[mip];
        const bool pinned =
            entry.descriptor.mode == texture_streaming_mode::resident || mip >= descriptor.artifact.tail_first_mip;
        entry.mips.push_back({.kind = texture_subresource_kind::mip,
                              .mip = mip,
                              .byte_offset = range.offset,
                              .byte_size = range.stored_size,
                              .decoded_size = range.decoded_size,
                              .content_hash = range.content_hash,
                              .last_used_frame = implementation_->frame_index,
                              .priority = mip_priority(descriptor.artifact.mip_count, mip, 1.0f, pinned),
                              .pinned = pinned});
        if (pinned) entry.mips.back().state = texture_residency_state::requested;
    }
    entry.tiles.reserve(descriptor.artifact.tiles.size());
    for (const auto& range : descriptor.artifact.tiles)
    {
        const auto index = static_cast<std::uint32_t>(entry.tiles.size());
        entry.tile_lookup.emplace(tile_key(range.mip, range.x, range.y), index);
        entry.tiles.push_back({.kind = texture_subresource_kind::tile,
                               .mip = range.mip,
                               .x = range.x,
                               .y = range.y,
                               .byte_offset = range.offset,
                               .byte_size = range.stored_size,
                               .decoded_size = range.decoded_size,
                               .content_hash = range.content_hash});
    }
    implementation_->resources.emplace(resource_key(resource), std::move(entry));
}

void texture_residency_manager::unregister_resource(texture_handle resource)
{
    const auto found = implementation_->resources.find(resource_key(resource));
    if (found == implementation_->resources.end()) return;
    for (const auto& mip : found->second.mips)
    {
        implementation_->gpu_bytes -= std::min<std::uint64_t>(implementation_->gpu_bytes, mip.gpu_bytes);
        implementation_->cpu_bytes -= std::min<std::uint64_t>(implementation_->cpu_bytes, mip.cpu_bytes);
    }
    for (const auto& tile : found->second.tiles)
    {
        implementation_->gpu_bytes -= std::min<std::uint64_t>(implementation_->gpu_bytes, tile.gpu_bytes);
        implementation_->cpu_bytes -= std::min<std::uint64_t>(implementation_->cpu_bytes, tile.cpu_bytes);
    }
    implementation_->resources.erase(found);
}

void texture_residency_manager::begin_frame(std::uint64_t frame_index)
{
    implementation_->frame_index = frame_index;
    implementation_->uploaded_bytes = 0;
    implementation_->deduplicated_requests = 0;
    implementation_->stale_requests = 0;
    implementation_->feedback_overflow = 0;
    implementation_->parent_fallbacks = 0;
    implementation_->trim();
}

void texture_residency_manager::request(std::span<const texture_mip_feedback> mips,
                                        std::span<const texture_tile_feedback> tiles)
{
    for (const auto& feedback : mips)
    {
        auto* resource = implementation_->find_resource(feedback.resource, feedback.content_generation);
        if (!resource)
        {
            ++implementation_->stale_requests;
            continue;
        }
        const auto requested = resource->forced_mip.value_or(feedback.desired_mip);
        const auto desired =
            std::min(requested, static_cast<std::uint32_t>(resource->mips.empty() ? 0 : resource->mips.size() - 1));
        resource->requested_mip = desired;
        for (std::uint32_t mip = desired; mip < resource->mips.size(); ++mip)
            implementation_->demand(resource->mips[mip],
                                    mip_priority(resource->descriptor.artifact.mip_count, mip, feedback.screen_coverage,
                                                 resource->mips[mip].pinned));
    }
    for (const auto& feedback : tiles)
    {
        auto* entry = implementation_->find(feedback.resource, feedback.content_generation,
                                            texture_subresource_kind::tile, feedback.mip, feedback.x, feedback.y);
        if (!entry)
        {
            ++implementation_->stale_requests;
            continue;
        }
        const auto* resource = implementation_->find_resource(feedback.resource, feedback.content_generation);
        implementation_->demand(
            *entry, tile_priority(resource->descriptor.artifact.mip_count, feedback.mip, feedback.screen_coverage));
    }
}

void texture_residency_manager::note_feedback_overflow(std::uint32_t count) noexcept
{
    implementation_->feedback_overflow += count;
}

std::vector<texture_stream_load> texture_residency_manager::take_load_requests(std::uint32_t maximum_requests)
{
    std::vector<texture_stream_load> result;
    for (const auto& [key, resource] : implementation_->resources)
    {
        const auto resource_handle = handle_from_resource_key(key);
        const auto content_generation = resource.descriptor.content_generation;
        const auto source = resource.descriptor.source;
        const auto append = [&](const implementation::subresource_entry& entry)
        {
            if (entry.state != texture_residency_state::requested) return;
            result.push_back({.resource = resource_handle,
                              .content_generation = content_generation,
                              .source = source,
                              .kind = entry.kind,
                              .mip = entry.mip,
                              .x = entry.x,
                              .y = entry.y,
                              .byte_offset = entry.byte_offset,
                              .byte_size = entry.byte_size,
                              .content_hash = entry.content_hash,
                              .priority = entry.priority});
        };
        for (const auto& mip : resource.mips)
            append(mip);
        for (const auto& tile : resource.tiles)
            append(tile);
    }
    std::stable_sort(result.begin(), result.end(),
                     [](const auto& lhs, const auto& rhs)
                     {
                         if (lhs.resource == rhs.resource && lhs.kind == texture_subresource_kind::mip &&
                             rhs.kind == texture_subresource_kind::mip && lhs.mip != rhs.mip)
                             return lhs.mip > rhs.mip;
                         if (lhs.priority != rhs.priority) return lhs.priority > rhs.priority;
                         if (lhs.resource.index != rhs.resource.index) return lhs.resource.index < rhs.resource.index;
                         if (lhs.kind != rhs.kind) return lhs.kind < rhs.kind;
                         if (lhs.mip != rhs.mip) return lhs.mip < rhs.mip;
                         if (lhs.y != rhs.y) return lhs.y < rhs.y;
                         return lhs.x < rhs.x;
                     });
    std::uint64_t bytes{};
    std::size_t count{};
    const auto request_limit = std::min(implementation_->config.maximum_requests_per_frame, maximum_requests);
    while (count < result.size() && count < request_limit)
    {
        if (count > 0 && bytes + result[count].byte_size > implementation_->config.upload_budget_per_frame) break;
        bytes += result[count].byte_size;
        ++count;
    }
    result.resize(count);
    return result;
}

void texture_residency_manager::mark_loading(const texture_stream_load& load)
{
    if (auto* entry =
            implementation_->find(load.resource, load.content_generation, load.kind, load.mip, load.x, load.y);
        entry && entry->state == texture_residency_state::requested)
        entry->state = texture_residency_state::loading;
}

void texture_residency_manager::mark_uploading(const texture_stream_upload& upload)
{
    if (auto* entry = implementation_->find(upload.resource, upload.content_generation, upload.kind, upload.mip,
                                            upload.x, upload.y);
        entry &&
        (entry->state == texture_residency_state::loading || entry->state == texture_residency_state::requested))
    {
        entry->state = texture_residency_state::uploading;
        entry->cpu_bytes = upload.stored_bytes;
    }
}

void texture_residency_manager::complete(const texture_stream_upload_result& result)
{
    auto* entry =
        implementation_->find(result.resource, result.content_generation, result.kind, result.mip, result.x, result.y);
    if (!entry)
    {
        ++implementation_->stale_requests;
        return;
    }
    if (!result.succeeded)
    {
        entry->state = texture_residency_state::failed;
        entry->retry_frame =
            implementation_->frame_index + std::min<std::uint64_t>(120u, 1ull << std::min(7u, ++entry->failures));
        return;
    }
    implementation_->gpu_bytes -= std::min<std::uint64_t>(implementation_->gpu_bytes, entry->gpu_bytes);
    implementation_->cpu_bytes -= std::min<std::uint64_t>(implementation_->cpu_bytes, entry->cpu_bytes);
    entry->state = texture_residency_state::resident;
    entry->gpu_bytes = result.gpu_bytes ? result.gpu_bytes : entry->decoded_size;
    entry->last_used_frame = implementation_->frame_index;
    implementation_->gpu_bytes += entry->gpu_bytes;
    implementation_->cpu_bytes += entry->cpu_bytes;
    implementation_->uploaded_bytes += entry->gpu_bytes;
    implementation_->trim();
}

void texture_residency_manager::fail(const texture_stream_load& load)
{
    if (auto* entry =
            implementation_->find(load.resource, load.content_generation, load.kind, load.mip, load.x, load.y))
    {
        entry->state = texture_residency_state::failed;
        entry->retry_frame =
            implementation_->frame_index + std::min<std::uint64_t>(120u, 1ull << std::min(7u, ++entry->failures));
    }
}

std::vector<texture_stream_eviction> texture_residency_manager::take_evictions()
{
    auto result = std::move(implementation_->evictions);
    implementation_->evictions.clear();
    return result;
}

bool texture_residency_manager::resident(texture_handle resource, std::uint32_t generation,
                                         texture_subresource_kind kind, std::uint32_t mip, std::uint32_t x,
                                         std::uint32_t y) const noexcept
{
    const auto* entry = implementation_->find(resource, generation, kind, mip, x, y);
    return entry && entry->state == texture_residency_state::resident;
}

void texture_residency_manager::note_parent_fallback() noexcept
{
    ++implementation_->parent_fallbacks;
}

void texture_residency_manager::set_forced_mip(texture_handle resource, std::uint32_t generation,
                                               std::optional<std::uint32_t> mip) noexcept
{
    auto* entry = implementation_->find_resource(resource, generation);
    if (!entry) return;
    if (mip && !entry->mips.empty()) *mip = std::min(*mip, static_cast<std::uint32_t>(entry->mips.size() - 1));
    entry->forced_mip = mip;
    if (!mip) return;
    entry->requested_mip = *mip;
    for (std::uint32_t level = *mip; level < entry->mips.size(); ++level)
        implementation_->demand(entry->mips[level], mip_priority(entry->descriptor.artifact.mip_count, level, 1.0f,
                                                                 entry->mips[level].pinned));
}

texture_residency_snapshot texture_residency_manager::snapshot() const noexcept
{
    texture_residency_snapshot result{.frame_index = implementation_->frame_index,
                                      .gpu_budget_bytes = implementation_->config.gpu_budget_bytes,
                                      .gpu_resident_bytes = implementation_->gpu_bytes,
                                      .cpu_cache_budget_bytes = implementation_->config.cpu_cache_budget_bytes,
                                      .cpu_cached_bytes = implementation_->cpu_bytes,
                                      .upload_budget_per_frame = implementation_->config.upload_budget_per_frame,
                                      .uploaded_bytes = implementation_->uploaded_bytes,
                                      .resource_count = static_cast<std::uint32_t>(implementation_->resources.size()),
                                      .evictions = implementation_->eviction_count,
                                      .deduplicated_requests = implementation_->deduplicated_requests,
                                      .stale_requests = implementation_->stale_requests,
                                      .feedback_overflow = implementation_->feedback_overflow,
                                      .parent_fallbacks = implementation_->parent_fallbacks,
                                      .over_budget =
                                          implementation_->gpu_bytes > implementation_->config.gpu_budget_bytes ||
                                          implementation_->cpu_bytes > implementation_->config.cpu_cache_budget_bytes};
    for (const auto& [_, resource] : implementation_->resources)
    {
        if (resource.descriptor.mode == texture_streaming_mode::streamed_mips) ++result.streamed_mip_resources;
        if (resource.descriptor.mode == texture_streaming_mode::virtual_tiles) ++result.virtual_texture_resources;
        const auto accumulate = [&](const implementation::subresource_entry& entry)
        {
            if (entry.state == texture_residency_state::resident)
            {
                if (entry.kind == texture_subresource_kind::mip)
                    ++result.resident_mips;
                else
                    ++result.resident_tiles;
            }
            if (entry.state == texture_residency_state::requested || entry.state == texture_residency_state::loading ||
                entry.state == texture_residency_state::uploading)
                ++result.requested_subresources;
            if (entry.state == texture_residency_state::failed) ++result.failed_subresources;
        };
        for (const auto& mip : resource.mips)
            accumulate(mip);
        for (const auto& tile : resource.tiles)
            accumulate(tile);
    }
    return result;
}

std::vector<texture_streaming_resource_snapshot> texture_residency_manager::resource_snapshots() const
{
    std::vector<texture_streaming_resource_snapshot> result;
    result.reserve(implementation_->resources.size());
    for (const auto& [_, resource] : implementation_->resources)
    {
        std::uint32_t resident_first = resource.mips.empty() ? 0 : static_cast<std::uint32_t>(resource.mips.size());
        std::uint64_t resident_bytes{};
        for (const auto& mip : resource.mips)
            if (mip.state == texture_residency_state::resident)
            {
                resident_first = std::min(resident_first, mip.mip);
                resident_bytes += mip.gpu_bytes;
            }
        for (const auto& tile : resource.tiles)
            if (tile.state == texture_residency_state::resident) resident_bytes += tile.gpu_bytes;
        if (resident_first == resource.mips.size()) resident_first = resource.descriptor.artifact.tail_first_mip;
        result.push_back({.resource = resource.handle,
                          .content_generation = resource.descriptor.content_generation,
                          .authored_mode = resource.authored_mode,
                          .resolved_mode = resource.descriptor.mode,
                          .requested_mip = resource.requested_mip,
                          .resident_first_mip = resident_first,
                          .tail_first_mip = resource.descriptor.artifact.tail_first_mip,
                          .resident_bytes = resident_bytes,
                          .forced_mip = resource.forced_mip});
    }
    std::sort(result.begin(), result.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.resource.index < rhs.resource.index; });
    return result;
}

} // namespace arc::render

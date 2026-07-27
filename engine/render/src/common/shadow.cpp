#include <arc/render/shadow.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

namespace arc::render
{
namespace
{

constexpr bool power_of_two(std::uint32_t value) noexcept
{
    return value != 0 && (value & (value - 1u)) == 0;
}

std::uint32_t floor_power_of_two(std::uint32_t value) noexcept
{
    if (value == 0)
        return 0;
    return std::bit_floor(value);
}

math::matrix4f look_at_rh(
    const math::vector3f& eye,
    const math::vector3f& center,
    const math::vector3f& up) noexcept
{
    const auto forward = math::normalize(math::sub(center, eye), 0.0f);
    const auto right = math::normalize(math::cross(forward, up), 0.0f);
    const auto corrected_up = math::cross(right, forward);
    math::matrix4f result = math::identity<float, 4>();
    result(0, 0) = right[0]; result(0, 1) = right[1]; result(0, 2) = right[2];
    result(1, 0) = corrected_up[0]; result(1, 1) = corrected_up[1]; result(1, 2) = corrected_up[2];
    result(2, 0) = -forward[0]; result(2, 1) = -forward[1]; result(2, 2) = -forward[2];
    result(0, 3) = -math::dot(right, eye);
    result(1, 3) = -math::dot(corrected_up, eye);
    result(2, 3) = math::dot(forward, eye);
    return result;
}

math::matrix4f orthographic_rh_zo(float extent, float near_plane, float far_plane) noexcept
{
    const float half = std::max(extent * 0.5f, 0.001f);
    const float depth = std::max(far_plane - near_plane, 0.001f);
    math::matrix4f result{};
    result(0, 0) = 1.0f / half;
    result(1, 1) = 1.0f / half;
    result(2, 2) = -1.0f / depth;
    result(2, 3) = -near_plane / depth;
    result(3, 3) = 1.0f;
    return result;
}

std::array<math::vector3f, 8> frustum_corners(const math::matrix4f& inverse_view_projection) noexcept
{
    std::array<math::vector3f, 8> result{};
    std::size_t index{};
    for (float z : { 0.0f, 1.0f })
    {
        for (float y : { -1.0f, 1.0f })
        {
            for (float x : { -1.0f, 1.0f })
                result[index++] = math::transform_point(inverse_view_projection, math::vector3f{ x, y, z });
        }
    }
    return result;
}

} // namespace

struct shadow_atlas_allocator::slot
{
    shadow_atlas_allocation allocation{};
    std::uint32_t generation{ 1 };
    bool occupied{};
};

shadow_atlas_allocator::~shadow_atlas_allocator() = default;
shadow_atlas_allocator::shadow_atlas_allocator(shadow_atlas_allocator&&) noexcept = default;
shadow_atlas_allocator& shadow_atlas_allocator::operator=(shadow_atlas_allocator&&) noexcept = default;

shadow_atlas_allocator::shadow_atlas_allocator(
    std::uint32_t atlas_size,
    std::uint32_t minimum_tile_size,
    std::uint32_t guard_texels)
    : atlas_size_(floor_power_of_two(std::max(1u, atlas_size)))
    , minimum_tile_size_(floor_power_of_two(std::max(1u, minimum_tile_size)))
    , guard_texels_(guard_texels)
{
    minimum_tile_size_ = std::min(minimum_tile_size_, atlas_size_);
    cells_per_axis_ = std::max(1u, atlas_size_ / minimum_tile_size_);
    occupancy_.resize(static_cast<std::size_t>(cells_per_axis_) * cells_per_axis_);
}

bool shadow_atlas_allocator::region_free(std::uint32_t x, std::uint32_t y, std::uint32_t cells) const noexcept
{
    if (x + cells > cells_per_axis_ || y + cells > cells_per_axis_)
        return false;
    for (std::uint32_t row = y; row < y + cells; ++row)
        for (std::uint32_t column = x; column < x + cells; ++column)
            if (occupancy_[static_cast<std::size_t>(row) * cells_per_axis_ + column] != 0)
                return false;
    return true;
}

void shadow_atlas_allocator::mark_region(const shadow_atlas_rect& rect, bool occupied) noexcept
{
    const std::uint32_t first_x = rect.x / minimum_tile_size_;
    const std::uint32_t first_y = rect.y / minimum_tile_size_;
    const std::uint32_t cells = rect.size / minimum_tile_size_;
    for (std::uint32_t row = first_y; row < first_y + cells; ++row)
        for (std::uint32_t column = first_x; column < first_x + cells; ++column)
            occupancy_[static_cast<std::size_t>(row) * cells_per_axis_ + column] = occupied ? 1u : 0u;
}

std::optional<shadow_atlas_allocation> shadow_atlas_allocator::try_allocate(
    const shadow_atlas_request& request,
    std::uint32_t resolution)
{
    // Resolution denotes the complete power-of-two atlas tile. Guard texels
    // are reserved inside that tile so a 512 request does not consume a 1024
    // buddy block merely to gain a two-texel sampling border.
    const std::uint32_t tile_size = std::max(
        minimum_tile_size_,
        std::bit_ceil(resolution));
    if (tile_size > atlas_size_)
        return std::nullopt;

    const std::uint32_t cells = tile_size / minimum_tile_size_;
    const std::uint32_t face_count =
        request.kind == shadow_light_kind::point ? point_shadow_face_count : 1u;
    std::array<shadow_atlas_rect, point_shadow_face_count> faces{};
    std::vector<shadow_atlas_rect> reserved;
    reserved.reserve(face_count);

    for (std::uint32_t face = 0; face < face_count; ++face)
    {
        bool found{};
        for (std::uint32_t y = 0; y + cells <= cells_per_axis_ && !found; y += cells)
        {
            for (std::uint32_t x = 0; x + cells <= cells_per_axis_; x += cells)
            {
                if (!region_free(x, y, cells))
                    continue;
                faces[face] = {
                    .x = x * minimum_tile_size_,
                    .y = y * minimum_tile_size_,
                    .size = tile_size,
                    .guard = guard_texels_
                };
                mark_region(faces[face], true);
                reserved.push_back(faces[face]);
                found = true;
                break;
            }
        }
        if (!found)
        {
            for (const auto& rect : reserved)
                mark_region(rect, false);
            return std::nullopt;
        }
    }

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

    auto& slot = slots_[index];
    slot.occupied = true;
    slot.allocation = {
        .handle = { index, slot.generation },
        .kind = request.kind,
        .light_key = request.light_key,
        .faces = faces,
        .face_count = face_count,
        .resolved_resolution = resolution,
        .priority = request.priority,
        .last_used_frame = request.frame_index
    };
    return slot.allocation;
}

std::optional<std::size_t> shadow_atlas_allocator::eviction_candidate(
    std::uint16_t incoming_priority,
    std::uint64_t protected_light_key) const noexcept
{
    std::optional<std::size_t> candidate;
    for (std::size_t index = 0; index < slots_.size(); ++index)
    {
        const auto& slot = slots_[index];
        if (!slot.occupied || slot.allocation.light_key == protected_light_key ||
            slot.allocation.priority > incoming_priority)
            continue;
        if (!candidate ||
            slot.allocation.priority < slots_[*candidate].allocation.priority ||
            (slot.allocation.priority == slots_[*candidate].allocation.priority &&
             slot.allocation.last_used_frame < slots_[*candidate].allocation.last_used_frame))
            candidate = index;
    }
    return candidate;
}

std::optional<shadow_atlas_allocation> shadow_atlas_allocator::allocate(const shadow_atlas_request& request)
{
    if (const auto* existing = find_light(request.kind, request.light_key))
    {
        if (existing->resolved_resolution >= request.minimum_resolution)
        {
            touch(existing->handle, request.frame_index, request.priority);
            return *find(existing->handle);
        }
    }

    const std::uint32_t maximum = std::min(
        atlas_size_,
        std::bit_ceil(std::max(request.minimum_resolution, request.requested_resolution)));
    const std::uint32_t minimum = std::max(
        minimum_tile_size_ > guard_texels_ * 2u ? minimum_tile_size_ - guard_texels_ * 2u : 1u,
        std::bit_ceil(std::max(1u, request.minimum_resolution)));

    for (std::uint32_t resolution = maximum; resolution >= minimum; resolution /= 2u)
    {
        if (auto allocation = try_allocate(request, resolution))
        {
            if (resolution < maximum)
                ++resolution_reduction_count_;
            return allocation;
        }
        if (resolution == 1u)
            break;
    }

    while (const auto candidate = eviction_candidate(request.priority, request.light_key))
    {
        const auto handle = slots_[*candidate].allocation.handle;
        release(handle);
        ++eviction_count_;
        for (std::uint32_t resolution = maximum; resolution >= minimum; resolution /= 2u)
        {
            if (auto allocation = try_allocate(request, resolution))
            {
                if (resolution < maximum)
                    ++resolution_reduction_count_;
                return allocation;
            }
            if (resolution == 1u)
                break;
        }
    }
    return std::nullopt;
}

const shadow_atlas_allocation* shadow_atlas_allocator::find(shadow_allocation_handle handle) const noexcept
{
    if (!handle.valid() || handle.index >= slots_.size())
        return nullptr;
    const auto& slot = slots_[handle.index];
    return slot.occupied && slot.generation == handle.generation ? &slot.allocation : nullptr;
}

const shadow_atlas_allocation* shadow_atlas_allocator::find_light(
    shadow_light_kind kind,
    std::uint64_t light_key) const noexcept
{
    for (const auto& slot : slots_)
        if (slot.occupied && slot.allocation.kind == kind && slot.allocation.light_key == light_key)
            return &slot.allocation;
    return nullptr;
}

bool shadow_atlas_allocator::touch(
    shadow_allocation_handle handle,
    std::uint64_t frame_index,
    std::uint16_t priority) noexcept
{
    if (!handle.valid() || handle.index >= slots_.size())
        return false;
    auto& slot = slots_[handle.index];
    if (!slot.occupied || slot.generation != handle.generation)
        return false;
    slot.allocation.last_used_frame = frame_index;
    slot.allocation.priority = priority;
    return true;
}

bool shadow_atlas_allocator::release(shadow_allocation_handle handle) noexcept
{
    if (!handle.valid() || handle.index >= slots_.size())
        return false;
    auto& slot = slots_[handle.index];
    if (!slot.occupied || slot.generation != handle.generation)
        return false;
    for (std::uint32_t face = 0; face < slot.allocation.face_count; ++face)
        mark_region(slot.allocation.faces[face], false);
    slot.occupied = false;
    slot.allocation = {};
    ++slot.generation;
    if (slot.generation == 0)
        slot.generation = 1;
    free_slots_.push_back(handle.index);
    return true;
}

void shadow_atlas_allocator::clear() noexcept
{
    std::fill(occupancy_.begin(), occupancy_.end(), std::uint8_t{});
    free_slots_.clear();
    for (std::uint32_t index = 0; index < slots_.size(); ++index)
    {
        auto& slot = slots_[index];
        slot.occupied = false;
        slot.allocation = {};
        ++slot.generation;
        if (slot.generation == 0)
            slot.generation = 1;
        free_slots_.push_back(index);
    }
}

shadow_atlas_statistics shadow_atlas_allocator::statistics() const noexcept
{
    shadow_atlas_statistics result{
        .atlas_size = atlas_size_,
        .minimum_tile_size = minimum_tile_size_,
        .eviction_count = eviction_count_,
        .resolution_reduction_count = resolution_reduction_count_
    };
    for (const auto value : occupancy_)
        if (value != 0)
            result.occupied_texels += minimum_tile_size_ * minimum_tile_size_;
    for (const auto& slot : slots_)
        result.allocation_count += slot.occupied ? 1u : 0u;
    return result;
}

directional_shadow_layout fit_directional_shadow_cascades(
    const directional_shadow_camera& camera,
    const math::vector3f& authored_light_direction,
    const directional_shadow_settings& settings,
    std::uint32_t resolution) noexcept
{
    directional_shadow_layout result{};
    result.cascade_count = std::clamp(
        settings.cascade_count,
        1u,
        maximum_directional_shadow_cascades);
    resolution = std::max(1u, resolution);

    const float near_plane = std::max(0.001f, camera.near_plane);
    const float far_plane = std::max(
        near_plane + 0.001f,
        std::min(camera.far_plane, std::max(near_plane, settings.maximum_distance)));
    const float lambda = std::clamp(settings.split_lambda, 0.0f, 1.0f);
    const float blend = std::clamp(settings.blend_fraction, 0.0f, 0.3f);
    std::array<float, maximum_directional_shadow_cascades> splits{};
    for (std::uint32_t index = 0; index < result.cascade_count; ++index)
    {
        const float p = static_cast<float>(index + 1u) / static_cast<float>(result.cascade_count);
        const float logarithmic = near_plane * std::pow(far_plane / near_plane, p);
        const float uniform = near_plane + (far_plane - near_plane) * p;
        splits[index] = lambda * logarithmic + (1.0f - lambda) * uniform;
    }
    splits[result.cascade_count - 1u] = far_plane;

    auto light_direction = math::normalize(authored_light_direction, 0.0f);
    if (math::length_squared(light_direction) < 1.0e-6f)
        light_direction = math::normalize(math::vector3f{ 0.35f, -0.85f, -0.40f }, 0.0f);
    const math::vector3f up =
        std::abs(math::dot(light_direction, math::vector3f{ 0.0f, 1.0f, 0.0f })) > 0.95f
        ? math::vector3f{ 0.0f, 0.0f, 1.0f }
        : math::vector3f{ 0.0f, 1.0f, 0.0f };

    const auto full_corners = frustum_corners(camera.inverse_view_projection);
    const float full_range = std::max(camera.far_plane - camera.near_plane, 0.001f);
    float previous_split = near_plane;
    for (std::uint32_t cascade_index = 0; cascade_index < result.cascade_count; ++cascade_index)
    {
        const float split = splits[cascade_index];
        const float near_t = std::clamp((previous_split - camera.near_plane) / full_range, 0.0f, 1.0f);
        const float far_t = std::clamp((split - camera.near_plane) / full_range, 0.0f, 1.0f);
        std::array<math::vector3f, 8> corners{};
        for (std::size_t corner = 0; corner < 4; ++corner)
        {
            const auto ray = math::sub(full_corners[corner + 4], full_corners[corner]);
            corners[corner] = math::add(full_corners[corner], math::mul(ray, near_t));
            corners[corner + 4] = math::add(full_corners[corner], math::mul(ray, far_t));
        }

        math::vector3f center{};
        for (const auto& corner : corners)
            center = math::add(center, corner);
        center = math::div(center, static_cast<float>(corners.size()));

        float radius{};
        for (const auto& corner : corners)
            radius = std::max(radius, math::length(math::sub(corner, center)));
        radius = std::max(radius, 0.01f);
        if (settings.stable)
            radius = std::ceil(radius * 16.0f) / 16.0f;

        const float texel_size = (radius * 2.0f) / static_cast<float>(resolution);
        if (settings.stable && texel_size > 0.0f)
        {
            // Quantize against a world-anchored light basis. Quantizing after
            // building a view centered on `center` always observes x=y=0 and
            // silently disables stabilization.
            const auto light_basis = look_at_rh(
                math::mul(light_direction, -1.0f),
                math::vector3f::zero,
                up);
            const auto light_center = math::transform_point(light_basis, center);
            const float snapped_x = std::round(light_center[0] / texel_size) * texel_size;
            const float snapped_y = std::round(light_center[1] / texel_size) * texel_size;
            center = math::add(center, math::vector3f{
                light_basis(0, 0) * (snapped_x - light_center[0]) +
                    light_basis(1, 0) * (snapped_y - light_center[1]),
                light_basis(0, 1) * (snapped_x - light_center[0]) +
                    light_basis(1, 1) * (snapped_y - light_center[1]),
                light_basis(0, 2) * (snapped_x - light_center[0]) +
                    light_basis(1, 2) * (snapped_y - light_center[1])
            });
        }

        const auto view = look_at_rh(
            math::sub(center, math::mul(light_direction, radius * 2.5f)),
            center,
            up);
        const float guard_extent = radius * 2.0f + texel_size * 4.0f;
        const auto projection = orthographic_rh_zo(guard_extent, 0.01f, radius * 5.0f);
        result.cascades[cascade_index] = {
            .light_view_projection = math::matmul(projection, view),
            .center = center,
            .radius = radius,
            .near_depth = previous_split,
            .split_depth = split,
            .blend_start_depth = split - (split - previous_split) * blend,
            .texel_world_size = texel_size
        };
        previous_split = split;
    }
    return result;
}

} // namespace arc::render

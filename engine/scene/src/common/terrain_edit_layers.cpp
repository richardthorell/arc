#include <arc/scene/terrain_asset.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace arc::scene
{
namespace
{

bool sample_less(const terrain_sculpt_sample_delta& lhs, const terrain_sculpt_sample_delta& rhs) noexcept
{
    return lhs.z < rhs.z || (lhs.z == rhs.z && lhs.x < rhs.x);
}

bool sample_less(const terrain_paint_sample_delta& lhs, const terrain_paint_sample_delta& rhs) noexcept
{
    return lhs.z < rhs.z || (lhs.z == rhs.z && lhs.x < rhs.x);
}

template <class Sample> bool has_duplicate_samples(const std::vector<Sample>& samples) noexcept
{
    if (samples.size() < 2u) return false;
    for (std::size_t index = 1u; index < samples.size(); ++index)
        if (samples[index - 1u].x == samples[index].x && samples[index - 1u].z == samples[index].z) return true;
    return false;
}

template <class Sample> void sort_samples(std::vector<Sample>& samples)
{
    std::sort(samples.begin(), samples.end(),
              [](const Sample& lhs, const Sample& rhs) { return sample_less(lhs, rhs); });
}

bool zero_paint_delta(const terrain_paint_sample_delta& sample) noexcept
{
    return std::all_of(sample.delta.begin(), sample.delta.end(), [](std::int16_t value) { return value == 0; });
}

void recompute_affected_bounds(terrain_asset& asset, terrain_modifier_descriptor& modifier)
{
    if (modifier.region_payloads.empty())
    {
        modifier.affected_bounds.reset();
        return;
    }

    auto bounds = terrain_region_bounds(asset.coordinates, asset.partition, modifier.region_payloads.front().region);
    for (std::size_t index = 1u; index < modifier.region_payloads.size(); ++index)
    {
        const auto region =
            terrain_region_bounds(asset.coordinates, asset.partition, modifier.region_payloads[index].region);
        bounds.min_x = std::min(bounds.min_x, region.min_x);
        bounds.min_y = std::min(bounds.min_y, region.min_y);
        bounds.min_z = std::min(bounds.min_z, region.min_z);
        bounds.max_x = std::max(bounds.max_x, region.max_x);
        bounds.max_y = std::max(bounds.max_y, region.max_y);
        bounds.max_z = std::max(bounds.max_z, region.max_z);
    }
    modifier.affected_bounds = bounds;
}

terrain_modifier_descriptor& add_builtin_layer(terrain_asset& asset, std::string_view type_id, std::string name,
                                               terrain_domain domains)
{
    terrain_modifier_descriptor modifier;
    modifier.id = generate_terrain_stable_id();
    modifier.type_id = std::string(type_id);
    modifier.name = std::move(name);
    modifier.domains = domains;
    modifier.canonical_parameters = "{}";
    asset.modifiers.push_back(std::move(modifier));
    if (asset.authoring_revision != std::numeric_limits<std::uint64_t>::max()) ++asset.authoring_revision;
    return asset.modifiers.back();
}

template <class Payload>
terrain_modifier_region_payload* find_payload(terrain_modifier_descriptor& modifier, terrain_region_id region) noexcept
{
    const auto found = std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                                    [region](const auto& value) { return value.region == region; });
    if (found == modifier.region_payloads.end() || !std::holds_alternative<Payload>(found->data)) return nullptr;
    return &*found;
}

template <class Payload>
terrain_modifier_region_payload& ensure_payload(terrain_modifier_descriptor& modifier, terrain_region_id region)
{
    const auto found = std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                                    [region](const auto& value) { return value.region == region; });
    if (found != modifier.region_payloads.end()) return *found;
    terrain_modifier_region_payload payload;
    payload.region = region;
    payload.data = Payload{};
    modifier.region_payloads.push_back(std::move(payload));
    std::sort(modifier.region_payloads.begin(), modifier.region_payloads.end(), [](const auto& lhs, const auto& rhs)
              { return lhs.region.z < rhs.region.z || (lhs.region.z == rhs.region.z && lhs.region.x < rhs.region.x); });
    return *std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                         [region](const auto& value) { return value.region == region; });
}

void erase_empty_payload(terrain_modifier_descriptor& modifier, terrain_region_id region)
{
    const auto found = std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                                    [region](const auto& value) { return value.region == region; });
    if (found == modifier.region_payloads.end()) return;
    const bool empty = std::visit([](const auto& value) { return value.samples.empty(); }, found->data);
    if (empty) modifier.region_payloads.erase(found);
}

} // namespace

terrain_modifier_descriptor* find_terrain_modifier(terrain_asset& asset, terrain_stable_id id) noexcept
{
    const auto found = std::find_if(asset.modifiers.begin(), asset.modifiers.end(),
                                    [id](const auto& value) { return value.id == id; });
    return found == asset.modifiers.end() ? nullptr : &*found;
}

const terrain_modifier_descriptor* find_terrain_modifier(const terrain_asset& asset, terrain_stable_id id) noexcept
{
    const auto found = std::find_if(asset.modifiers.begin(), asset.modifiers.end(),
                                    [id](const auto& value) { return value.id == id; });
    return found == asset.modifiers.end() ? nullptr : &*found;
}

terrain_modifier_descriptor& add_terrain_sculpt_layer(terrain_asset& asset, std::string name)
{
    if (name.empty()) name = "Sculpt Layer";
    return add_builtin_layer(asset, terrain_builtin_modifier_types::sculpt_layer, std::move(name),
                             terrain_domain::geometry);
}

terrain_modifier_descriptor& add_terrain_paint_layer(terrain_asset& asset, std::string name)
{
    if (name.empty()) name = "Paint Layer";
    return add_builtin_layer(asset, terrain_builtin_modifier_types::paint_layer, std::move(name),
                             terrain_domain::attributes);
}

terrain_modifier_region_payload* find_terrain_modifier_payload(terrain_modifier_descriptor& modifier,
                                                               terrain_region_id region) noexcept
{
    const auto found = std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                                    [region](const auto& value) { return value.region == region; });
    return found == modifier.region_payloads.end() ? nullptr : &*found;
}

const terrain_modifier_region_payload* find_terrain_modifier_payload(const terrain_modifier_descriptor& modifier,
                                                                     terrain_region_id region) noexcept
{
    const auto found = std::find_if(modifier.region_payloads.begin(), modifier.region_payloads.end(),
                                    [region](const auto& value) { return value.region == region; });
    return found == modifier.region_payloads.end() ? nullptr : &*found;
}

terrain_dirty_update set_terrain_sculpt_region_samples(terrain_asset& asset, terrain_stable_id modifier_id,
                                                       terrain_region_id region,
                                                       std::vector<terrain_sculpt_sample_delta> samples)
{
    auto* modifier = find_terrain_modifier(asset, modifier_id);
    if (!modifier || modifier->type_id != terrain_builtin_modifier_types::sculpt_layer) return {};

    samples.erase(std::remove_if(samples.begin(), samples.end(),
                                 [](const auto& sample)
                                 {
                                     return !std::isfinite(sample.delta) || sample.delta == 0.0f ||
                                            sample.x > terrain_modifier_sample_coordinate_max ||
                                            sample.z > terrain_modifier_sample_coordinate_max;
                                 }),
                  samples.end());
    sort_samples(samples);
    if (has_duplicate_samples(samples)) return {};

    const auto* existing = find_payload<terrain_sculpt_region_payload>(*modifier, region);
    if (existing && std::get<terrain_sculpt_region_payload>(existing->data).samples == samples) return {};
    if (!existing && samples.empty()) return {};

    auto& payload = ensure_payload<terrain_sculpt_region_payload>(*modifier, region);
    if (!std::holds_alternative<terrain_sculpt_region_payload>(payload.data)) return {};
    std::get<terrain_sculpt_region_payload>(payload.data).samples = std::move(samples);
    erase_empty_payload(*modifier, region);
    recompute_affected_bounds(asset, *modifier);
    return mark_terrain_dirty(asset, terrain_region_bounds(asset.coordinates, asset.partition, region),
                              modifier->domains);
}

terrain_dirty_update accumulate_terrain_sculpt_samples(terrain_asset& asset, terrain_stable_id modifier_id,
                                                       std::span<const terrain_sculpt_sample_edit> edits)
{
    auto* modifier = find_terrain_modifier(asset, modifier_id);
    if (!modifier || modifier->type_id != terrain_builtin_modifier_types::sculpt_layer || edits.empty()) return {};

    std::vector<terrain_sculpt_sample_edit> pending;
    pending.reserve(edits.size());
    for (const auto& edit : edits)
    {
        if (!std::isfinite(edit.sample.delta) || edit.sample.delta == 0.0f ||
            edit.sample.x > terrain_modifier_sample_coordinate_max ||
            edit.sample.z > terrain_modifier_sample_coordinate_max)
            return {};
        pending.push_back(edit);
    }

    const auto edit_less = [](const terrain_sculpt_sample_edit& lhs, const terrain_sculpt_sample_edit& rhs)
    {
        if (lhs.region.z != rhs.region.z) return lhs.region.z < rhs.region.z;
        if (lhs.region.x != rhs.region.x) return lhs.region.x < rhs.region.x;
        return sample_less(lhs.sample, rhs.sample);
    };
    std::sort(pending.begin(), pending.end(), edit_less);

    std::vector<terrain_sculpt_sample_edit> compact;
    compact.reserve(pending.size());
    for (const auto& edit : pending)
    {
        if (!compact.empty() && compact.back().region == edit.region && compact.back().sample.x == edit.sample.x &&
            compact.back().sample.z == edit.sample.z)
        {
            const auto combined = compact.back().sample.delta + edit.sample.delta;
            if (!std::isfinite(combined)) return {};
            compact.back().sample.delta = combined;
        }
        else
        {
            compact.push_back(edit);
        }
    }
    compact.erase(
        std::remove_if(compact.begin(), compact.end(), [](const auto& edit) { return edit.sample.delta == 0.0f; }),
        compact.end());
    if (compact.empty()) return {};

    std::vector<terrain_region_id> changed_regions;
    for (std::size_t begin = 0; begin < compact.size();)
    {
        const auto region = compact[begin].region;
        auto end = begin + 1u;
        while (end < compact.size() && compact[end].region == region)
            ++end;

        std::vector<terrain_sculpt_sample_delta> samples;
        if (const auto* existing = find_terrain_modifier_payload(*modifier, region);
            existing && std::holds_alternative<terrain_sculpt_region_payload>(existing->data))
            samples = std::get<terrain_sculpt_region_payload>(existing->data).samples;
        sort_samples(samples);

        bool changed{};
        for (auto edit = begin; edit < end; ++edit)
        {
            const auto found = std::lower_bound(samples.begin(), samples.end(), compact[edit].sample,
                                                [](const auto& lhs, const auto& rhs) { return sample_less(lhs, rhs); });
            if (found != samples.end() && found->x == compact[edit].sample.x && found->z == compact[edit].sample.z)
            {
                const auto combined = found->delta + compact[edit].sample.delta;
                if (!std::isfinite(combined)) return {};
                if (combined == found->delta) continue;
                if (combined == 0.0f)
                    samples.erase(found);
                else
                    found->delta = combined;
                changed = true;
            }
            else
            {
                samples.insert(found, compact[edit].sample);
                changed = true;
            }
        }

        if (changed)
        {
            auto& payload = ensure_payload<terrain_sculpt_region_payload>(*modifier, region);
            payload.data = terrain_sculpt_region_payload{std::move(samples)};
            erase_empty_payload(*modifier, region);
            changed_regions.push_back(region);
        }
        begin = end;
    }

    if (changed_regions.empty()) return {};
    recompute_affected_bounds(asset, *modifier);
    if (asset.authoring_revision != std::numeric_limits<std::uint64_t>::max()) ++asset.authoring_revision;

    terrain_dirty_update result;
    result.revision = asset.authoring_revision;
    result.regions = std::move(changed_regions);
    for (const auto region : result.regions)
    {
        auto& record = ensure_terrain_region(asset, region);
        record.dirty_revision = result.revision;
        record.dirty_domains |= terrain_domain::geometry;
    }
    return result;
}

terrain_dirty_update set_terrain_paint_region_samples(terrain_asset& asset, terrain_stable_id modifier_id,
                                                      terrain_region_id region,
                                                      std::vector<terrain_paint_sample_delta> samples)
{
    auto* modifier = find_terrain_modifier(asset, modifier_id);
    if (!modifier || modifier->type_id != terrain_builtin_modifier_types::paint_layer) return {};

    samples.erase(std::remove_if(samples.begin(), samples.end(), zero_paint_delta), samples.end());
    sort_samples(samples);
    if (has_duplicate_samples(samples)) return {};

    const auto* existing = find_payload<terrain_paint_region_payload>(*modifier, region);
    if (existing && std::get<terrain_paint_region_payload>(existing->data).samples == samples) return {};
    if (!existing && samples.empty()) return {};

    auto& payload = ensure_payload<terrain_paint_region_payload>(*modifier, region);
    if (!std::holds_alternative<terrain_paint_region_payload>(payload.data)) return {};
    std::get<terrain_paint_region_payload>(payload.data).samples = std::move(samples);
    erase_empty_payload(*modifier, region);
    recompute_affected_bounds(asset, *modifier);
    return mark_terrain_dirty(asset, terrain_region_bounds(asset.coordinates, asset.partition, region),
                              modifier->domains);
}

bool validate_terrain_modifier_payloads(const terrain_modifier_descriptor& modifier) noexcept
{
    std::unordered_set<std::uint64_t> regions;
    for (const auto& payload : modifier.region_payloads)
    {
        if (payload.schema_version == 0u) return false;
        const auto x = static_cast<std::uint64_t>(payload.region.x);
        const auto z = static_cast<std::uint64_t>(payload.region.z);
        const auto key = x ^ (z + 0x9e3779b97f4a7c15ull + (x << 6u) + (x >> 2u));
        if (!regions.insert(key).second) return false;

        if (modifier.type_id == terrain_builtin_modifier_types::sculpt_layer)
        {
            if (!std::holds_alternative<terrain_sculpt_region_payload>(payload.data)) return false;
            const auto& samples = std::get<terrain_sculpt_region_payload>(payload.data).samples;
            if (samples.empty()) return false;
            for (const auto& sample : samples)
                if (!std::isfinite(sample.delta) || sample.delta == 0.0f ||
                    sample.x > terrain_modifier_sample_coordinate_max ||
                    sample.z > terrain_modifier_sample_coordinate_max)
                    return false;
            auto sorted = samples;
            sort_samples(sorted);
            if (has_duplicate_samples(sorted)) return false;
        }
        else if (modifier.type_id == terrain_builtin_modifier_types::paint_layer)
        {
            if (!std::holds_alternative<terrain_paint_region_payload>(payload.data)) return false;
            const auto& samples = std::get<terrain_paint_region_payload>(payload.data).samples;
            if (samples.empty()) return false;
            if (std::any_of(samples.begin(), samples.end(),
                            [](const auto& sample)
                            {
                                return zero_paint_delta(sample) || sample.x > terrain_modifier_sample_coordinate_max ||
                                       sample.z > terrain_modifier_sample_coordinate_max;
                            }))
                return false;
            auto sorted = samples;
            sort_samples(sorted);
            if (has_duplicate_samples(sorted)) return false;
        }
        else
        {
            return false;
        }
    }
    return true;
}

} // namespace arc::scene

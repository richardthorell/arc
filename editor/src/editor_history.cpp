#include <arc/editor/editor_history.h>

#include <algorithm>

namespace arc::editor
{
namespace
{
constexpr std::size_t max_history_entries = 256;
constexpr std::size_t max_history_bytes = 64u * 1024u * 1024u;
}

std::size_t editor_history::estimate(const editor_scene_state& scene) noexcept
{
    std::size_t result = scene.scene.live_count() * 768u;
    for (const auto& binding : scene.asset_bindings)
        result += sizeof(binding) + binding.source_kind.size() + binding.source_path.native().size() * sizeof(std::filesystem::path::value_type)
            + binding.subresource.size() + binding.material_path.native().size() * sizeof(std::filesystem::path::value_type);
    for (const auto& [_, unknown] : scene.unknown_component_records)
        result += unknown.size();
    scene.scene.view<scene::terrain_component>().each([&](scene::entity, const scene::terrain_component& terrain) {
        result += terrain.heights.size() * sizeof(float) +
            terrain.layer_weights.size() * sizeof(std::array<std::uint8_t, 4>);
    });
    return result;
}

void editor_history::clear(const editor_scene_state&, bool mark_as_saved)
{
    entries_.clear();
    cursor_ = 0;
    bytes_ = 0;
    revision_ = next_revision_++;
    saved_revision_ = mark_as_saved ? revision_ : static_cast<std::uint64_t>(-1);
    transaction_.reset();
    last_terrain_change_.reset();
}

void editor_history::record(std::string label, editor_scene_state before, const editor_scene_state& after)
{
    if (transaction_)
        return;
    while (entries_.size() > cursor_)
    {
        bytes_ -= entries_.back().estimated_bytes;
        entries_.pop_back();
    }
    entry value;
    value.label = std::move(label);
    value.before = std::move(before);
    value.after = after;
    value.before_revision = revision_;
    value.after_revision = next_revision_++;
    value.estimated_bytes = estimate(value.before) + estimate(value.after);
    revision_ = value.after_revision;
    bytes_ += value.estimated_bytes;
    entries_.push_back(std::move(value));
    cursor_ = entries_.size();
    enforce_limits();
}

void editor_history::enforce_limits()
{
    while (!entries_.empty() && (entries_.size() > max_history_entries || bytes_ > max_history_bytes))
    {
        bytes_ -= entries_.front().estimated_bytes;
        entries_.erase(entries_.begin());
        if (cursor_ > 0) --cursor_;
    }
}

bool editor_history::undo(editor_scene_state& scene)
{
    last_terrain_change_.reset();
    if (transaction_ || cursor_ == 0)
        return false;
    const auto& value = entries_[cursor_ - 1];
    if (value.terrain)
    {
        if (!apply_terrain_delta(scene, *value.terrain, false))
            return false;
        last_terrain_change_ = editor_terrain_history_change{
            .entity = value.terrain->entity,
            .region = { value.terrain->min_x, value.terrain->min_z, value.terrain->max_x, value.terrain->max_z, true }
        };
    }
    else
    {
        scene = value.before;
    }
    revision_ = value.before_revision;
    --cursor_;
    return true;
}

bool editor_history::redo(editor_scene_state& scene)
{
    last_terrain_change_.reset();
    if (transaction_ || cursor_ >= entries_.size())
        return false;
    const auto& value = entries_[cursor_];
    if (value.terrain)
    {
        if (!apply_terrain_delta(scene, *value.terrain, true))
            return false;
        last_terrain_change_ = editor_terrain_history_change{
            .entity = value.terrain->entity,
            .region = { value.terrain->min_x, value.terrain->min_z, value.terrain->max_x, value.terrain->max_z, true }
        };
    }
    else
    {
        scene = value.after;
    }
    revision_ = value.after_revision;
    ++cursor_;
    return true;
}

bool editor_history::begin(std::uint64_t transaction_id, std::string label, const editor_scene_state& scene)
{
    if (transaction_id == 0 || transaction_)
        return false;
    transaction_ = transaction{ transaction_id, std::move(label), scene };
    return true;
}

bool editor_history::commit(std::uint64_t transaction_id, const editor_scene_state& scene)
{
    if (!transaction_matches(transaction_id))
        return false;
    auto value = std::move(*transaction_);
    transaction_.reset();
    record(std::move(value.label), std::move(value.before), scene);
    return true;
}

bool editor_history::commit_terrain(
    std::uint64_t transaction_id,
    const editor_scene_state& current_scene,
    scene::entity_guid terrain_guid)
{
    if (!transaction_matches(transaction_id) || !terrain_guid.valid())
        return false;

    auto transaction = std::move(*transaction_);
    transaction_.reset();
    const auto before_entity = find_entity_by_guid(transaction.before, terrain_guid);
    const auto after_entity = find_entity_by_guid(current_scene, terrain_guid);
    const auto* before = transaction.before.scene.try_get<scene::terrain_component>(before_entity);
    const auto* after = current_scene.scene.try_get<scene::terrain_component>(after_entity);
    if (!before || !after || before->subdivisions != after->subdivisions ||
        before->heights.size() != after->heights.size() || before->layer_weights.size() != after->layer_weights.size())
        return false;

    const auto resolution = after->subdivisions + 1u;
    std::uint32_t min_x = resolution;
    std::uint32_t min_z = resolution;
    std::uint32_t max_x = 0u;
    std::uint32_t max_z = 0u;
    bool changed = false;
    for (std::uint32_t z = 0; z < resolution; ++z)
    {
        for (std::uint32_t x = 0; x < resolution; ++x)
        {
            const auto sample = static_cast<std::size_t>(z) * resolution + x;
            bool differs = before->heights[sample] != after->heights[sample];
            for (std::size_t channel = 0; channel < 4u && !differs; ++channel)
                differs = before->layer_weights[sample][channel] != after->layer_weights[sample][channel];
            if (!differs)
                continue;
            changed = true;
            min_x = std::min(min_x, x);
            min_z = std::min(min_z, z);
            max_x = std::max(max_x, x);
            max_z = std::max(max_z, z);
        }
    }
    if (!changed)
        return true;

    entry value;
    value.label = std::move(transaction.label);
    value.before_revision = revision_;
    value.after_revision = next_revision_++;
    value.terrain.emplace();
    auto& delta = *value.terrain;
    delta.entity = terrain_guid;
    delta.min_x = min_x;
    delta.min_z = min_z;
    delta.max_x = max_x;
    delta.max_z = max_z;
    delta.before_content_revision = before->content_revision;
    delta.after_content_revision = after->content_revision;
    const auto sample_count = static_cast<std::size_t>(max_x - min_x + 1u) * (max_z - min_z + 1u);
    delta.before_heights.reserve(sample_count);
    delta.after_heights.reserve(sample_count);
    delta.before_weights.reserve(sample_count * 4u);
    delta.after_weights.reserve(sample_count * 4u);
    for (std::uint32_t z = min_z; z <= max_z; ++z)
    {
        for (std::uint32_t x = min_x; x <= max_x; ++x)
        {
            const auto sample = static_cast<std::size_t>(z) * resolution + x;
            delta.before_heights.push_back(before->heights[sample]);
            delta.after_heights.push_back(after->heights[sample]);
            for (std::size_t channel = 0; channel < 4u; ++channel)
            {
                delta.before_weights.push_back(before->layer_weights[sample][channel]);
                delta.after_weights.push_back(after->layer_weights[sample][channel]);
            }
        }
    }
    value.estimated_bytes = sizeof(value) + value.label.size() +
        (delta.before_heights.size() + delta.after_heights.size()) * sizeof(float) +
        delta.before_weights.size() + delta.after_weights.size();

    while (entries_.size() > cursor_)
    {
        bytes_ -= entries_.back().estimated_bytes;
        entries_.pop_back();
    }
    revision_ = value.after_revision;
    bytes_ += value.estimated_bytes;
    entries_.push_back(std::move(value));
    cursor_ = entries_.size();
    enforce_limits();
    return true;
}

bool editor_history::apply_terrain_delta(
    editor_scene_state& scene_state,
    const entry::terrain_delta& delta,
    bool after)
{
    const auto entity = find_entity_by_guid(scene_state, delta.entity);
    auto* terrain = scene_state.scene.try_get<scene::terrain_component>(entity);
    const auto resolution = terrain ? terrain->subdivisions + 1u : 0u;
    if (!terrain || delta.max_x >= resolution || delta.max_z >= resolution)
        return false;
    const auto& heights = after ? delta.after_heights : delta.before_heights;
    const auto& weights = after ? delta.after_weights : delta.before_weights;
    const auto width = delta.max_x - delta.min_x + 1u;
    const auto height = delta.max_z - delta.min_z + 1u;
    if (heights.size() != static_cast<std::size_t>(width) * height || weights.size() != heights.size() * 4u)
        return false;

    std::size_t source = 0u;
    for (std::uint32_t z = delta.min_z; z <= delta.max_z; ++z)
    {
        for (std::uint32_t x = delta.min_x; x <= delta.max_x; ++x, ++source)
        {
            const auto destination = static_cast<std::size_t>(z) * resolution + x;
            terrain->heights[destination] = heights[source];
            for (std::size_t channel = 0; channel < 4u; ++channel)
                terrain->layer_weights[destination][channel] = weights[source * 4u + channel];
        }
    }
    terrain->content_revision = after ? delta.after_content_revision : delta.before_content_revision;
    return true;
}

bool editor_history::cancel(std::uint64_t transaction_id, editor_scene_state& scene)
{
    if (!transaction_matches(transaction_id))
        return false;
    scene = transaction_->before;
    transaction_.reset();
    return true;
}

bool editor_history::transaction_matches(std::uint64_t transaction_id) const noexcept
{
    return transaction_ && transaction_->id == transaction_id;
}

void editor_history::mark_saved() noexcept
{
    saved_revision_ = revision_;
}

editor_history_snapshot editor_history::snapshot() const
{
    return {
        .can_undo = !transaction_ && cursor_ > 0,
        .can_redo = !transaction_ && cursor_ < entries_.size(),
        .dirty = revision_ != saved_revision_,
        .transaction_active = transaction_.has_value(),
        .undo_label = cursor_ > 0 ? entries_[cursor_ - 1].label : std::string{},
        .redo_label = cursor_ < entries_.size() ? entries_[cursor_].label : std::string{},
        .revision = revision_,
        .saved_revision = saved_revision_
    };
}

const std::optional<editor_terrain_history_change>& editor_history::last_terrain_change() const noexcept
{
    return last_terrain_change_;
}

} // namespace arc::editor

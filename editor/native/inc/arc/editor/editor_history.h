#pragma once

#include <arc/editor/editor_state.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace arc::editor
{

enum class edit_transaction_phase : std::uint8_t { none, begin, update, commit, cancel };

struct editor_history_snapshot
{
    bool can_undo{};
    bool can_redo{};
    bool dirty{};
    bool transaction_active{};
    std::string undo_label;
    std::string redo_label;
    std::uint64_t revision{};
    std::uint64_t saved_revision{};
};

struct editor_terrain_history_change
{
    scene::entity_guid entity;
    scene::terrain_dirty_region region;
};

class editor_history
{
public:
    void clear(const editor_scene_state& scene, bool mark_saved = false);
    void record(std::string label, editor_scene_state before, const editor_scene_state& after);
    bool undo(editor_scene_state& scene);
    bool redo(editor_scene_state& scene);
    bool begin(std::uint64_t transaction_id, std::string label, const editor_scene_state& scene);
    bool commit(std::uint64_t transaction_id, const editor_scene_state& scene);
    bool commit_terrain(
        std::uint64_t transaction_id,
        const editor_scene_state& current_scene,
        scene::entity_guid terrain_guid);
    bool cancel(std::uint64_t transaction_id, editor_scene_state& scene);
    bool transaction_matches(std::uint64_t transaction_id) const noexcept;
    void mark_saved() noexcept;
    editor_history_snapshot snapshot() const;
    const std::optional<editor_terrain_history_change>& last_terrain_change() const noexcept;

private:
    struct entry
    {
        struct terrain_delta
        {
            scene::entity_guid entity;
            std::uint32_t min_x{};
            std::uint32_t min_z{};
            std::uint32_t max_x{};
            std::uint32_t max_z{};
            std::uint64_t before_content_revision{};
            std::uint64_t after_content_revision{};
            std::vector<float> before_heights;
            std::vector<float> after_heights;
            std::vector<std::uint8_t> before_weights;
            std::vector<std::uint8_t> after_weights;
        };

        std::string label;
        editor_scene_state before;
        editor_scene_state after;
        std::optional<terrain_delta> terrain;
        std::uint64_t before_revision{};
        std::uint64_t after_revision{};
        std::size_t estimated_bytes{};
    };
    struct transaction
    {
        std::uint64_t id{};
        std::string label;
        editor_scene_state before;
    };

    void enforce_limits();
    static std::size_t estimate(const editor_scene_state& scene) noexcept;
    static bool apply_terrain_delta(editor_scene_state& scene, const entry::terrain_delta& delta, bool after);

    std::vector<entry> entries_;
    std::size_t cursor_{};
    std::size_t bytes_{};
    std::uint64_t revision_{};
    std::uint64_t next_revision_{ 1 };
    std::uint64_t saved_revision_{ static_cast<std::uint64_t>(-1) };
    std::optional<transaction> transaction_;
    std::optional<editor_terrain_history_change> last_terrain_change_;
};

} // namespace arc::editor

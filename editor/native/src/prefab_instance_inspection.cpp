#include <arc/editor/prefab_document.h>

namespace arc::editor
{

bool is_prefab_instance(const editor_scene_state& state, ecs::entity root)
{
    return state.scene.alive(root) && state.scene.try_get<scene::prefab_instance_component>(root) != nullptr;
}

std::size_t prefab_override_count(const editor_scene_state& state, ecs::entity root)
{
    if (!state.scene.alive(root)) return 0;
    const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    return instance ? instance->overrides.size() : 0;
}

bool prefab_has_overrides(const editor_scene_state& state, ecs::entity root)
{
    return prefab_override_count(state, root) != 0;
}

std::optional<prefab_instance_summary> inspect_prefab_instance(const editor_scene_state& state, ecs::entity root)
{
    if (!state.scene.alive(root)) return std::nullopt;
    const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    if (!instance) return std::nullopt;
    return prefab_instance_summary{
        .prefab_path = instance->prefab_path,
        .override_count = instance->overrides.size(),
        .nested = instance->nested,
    };
}

} // namespace arc::editor

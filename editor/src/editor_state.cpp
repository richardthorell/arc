#include <arc/editor/editor_state.h>

namespace arc::editor
{

const char* selected_entity_name(const editor_scene_state& scene, const char* fallback)
{
    if (!scene.scene.alive(scene.selected_entity))
        return fallback;

    const auto* name = scene.scene.try_get<arc::scene::name_component>(scene.selected_entity);
    return name ? name->value.c_str() : fallback;
}

} // namespace arc::editor

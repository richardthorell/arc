#pragma once

#include <arc/assets/assets.h>
#include <arc/ecs/entity.h>
#include <arc/render/shader.h>

#include <span>
#include <string>

namespace arc::render
{
class renderer;
}
namespace arc::editor
{
struct editor_scene_state;

/** Native entity-local operation; host authority/history checks must precede this call. */
[[nodiscard]] bool set_entity_material_parameter(editor_scene_state& scene, render::renderer& renderer,
                                                 ecs::entity entity, render::shader_parameter_id parameter,
                                                 std::span<const float> values, const assets::asset_reference* texture,
                                                 bool reset, std::string& error, std::string slot_id = {},
                                                 bool clear_texture = false);

[[nodiscard]] bool select_entity_material_function(editor_scene_state& scene, render::renderer& renderer,
                                                   ecs::entity entity, const std::string& slot,
                                                   const assets::asset_reference& function, std::string& error);

/** Remove the local selection and its inputs; the host validates the slot before calling. */
[[nodiscard]] bool reset_entity_material_function(editor_scene_state& scene, render::renderer& renderer,
                                                  ecs::entity entity, const std::string& slot, std::string& error);

void clear_entity_material_overrides(editor_scene_state& scene, ecs::entity entity);

/** Structured readback independent of the compatibility display-name encoding. */
[[nodiscard]] std::string entity_material_overrides_json(const editor_scene_state& scene, ecs::entity entity);
} // namespace arc::editor

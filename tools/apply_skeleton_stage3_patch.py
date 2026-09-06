#!/usr/bin/env python3
from pathlib import Path


def replace(path: str, old: str, new: str, label: str):
    p = Path(path)
    text = p.read_text()
    if old not in text:
        raise SystemExit(f'{label} marker not found in {path}')
    p.write_text(text.replace(old, new, 1))

# Persist imported skeleton authoring data in editor-only state.
replace(
    'editor/native/inc/arc/editor/editor_state.h',
    '''    std::vector<ecs::entity> imported_scene_entities;\n    std::vector<ecs::entity> world_feature_entities;\n''',
    '''    std::vector<ecs::entity> imported_scene_entities;\n    struct imported_skeleton_binding\n    {\n        ecs::entity_guid entity;\n        render::skeleton_asset skeleton;\n    };\n    std::vector<imported_skeleton_binding> imported_skeletons;\n    std::vector<ecs::entity> world_feature_entities;\n''',
    'editor skeleton state',
)
replace(
    'editor/native/inc/arc/editor/editor_state.h',
    '''const editor_scene_state::asset_binding* find_asset_binding(const editor_scene_state& scene,\n                                                            ecs::entity_guid guid) noexcept;\n''',
    '''const editor_scene_state::asset_binding* find_asset_binding(const editor_scene_state& scene,\n                                                            ecs::entity_guid guid) noexcept;\nconst render::skeleton_asset* find_imported_skeleton(const editor_scene_state& scene, ecs::entity entity) noexcept;\n''',
    'skeleton lookup declaration',
)

replace(
    'editor/native/src/editor_state.cpp',
    '''    destroy_entities(state, state.imported_scene_entities);\n    state.selected_entity = {};\n}\n''',
    '''    destroy_entities(state, state.imported_scene_entities);\n    state.imported_skeletons.clear();\n    state.selected_entity = {};\n}\n''',
    'skeleton cleanup',
)
replace(
    'editor/native/src/editor_state.cpp',
    '''        scene.scene.emplace<scene::persistent_id_component>(entity, ecs::generate_entity_guid());\n        scene.scene.emplace<scene::hierarchy_component>(entity);\n        scene.asset_bindings.push_back({.entity = scene.scene.get<scene::persistent_id_component>(entity).value,\n''',
    '''        scene.scene.emplace<scene::persistent_id_component>(entity, ecs::generate_entity_guid());\n        scene.scene.emplace<scene::hierarchy_component>(entity);\n        if (skinned_bound)\n            scene.imported_skeletons.push_back(\n                {.entity = scene.scene.get<scene::persistent_id_component>(entity).value,\n                 .skeleton = imported.skeletons[node.skin_index]});\n        scene.asset_bindings.push_back({.entity = scene.scene.get<scene::persistent_id_component>(entity).value,\n''',
    'store imported skeleton',
)
# Add lookup beside existing asset-binding helpers.
replace(
    'editor/native/src/editor_state.cpp',
    '''const char* selected_entity_name(const editor_scene_state& scene, const char* fallback)\n{\n''',
    '''const render::skeleton_asset* find_imported_skeleton(const editor_scene_state& scene, ecs::entity entity) noexcept\n{\n    if (!scene.scene.alive(entity)) return nullptr;\n    const auto guid = entity_guid_of(scene, entity);\n    if (!guid.valid()) return nullptr;\n    const auto found = std::find_if(scene.imported_skeletons.begin(), scene.imported_skeletons.end(),\n                                    [guid](const editor_scene_state::imported_skeleton_binding& binding)\n                                    { return binding.entity == guid; });\n    return found == scene.imported_skeletons.end() ? nullptr : &found->skeleton;\n}\n\nconst char* selected_entity_name(const editor_scene_state& scene, const char* fallback)\n{\n''',
    'skeleton lookup implementation',
)

# Debug overlay visualization.
replace(
    'editor/native/inc/arc/editor/editor_gizmo.h',
    '''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,\n                                const scene::transform_component& camera_transform, std::uint32_t viewport_height);\n\nrender::debug_overlay_stream build_editor_gizmo_overlay''',
    '''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,\n                                const scene::transform_component& camera_transform, std::uint32_t viewport_height);\n\n/** Append an editor-only bind-pose skeleton visualization without creating bone entities. */\nvoid append_editor_skeleton_overlay(render::debug_overlay_stream& stream, const ecs::world& registry, ecs::entity entity,\n                                    const render::skeleton_asset& skeleton, ecs::entity camera_entity,\n                                    std::uint32_t viewport_height,\n                                    std::uint32_t selected_joint = render::skeleton_asset::invalid_joint);\n\nrender::debug_overlay_stream build_editor_gizmo_overlay''',
    'skeleton overlay declaration',
)

p = Path('editor/native/src/editor_gizmo.cpp')
text = p.read_text()
insert_at = text.rfind('\n} // namespace arc::editor')
if insert_at < 0:
    raise SystemExit('editor_gizmo namespace end not found')
impl = r'''

void append_editor_skeleton_overlay(render::debug_overlay_stream& stream, const ecs::world& registry, ecs::entity entity,
                                    const render::skeleton_asset& skeleton, ecs::entity camera_entity,
                                    std::uint32_t viewport_height, std::uint32_t selected_joint)
{
    if (!skeleton.valid() || !registry.alive(entity)) return;
    const auto* entity_transform = registry.try_get<scene::transform_component>(entity);
    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);
    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);
    if (!entity_transform || !camera || !camera_transform) return;

    const auto owner_world = entity_transform->dirty ? scene::local_matrix(*entity_transform) : entity_transform->world;
    std::vector<math::matrix4f> joint_world(skeleton.joints.size(), math::identity<float, 4>());
    std::vector<std::uint8_t> state(skeleton.joints.size());
    std::function<bool(std::size_t)> evaluate = [&](std::size_t index)
    {
        if (index >= skeleton.joints.size()) return false;
        if (state[index] == 2u) return true;
        if (state[index] == 1u) return false;
        state[index] = 1u;
        const auto& joint = skeleton.joints[index];
        scene::transform_component local;
        local.position = joint.bind_position;
        local.rotation = joint.bind_rotation;
        local.scale = joint.bind_scale;
        auto world = scene::local_matrix(local);
        if (joint.parent >= 0)
        {
            const auto parent = static_cast<std::size_t>(joint.parent);
            if (parent >= skeleton.joints.size() || !evaluate(parent)) return false;
            world = math::matmul(joint_world[parent], world);
        }
        joint_world[index] = world;
        state[index] = 2u;
        return true;
    };
    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
        if (!evaluate(joint)) return;

    const auto is_descendant = [&](std::size_t joint)
    {
        if (selected_joint >= skeleton.joints.size() || joint == selected_joint) return false;
        auto parent = skeleton.joints[joint].parent;
        while (parent >= 0)
        {
            if (static_cast<std::uint32_t>(parent) == selected_joint) return true;
            if (static_cast<std::size_t>(parent) >= skeleton.joints.size()) return false;
            parent = skeleton.joints[static_cast<std::size_t>(parent)].parent;
        }
        return false;
    };

    constexpr math::vector4f normal_color{0.45f, 0.72f, 0.86f, 0.72f};
    constexpr math::vector4f descendant_color{0.55f, 0.82f, 0.96f, 0.95f};
    constexpr math::vector4f selected_color{1.0f, 0.78f, 0.16f, 1.0f};
    for (std::size_t joint = 0; joint < skeleton.joints.size(); ++joint)
    {
        const auto position = math::transform_point(owner_world, math::transform_point(joint_world[joint], math::vector3f::zero));
        const bool selected = joint == selected_joint;
        const bool descendant = is_descendant(joint);
        const auto color = selected ? selected_color : descendant ? descendant_color : normal_color;
        if (skeleton.joints[joint].parent >= 0)
        {
            const auto parent = static_cast<std::size_t>(skeleton.joints[joint].parent);
            if (parent < joint_world.size())
            {
                const auto parent_position = math::transform_point(
                    owner_world, math::transform_point(joint_world[parent], math::vector3f::zero));
                const auto bone_color = (selected || descendant) ? descendant_color : normal_color;
                append_overlay_line(stream, parent_position, position, bone_color);
            }
        }

        const float marker = editor_gizmo_world_scale(*camera, *camera_transform, position, viewport_height) *
                             (selected ? 0.055f : 0.035f);
        append_overlay_line(stream, math::add(position, {-marker, 0.0f, 0.0f}),
                            math::add(position, {marker, 0.0f, 0.0f}), color);
        append_overlay_line(stream, math::add(position, {0.0f, -marker, 0.0f}),
                            math::add(position, {0.0f, marker, 0.0f}), color);
        append_overlay_line(stream, math::add(position, {0.0f, 0.0f, -marker}),
                            math::add(position, {0.0f, 0.0f, marker}), color);
    }
}
'''
p.write_text(text[:insert_at] + impl + text[insert_at:])

# Viewport protocol: global skeleton visibility plus selected bone sub-selection.
replace(
    'editor/native/inc/arc/editor/host_protocol_base.h',
    '''    bool shadows{true};\n    bool grid{true};\n    bool realtime{true};\n    float camera_speed{4.0f};\n''',
    '''    bool shadows{true};\n    bool grid{true};\n    bool skeletons{};\n    bool realtime{true};\n    float camera_speed{4.0f};\n''',
    'render options skeleton toggle',
)
replace(
    'editor/native/inc/arc/editor/host_protocol_base.h',
    '''struct host_viewport_camera_input_command\n{\n''',
    '''struct host_viewport_set_skeleton_joint_command\n{\n    std::string viewport_id{"viewport-1"};\n    std::uint32_t joint_index{render::skeleton_asset::invalid_joint};\n};\n\nstruct host_viewport_camera_input_command\n{\n''',
    'selected skeleton joint command',
)
replace(
    'editor/native/inc/arc/editor/host_protocol_base.h',
    '''    host_viewport_pointer_command, host_viewport_key_command, host_viewport_set_camera_mode_command,\n    host_viewport_set_render_options_command, host_viewport_camera_input_command, host_viewport_set_pose_command,\n''',
    '''    host_viewport_pointer_command, host_viewport_key_command, host_viewport_set_camera_mode_command,\n    host_viewport_set_render_options_command, host_viewport_set_skeleton_joint_command, host_viewport_camera_input_command,\n    host_viewport_set_pose_command,\n''',
    'command variant skeleton selection',
)
# second bool block belongs to host_viewport_request; replace remaining occurrence.
replace(
    'editor/native/inc/arc/editor/host_protocol_base.h',
    '''    bool shadows{true};\n    bool grid{true};\n    bool realtime{true};\n    float camera_speed{4.0f};\n''',
    '''    bool shadows{true};\n    bool grid{true};\n    bool skeletons{};\n    bool realtime{true};\n    float camera_speed{4.0f};\n    std::uint32_t selected_skeleton_joint{render::skeleton_asset::invalid_joint};\n''',
    'viewport request skeleton fields',
)

# Protocol serializer/parser edits.
replace(
    'editor/native/src/host_protocol_base.inc',
    '''            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)\n                return "viewport.setRenderOptions";\n            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n''',
    '''            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)\n                return "viewport.setRenderOptions";\n            else if constexpr (std::is_same_v<type, host_viewport_set_skeleton_joint_command>)\n                return "viewport.setSkeletonJoint";\n            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n''',
    'command type skeleton joint',
)
# There are duplicate visit blocks; update all remaining command type occurrence.
p = Path('editor/native/src/host_protocol_base.inc')
text = p.read_text().replace(
    '''            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)\n                return "viewport.setRenderOptions";\n            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n''',
    '''            else if constexpr (std::is_same_v<type, host_viewport_set_render_options_command>)\n                return "viewport.setRenderOptions";\n            else if constexpr (std::is_same_v<type, host_viewport_set_skeleton_joint_command>)\n                return "viewport.setSkeletonJoint";\n            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n''')
p.write_text(text)

# JSON serialization of setRenderOptions, add skeletons and new command.
p = Path('editor/native/src/host_protocol_base.inc')
text = p.read_text()
text = text.replace(
    ''',\\"shadows\\":" + bool_json(payload.shadows) + ",\\"grid\\":" + bool_json(payload.grid) +''',
    ''',\\"shadows\\":" + bool_json(payload.shadows) + ",\\"grid\\":" + bool_json(payload.grid) +\n                       ",\\"skeletons\\":" + bool_json(payload.skeletons) +''')
text = text.replace(
    '''            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n                return "{\\"viewportId\\":" + quote(payload.viewport_id) +''',
    '''            else if constexpr (std::is_same_v<type, host_viewport_set_skeleton_joint_command>)\n                return "{\\"viewportId\\":" + quote(payload.viewport_id) + ",\\"jointIndex\\":" +\n                       std::to_string(payload.joint_index) + '}';\n            else if constexpr (std::is_same_v<type, host_viewport_camera_input_command>)\n                return "{\\"viewportId\\":" + quote(payload.viewport_id) +''')
# Parser: skeleton toggle and dedicated command.
text = text.replace(
    '''        bool_value(payload, "shadows", command.shadows);\n        bool_value(payload, "grid", command.grid);\n''',
    '''        bool_value(payload, "shadows", command.shadows);\n        bool_value(payload, "grid", command.grid);\n        bool_value(payload, "skeletons", command.skeletons);\n''')
needle = '''    if (type == "viewport.cameraInput")\n    {\n'''
addition = '''    if (type == "viewport.setSkeletonJoint")\n    {\n        host_viewport_set_skeleton_joint_command command;\n        string_value(payload, "viewportId", command.viewport_id);\n        if (!number_value(payload, "jointIndex", command.joint_index))\n        {\n            error = "viewport.setSkeletonJoint requires jointIndex";\n            return false;\n        }\n        envelope.payload = command;\n        return true;\n    }\n'''
if needle not in text:
    raise SystemExit('viewport camera input parser marker not found')
text = text.replace(needle, addition + needle, 1)
p.write_text(text)

# Base host updates options and handles sub-selection command.
replace(
    'editor/native/src/arc_host_base.inc',
    '''                state_->viewport_options.grid = payload.grid;\n                state_->viewport_options.realtime = payload.realtime;\n''',
    '''                state_->viewport_options.grid = payload.grid;\n                state_->viewport_options.skeletons = payload.skeletons;\n                state_->viewport_options.realtime = payload.realtime;\n''',
    'store skeleton visibility',
)
replace(
    'editor/native/src/arc_host_base.inc',
    '''            else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)\n''',
    '''            else if constexpr (std::is_same_v<command_type, host_viewport_set_skeleton_joint_command>)\n            {\n                if (payload.viewport_id != state_->active_viewport_id) return fail("Viewport is not attached");\n                const auto* skeleton = find_imported_skeleton(state_->scene, state_->scene.selected_entity);\n                if (!skeleton) return fail("Selected entity has no imported skeleton");\n                if (payload.joint_index != render::skeleton_asset::invalid_joint &&\n                    payload.joint_index >= skeleton->joints.size())\n                    return fail("Skeleton joint index is out of range");\n                state_->viewport_options.selected_skeleton_joint = payload.joint_index;\n                return success();\n            }\n            else if constexpr (std::is_same_v<command_type, host_viewport_camera_input_command>)\n''',
    'handle skeleton joint selection',
)
# Rendering overlays: selected skeleton always; all when Show Skeletons enabled.
replace(
    'editor/native/src/arc_host_base.inc',
    '''    if (state_->viewport_options.grid)\n    {\n''',
    '''    const auto append_skeleton = [&](ecs::entity entity, std::uint32_t selected_joint)\n    {\n        if (const auto* skeleton = find_imported_skeleton(state_->scene, entity))\n            append_editor_skeleton_overlay(debug_overlay, state_->scene.scene, entity, *skeleton, state_->scene.camera_entity,\n                                           request.height, selected_joint);\n    };\n    if (state_->viewport_options.skeletons)\n    {\n        for (const auto& binding : state_->scene.imported_skeletons)\n        {\n            const auto entity = find_entity_by_guid(state_->scene, binding.entity);\n            if (!state_->scene.scene.alive(entity)) continue;\n            append_skeleton(entity, entity == state_->scene.selected_entity ? state_->viewport_options.selected_skeleton_joint\n                                                                            : render::skeleton_asset::invalid_joint);\n        }\n    }\n    else if (find_imported_skeleton(state_->scene, state_->scene.selected_entity))\n    {\n        append_skeleton(state_->scene.selected_entity, state_->viewport_options.selected_skeleton_joint);\n    }\n    if (state_->viewport_options.grid)\n    {\n''',
    'append skeleton overlays',
)

# Extend selected-entity JSON dynamically with read-only skeleton details.
replace(
    'editor/native/src/arc_host.cpp',
    '''                if (const auto* procedural = const_query_scene.try_get<procedural_mesh_component>(entity))\n                {\n                    auto procedural_json =\n                        nlohmann::json::parse(procedural_mesh_snapshot_json(*procedural), nullptr, false);\n                    if (!procedural_json.is_discarded()) payload["proceduralMesh"] = std::move(procedural_json);\n                }\n''',
    '''                if (const auto* procedural = const_query_scene.try_get<procedural_mesh_component>(entity))\n                {\n                    auto procedural_json =\n                        nlohmann::json::parse(procedural_mesh_snapshot_json(*procedural), nullptr, false);\n                    if (!procedural_json.is_discarded()) payload["proceduralMesh"] = std::move(procedural_json);\n                }\n                if (const auto* skeleton = find_imported_skeleton(query_scene, entity))\n                {\n                    nlohmann::json skeleton_json;\n                    skeleton_json["name"] = skeleton->name.empty() ? "Imported Skeleton" : skeleton->name;\n                    skeleton_json["selectedJoint"] = viewport_surface->options.selected_skeleton_joint;\n                    skeleton_json["joints"] = nlohmann::json::array();\n                    for (std::size_t index = 0; index < skeleton->joints.size(); ++index)\n                    {\n                        const auto& joint = skeleton->joints[index];\n                        skeleton_json["joints"].push_back(\n                            {{"index", index},\n                             {"name", joint.name.empty() ? "Joint " + std::to_string(index) : joint.name},\n                             {"parent", joint.parent},\n                             {"bindPosition", {joint.bind_position[0], joint.bind_position[1], joint.bind_position[2]}},\n                             {"bindRotation",\n                              {joint.bind_rotation[0], joint.bind_rotation[1], joint.bind_rotation[2], joint.bind_rotation[3]}},\n                             {"bindScale", {joint.bind_scale[0], joint.bind_scale[1], joint.bind_scale[2]}}});\n                    }\n                    payload["skeleton"] = std::move(skeleton_json);\n                }\n''',
    'selected entity skeleton json',
)

# Frontend render option.
replace(
    'editor/src/renderer/src/viewport/ViewportPanel.tsx',
    '''  shadows: boolean;\n  grid: boolean;\n  realtime: boolean;\n''',
    '''  shadows: boolean;\n  grid: boolean;\n  skeletons: boolean;\n  realtime: boolean;\n''',
    'viewport skeleton type',
)
replace(
    'editor/src/renderer/src/viewport/ViewportPanel.tsx',
    '''  shadows: true,\n  grid: true,\n  realtime: true,\n''',
    '''  shadows: true,\n  grid: true,\n  skeletons: false,\n  realtime: true,\n''',
    'viewport skeleton default',
)
replace(
    'editor/src/renderer/src/viewport/ViewportPanel.tsx',
    '''              <button\n                role="menuitemcheckbox"\n                aria-checked={renderOptions.shadows}\n''',
    '''              <button\n                role="menuitemcheckbox"\n                aria-checked={renderOptions.skeletons}\n                onClick={() => void updateRenderOptions({ skeletons: !renderOptions.skeletons })}\n              >\n                <span className="arc-viewport-menu-check">{renderOptions.skeletons ? '✓' : ''}</span>Skeletons\n              </button>\n              <button\n                role="menuitemcheckbox"\n                aria-checked={renderOptions.shadows}\n''',
    'show skeletons menu',
)

# Inspector snapshot skeleton types and parsing.
replace(
    'editor/src/renderer/src/inspector/inspectorTypes.ts',
    '''export type InspectorEntitySnapshot = Omit<BaseInspectorEntitySnapshot, 'meshRenderer'> & {\n  meshRenderer: InspectorMeshRenderer | null;\n  proceduralMesh?: InspectorProceduralMesh | null;\n};\n''',
    '''export type InspectorSkeletonJoint = {\n  index: number;\n  name: string;\n  parent: number;\n  bindPosition: [number, number, number];\n  bindRotation: [number, number, number, number];\n  bindScale: [number, number, number];\n};\n\nexport type InspectorSkeleton = {\n  name: string;\n  selectedJoint: number;\n  joints: InspectorSkeletonJoint[];\n};\n\nexport type InspectorEntitySnapshot = Omit<BaseInspectorEntitySnapshot, 'meshRenderer'> & {\n  meshRenderer: InspectorMeshRenderer | null;\n  proceduralMesh?: InspectorProceduralMesh | null;\n  skeleton?: InspectorSkeleton | null;\n};\n''',
    'inspector skeleton types',
)
replace(
    'editor/src/renderer/src/inspector/inspectorTypes.ts',
    '''function parseProceduralMesh(value: unknown): InspectorProceduralMesh | null {\n''',
    '''const tuple = (value: unknown, length: number): number[] | null => {\n  if (!Array.isArray(value) || value.length !== length || value.some((entry) => typeof entry !== 'number')) return null;\n  return value as number[];\n};\n\nfunction parseSkeleton(value: unknown): InspectorSkeleton | null {\n  if (!value || typeof value !== 'object') return null;\n  const raw = value as Record<string, unknown>;\n  if (!Array.isArray(raw.joints)) return null;\n  const joints = raw.joints.flatMap((entry) => {\n    if (!entry || typeof entry !== 'object') return [];\n    const joint = entry as Record<string, unknown>;\n    const bindPosition = tuple(joint.bindPosition, 3);\n    const bindRotation = tuple(joint.bindRotation, 4);\n    const bindScale = tuple(joint.bindScale, 3);\n    if (typeof joint.index !== 'number' || typeof joint.parent !== 'number' || !bindPosition || !bindRotation || !bindScale)\n      return [];\n    return [{\n      index: joint.index,\n      name: typeof joint.name === 'string' ? joint.name : `Joint ${joint.index}`,\n      parent: joint.parent,\n      bindPosition: bindPosition as [number, number, number],\n      bindRotation: bindRotation as [number, number, number, number],\n      bindScale: bindScale as [number, number, number],\n    }];\n  });\n  if (!joints.length) return null;\n  return {\n    name: typeof raw.name === 'string' ? raw.name : 'Skeleton',\n    selectedJoint: typeof raw.selectedJoint === 'number' ? raw.selectedJoint : joints[0].index,\n    joints,\n  };\n}\n\nfunction parseProceduralMesh(value: unknown): InspectorProceduralMesh | null {\n''',
    'parse skeleton helper',
)
replace(
    'editor/src/renderer/src/inspector/inspectorTypes.ts',
    '''    proceduralMesh: parseProceduralMesh(raw?.proceduralMesh),\n  };\n''',
    '''    proceduralMesh: parseProceduralMesh(raw?.proceduralMesh),\n    skeleton: parseSkeleton(raw?.skeleton),\n  };\n''',
    'parse skeleton snapshot',
)

# New inspector component.
Path('editor/src/renderer/src/inspector/SkeletonInspector.tsx').write_text(r'''import { useEffect, useMemo, useState } from 'react';

import { UiTreeView } from '../ui';
import type { UiTreeNode } from '../ui';
import type { InspectorCommand } from './InspectorPanel';
import type { InspectorSkeleton } from './inspectorTypes';

import './skeletonInspector.css';

type SkeletonInspectorProps = {
  skeleton: InspectorSkeleton;
  command: InspectorCommand;
  viewportId?: string;
  onStatus?: (message: string) => void;
};

const formatVector = (values: readonly number[]) => values.map((value) => Number(value.toFixed(4))).join(', ');

function buildTree(skeleton: InspectorSkeleton): UiTreeNode[] {
  const children = new Map<number, InspectorSkeleton['joints']>();
  for (const joint of skeleton.joints) {
    const list = children.get(joint.parent) ?? [];
    list.push(joint);
    children.set(joint.parent, list);
  }
  const build = (joint: InspectorSkeleton['joints'][number]): UiTreeNode => ({
    id: String(joint.index),
    label: joint.name,
    keywords: [joint.name],
    children: (children.get(joint.index) ?? []).map(build),
  });
  return (children.get(-1) ?? skeleton.joints.filter((joint) => !skeleton.joints.some((candidate) => candidate.index === joint.parent))).map(build);
}

export function SkeletonInspector({ skeleton, command, viewportId = 'viewport-1', onStatus }: SkeletonInspectorProps) {
  const fallback = skeleton.joints[0]?.index ?? -1;
  const initial = skeleton.joints.some((joint) => joint.index === skeleton.selectedJoint) ? skeleton.selectedJoint : fallback;
  const [selectedJoint, setSelectedJoint] = useState(initial);
  useEffect(() => setSelectedJoint(initial), [initial, skeleton.name]);
  const nodes = useMemo(() => buildTree(skeleton), [skeleton]);
  const joint = skeleton.joints.find((candidate) => candidate.index === selectedJoint) ?? skeleton.joints[0];
  const parent = joint?.parent >= 0 ? skeleton.joints.find((candidate) => candidate.index === joint.parent) : undefined;
  const expanded = useMemo(() => new Set(skeleton.joints.filter((candidate) => candidate.parent >= 0).map((candidate) => String(candidate.parent))), [skeleton]);
  if (!joint) return null;

  return (
    <section className="inspector-component-card skeleton-inspector" aria-label="Skeleton">
      <header><strong>Skeleton</strong><span>{skeleton.name}</span></header>
      <UiTreeView
        ariaLabel="Skeleton hierarchy"
        nodes={nodes}
        defaultExpandedIds={[...expanded]}
        selectedId={String(selectedJoint)}
        onSelect={(node) => {
          const next = Number(node.id);
          setSelectedJoint(next);
          void command('viewport.setSkeletonJoint', { viewportId, jointIndex: next }).then((response) => {
            if (!response.succeeded) onStatus?.(response.error || 'Could not select skeleton bone');
            else onStatus?.(`Selected bone ${skeleton.joints.find((candidate) => candidate.index === next)?.name ?? next}`);
          });
        }}
      />
      <div className="skeleton-inspector-details">
        <label><span>Selected Bone</span><output>{joint.name}</output></label>
        <label><span>Parent</span><output>{parent?.name ?? 'None'}</output></label>
        <fieldset>
          <legend>Bind Transform</legend>
          <label><span>Position</span><output>{formatVector(joint.bindPosition)}</output></label>
          <label><span>Rotation</span><output>{formatVector(joint.bindRotation)}</output></label>
          <label><span>Scale</span><output>{formatVector(joint.bindScale)}</output></label>
        </fieldset>
      </div>
    </section>
  );
}
''')
Path('editor/src/renderer/src/inspector/skeletonInspector.css').write_text(r'''.skeleton-inspector {
  display: grid;
  gap: 8px;
  padding: 10px;
}

.skeleton-inspector > header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.skeleton-inspector > header > span,
.skeleton-inspector-details output {
  color: var(--arc-color-text-muted);
}

.skeleton-inspector .ui-tree-view {
  max-height: 240px;
  overflow: auto;
  border: 1px solid var(--arc-color-border-subtle);
  border-radius: var(--arc-radius-sm);
}

.skeleton-inspector-details,
.skeleton-inspector-details fieldset {
  display: grid;
  gap: 6px;
}

.skeleton-inspector-details > label,
.skeleton-inspector-details fieldset > label {
  display: grid;
  grid-template-columns: 96px minmax(0, 1fr);
  gap: 8px;
}

.skeleton-inspector-details fieldset {
  margin: 2px 0 0;
  padding: 8px;
  border: 1px solid var(--arc-color-border-subtle);
  border-radius: var(--arc-radius-sm);
}
''')
replace(
    'editor/src/renderer/src/inspector/InspectorPanel.tsx',
    '''import { SchemaComponentCard } from './SchemaComponents';\n''',
    '''import { SchemaComponentCard } from './SchemaComponents';\nimport { SkeletonInspector } from './SkeletonInspector';\n''',
    'skeleton inspector import',
)
replace(
    'editor/src/renderer/src/inspector/InspectorPanel.tsx',
    '''      <div className="inspector-component-list">\n        {draft.prefab && (\n''',
    '''      <div className="inspector-component-list">\n        {draft.skeleton && <SkeletonInspector skeleton={draft.skeleton} command={command} onStatus={onStatus} />}\n        {draft.prefab && (\n''',
    'skeleton inspector card',
)

# Frontend test for skeleton inspector.
Path('editor/src/renderer/src/inspector/SkeletonInspector.test.tsx').write_text(r'''// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SkeletonInspector } from './SkeletonInspector';

afterEach(cleanup);

describe('SkeletonInspector', () => {
  it('shows the hierarchy and selects a bone without creating an entity', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, error: '' });
    render(
      <SkeletonInspector
        command={command}
        skeleton={{
          name: 'Character',
          selectedJoint: 0,
          joints: [
            { index: 0, name: 'Hips', parent: -1, bindPosition: [0, 0, 0], bindRotation: [0, 0, 0, 1], bindScale: [1, 1, 1] },
            { index: 1, name: 'Spine', parent: 0, bindPosition: [0, 1, 0], bindRotation: [0, 0, 0, 1], bindScale: [1, 1, 1] },
          ],
        }}
      />,
    );

    expect(screen.getByRole('tree', { name: 'Skeleton hierarchy' })).toBeVisible();
    expect(screen.getByText('Hips')).toBeVisible();
    fireEvent.click(screen.getByText('Spine'));
    await waitFor(() => expect(command).toHaveBeenCalledWith('viewport.setSkeletonJoint', { viewportId: 'viewport-1', jointIndex: 1 }));
    expect(screen.getByText('Selected Bone').nextSibling).toHaveTextContent('Spine');
    expect(screen.getByText('Parent').nextSibling).toHaveTextContent('Hips');
  });
});
''')

# Native Stage 3 regression: imported skeleton metadata retained/cleaned.
p = Path('editor/native/tests/skinned_scene_import_tests.cpp')
text = p.read_text()
append = r'''

TEST_CASE("imported skinned entities retain editor skeleton metadata", "[editor][skeleton][visualization]")
{
    arc::editor::editor_scene_state state;
    arc::render::renderer renderer;
    REQUIRE(arc::editor::apply_scene_import_result_to_editor(
                state, renderer, "assets/character.glb", make_skinned_scene(), arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    REQUIRE(state.imported_scene_entities.size() == 1u);
    const auto entity = state.imported_scene_entities.front();
    const auto* skeleton = arc::editor::find_imported_skeleton(state, entity);
    REQUIRE(skeleton != nullptr);
    CHECK(skeleton->name == "CharacterRig");
    REQUIRE(skeleton->joints.size() == 2u);
    CHECK(skeleton->joints[1].parent == 0);

    REQUIRE(arc::editor::apply_scene_import_result_to_editor(
                state, renderer, "assets/prop.glb", make_static_scene(), arc::editor::editor_scene_open_mode::replace)
                .succeeded);
    CHECK(state.imported_skeletons.empty());
}
'''
p.write_text(text + append)

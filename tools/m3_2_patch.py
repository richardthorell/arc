from pathlib import Path
import re


def replace_once(path: str, old: str, new: str) -> None:
    source = Path(path)
    text = source.read_text()
    if old not in text:
        raise SystemExit(f"expected block not found in {path}: {old[:100]!r}")
    source.write_text(text.replace(old, new, 1))


def regex_once(path: str, pattern: str, replacement: str, flags: int = 0) -> None:
    source = Path(path)
    text = source.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise SystemExit(f"expected regex not found in {path}: {pattern[:100]!r}")
    source.write_text(updated)


# Native command contract.
protocol_header = "editor/native/inc/arc/editor/host_protocol_base.h"
replace_once(
    protocol_header,
    "struct host_set_terrain_layer_command\n{\n    host_entity_id entity{};\n    std::uint32_t layer{};\n    std::filesystem::path path;\n};",
    """struct host_terrain_modifier_stack_command
{
    host_entity_id entity{};
    std::string operation{"inspect"};
    std::string modifier;
    std::string name;
    bool enabled{true};
    std::int32_t index{-1};
};

struct host_set_terrain_layer_command
{
    host_entity_id entity{};
    std::uint32_t layer{};
    std::filesystem::path path;
};""",
)
replace_once(
    protocol_header,
    "host_set_water_command, host_set_terrain_brush_command, host_set_terrain_layer_command, host_create_terrain_command,",
    "host_set_water_command, host_set_terrain_brush_command, host_terrain_modifier_stack_command, "
    "host_set_terrain_layer_command, host_create_terrain_command,",
)

protocol_source = "editor/native/src/host_protocol_base.inc"
replace_once(
    protocol_source,
    """            else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>)
                return "terrain.setBrush";
            else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>)
                return "terrain.assignLayer";""",
    """            else if constexpr (std::is_same_v<type, host_set_terrain_brush_command>)
                return "terrain.setBrush";
            else if constexpr (std::is_same_v<type, host_terrain_modifier_stack_command>)
                return "terrain.modifierStack";
            else if constexpr (std::is_same_v<type, host_set_terrain_layer_command>)
                return "terrain.assignLayer";""",
)
replace_once(
    protocol_source,
    """    else if (type == "terrain.create")
    {""",
    """    else if (type == "terrain.modifierStack")
    {
        host_terrain_modifier_stack_command command;
        if (!entity_field_value(payload, "entity", command.entity) ||
            !string_value(payload, "operation", command.operation))
        {
            error = "Terrain modifier stack requires entity and operation";
            return false;
        }
        string_value(payload, "modifier", command.modifier);
        string_value(payload, "name", command.name);
        bool_value(payload, "enabled", command.enabled);
        number_value(payload, "index", command.index);
        envelope.payload = std::move(command);
    }
    else if (type == "terrain.create")
    {""",
)

# Native editor host implementation.
host_source = "editor/native/src/arc_host_base.inc"
replace_once(host_source, "#include <filesystem>\n", "#include <filesystem>\n#include <fstream>\n")
replace_once(
    host_source,
    """           std::is_same_v<Command, host_set_terrain_command> || std::is_same_v<Command, host_set_water_command> ||
           std::is_same_v<Command, host_terrain_stroke_command> ||""",
    """           std::is_same_v<Command, host_set_terrain_command> || std::is_same_v<Command, host_set_water_command> ||
           std::is_same_v<Command, host_terrain_modifier_stack_command> ||
           std::is_same_v<Command, host_terrain_stroke_command> ||""",
)

terrain_stack_handler = r'''            else if constexpr (std::is_same_v<command_type, host_terrain_modifier_stack_command>)
            {
                const auto entity = to_scene_entity(payload.entity);
                auto* terrain = state_->scene.scene.try_get<scene::terrain_component>(entity);
                if (!terrain) return fail("Terrain modifier stack requires a terrain entity", entity);

                const auto stack_json = [&](const scene::terrain_asset* asset, bool asset_backed, bool read_only,
                                            std::string_view asset_path)
                {
                    nlohmann::json json;
                    json["entity"] = {{"index", payload.entity.index}, {"generation", payload.entity.generation}};
                    json["assetBacked"] = asset_backed;
                    json["readOnly"] = read_only;
                    json["assetPath"] = asset_path;
                    json["authoringRevision"] = asset ? asset->authoring_revision : 0u;
                    json["modifiers"] = nlohmann::json::array();
                    if (asset)
                    {
                        for (const auto& modifier : asset->modifiers)
                        {
                            std::string type = "unknown";
                            if (modifier.type_id == scene::terrain_builtin_modifier_types::sculpt_layer)
                                type = "sculpt";
                            else if (modifier.type_id == scene::terrain_builtin_modifier_types::paint_layer)
                                type = "paint";
                            json["modifiers"].push_back({{"id", scene::to_string(modifier.id)},
                                                         {"name", modifier.name},
                                                         {"type", type},
                                                         {"typeId", modifier.type_id},
                                                         {"enabled", modifier.enabled},
                                                         {"regionPayloads", modifier.region_payloads.size()}});
                        }
                    }
                    return json.dump();
                };

                if (!terrain->asset.guid.valid() && terrain->asset.path_hint.empty())
                    return success(stack_json(nullptr, false, false, std::string_view{}));
                if (!state_->asset_registry) return fail("Terrain asset registry is unavailable", entity);

                auto reference = terrain->asset;
                reference.expected_type = assets::asset_types::terrain;
                if (!reference.guid.valid() && !reference.path_hint.empty())
                    reference = state_->asset_registry->resolve(reference.path_hint, assets::asset_types::terrain);
                if (!reference.guid.valid()) return fail("Terrain asset reference could not be resolved", entity);

                auto pending = state_->asset_registry->load<scene::terrain_asset>(
                    {.reference = reference,
                     .priority = assets::asset_streaming_priority::high,
                     .residency = assets::asset_residency::cpu,
                     .allow_fallback = false});
                auto loaded = pending.get();
                if (!loaded || !loaded.asset.get())
                    return fail(loaded.error.message.empty() ? "Terrain asset could not be loaded" : loaded.error.message,
                                entity);

                auto authored = *loaded.asset.get();
                const auto asset_snapshot = state_->asset_registry->find(loaded.asset.resolved_guid());
                const bool read_only = asset_snapshot && asset_snapshot->read_only;
                const auto source_reference = !terrain->asset.path_hint.empty()
                                                  ? std::filesystem::path(terrain->asset.path_hint)
                                                  : asset_snapshot ? asset_snapshot->source_path
                                                                   : std::filesystem::path{};
                if (payload.operation == "inspect")
                    return success(stack_json(&authored, true, read_only, source_reference.generic_string()));
                if (read_only) return fail("Terrain asset is read only", entity);

                const auto bump_revision = [&]
                {
                    if (authored.authoring_revision != std::numeric_limits<std::uint64_t>::max())
                        ++authored.authoring_revision;
                };
                const auto parsed_id = scene::parse_terrain_stable_id(payload.modifier);
                const auto find_modifier = [&]() -> scene::terrain_modifier_descriptor*
                { return parsed_id ? scene::find_terrain_modifier(authored, *parsed_id) : nullptr; };

                if (payload.operation == "add_sculpt")
                    scene::add_terrain_sculpt_layer(authored);
                else if (payload.operation == "add_paint")
                    scene::add_terrain_paint_layer(authored);
                else if (payload.operation == "rename")
                {
                    auto* modifier = find_modifier();
                    if (!modifier || payload.name.empty())
                        return fail("Terrain modifier rename requires a valid layer and name", entity);
                    if (modifier->name != payload.name)
                    {
                        modifier->name = payload.name;
                        bump_revision();
                    }
                }
                else if (payload.operation == "set_enabled")
                {
                    auto* modifier = find_modifier();
                    if (!modifier) return fail("Terrain modifier was not found", entity);
                    if (modifier->enabled != payload.enabled)
                    {
                        modifier->enabled = payload.enabled;
                        bump_revision();
                    }
                }
                else if (payload.operation == "move")
                {
                    if (!parsed_id || authored.modifiers.empty()) return fail("Terrain modifier was not found", entity);
                    const auto found = std::find_if(authored.modifiers.begin(), authored.modifiers.end(),
                                                    [&](const auto& value) { return value.id == *parsed_id; });
                    if (found == authored.modifiers.end()) return fail("Terrain modifier was not found", entity);
                    const auto target = static_cast<std::size_t>(std::clamp<std::int32_t>(
                        payload.index, 0, static_cast<std::int32_t>(authored.modifiers.size() - 1u)));
                    const auto current = static_cast<std::size_t>(std::distance(authored.modifiers.begin(), found));
                    if (current != target)
                    {
                        auto moved = std::move(*found);
                        authored.modifiers.erase(authored.modifiers.begin() + static_cast<std::ptrdiff_t>(current));
                        authored.modifiers.insert(authored.modifiers.begin() + static_cast<std::ptrdiff_t>(target),
                                                 std::move(moved));
                        bump_revision();
                    }
                }
                else if (payload.operation == "erase")
                {
                    if (!parsed_id) return fail("Terrain modifier was not found", entity);
                    const auto found = std::find_if(authored.modifiers.begin(), authored.modifiers.end(),
                                                    [&](const auto& value) { return value.id == *parsed_id; });
                    if (found == authored.modifiers.end()) return fail("Terrain modifier was not found", entity);
                    authored.modifiers.erase(found);
                    bump_revision();
                }
                else
                {
                    return fail("Unknown terrain modifier stack operation", entity);
                }

                const auto encoded = scene::write_terrain_asset_json(authored, true);
                if (!encoded) return fail("Terrain asset could not be serialized", entity);
                const auto resolved = resolve_editor_asset(state_->assets, state_->asset_registry.get(),
                                                            state_->project.root, source_reference);
                if (!resolved || resolved->read_only) return fail("Terrain asset source is not writable", entity);
                {
                    std::ofstream output(resolved->path, std::ios::binary | std::ios::trunc);
                    if (!output) return fail("Terrain asset source could not be opened for writing", entity);
                    output.write(encoded.value().data(), static_cast<std::streamsize>(encoded.value().size()));
                    if (!output) return fail("Terrain asset source could not be written", entity);
                }

                terrain->asset.guid = loaded.asset.requested_guid();
                terrain->asset.expected_type = assets::asset_types::terrain;
                terrain->asset.path_hint = assets::normalize_asset_path(source_reference);
                terrain->asset_authoring_revision = authored.authoring_revision;
                state_->asset_registry->mark_stale(terrain->asset.guid, "Terrain modifier stack changed");
                [[maybe_unused]] const auto reimport =
                    state_->asset_registry->reimport(terrain->asset.guid, assets::asset_streaming_priority::high);
                push_event(state_->events, state_->event_sequence, host_event_type::component_changed,
                           "Terrain modifier stack changed", entity);
                return success(stack_json(&authored, true, false, source_reference.generic_string()));
            }
'''
replace_once(
    host_source,
    "            else if constexpr (std::is_same_v<command_type, host_set_terrain_layer_command>)\n",
    terrain_stack_handler + "            else if constexpr (std::is_same_v<command_type, host_set_terrain_layer_command>)\n",
)

# Viewport overlay host.
viewport = "editor/src/renderer/src/viewport/ViewportPanel.tsx"
replace_once(
    viewport,
    "import type { DragEvent, KeyboardEvent, PointerEvent, WheelEvent } from 'react';",
    "import type { DragEvent, KeyboardEvent, PointerEvent, ReactNode, WheelEvent } from 'react';",
)
replace_once(viewport, "  active?: boolean;\n};", "  active?: boolean;\n  overlay?: ReactNode;\n};")
replace_once(viewport, "  active = true,\n}: ViewportPanelProps) {", "  active = true,\n  overlay,\n}: ViewportPanelProps) {")
replace_once(
    viewport,
    """        {streamedAvailable && (
          <canvas id={surfaceId} className="arc-viewport-shared-surface" aria-label="ARC 3D viewport" />
        )}

        {!viewportActive && !viewportAvailable && (""",
    """        {streamedAvailable && (
          <canvas id={surfaceId} className="arc-viewport-shared-surface" aria-label="ARC 3D viewport" />
        )}

        {overlay && <div className="arc-viewport-tool-overlay">{overlay}</div>}

        {!viewportActive && !viewportAvailable && (""",
)

viewport_css = Path("editor/src/renderer/src/viewport/viewport.css")
css = viewport_css.read_text()
if ".arc-viewport-tool-overlay" not in css:
    css += """

.arc-viewport-tool-overlay {
  position: absolute;
  z-index: 9;
  top: 58px;
  left: 14px;
  max-width: calc(100% - 28px);
  pointer-events: none;
}
"""
viewport_css.write_text(css)

# Workbench placement: tools in viewport, stack in right inspector, hierarchy remains left.
workbench = Path("editor/src/renderer/src/app/Workbench.tsx")
text = workbench.read_text()
old_import = """import { TerrainToolsPanel } from '../terrain/TerrainToolsPanel';
import type { TerrainToolState } from '../terrain/TerrainToolsPanel';"""
new_import = """import { TerrainViewportOverlay } from '../terrain/TerrainViewportOverlay';
import type { TerrainToolState } from '../terrain/TerrainViewportOverlay';
import { TerrainStackPanel } from '../terrain/TerrainStackPanel';
import type { TerrainModifierStackSnapshot } from '../terrain/TerrainStackPanel';"""
if old_import not in text:
    raise SystemExit("Workbench terrain imports were not found")
text = text.replace(old_import, new_import, 1)

left_pattern = re.compile(
    r"\n    if \(\(!requestedPanel \|\| requestedPanel === 'hierarchy'\) && activeTool === 'terrain' && selectedSnapshot\?\.terrain\) \{.*?\n    \}\n\n    if \(requestedPanel === 'hierarchy'",
    re.S,
)
text, count = left_pattern.subn("\n    if (requestedPanel === 'hierarchy'", text, count=1)
if count != 1:
    raise SystemExit("Workbench terrain left-panel override was not found")

overlay_helper = """
  const renderTerrainViewportOverlay = () => {
    if (activeTool !== 'terrain' || !selectedSnapshot?.terrain || !project) return undefined;
    const selectedKey = hostEntityKey(selectedSnapshot.entity);
    const visibleTerrainState =
      terrainToolState && hostEntityKey(terrainToolState.entity) === selectedKey
        ? terrainToolState
        : terrainToolStateFromSnapshot(selectedSnapshot)!;
    return (
      <TerrainViewportOverlay
        terrain={selectedSnapshot.terrain}
        state={visibleTerrainState}
        assets={project.assets}
        thumbnailProvider={loadAssetThumbnail}
        onStateChange={setTerrainToolState}
        onStatus={setLastCommand}
        command={async (type, payload) => {
          if (!startupState?.engineHostConnected)
            return { succeeded: false, error: 'Native editor host is unavailable' };
          return window.arc.host.command(type, payload as Record<string, unknown>) as Promise<HostResponse<TerrainToolState>>;
        }}
      />
    );
  };

"""
registry_marker = "  const editorRegistry = createEditorRegistry({"
if registry_marker not in text:
    raise SystemExit("Workbench editor registry marker was not found")
text = text.replace(registry_marker, overlay_helper + registry_marker, 1)

viewport_active = "            active={!createTerrainOpen && !settingsOpen && (context.instanceId ?? 'viewport-1') === activeViewportId}\n"
if viewport_active not in text:
    raise SystemExit("Workbench viewport active prop was not found")
text = text.replace(viewport_active, viewport_active + "            overlay={renderTerrainViewportOverlay()}\n", 1)

right_marker = """    if (panel === 'inspector') {
      return (
        <DataDrivenInspector"""
right_replacement = """    if (panel === 'inspector') {
      if (activeTool === 'terrain' && selectedSnapshot?.terrain) {
        return (
          <TerrainStackPanel
            entity={selectedSnapshot.entity}
            onStatus={setLastCommand}
            command={async (type, payload) => {
              if (!startupState?.engineHostConnected)
                return { succeeded: false, error: 'Native editor host is unavailable' };
              return window.arc.host.command(type, payload as Record<string, unknown>) as Promise<
                HostResponse<TerrainModifierStackSnapshot>
              >;
            }}
          />
        );
      }
      return (
        <DataDrivenInspector"""
if right_marker not in text:
    raise SystemExit("Workbench right inspector marker was not found")
text = text.replace(right_marker, right_replacement, 1)
workbench.write_text(text)

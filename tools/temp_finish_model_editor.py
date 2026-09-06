from pathlib import Path

preview = Path('editor/src/renderer/src/assetPreview/AssetPreviewViewport.tsx')
text = preview.read_text()
text = text.replace("kind: 'material' | 'shader';", "kind: 'material' | 'shader' | 'model';")
preview.write_text(text)

model = Path('editor/src/renderer/src/model/ModelEditor.tsx')
text = model.read_text()
text = text.replace(
    "import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';",
    "import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';\nimport { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';",
)
old = '''        <AssetPreviewPlaceholder
          label="Model preview"
          description={skeleton ? 'Skinned model with imported skeleton metadata.' : 'Static model preview.'}
        />'''
new = '''        <AssetPreviewViewport
          kind="model"
          assetGuid={asset.guid}
          label={`${asset.name} model preview`}
          fallback={
            <AssetPreviewPlaceholder
              label="Model preview"
              description={
                asset.guid
                  ? 'Waiting for the native model preview viewport.'
                  : 'The model must be registered before a live preview can be created.'
              }
            />
          }
        />'''
if old not in text:
    raise SystemExit('ModelEditor preview placeholder pattern not found')
model.write_text(text.replace(old, new, 1))

header = Path('editor/native/inc/arc/editor/editor_state.h')
text = header.read_text()
marker = '''editor_scene_open_result apply_scene_import_result_to_editor(editor_scene_state& scene, render::renderer& renderer,
                                                             const std::filesystem::path& source_path,
                                                             render::scene_import_result imported,
                                                             editor_scene_open_mode mode);'''
addition = marker + '''

/** Release editor-only imported scene entities and skin palettes. */
void clear_imported_scene_content(editor_scene_state& scene, render::renderer& renderer);'''
if marker not in text:
    raise SystemExit('editor_state.h insertion point missing')
header.write_text(text.replace(marker, addition, 1))

state = Path('editor/native/src/editor_state.cpp')
text = state.read_text()
marker = '''editor_scene_open_result open_scene_asset_in_editor(editor_scene_state& scene, render::renderer& renderer,
                                                    const std::filesystem::path& asset_root,
                                                    const std::filesystem::path& path, editor_scene_open_mode mode)'''
wrapper = '''void clear_imported_scene_content(editor_scene_state& scene, render::renderer& renderer)
{
    clear_imported_content(scene, &renderer);
}

''' + marker
if marker not in text:
    raise SystemExit('editor_state.cpp insertion point missing')
state.write_text(text.replace(marker, wrapper, 1))

host = Path('editor/native/src/arc_host.cpp')
text = host.read_text()
text = text.replace(
    'constexpr std::string_view shader_preview_viewport_prefix = "asset-preview-shader-";',
    'constexpr std::string_view shader_preview_viewport_prefix = "asset-preview-shader-";\nconstexpr std::string_view model_preview_viewport_prefix = "asset-preview-model-";',
    1,
)
text = text.replace('''    material,
    shader
};''', '''    material,
    shader,
    model
};''', 1)
parser = '''    else if (viewport_id.starts_with(shader_preview_viewport_prefix))
    {
        result.kind = asset_preview_kind::shader;
        guid_text = viewport_id.substr(shader_preview_viewport_prefix.size());
    }'''
parser_new = parser + '''
    else if (viewport_id.starts_with(model_preview_viewport_prefix))
    {
        result.kind = asset_preview_kind::model;
        guid_text = viewport_id.substr(model_preview_viewport_prefix.size());
    }'''
if parser not in text:
    raise SystemExit('preview parser insertion point missing')
text = text.replace(parser, parser_new, 1)
text = text.replace('''        case asset_preview_kind::shader:
            return "shader";''', '''        case asset_preview_kind::shader:
            return "shader";
        case asset_preview_kind::model:
            return "model";''', 1)
text = text.replace('''        std::uint64_t preview_material_generation{};
        std::string preview_error;''', '''        std::uint64_t preview_material_generation{};
        std::uint64_t preview_asset_generation{};
        std::string preview_error;''', 1)

ensure_marker = '''template <class HostState>
bool ensure_asset_preview_scene(HostState& host, viewport_surface_registry::surface_state& surface)
{'''
model_helper = r'''template <class HostState>
bool refresh_asset_preview_model(HostState& host, viewport_surface_registry::surface_state& surface)
{
    if (surface.preview_kind != asset_preview_kind::model) return true;
    if (!surface.preview_guid.valid())
    {
        surface.preview_error = "Model preview viewport has an invalid asset GUID";
        return false;
    }
    if (!host.asset_registry)
    {
        surface.preview_error = "Model preview is waiting for the project asset registry";
        return false;
    }
    if (!host.renderer)
    {
        surface.preview_error = "Model preview renderer is unavailable";
        return false;
    }

    const auto asset = host.asset_registry->find(surface.preview_guid);
    if (!asset)
    {
        surface.preview_error = "Model preview asset is not registered";
        return false;
    }
    const auto generation = asset->generation;
    if (surface.preview_scene && generation == surface.preview_asset_generation &&
        !surface.preview_scene->imported_scene_entities.empty())
        return true;

    const auto resolved =
        resolve_editor_asset(host.assets, host.asset_registry.get(), host.project.root, asset->source_path);
    if (!resolved)
    {
        surface.preview_error = "Model preview source path could not be resolved: " + asset->source_path.generic_string();
        return false;
    }

    auto preview = std::make_unique<editor_scene_state>(create_blank_scene(*host.renderer, false, nullptr));
    preview->scene_name = "Asset Preview: model";
    preview->primitive_material =
        host.scene.primitive_material.valid() ? host.scene.primitive_material : host.scene.default_material;
    const auto opened = open_scene_asset_in_editor(*preview, *host.renderer, resolved->asset_root, resolved->path,
                                                   editor_scene_open_mode::replace);
    if (!opened.succeeded)
    {
        surface.preview_error = opened.message.empty() ? "Model preview import failed" : opened.message;
        return false;
    }

    surface.preview_camera = {};
    if (!focus_selected_entity(preview->scene, preview->selected_entity, surface.preview_camera))
        (void)surface.preview_camera.place({2.5f, 1.5f, 2.5f}, {0.0f, 0.0f, 0.0f});
    if (auto* camera_transform = preview->scene.try_get<scene::transform_component>(preview->camera_entity))
        surface.preview_camera.apply_to(*camera_transform);
    clear_selection(preview->scene, preview->selected_entity);

    if (surface.preview_scene) clear_imported_scene_content(*surface.preview_scene, *host.renderer);
    surface.preview_scene = std::move(preview);
    surface.preview_entity = {};
    surface.preview_mesh = {};
    surface.preview_asset_generation = generation;
    surface.preview_error.clear();
    return true;
}

'''
if ensure_marker not in text:
    raise SystemExit('ensure preview insertion point missing')
text = text.replace(ensure_marker, model_helper + ensure_marker, 1)
old_ensure = '''    if (surface.preview_kind == asset_preview_kind::none) return true;
    if (surface.preview_scene) return refresh_asset_preview_material(host, surface);'''
new_ensure = '''    if (surface.preview_kind == asset_preview_kind::none) return true;
    if (surface.preview_kind == asset_preview_kind::model) return refresh_asset_preview_model(host, surface);
    if (surface.preview_scene) return refresh_asset_preview_material(host, surface);'''
if old_ensure not in text:
    raise SystemExit('ensure preview model dispatch point missing')
text = text.replace(old_ensure, new_ensure, 1)
cleanup = '''    if (!surface.preview_scene) return;
    surface.preview_scene->terrain_render_proxies.clear(renderer);'''
cleanup_new = '''    if (!surface.preview_scene) return;
    clear_imported_scene_content(*surface.preview_scene, renderer);
    surface.preview_scene->terrain_render_proxies.clear(renderer);'''
if cleanup not in text:
    raise SystemExit('preview cleanup insertion point missing')
text = text.replace(cleanup, cleanup_new, 1)
text = text.replace('''    surface.preview_material_generation = 0;''', '''    surface.preview_material_generation = 0;
    surface.preview_asset_generation = 0;''', 1)
host.write_text(text)

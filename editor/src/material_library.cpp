#include <arc/editor/material_library.h>

#include <arc/editor/editor_interaction.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/render/primitives.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <cctype>
#include <cstddef>

namespace arc::editor
{
namespace
{

std::filesystem::path canonical_key(const std::filesystem::path& path)
{
    std::error_code ec;
    const auto absolute = std::filesystem::absolute(path, ec);
    return ec ? path.lexically_normal() : absolute.lexically_normal();
}

render::texture_handle ensure_texture(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& path)
{
    if (path.empty())
        return {};
    const auto key = canonical_key(path);
    for (const auto& [texture_path, handle] : library.textures)
    {
        if (texture_path == key)
            return handle;
    }

    auto texture_result = render::load_texture_asset(path);
    if (!texture_result.succeeded())
    {
        arc::warn(
            "editor.materials",
            "Texture asset could not be loaded: " + path.string() + " (" + texture_result.message + ")");
        return {};
    }

    const auto handle = renderer.create_texture(std::move(texture_result.texture));
    library.textures.push_back({ key, handle });
    return handle;
}

void resolve_texture_handles(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    material_asset& asset)
{
    asset.material.base_color_texture = ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.base_color));
    asset.material.metallic_roughness_texture = ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.metallic_roughness));
    asset.material.normal_texture = ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.normal));
    asset.material.occlusion_texture = ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.ao));
    asset.material.emissive_texture = ensure_texture(library, renderer, resolve_material_texture_path(asset_root, asset.textures.emissive));
}

editor_material_record* find_record(editor_material_library& library, const std::filesystem::path& path)
{
    const auto key = canonical_key(path);
    for (auto& record : library.materials)
    {
        if (canonical_key(record.path) == key)
            return &record;
    }
    return nullptr;
}

} // namespace

bool is_material_asset_path(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext == ".arcmat";
}

bool is_texture_asset_path(const std::filesystem::path& path)
{
    return render::is_supported_texture_asset(path);
}

bool assign_texture_to_material_slot(
    material_editor_state& editor,
    material_texture_slot slot,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& texture_path,
    std::string* message)
{
    if (!editor.open)
    {
        if (message)
            *message = "material editor is not open";
        return false;
    }

    std::filesystem::path relative_path = texture_path;
    if (texture_path.is_absolute())
    {
        std::error_code ec;
        relative_path = std::filesystem::relative(texture_path, asset_root, ec);
        if (ec)
            relative_path = texture_path.filename();
    }
    relative_path = relative_path.lexically_normal();
    if (!is_texture_asset_path(relative_path))
    {
        if (message)
            *message = "asset is not a supported texture";
        return false;
    }

    auto& textures = editor.working.textures;
    const auto value = relative_path.generic_string();
    switch (slot)
    {
    case material_texture_slot::base_color:
        textures.base_color = value;
        break;
    case material_texture_slot::metallic_roughness:
        textures.metallic_roughness = value;
        break;
    case material_texture_slot::normal:
        textures.normal = value;
        break;
    case material_texture_slot::ao:
        textures.ao = value;
        break;
    case material_texture_slot::emissive:
        textures.emissive = value;
        break;
    case material_texture_slot::height:
        textures.height = value;
        break;
    }

    editor.dirty = true;
    if (message)
        *message = "assigned texture " + value;
    return true;
}

bool create_default_material_asset(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_root,
    std::string& message)
{
    auto asset = make_default_material_asset(path.stem().string());
    asset.path = path;
    return save_material_asset(asset, asset_root, message);
}

render::material_handle load_material_for_editor(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    material_asset* out_asset)
{
    if (auto* record = find_record(library, path))
    {
        if (out_asset)
            *out_asset = record->asset;
        return record->material;
    }

    material_asset asset;
    std::string message;
    if (!load_material_asset(path, asset_root, asset, message))
    {
        arc::error("editor.materials", "Failed to load material '" + path.string() + "': " + message);
        return {};
    }

    resolve_texture_handles(library, renderer, asset_root, asset);
    const auto handle = renderer.create_material(asset.material);
    library.materials.push_back({ canonical_key(path), asset, handle });
    if (out_asset)
        *out_asset = asset;
    return handle;
}

bool open_material_editor(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    std::string& message)
{
    material_asset asset;
    const auto handle = load_material_for_editor(library, renderer, asset_root, path, &asset);
    if (!handle.valid())
    {
        message = "material could not be loaded";
        return false;
    }

    editor.open = true;
    editor.dirty = false;
    editor.working = asset;
    editor.saved = asset;
    editor.material = handle;
    if (!editor.preview_sphere.valid())
        editor.preview_sphere = renderer.create_mesh(render::make_uv_sphere_mesh(0.75f, 48, 24));
    if (!editor.preview_plane.valid())
        editor.preview_plane = renderer.create_mesh(render::make_plane_mesh(1.6f));
    if (!editor.preview_cube.valid())
        editor.preview_cube = renderer.create_mesh(render::make_cube_mesh(1.1f));
    message = "opened material editor";
    return true;
}

bool save_material_editor(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    std::string& message)
{
    if (!editor.open)
    {
        message = "material editor is not open";
        return false;
    }

    editor.working.material.name = editor.working.name;
    if (!save_material_asset(editor.working, asset_root, message))
        return false;

    resolve_texture_handles(library, renderer, asset_root, editor.working);
    if (editor.material.valid())
        renderer.update_material(editor.material, editor.working.material);
    else
        editor.material = renderer.create_material(editor.working.material);

    if (auto* record = find_record(library, editor.working.path))
    {
        record->asset = editor.working;
        record->material = editor.material;
    }
    else
    {
        library.materials.push_back({ canonical_key(editor.working.path), editor.working, editor.material });
    }

    editor.saved = editor.working;
    editor.dirty = false;
    return true;
}

bool update_material_editor_live_material(
    material_editor_state& editor,
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    std::string* message)
{
    if (!editor.open)
    {
        if (message)
            *message = "material editor is not open";
        return false;
    }

    editor.working.material.name = editor.working.name;
    resolve_texture_handles(library, renderer, asset_root, editor.working);
    if (editor.material.valid())
        renderer.update_material(editor.material, editor.working.material);
    else
        editor.material = renderer.create_material(editor.working.material);

    if (auto* record = find_record(library, editor.working.path))
    {
        record->asset = editor.working;
        record->material = editor.material;
    }

    if (message)
        *message = "updated live material";
    return true;
}

bool apply_material_to_selected(
    scene::registry& scene,
    scene::entity selected,
    render::material_handle material)
{
    if (!material.valid())
        return false;
    auto* mesh = scene.try_get<scene::mesh_renderer_component>(selected);
    if (!mesh)
        return false;
    mesh->material = material;
    return true;
}

bool apply_material_asset_to_entity(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& material_path,
    scene::registry& scene,
    scene::entity entity,
    std::string* message)
{
    if (!is_material_asset_path(material_path))
    {
        if (message)
            *message = "asset is not a material";
        return false;
    }

    auto* mesh = scene.try_get<scene::mesh_renderer_component>(entity);
    if (!mesh)
    {
        if (message)
            *message = "target entity has no mesh renderer";
        return false;
    }

    const auto resolved_path = material_path.is_absolute() ? material_path : asset_root / material_path;
    material_asset asset;
    const auto material = load_material_for_editor(library, renderer, asset_root, resolved_path, &asset);
    if (!material.valid())
    {
        if (message)
            *message = "material could not be loaded";
        return false;
    }

    mesh->material = material;
    if (message)
        *message = "applied material " + asset.name;
    return true;
}

scene::entity apply_material_asset_to_viewport_hit(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const std::filesystem::path& material_path,
    scene::registry& scene,
    const editor_ray& ray,
    scene::entity& selected,
    std::string* message)
{
    const auto picked = pick_bounded_entity(scene, ray);
    if (!picked.valid())
    {
        if (message)
            *message = "material drop did not hit a renderable entity";
        return {};
    }

    if (!apply_material_asset_to_entity(library, renderer, asset_root, material_path, scene, picked, message))
        return {};

    select_entity(scene, picked, selected);
    return picked;
}

} // namespace arc::editor

#include <arc/editor/material_library.h>

#include <arc/editor/editor_interaction.h>
#include <arc/diagnostics/diagnostics.h>
#include <arc/render/primitives.h>
#include <arc/render/texture.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <optional>

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
    const std::filesystem::path& path,
    render::texture_semantic semantic = render::texture_semantic::generic_color)
{
    if (path.empty())
        return {};
    auto key = canonical_key(path);
    key += "#" + std::to_string(static_cast<unsigned>(semantic));
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

    texture_result.texture.semantic = semantic;
    texture_result.texture.color_space = render::required_color_space(semantic);
    if (texture_result.texture.format == render::texture_format::rgba8_srgb ||
        texture_result.texture.format == render::texture_format::rgba8_unorm)
    {
        texture_result.texture.format = texture_result.texture.color_space == render::texture_color_space::srgb
            ? render::texture_format::rgba8_srgb
            : render::texture_format::rgba8_unorm;
    }
    const auto handle = renderer.create_texture(std::move(texture_result.texture));
    library.textures.push_back({ key, handle });
    return handle;
}

render::texture_handle ensure_packed_terrain_surface(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    const material_asset& asset,
    std::size_t layer_index)
{
    const auto& paths = asset.terrain_layers[layer_index];
    if (!paths.packed_aorh.empty())
        return ensure_texture(library, renderer, resolve_material_texture_path(asset_root, paths.packed_aorh),
            render::texture_semantic::metallic_roughness);

    auto cache_key = canonical_key(asset.path);
    cache_key += ".terrain-aorh-" + std::to_string(layer_index);
    for (const auto& [texture_path, handle] : library.textures)
    {
        if (texture_path == cache_key)
            return handle;
    }

    const auto load_channel = [&](const std::string& relative_path, std::string_view channel)
        -> std::optional<render::texture_data> {
        if (relative_path.empty())
            return std::nullopt;
        const auto path = resolve_material_texture_path(asset_root, relative_path);
        auto loaded = render::load_texture_asset(path);
        if (!loaded.succeeded() || !loaded.texture.has_pixels())
        {
            arc::warn("editor.materials", "Terrain " + std::string(channel) +
                " map could not be packed: " + path.generic_string());
            return std::nullopt;
        }
        return std::move(loaded.texture);
    };

    auto ao = load_channel(paths.ao, "AO");
    auto roughness = load_channel(paths.roughness, "roughness");
    auto height = load_channel(paths.height, "height");
    const render::texture_data* reference = ao ? &*ao : roughness ? &*roughness : height ? &*height : nullptr;
    if (reference == nullptr)
        return {};

    const auto pixel_count = static_cast<std::size_t>(reference->width) * reference->height;
    const auto channel_usable = [&](const std::optional<render::texture_data>& source) {
        return source && source->width == reference->width && source->height == reference->height &&
            source->pixels.size() >= pixel_count * 4u;
    };
    if ((ao && !channel_usable(ao)) || (roughness && !channel_usable(roughness)) || (height && !channel_usable(height)))
        arc::warn("editor.materials", "Terrain AORH source dimensions differ; mismatched channels use explicit defaults");

    render::texture_data packed;
    packed.name = asset.name + " " + asset.material.terrain_layers[layer_index].name + " AORH";
    packed.source_path = cache_key;
    packed.width = reference->width;
    packed.height = reference->height;
    packed.format = render::texture_format::rgba8_unorm;
    packed.color_space = render::texture_color_space::linear;
    packed.semantic = render::texture_semantic::metallic_roughness;
    packed.pixels.resize(pixel_count * 4u);
    const auto channel_value = [&](const std::optional<render::texture_data>& source,
                                   std::size_t pixel, std::uint8_t fallback) {
        return channel_usable(source)
            ? std::to_integer<std::uint8_t>(source->pixels[pixel * 4u])
            : fallback;
    };
    const auto roughness_fallback = static_cast<std::uint8_t>(std::clamp(
        asset.material.terrain_layers[layer_index].roughness, 0.0f, 1.0f) * 255.0f + 0.5f);
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
    {
        packed.pixels[pixel * 4u + 0u] = std::byte{ channel_value(ao, pixel, 255u) };
        packed.pixels[pixel * 4u + 1u] = std::byte{ channel_value(roughness, pixel, roughness_fallback) };
        packed.pixels[pixel * 4u + 2u] = std::byte{ channel_value(height, pixel, 128u) };
        packed.pixels[pixel * 4u + 3u] = std::byte{ 255u };
    }

    const auto handle = renderer.create_texture(std::move(packed));
    if (handle.valid())
        library.textures.push_back({ std::move(cache_key), handle });
    return handle;
}

void resolve_texture_handles(
    editor_material_library& library,
    render::renderer& renderer,
    const std::filesystem::path& asset_root,
    material_asset& asset)
{
    if (asset.material.domain == render::material_domain::terrain)
    {
        for (std::size_t layer_index = 0; layer_index < asset.terrain_layers.size(); ++layer_index)
        {
            const auto& paths = asset.terrain_layers[layer_index];
            auto& layer = asset.material.terrain_layers[layer_index];
            layer.base_color_texture = ensure_texture(
                library, renderer, resolve_material_texture_path(asset_root, paths.base_color),
                render::texture_semantic::base_color);
            layer.normal_texture = ensure_texture(
                library, renderer, resolve_material_texture_path(asset_root, paths.normal),
                render::texture_semantic::normal);
            layer.packed_surface_texture = ensure_packed_terrain_surface(
                library, renderer, asset_root, asset, layer_index);
        }
        return;
    }

    asset.material.base_color_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.base_color), render::texture_semantic::base_color);
    asset.material.metallic_roughness_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.metallic_roughness), render::texture_semantic::metallic_roughness);
    asset.material.normal_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.normal), render::texture_semantic::normal);
    asset.material.occlusion_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.ao), render::texture_semantic::occlusion);
    asset.material.emissive_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.emissive), render::texture_semantic::emissive);
    asset.material.clear_coat_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.clear_coat), render::texture_semantic::clear_coat);
    asset.material.clear_coat_roughness_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.clear_coat_roughness), render::texture_semantic::clear_coat);
    asset.material.clear_coat_normal_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.clear_coat_normal), render::texture_semantic::normal);
    asset.material.anisotropy_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.anisotropy), render::texture_semantic::anisotropy);
    asset.material.subsurface_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.subsurface), render::texture_semantic::thickness);
    asset.material.thickness_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.thickness), render::texture_semantic::thickness);
    asset.material.transmission_texture = ensure_texture(library, renderer,
        resolve_material_texture_path(asset_root, asset.textures.transmission), render::texture_semantic::transmission);
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
    case material_texture_slot::clear_coat:
        textures.clear_coat = value;
        break;
    case material_texture_slot::clear_coat_roughness:
        textures.clear_coat_roughness = value;
        break;
    case material_texture_slot::clear_coat_normal:
        textures.clear_coat_normal = value;
        break;
    case material_texture_slot::anisotropy:
        textures.anisotropy = value;
        break;
    case material_texture_slot::subsurface:
        textures.subsurface = value;
        break;
    case material_texture_slot::thickness:
        textures.thickness = value;
        break;
    case material_texture_slot::transmission:
        textures.transmission = value;
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

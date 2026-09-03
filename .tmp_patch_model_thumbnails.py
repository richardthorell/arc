from pathlib import Path

p = Path('editor/native/src/arc_host_base.inc')
s = p.read_text()

needle = '#include <arc/editor/material_preview.h>\n'
if '#include <arc/editor/model_preview.h>' not in s:
    s = s.replace(needle, needle + '#include <arc/editor/model_preview.h>\n', 1)

if '#include <unordered_map>' not in s:
    s = s.replace('#include <type_traits>\n', '#include <type_traits>\n#include <unordered_map>\n', 1)

start = s.index('std::optional<host_asset_thumbnail_snapshot> arc_host::asset_thumbnail')
end = s.index('std::vector<host_event> arc_host::poll_events()', start)
replacement = r'''std::optional<host_asset_thumbnail_snapshot> arc_host::asset_thumbnail(std::string_view path,
                                                                       std::uint32_t max_size) const
{
    const auto resolved = resolve_editor_asset(state_->assets, state_->asset_registry.get(), state_->project.root,
                                               std::filesystem::path{path});
    if (!resolved) return std::nullopt;

    const auto requested_size = std::clamp(max_size, 32u, 256u);
    auto extension = resolved->path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const bool model_asset = extension == ".fbx" || extension == ".obj" || extension == ".glb" ||
                             extension == ".gltf";

    render::texture_data preview;
    if (model_asset)
    {
        // Model thumbnails are expensive enough that both the React thumbnail provider
        // and the native host keep them cached. The file timestamp makes reimport/source
        // edits invalidate this cache without coupling the renderer to asset generations.
        struct model_thumbnail_cache_entry
        {
            std::filesystem::file_time_type source_time{};
            host_asset_thumbnail_snapshot snapshot;
        };
        static std::unordered_map<std::string, model_thumbnail_cache_entry> model_thumbnail_cache;

        std::error_code timestamp_error;
        const auto source_time = std::filesystem::last_write_time(resolved->path, timestamp_error);
        const auto cache_key = resolved->path.lexically_normal().generic_string() + ":" + std::to_string(requested_size);
        if (!timestamp_error)
            if (const auto cached = model_thumbnail_cache.find(cache_key);
                cached != model_thumbnail_cache.end() && cached->second.source_time == source_time)
                return cached->second.snapshot;

        render::scene_import_options import_options;
        import_options.asset_root = resolved->asset_root;
        import_options.import_directory = resolved->path.parent_path();
        import_options.copy_assets = false;
        const auto imported = render::load_scene_asset(resolved->path, import_options);
        if (!imported.succeeded()) return std::nullopt;

        model_preview_options preview_options{.size = requested_size};
        // V1 intentionally uses the neutral/default Phong-style material. The
        // model_preview_options contract already accepts a material override so
        // model/material binding resolution can opt into an imported or assigned
        // material later without changing asset.thumbnail or the Content Browser.
        auto rendered = render_model_preview(imported, preview_options);
        if (!rendered.succeeded()) return std::nullopt;
        preview = std::move(rendered.texture);

        const auto bmp = texture_preview_bmp(preview, requested_size);
        if (bmp.empty()) return std::nullopt;
        const float scale = std::min(1.0f, static_cast<float>(requested_size) /
                                               static_cast<float>(std::max(preview.width, preview.height)));
        host_asset_thumbnail_snapshot snapshot{
            .path = std::string(path),
            .width = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.width * scale))),
            .height = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.height * scale))),
            .data_url = "data:image/bmp;base64," + base64_encode(bmp)};
        if (!timestamp_error) model_thumbnail_cache[cache_key] = {.source_time = source_time, .snapshot = snapshot};
        return snapshot;
    }

    if (is_material_asset_path(resolved->path))
    {
        material_asset material;
        std::string message;
        if (!load_material_asset(resolved->path, resolved->asset_root, material, message)) return std::nullopt;
        auto rendered = render_material_preview(material, resolved->asset_root, requested_size);
        if (!rendered.succeeded()) return std::nullopt;
        preview = std::move(rendered.texture);
    }
    else
    {
        if (!render::is_supported_texture_asset(resolved->path)) return std::nullopt;
        const auto loaded = render::load_texture_asset(resolved->path);
        if (!loaded.succeeded()) return std::nullopt;
        preview = loaded.texture;
    }
    const auto bmp = texture_preview_bmp(preview, requested_size);
    if (bmp.empty()) return std::nullopt;
    const float scale = std::min(1.0f, static_cast<float>(requested_size) /
                                           static_cast<float>(std::max(preview.width, preview.height)));
    return host_asset_thumbnail_snapshot{
        .path = std::string(path),
        .width = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.width * scale))),
        .height = std::max(1u, static_cast<std::uint32_t>(std::lround(preview.height * scale))),
        .data_url = "data:image/bmp;base64," + base64_encode(bmp)};
}

'''
s = s[:start] + replacement + s[end:]
p.write_text(s)

#include <arc/editor/texture_preview.h>

#ifdef query
#undef query
#endif
#define query(...) query_raw(__VA_ARGS__)
#define asset_thumbnail asset_thumbnail_base
#include "arc_host_base_core.inc"
#undef asset_thumbnail
#undef query

namespace arc::editor
{

std::optional<host_asset_thumbnail_snapshot> arc_host::asset_thumbnail(std::string_view path,
                                                                       std::uint32_t max_size) const
{
    const auto resolved = resolve_editor_asset(state_->assets, state_->asset_registry.get(), state_->project.root,
                                               std::filesystem::path{path});
    if (!resolved) return std::nullopt;

    auto extension = resolved->path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    const bool model_asset =
        extension == ".fbx" || extension == ".obj" || extension == ".glb" || extension == ".gltf";
    if (model_asset || is_material_asset_path(resolved->path) || !render::is_supported_texture_asset(resolved->path))
        return asset_thumbnail_base(path, max_size);

    const auto loaded = render::load_texture_asset(resolved->path);
    if (!loaded.succeeded()) return std::nullopt;

    const auto requested_size = std::clamp(max_size, 32u, 256u);
    const auto preview = build_texture_preview_image(loaded.texture, requested_size);
    if (!preview.valid()) return std::nullopt;
    const auto bmp = encode_texture_preview_bmp(preview);
    if (bmp.empty()) return std::nullopt;

    return host_asset_thumbnail_snapshot{.path = std::string(path),
                                         .width = preview.width,
                                         .height = preview.height,
                                         .data_url = "data:image/bmp;base64," + base64_encode(bmp)};
}

host_response arc_host::query_base(const host_query_envelope& query) const
{
    if (const auto* thumbnail_query = std::get_if<host_asset_thumbnail_query>(&query.payload))
    {
        const auto thumbnail = asset_thumbnail(thumbnail_query->path, thumbnail_query->max_size);
        host_response response;
        response.request_id = query.request_id;
        response.succeeded = thumbnail.has_value();
        if (thumbnail)
            response.payload_json = to_json(*thumbnail);
        else
            response.error = "Texture thumbnail could not be generated";
        response.scene_revision = state_->scene_revision;
        response.world_epoch = state_->world_epoch;
        response.frame_revision = state_->viewport_frame_index;
        return response;
    }
    return query_raw(query);
}

} // namespace arc::editor

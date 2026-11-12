#include <arc/scene/cooked_texture_hydration.h>

#include <unordered_set>
#include <utility>

namespace arc::scene
{
std::string hydrate_cooked_scene_textures(render::renderer& renderer, std::span<const cooked_texture_upload> uploads,
                                          cooked_texture_residencies& residency)
{
    if (!residency.empty()) return "Cooked texture residency still belongs to an existing renderer generation";
    if (uploads.empty()) return {};

    std::unordered_set<assets::asset_guid, assets::asset_guid_hash> seen;
    std::vector<const cooked_texture_upload*> unique;
    unique.reserve(uploads.size());
    for (const auto& upload : uploads)
    {
        if (!upload.asset.valid()) return "Cooked texture has no stable asset GUID";
        const auto& texture = upload.texture;
        if (texture.width == 0 || texture.height == 0 || texture.mips.empty() ||
            (!texture.has_pixels() && !texture.has_encoded_mips()))
            return "Cooked texture has no resident mip payload";
        if (seen.insert(upload.asset).second) unique.push_back(&upload);
    }

    cooked_texture_residencies staged;
    staged.reserve(unique.size());
    for (const auto* upload : unique)
    {
        const auto handle = renderer.create_texture(upload->texture);
        if (!handle.valid())
        {
            for (const auto& item : staged)
                static_cast<void>(renderer.destroy_texture(item.handle));
            return "Renderer failed to allocate cooked texture";
        }
        staged.push_back({.asset = upload->asset, .handle = handle});
    }
    residency = std::move(staged);
    return {};
}

void release_cooked_scene_textures(render::renderer& renderer, cooked_texture_residencies& residency) noexcept
{
    for (const auto& item : residency)
        static_cast<void>(renderer.destroy_texture(item.handle));
    residency.clear();
}

render::texture_handle find_cooked_scene_texture(std::span<const cooked_texture_residency> residency,
                                                 assets::asset_guid asset) noexcept
{
    for (const auto& item : residency)
        if (item.asset == asset) return item.handle;
    return {};
}
} // namespace arc::scene

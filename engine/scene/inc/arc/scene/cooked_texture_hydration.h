#pragma once

/** @file
 * @brief Renderer-generation residency of cooked, integrity-verified textures.
 * GUID identity is retained here; Material ABI binding is a separate phase.
 */

#include <arc/assets/assets.h>
#include <arc/render/renderer.h>

#include <span>
#include <string>
#include <vector>

namespace arc::scene
{
struct cooked_texture_upload
{
    assets::asset_guid asset{};
    render::texture_data texture;
};

struct cooked_texture_residency
{
    assets::asset_guid asset{};
    render::texture_handle handle{};
};

using cooked_texture_residencies = std::vector<cooked_texture_residency>;

/** Deduplicate by cooked catalog GUID, validate before any renderer mutation,
 * and roll back all newly allocated textures if a resource cannot be created.
 * Existing residencies are not replaced in place: release the old generation first.
 */
[[nodiscard]] std::string hydrate_cooked_scene_textures(render::renderer& renderer,
                                                        std::span<const cooked_texture_upload> uploads,
                                                        cooked_texture_residencies& residency);

/** Retire every renderer-owned texture before discarding this renderer generation. */
void release_cooked_scene_textures(render::renderer& renderer, cooked_texture_residencies& residency) noexcept;

/** Resolve a GUID to a renderer-local handle, or an invalid handle if absent. */
[[nodiscard]] render::texture_handle find_cooked_scene_texture(std::span<const cooked_texture_residency> residency,
                                                               assets::asset_guid asset) noexcept;

} // namespace arc::scene

#pragma once

#include <arc/assets/assets.h>
#include <arc/render/cooked_material_texture_bindings.h>
#include <arc/render_tools/material_graph.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace arc::tools::detail
{
inline std::string material_texture_path(const std::filesystem::path& input)
{
    auto text = input.lexically_normal().generic_string();
    while (text.starts_with("./"))
        text.erase(0, 2);
    if (text.starts_with("assets/")) text.erase(0, 7);
    return text;
}

/** Resolve each reachable graph texture against the exact registered cooked
 * dependency, rejecting ambiguous basename/suffix matches rather than relying
 * on catalog enumeration order.
 */
inline core::result<render::cooked_material_texture_manifest, std::string>
make_material_texture_manifest(assets::asset_guid material, const render::tools::material_graph_descriptor& graph,
                               const std::vector<assets::asset_snapshot>& dependencies)
{
    using result = core::result<render::cooked_material_texture_manifest, std::string>;
    render::cooked_material_texture_manifest manifest;
    manifest.material = {.high = material.high, .low = material.low};
    for (const auto& binding : graph.textures)
    {
        assets::asset_guid guid{};
        if (!binding.default_source.empty())
        {
            const auto path = material_texture_path(binding.default_source);
            const assets::asset_snapshot* selected{};
            for (const auto& dependency : dependencies)
            {
                if (dependency.type != assets::asset_types::texture_2d) continue;
                const auto candidate = material_texture_path(dependency.source_path);
                const bool suffix = candidate.size() > path.size() && candidate.ends_with(path) &&
                                    candidate[candidate.size() - path.size() - 1] == '/';
                if (candidate != path && !suffix) continue;
                if (selected)
                    return result::failure("Ambiguous material texture dependency: " + binding.default_source);
                selected = &dependency;
            }
            if (!selected)
                return result::failure("Missing registered material texture dependency: " + binding.default_source);
            guid = selected->guid;
        }
        render::cooked_material_texture_binding entry;
        entry.binding.slot = binding.slot;
        entry.binding.parameter_id = binding.parameter_id;
        entry.binding.type = binding.type;
        entry.binding.dimension_slot = binding.dimension_slot;
        entry.binding.semantic = binding.semantic;
        entry.asset = guid;
        manifest.textures.push_back(entry);
    }
    return result::success(std::move(manifest));
}
} // namespace arc::tools::detail

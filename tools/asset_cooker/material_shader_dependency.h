#pragma once

#include <arc/assets/assets.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace arc::tools::detail
{

struct material_shader_dependency_lookup
{
    const assets::asset_snapshot* dependency{};
    bool ambiguous{};
};

inline std::string normalized_material_shader_path(const std::filesystem::path& path)
{
    auto text = path.lexically_normal().generic_string();
    while (text.starts_with("./"))
        text.erase(0, 2);
    return text;
}

/** Resolve the authored Material Shader path, never an unrelated first shader dependency.
 * Prefer an exact registered path, then a unique asset-root-relative suffix. Ambiguous
 * suffixes must fail rather than depending on dependency enumeration order.
 */
inline material_shader_dependency_lookup find_material_shader_dependency(
    const std::vector<assets::asset_snapshot>& dependencies, std::string_view shader_path)
{
    auto authored = normalized_material_shader_path(std::filesystem::path(shader_path));
    if (authored.starts_with("assets/"))
        authored.erase(0, 7);
    if (authored.empty() || authored == ".")
        return {};

    const assets::asset_snapshot* exact{};
    const assets::asset_snapshot* suffix{};
    bool duplicate_exact{};
    bool duplicate_suffix{};
    for (const auto& dependency : dependencies)
    {
        if (dependency.type != assets::asset_types::shader)
            continue;
        const auto path = normalized_material_shader_path(dependency.source_path);
        if (path == authored)
        {
            if (exact)
                duplicate_exact = true;
            exact = &dependency;
        }
        else if (path.size() > authored.size() && path.ends_with(authored) &&
                 path[path.size() - authored.size() - 1] == '/')
        {
            if (suffix)
                duplicate_suffix = true;
            suffix = &dependency;
        }
    }

    if (exact)
        return {.dependency = duplicate_exact ? nullptr : exact, .ambiguous = duplicate_exact};
    if (duplicate_suffix)
        return {.dependency = nullptr, .ambiguous = true};
    return {.dependency = suffix};
}

} // namespace arc::tools::detail

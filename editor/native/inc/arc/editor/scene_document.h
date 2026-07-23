#pragma once

#include <arc/editor/editor_state.h>

namespace arc::editor
{

inline constexpr std::uint32_t arc_scene_format_version = 1;

struct scene_document_result
{
    bool succeeded{};
    std::size_t entity_count{};
    std::string message;
    std::vector<std::string> diagnostics;
};

struct scene_document_text_result
{
    bool succeeded{};
    std::size_t entity_count{};
    std::string text;
    std::string message;
};

scene_document_text_result serialize_scene_subtree_as_prefab(
    editor_scene_state& scene,
    const std::filesystem::path& project_root,
    scene::entity root,
    scene::entity_guid prefab_guid,
    std::string_view prefab_name);

scene_document_result save_scene_document(
    editor_scene_state& scene,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path);

scene_document_result load_scene_document(
    editor_scene_state& scene,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path);

} // namespace arc::editor

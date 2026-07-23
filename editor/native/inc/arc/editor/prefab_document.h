#pragma once

#include <arc/editor/editor_state.h>

namespace arc::editor
{

struct prefab_document_result
{
    bool succeeded{};
    scene::entity root{};
    std::size_t entity_count{};
    std::string message;
    std::vector<std::string> diagnostics;
};

prefab_document_result save_prefab_document(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    scene::entity root,
    const std::filesystem::path& path);

prefab_document_result instantiate_prefab_document(
    editor_scene_state& state,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path,
    scene::entity parent = {});

prefab_document_result apply_prefab_instance(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    scene::entity root);

prefab_document_result revert_prefab_instance(
    editor_scene_state& state,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    scene::entity root);

bool unpack_prefab_instance(editor_scene_state& state, scene::entity root);

} // namespace arc::editor

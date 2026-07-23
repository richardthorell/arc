#include <arc/editor/prefab_document.h>

#include <arc/editor/scene_document.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <unordered_map>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

bool atomic_write(const std::filesystem::path& path, std::string_view text, std::string& error)
{
    std::error_code directory_error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), directory_error);
    if (directory_error)
    {
        error = "could not create prefab directory: " + directory_error.message();
        return false;
    }
    const auto temporary = path.parent_path() / (path.filename().string() + ".tmp");
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream)
        {
            error = "could not open temporary prefab file";
            return false;
        }
        stream.write(text.data(), static_cast<std::streamsize>(text.size()));
        stream.flush();
        if (!stream)
        {
            error = "failed while writing temporary prefab file";
            return false;
        }
    }
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    {
        error = "atomic prefab replacement failed with Win32 error " + std::to_string(GetLastError());
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
#else
    std::error_code move_error;
    std::filesystem::rename(temporary, path, move_error);
    if (move_error)
    {
        error = move_error.message();
        return false;
    }
#endif
    return true;
}

template <class Component>
void copy_component(const editor_scene_state& source, editor_scene_state& target, scene::entity from, scene::entity to)
{
    if (const auto* value = source.scene.try_get<Component>(from))
        target.scene.emplace<Component>(to, *value);
}

scene::entity clone_subtree(
    const editor_scene_state& source,
    editor_scene_state& target,
    scene::entity source_entity,
    scene::entity parent,
    std::vector<std::pair<scene::entity_guid, scene::entity_guid>>& mapping)
{
    const scene::entity result = target.scene.create();
    copy_component<scene::name_component>(source, target, source_entity, result);
    copy_component<scene::tag_component>(source, target, source_entity, result);
    copy_component<scene::active_component>(source, target, source_entity, result);
    copy_component<scene::transform_component>(source, target, source_entity, result);
    copy_component<scene::bounds_component>(source, target, source_entity, result);
    copy_component<scene::camera_component>(source, target, source_entity, result);
    copy_component<scene::mesh_renderer_component>(source, target, source_entity, result);
    copy_component<scene::virtual_mesh_renderer_component>(source, target, source_entity, result);
    copy_component<scene::skinned_mesh_renderer_component>(source, target, source_entity, result);
    copy_component<scene::lod_component>(source, target, source_entity, result);
    copy_component<scene::render_layer_component>(source, target, source_entity, result);
    copy_component<scene::directional_light_component>(source, target, source_entity, result);
    copy_component<scene::point_light_component>(source, target, source_entity, result);
    copy_component<scene::spot_light_component>(source, target, source_entity, result);
    copy_component<scene::world_environment_component>(source, target, source_entity, result);
    copy_component<scene::sky_atmosphere_component>(source, target, source_entity, result);
    copy_component<scene::celestial_sky_component>(source, target, source_entity, result);
    copy_component<scene::cloud_layers_component>(source, target, source_entity, result);
    copy_component<scene::environment_lighting_component>(source, target, source_entity, result);
    copy_component<scene::height_fog_component>(source, target, source_entity, result);
    copy_component<scene::terrain_component>(source, target, source_entity, result);
    copy_component<scene::water_component>(source, target, source_entity, result);
    copy_component<scene::vegetation_component>(source, target, source_entity, result);
    copy_component<scene::decal_component>(source, target, source_entity, result);
    copy_component<scene::world_region_component>(source, target, source_entity, result);
    target.scene.emplace<scene::persistent_id_component>(result, scene::generate_entity_guid());
    target.scene.emplace<scene::hierarchy_component>(result);
    target.scene.emplace<scene::selection_component>(result, false);

    const auto source_guid = entity_guid_of(source, source_entity);
    const auto result_guid = entity_guid_of(target, result);
    mapping.emplace_back(source_guid, result_guid);
    if (const auto* binding = find_asset_binding(source, source_guid))
    {
        auto copied = *binding;
        copied.entity = result_guid;
        target.asset_bindings.push_back(std::move(copied));
    }
    if (target.scene.alive(parent))
        scene::reparent(target.scene, result, parent, {}, scene::reparent_transform_policy::preserve_local);
    for (const scene::entity child : scene::children(source.scene, source_entity))
        clone_subtree(source, target, child, result, mapping);
    return result;
}

std::optional<json> read_prefab(const std::filesystem::path& path, std::string& error)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        error = "could not open prefab file";
        return std::nullopt;
    }
    json document;
    try { stream >> document; }
    catch (const std::exception& exception)
    {
        error = std::string("invalid prefab JSON: ") + exception.what();
        return std::nullopt;
    }
    if (!document.is_object() || document.value("format", "") != "arc.prefab" ||
        document.value("formatVersion", 0u) != ecs::prefab_asset::current_format_version ||
        !document.contains("prefab") || !document["prefab"].is_object() ||
        !document.contains("entities") || !document["entities"].is_array())
    {
        error = "unsupported or malformed ARC prefab";
        return std::nullopt;
    }
    const auto& metadata = document["prefab"];
    if (!metadata.contains("id") || !metadata["id"].is_string() ||
        !metadata.contains("root") || !metadata["root"].is_string() ||
        !scene::parse_entity_guid(metadata["id"].get<std::string>()) ||
        !scene::parse_entity_guid(metadata["root"].get<std::string>()))
    {
        error = "prefab identity is invalid";
        return std::nullopt;
    }
    return document;
}

std::string project_relative_path(
    const std::filesystem::path& project_root,
    const std::filesystem::path& path)
{
    std::error_code error;
    const auto relative = std::filesystem::relative(path, project_root, error);
    return error || relative.empty() ? path.generic_string() : relative.lexically_normal().generic_string();
}

std::filesystem::path resolve_prefab_path(
    const std::filesystem::path& project_root,
    const std::filesystem::path& path)
{
    return path.is_absolute() ? path : (project_root / path).lexically_normal();
}

} // namespace

prefab_document_result save_prefab_document(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    scene::entity root,
    const std::filesystem::path& path)
{
    if (path.empty() || path.extension() != ".arcprefab")
        return { .message = "prefab path must use the .arcprefab extension" };
    scene::entity_guid prefab_guid = scene::generate_entity_guid();
    if (const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
        instance && instance->prefab_guid.valid())
        prefab_guid = instance->prefab_guid;
    const auto serialized = serialize_scene_subtree_as_prefab(
        state, project_root, root, prefab_guid, path.stem().string());
    if (!serialized.succeeded)
        return { .message = serialized.message };
    std::string error;
    if (!atomic_write(path, serialized.text, error))
        return { .message = std::move(error) };
    auto& instance = state.scene.emplace<scene::prefab_instance_component>(root);
    instance.prefab_guid = prefab_guid;
    instance.prefab_path = project_relative_path(project_root, path);
    instance.source_root = entity_guid_of(state, root);
    instance.source_to_instance.clear();
    for (const scene::entity value : scene::subtree(state.scene, root))
    {
        const auto guid = entity_guid_of(state, value);
        instance.source_to_instance.emplace_back(guid, guid);
    }
    instance.overrides.clear();
    return { .succeeded = true, .root = root, .entity_count = serialized.entity_count, .message = "Prefab saved" };
}

prefab_document_result instantiate_prefab_document(
    editor_scene_state& state,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    const std::filesystem::path& path,
    scene::entity parent)
{
    std::string error;
    const auto document = read_prefab(path, error);
    if (!document)
        return { .message = std::move(error) };
    json scene_document{
        { "format", "arc.scene" },
        { "formatVersion", arc_scene_format_version },
        { "scene", {
            { "id", scene::to_string(scene::generate_entity_guid()) },
            { "name", document->at("prefab").value("name", path.stem().string()) }
        } },
        { "entities", document->at("entities") }
    };
    const auto temporary = path.parent_path() / ("." + path.filename().string() + ".instantiate.arcscene");
    if (!atomic_write(temporary, scene_document.dump(2) + '\n', error))
        return { .message = std::move(error) };

    editor_scene_state loaded = state;
    const auto loaded_result = load_scene_document(loaded, renderer, project_root, temporary);
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    if (!loaded_result.succeeded)
        return { .message = loaded_result.message, .diagnostics = loaded_result.diagnostics };

    const auto source_root_guid = *scene::parse_entity_guid(document->at("prefab").at("root").get<std::string>());
    const scene::entity source_root = find_entity_by_guid(loaded, source_root_guid);
    if (!loaded.scene.alive(source_root))
        return { .message = "prefab root entity is missing" };

    std::vector<std::pair<scene::entity_guid, scene::entity_guid>> mapping;
    const scene::entity root = clone_subtree(loaded, state, source_root, parent, mapping);
    auto& instance = state.scene.emplace<scene::prefab_instance_component>(root);
    instance.prefab_guid = *scene::parse_entity_guid(document->at("prefab").at("id").get<std::string>());
    instance.prefab_path = project_relative_path(project_root, path);
    instance.source_root = source_root_guid;
    instance.source_to_instance = std::move(mapping);
    return {
        .succeeded = true,
        .root = root,
        .entity_count = instance.source_to_instance.size(),
        .message = "Prefab instantiated"
    };
}

prefab_document_result apply_prefab_instance(
    editor_scene_state& state,
    const std::filesystem::path& project_root,
    scene::entity root)
{
    auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    if (!instance)
        return { .message = "entity is not a prefab instance" };
    const auto result = save_prefab_document(
        state, project_root, root, resolve_prefab_path(project_root, instance->prefab_path));
    if (result.succeeded)
        instance->overrides.clear();
    return result;
}

prefab_document_result revert_prefab_instance(
    editor_scene_state& state,
    render::renderer& renderer,
    const std::filesystem::path& project_root,
    scene::entity root)
{
    const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    const auto* hierarchy = state.scene.try_get<scene::hierarchy_component>(root);
    if (!instance)
        return { .message = "entity is not a prefab instance" };
    const std::filesystem::path source = resolve_prefab_path(project_root, instance->prefab_path);
    const scene::entity parent = hierarchy ? hierarchy->parent : scene::entity{};
    const auto replacement = instantiate_prefab_document(state, renderer, project_root, source, parent);
    if (!replacement.succeeded)
        return replacement;
    const auto removed = scene::subtree(state.scene, root);
    std::vector<scene::entity_guid> removed_guids;
    removed_guids.reserve(removed.size());
    for (const scene::entity entity : removed)
        removed_guids.push_back(entity_guid_of(state, entity));
    scene::destroy_subtree(state.scene, root);
    for (const scene::entity_guid guid : removed_guids)
    {
        state.asset_bindings.erase(std::remove_if(state.asset_bindings.begin(), state.asset_bindings.end(),
            [guid](const auto& binding) { return binding.entity == guid; }), state.asset_bindings.end());
    }
    return replacement;
}

bool unpack_prefab_instance(editor_scene_state& state, scene::entity root)
{
    return state.scene.remove<scene::prefab_instance_component>(root);
}

} // namespace arc::editor

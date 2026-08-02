#include <arc/editor/prefab_document.h>

#include <arc/editor/scene_document.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <unordered_map>

namespace arc::editor
{
namespace
{
using json = nlohmann::json;

template <class Component>
void copy_component(const editor_scene_state& source, editor_scene_state& target, ecs::entity from, ecs::entity to)
{
    if (const auto* value = source.scene.try_get<Component>(from)) target.scene.emplace<Component>(to, *value);
}

ecs::entity clone_subtree(const editor_scene_state& source, editor_scene_state& target, ecs::entity source_entity,
                          ecs::entity parent, std::vector<std::pair<ecs::entity_guid, ecs::entity_guid>>& mapping)
{
    const ecs::entity result = target.scene.create();
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
    copy_component<scene::prefab_instance_component>(source, target, source_entity, result);
    target.scene.emplace<scene::persistent_id_component>(result, ecs::generate_entity_guid());
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
    for (const auto& [entity, record] : source.unknown_component_records)
        if (entity == source_guid) target.unknown_component_records.emplace_back(result_guid, record);
    for (const auto& preserved : source.preserved_component_records)
        if (preserved.entity == source_guid)
            target.preserved_component_records.push_back({result_guid, preserved.component_name, preserved.json});
    if (target.scene.alive(parent))
        scene::reparent(target.scene, result, parent, {}, scene::reparent_transform_policy::preserve_local);
    for (const ecs::entity child : scene::children(source.scene, source_entity))
        clone_subtree(source, target, child, result, mapping);
    return result;
}

void remap_nested_prefab_instances(editor_scene_state& state,
                                   const std::vector<std::pair<ecs::entity_guid, ecs::entity_guid>>& mapping)
{
    const std::unordered_map<ecs::entity_guid, ecs::entity_guid, ecs::entity_guid_hash> remap(mapping.begin(),
                                                                                              mapping.end());
    for (const auto& [source_guid, instance_guid] : mapping)
    {
        (void)source_guid;
        const auto entity = find_entity_by_guid(state, instance_guid);
        auto* nested = state.scene.try_get<scene::prefab_instance_component>(entity);
        if (!nested) continue;
        nested->nested = true;
        for (auto& [source, instance] : nested->source_to_instance)
        {
            (void)source;
            if (const auto found = remap.find(instance); found != remap.end()) instance = found->second;
        }
    }
}

std::optional<json> read_prefab(const std::filesystem::path& path, std::string& error)
{
    const auto stored = persistence::document_store{}.load_json(path);
    if (!stored.succeeded)
    {
        error = stored.error.empty() ? "could not open prefab file" : stored.error;
        return std::nullopt;
    }
    json document;
    try
    {
        document = json::parse(stored.text);
    }
    catch (const std::exception& exception)
    {
        error = std::string("invalid prefab JSON: ") + exception.what();
        return std::nullopt;
    }
    if (!document.is_object() || document.value("format", "") != "arc.prefab" ||
        document.value("formatVersion", 0u) < 1u ||
        document.value("formatVersion", 0u) > ecs::prefab_asset::current_format_version ||
        !document.contains("prefab") || !document["prefab"].is_object() || !document.contains("entities") ||
        !document["entities"].is_array())
    {
        error = "unsupported or malformed ARC prefab";
        return std::nullopt;
    }
    const auto& metadata = document["prefab"];
    if (!metadata.contains("id") || !metadata["id"].is_string() || !metadata.contains("root") ||
        !metadata["root"].is_string() || !ecs::parse_entity_guid(metadata["id"].get<std::string>()) ||
        !ecs::parse_entity_guid(metadata["root"].get<std::string>()))
    {
        error = "prefab identity is invalid";
        return std::nullopt;
    }
    return document;
}

std::string project_relative_path(const std::filesystem::path& project_root, const std::filesystem::path& path)
{
    std::error_code error;
    const auto relative = std::filesystem::relative(path, project_root, error);
    return error || relative.empty() ? path.generic_string() : relative.lexically_normal().generic_string();
}

std::filesystem::path resolve_prefab_path(const std::filesystem::path& project_root, const std::filesystem::path& path,
                                          ecs::entity_guid guid = {}, assets::asset_manager* asset_registry = nullptr)
{
    if (asset_registry && guid.valid())
    {
        if (const auto current = asset_registry->find({guid.high, guid.low});
            current && current->type == assets::asset_types::prefab)
            return (project_root / current->source_path).lexically_normal();
    }
    return path.is_absolute() ? path : (project_root / path).lexically_normal();
}

} // namespace

prefab_document_result save_prefab_document(editor_scene_state& state, const std::filesystem::path& project_root,
                                            ecs::entity root, const std::filesystem::path& path, assets::asset_manager*)
{
    if (path.empty() || path.extension() != ".arcprefab")
        return {.message = "prefab path must use the .arcprefab extension"};
    ecs::entity_guid prefab_guid = ecs::generate_entity_guid();
    if (const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
        instance && instance->prefab_guid.valid())
        prefab_guid = instance->prefab_guid;
    const auto serialized =
        serialize_scene_subtree_as_prefab(state, project_root, root, prefab_guid, path.stem().string());
    if (!serialized.succeeded) return {.message = serialized.message};
    const auto saved = persistence::document_store{}.save_json(path, serialized.text, true);
    if (!saved.succeeded) return {.message = saved.error};
    auto& instance = state.scene.emplace<scene::prefab_instance_component>(root);
    instance.prefab_guid = prefab_guid;
    instance.prefab_path = project_relative_path(project_root, path);
    instance.source_root = entity_guid_of(state, root);
    instance.source_to_instance.clear();
    for (const ecs::entity value : scene::subtree(state.scene, root))
    {
        const auto guid = entity_guid_of(state, value);
        instance.source_to_instance.emplace_back(guid, guid);
    }
    instance.overrides.clear();
    return {.succeeded = true, .root = root, .entity_count = serialized.entity_count, .message = "Prefab saved"};
}

prefab_document_result instantiate_prefab_document(editor_scene_state& state, render::renderer& renderer,
                                                   const std::filesystem::path& project_root,
                                                   const std::filesystem::path& path, ecs::entity parent,
                                                   assets::asset_manager* asset_registry)
{
    std::string error;
    const auto document = read_prefab(path, error);
    if (!document) return {.message = std::move(error)};
    json scene_document{{"format", "arc.scene"},
                        {"formatVersion", arc_scene_format_version},
                        {"scene",
                         {{"id", ecs::to_string(ecs::generate_entity_guid())},
                          {"name", document->at("prefab").value("name", path.stem().string())}}},
                        {"entities", document->at("entities")},
                        {"dependencies", document->value("dependencies", json::array())}};
    const auto sealed_scene = persistence::seal_json_document(scene_document.dump(), true);
    if (!sealed_scene.succeeded()) return {.message = sealed_scene.error};

    editor_scene_state loaded = state;
    const auto loaded_result =
        load_scene_document_text(loaded, renderer, project_root, path, sealed_scene.text, asset_registry);
    if (!loaded_result.succeeded) return {.message = loaded_result.message, .diagnostics = loaded_result.diagnostics};

    const auto source_root_guid = *ecs::parse_entity_guid(document->at("prefab").at("root").get<std::string>());
    const ecs::entity source_root = find_entity_by_guid(loaded, source_root_guid);
    if (!loaded.scene.alive(source_root)) return {.message = "prefab root entity is missing"};

    std::vector<std::pair<ecs::entity_guid, ecs::entity_guid>> mapping;
    const ecs::entity root = clone_subtree(loaded, state, source_root, parent, mapping);
    remap_nested_prefab_instances(state, mapping);
    auto& instance = state.scene.emplace<scene::prefab_instance_component>(root);
    instance.prefab_guid = *ecs::parse_entity_guid(document->at("prefab").at("id").get<std::string>());
    instance.prefab_path = project_relative_path(project_root, path);
    instance.source_root = source_root_guid;
    instance.source_to_instance = std::move(mapping);
    return {.succeeded = true,
            .root = root,
            .entity_count = instance.source_to_instance.size(),
            .message = "Prefab instantiated"};
}

prefab_document_result apply_prefab_instance(editor_scene_state& state, const std::filesystem::path& project_root,
                                             ecs::entity root, assets::asset_manager* asset_registry)
{
    auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    if (!instance) return {.message = "entity is not a prefab instance"};
    const auto result = save_prefab_document(
        state, project_root, root,
        resolve_prefab_path(project_root, instance->prefab_path, instance->prefab_guid, asset_registry),
        asset_registry);
    if (result.succeeded) instance->overrides.clear();
    return result;
}

prefab_document_result revert_prefab_instance(editor_scene_state& state, render::renderer& renderer,
                                              const std::filesystem::path& project_root, ecs::entity root,
                                              assets::asset_manager* asset_registry)
{
    const auto* instance = state.scene.try_get<scene::prefab_instance_component>(root);
    const auto* hierarchy = state.scene.try_get<scene::hierarchy_component>(root);
    if (!instance) return {.message = "entity is not a prefab instance"};
    const std::filesystem::path source =
        resolve_prefab_path(project_root, instance->prefab_path, instance->prefab_guid, asset_registry);
    const ecs::entity parent = hierarchy ? hierarchy->parent : ecs::entity{};
    const auto replacement = instantiate_prefab_document(state, renderer, project_root, source, parent, asset_registry);
    if (!replacement.succeeded) return replacement;
    const auto removed = scene::subtree(state.scene, root);
    std::vector<ecs::entity_guid> removed_guids;
    removed_guids.reserve(removed.size());
    for (const ecs::entity entity : removed)
        removed_guids.push_back(entity_guid_of(state, entity));
    scene::destroy_subtree(state.scene, root);
    for (const ecs::entity_guid guid : removed_guids)
    {
        state.asset_bindings.erase(std::remove_if(state.asset_bindings.begin(), state.asset_bindings.end(),
                                                  [guid](const auto& binding) { return binding.entity == guid; }),
                                   state.asset_bindings.end());
    }
    return replacement;
}

bool unpack_prefab_instance(editor_scene_state& state, ecs::entity root)
{
    return state.scene.remove<scene::prefab_instance_component>(root);
}

} // namespace arc::editor

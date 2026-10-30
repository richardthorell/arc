#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/assets/package_artifact_reader.h>
#include <arc/render_tools/material_asset.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace assets = arc::assets;
namespace materials = arc::render::tools;
using json = nlohmann::json;

void check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

json read_json(const std::filesystem::path& path)
{
    std::ifstream input(path);
    check(static_cast<bool>(input), "Unable to open fixture: " + path.generic_string());
    return json::parse(input);
}

assets::asset_guid guid_of(const std::string& text, std::string_view context)
{
    const auto guid = assets::parse_asset_guid(text);
    check(guid.has_value(), std::string(context) + ": invalid GUID " + text);
    return *guid;
}

const assets::cook_manifest_artifact* find_artifact(const assets::cook_manifest& manifest,
                                                    assets::asset_guid guid,
                                                    assets::artifact_schema_id schema)
{
    const auto it = std::find_if(manifest.artifacts.begin(), manifest.artifacts.end(),
                                 [&](const auto& entry) { return entry.asset == guid && entry.schema == schema; });
    return it == manifest.artifacts.end() ? nullptr : &*it;
}

std::vector<std::byte> require_artifact(const assets::cook_manifest& manifest,
                                        const assets::package_artifact_reader& reader, assets::asset_guid guid,
                                        assets::artifact_schema_id schema, const std::string& label)
{
    const auto* artifact = find_artifact(manifest, guid, schema);
    check(artifact != nullptr, label + " (" + assets::to_string(guid) + "): cooked artifact missing");
    const assets::cooked_artifact_address address{artifact->asset, artifact->schema, artifact->name};
    const auto loaded = reader.read(address);
    check(static_cast<bool>(loaded),
          label + " (" + assets::to_string(guid) + "): unable to load named artifact " + artifact->name);
    check(assets::hash_bytes(loaded.value()) == artifact->hash,
          label + " (" + assets::to_string(guid) + "): runtime package hash mismatch");
    return loaded.value();
}

void require_dependency(const assets::cook_manifest& manifest, assets::asset_guid guid,
                        const std::string& label)
{
    check(std::find(manifest.dependency_closure.begin(), manifest.dependency_closure.end(), guid) !=
              manifest.dependency_closure.end(),
          label + " (" + assets::to_string(guid) + "): missing from native cooker dependency closure");
}

void verify_texture_sources(const std::filesystem::path& project, const json& material_source,
                            const assets::cook_manifest& manifest, const assets::package_artifact_reader& reader,
                            const std::string& label, unsigned& texture_count)
{
    if (!material_source.contains("graph")) return;
    for (const auto& node : material_source["graph"].value("nodes", json::array()))
    {
        if (node.value("type", "") != "textureSample2D") continue;
        const auto& values = node.value("values", json::object());
        const auto path = values.value("texture", "");
        if (path.empty()) continue;
        const auto source = project / "assets" / path;
        check(std::filesystem::is_regular_file(source), label + ": missing source texture " + path);
        const auto meta = read_json(source.string() + ".arcmeta");
        const auto guid = guid_of(meta.at("guid").get<std::string>(), label);
        require_dependency(manifest, guid, label + ": texture " + path);
        require_artifact(manifest, reader, guid, assets::artifact_schemas::texture,
                         label + ": texture " + path);
        ++texture_count;
    }
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        check(argc == 3, "Usage: arc-materials-baseline-package-verify <package manifest> <isolated project root>");
        const std::filesystem::path manifest_path = argv[1];
        const std::filesystem::path project = argv[2];

        const auto manifest_result = assets::load_cook_manifest(manifest_path);
        check(static_cast<bool>(manifest_result), "Cannot load native cook manifest");
        const auto& manifest = manifest_result.value();
        assets::package_artifact_reader reader;
        check(static_cast<bool>(reader.mount(manifest_path)), "Cannot mount native packaged artifacts");

        // Exercise actual native package reads for *every named artifact*, including
        // each shader pass, not just the first artifact per material GUID.
        for (const auto& artifact : manifest.artifacts)
        {
            const assets::cooked_artifact_address address{artifact.asset, artifact.schema, artifact.name};
            auto payload = reader.read(address);
            check(static_cast<bool>(payload),
                  "Cannot load " + assets::to_string(artifact.asset) + ":" + artifact.name);
            check(assets::hash_bytes(payload.value()) == artifact.hash,
                  "Hash mismatch " + assets::to_string(artifact.asset) + ":" + artifact.name);
        }

        const auto scene = read_json(project / "assets/fixtures/materials_baseline.arcscene");
        const auto scene_meta = read_json(project / "assets/fixtures/materials_baseline.arcscene.arcmeta");
        const auto scene_guid = guid_of(scene_meta.at("guid").get<std::string>(), "Materials Baseline scene");
        require_dependency(manifest, scene_guid, "Materials Baseline scene");
        require_artifact(manifest, reader, scene_guid, assets::artifact_schemas::scene, "Materials Baseline scene");

        std::map<std::string, std::string> scene_dependency_paths;
        for (const auto& dependency : scene.at("dependencies"))
            scene_dependency_paths.emplace(dependency.at("ownerEntity").get<std::string>(),
                                           dependency.at("guid").get<std::string>());

        unsigned specimens = 0;
        unsigned instances = 0;
        unsigned specialized = 0;
        unsigned textures = 0;

        for (const auto& entity : scene.at("entities"))
        {
            if (!entity.contains("assetBinding")) continue;
            const auto& binding = entity.at("assetBinding");
            if (!binding.contains("material")) continue;
            const auto& reference = binding.at("material");
            const auto label = entity.at("components").at("Name").at("value").get<std::string>();
            const auto id = entity.at("id").get<std::string>();
            const auto guid_text = reference.at("guid").get<std::string>();
            const auto guid = guid_of(guid_text, label);
            const auto path_hint = reference.at("pathHint").get<std::string>();
            const auto source = project / path_hint;
            const auto metadata = read_json(source.string() + ".arcmeta");
            check(metadata.at("guid") == guid_text, label + ": source .arcmeta GUID disagrees with scene");
            check(scene_dependency_paths.contains(id) && scene_dependency_paths.at(id) == guid_text,
                  label + ": missing or stale authored scene dependency");
            require_dependency(manifest, guid, label);
            const auto authored = read_json(source);

            const bool is_instance =
                reference.at("expectedType").get<std::string>() == assets::to_string(assets::asset_types::material_instance);
            if (!is_instance)
            {
                check(reference.at("expectedType") == assets::to_string(assets::asset_types::material),
                      label + ": unexpected material asset type");
                const auto bytes =
                    require_artifact(manifest, reader, guid, assets::artifact_schemas::material, label);
                const auto decoded = materials::deserialize_material_package_v4(bytes);
                check(static_cast<bool>(decoded), label + ": cooked Material payload cannot be deserialized");
                check(!decoded.value().compiled.passes.empty(), label + ": cooked Material has no render passes");
                check(!decoded.value().canonical_document_json.empty(), label + ": missing canonical authoring data");
                verify_texture_sources(project, authored, manifest, reader, label, textures);
            }
            else
            {
                ++instances;
                const auto bytes =
                    require_artifact(manifest, reader, guid, assets::artifact_schemas::material_instance, label);
                const auto decoded = materials::deserialize_material_instance_package_v1(bytes);
                check(static_cast<bool>(decoded), label + ": cooked Material Instance cannot be deserialized");
                check(!decoded.value().material.compiled.passes.empty(),
                      label + ": Material Instance has no resolved render passes");

                const auto parent_guid_text = authored.at("parent").at("guid").get<std::string>();
                check(decoded.value().parent_guid == parent_guid_text, label + ": wrong cooked parent GUID");
                const auto parent_guid = guid_of(parent_guid_text, label);
                require_dependency(manifest, parent_guid, label + ": parent Material");
                const auto parent_bytes = require_artifact(manifest, reader, parent_guid,
                                                           assets::artifact_schemas::material,
                                                           label + ": parent Material");
                const auto parent = materials::deserialize_material_package_v4(parent_bytes);
                check(static_cast<bool>(parent), label + ": parent Material package failed to deserialize");

                const auto canonical_instance = json::parse(decoded.value().canonical_instance_json);
                check(canonical_instance.at("parameterOverrides") == authored.at("parameterOverrides"),
                      label + ": cooked parameter overrides differ from source");
                check(canonical_instance.at("functionOverrides") == authored.at("functionOverrides"),
                      label + ": cooked function overrides differ from source");

                const bool has_function_override = !authored.at("functionOverrides").empty();
                if (has_function_override)
                {
                    ++specialized;
                    check(decoded.value().function_specialization_key != 0,
                          label + ": missing Function Slot specialization key");
                    for (const auto& override_value : authored.at("functionOverrides"))
                    {
                        const auto& function = override_value.at("function");
                        const auto function_guid = guid_of(function.at("guid").get<std::string>(), label);
                        require_dependency(manifest, function_guid, label + ": selected Material Function");
                        require_artifact(manifest, reader, function_guid, assets::artifact_schemas::material_function,
                                         label + ": selected Material Function");
                    }
                }
                else
                {
                    check(decoded.value().function_specialization_key == 0,
                          label + ": value-only Material Instance unexpectedly compiled a specialization");
                    check(decoded.value().material.compiled.passes.size() == parent.value().compiled.passes.size(),
                          label + ": value-only Material Instance did not reuse parent passes");
                }
            }
            ++specimens;
        }

        check(specimens == 13, "Expected 12 material specimens plus ground; saw " + std::to_string(specimens));
        check(instances == 5, "Expected five authored Material Instances; saw " + std::to_string(instances));
        check(specialized == 1, "Expected exactly one Function Slot specialization");
        check(textures >= 1, "Expected at least one real 2D texture dependency to be cooked");
        std::cout << "Materials Baseline native runtime package: " << specimens << " assignments, "
                  << instances << " instances, " << specialized << " function specialization, "
                  << textures << " texture bindings, " << manifest.artifacts.size()
                  << " named artifacts loaded and hash-verified\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Materials Baseline verification failed: " << error.what() << '\n';
        return 1;
    }
}

#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/assets/package_artifact_reader.h>
#include <arc/render/cooked_material.h>
#include <arc/render/cooked_material_texture_bindings.h>
#include <arc/render/cooked_texture.h>
#include <arc/render/mesh_artifact.h>
#include <arc/scene/runtime_scene.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
namespace assets = arc::assets;

void require(bool ok, const std::string& message)
{
    if (!ok) throw std::runtime_error(message);
}

assets::asset_guid guid(std::string_view text)
{
    const auto parsed = assets::parse_asset_guid(text);
    require(parsed.has_value(), "Invalid fixture GUID");
    return *parsed;
}

std::vector<std::byte> load(const assets::cook_manifest& manifest, const assets::package_artifact_reader& reader,
                            assets::asset_guid asset, assets::artifact_schema_id schema, const std::string& label)
{
    const auto matching = [&](const auto& artifact) { return artifact.asset == asset && artifact.schema == schema; };
    const auto found = std::find_if(manifest.artifacts.begin(), manifest.artifacts.end(), matching);
    require(found != manifest.artifacts.end(), label + ": missing cooked artifact");
    const auto loaded = reader.read({found->asset, found->schema, found->name});
    require(static_cast<bool>(loaded), label + ": failed to read active cooked package");
    require(assets::hash_bytes(loaded.value()) == found->hash, label + ": invalid package payload hash");
    return loaded.value();
}

void require_closure(const assets::cook_manifest& manifest, assets::asset_guid asset, std::string_view label)
{
    require(std::find(manifest.dependency_closure.begin(), manifest.dependency_closure.end(), asset) !=
                manifest.dependency_closure.end(),
            std::string(label) + ": absent from dependency closure");
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        require(argc == 3, "Usage: arc-android-device-smoke-verify <manifest> <project>");
        const std::filesystem::path manifest_path = argv[1];
        const auto cooked = assets::load_cook_manifest(manifest_path);
        require(static_cast<bool>(cooked), "Cannot load Android cook manifest");
        const auto& manifest = cooked.value();
        assets::package_artifact_reader reader;
        require(static_cast<bool>(reader.mount(manifest_path)), "Cannot mount Android packaged content");

        // Check every named artifact, including all shader variants, against
        // the production content address and hash, not source paths.
        for (const auto& artifact : manifest.artifacts)
        {
            const auto bytes = reader.read({artifact.asset, artifact.schema, artifact.name});
            require(static_cast<bool>(bytes), "Unreadable cooked artifact: " + artifact.name);
            require(assets::hash_bytes(bytes.value()) == artifact.hash,
                    "Invalid cooked artifact hash: " + artifact.name);
        }

        const auto scene_guid = guid("b80a0000-0000-4000-8000-000000000005");
        const auto mesh_guid = guid("fa99c80b-dca6-4a36-a505-eb75c3c444ee");
        const auto material_guid = guid("c9430000-0000-4000-8000-000000000015");
        require_closure(manifest, scene_guid, "Startup scene");
        require_closure(manifest, mesh_guid, "Mesh");
        require_closure(manifest, material_guid, "Material");

        const auto scene_bytes = load(manifest, reader, scene_guid, assets::artifact_schemas::scene, "Scene");
        arc::ecs::world world;
        arc::scene::cooked_scene_mesh_bindings bindings;
        const auto error = arc::scene::materialize_cooked_scene(world, scene_bytes, &bindings);
        require(error.empty(), "Runtime scene materialization failed: " + error);
        require(bindings.size() == 1, "Expected exactly one scene mesh binding");
        require(bindings[0].source.guid == mesh_guid, "Scene mesh GUID differs from its package");
        require(bindings[0].material.guid == material_guid, "Scene material GUID differs from its package");

        const auto mesh_bytes = load(manifest, reader, mesh_guid, assets::artifact_schemas::mesh, "Mesh");
        const auto mesh = arc::render::decode_conventional_mesh_artifact(mesh_bytes, bindings[0].subresource);
        if (!mesh) throw std::runtime_error("Android mesh decoder rejected the cooked mesh: " + mesh.error());
        require(!mesh.value().mesh.vertices.empty(), "Cooked mesh contains no vertices");
        require(!mesh.value().mesh.indices.empty(), "Cooked mesh contains no triangles");

        const auto material_bytes =
            load(manifest, reader, material_guid, assets::artifact_schemas::material, "Material");
        const auto material = arc::render::decode_cooked_material_metadata(material_bytes);
        require(static_cast<bool>(material), "Runtime rejected cooked material metadata");
        require(!material.value().compiled.passes.empty(), "Cooked material contains no executable shader passes");

        const auto texture_manifest_bytes =
            load(manifest, reader, material_guid, assets::artifact_schemas::material_textures, "Texture slot manifest");
        const auto texture_manifest = arc::render::decode_cooked_material_texture_manifest(texture_manifest_bytes);
        require(static_cast<bool>(texture_manifest), "Runtime rejected cooked texture slot manifest");
        require(texture_manifest.value().material == material.value().compiled.package,
                "Texture binding manifest references a different shader package");
        require(!texture_manifest.value().textures.empty(), "Material fixture has no texture slots");
        std::size_t textured = 0;
        for (const auto& entry : texture_manifest.value().textures)
        {
            if (!entry.asset.valid()) continue;
            ++textured;
            require_closure(manifest, entry.asset, "Texture");
            const auto bytes = load(manifest, reader, entry.asset, assets::artifact_schemas::texture, "Texture");
            const auto texture = arc::render::decode_cooked_texture_data(bytes);
            require(static_cast<bool>(texture), "Runtime texture decoder rejected the cooked texture");
        }
        require(textured >= 1, "Fixture requires at least one bound, cooked source texture");

        std::vector<arc::render::cooked_material_shader> shaders;
        for (const auto& artifact : manifest.artifacts)
        {
            if (artifact.asset != material_guid || artifact.schema != assets::artifact_schemas::shader ||
                artifact.name.ends_with(".coat-neutral"))
                continue;
            auto bytes = reader.read({artifact.asset, artifact.schema, artifact.name});
            require(static_cast<bool>(bytes), "Unable to read material shader representation");
            auto parsed = arc::render::deserialize_shader_package(bytes.value());
            require(static_cast<bool>(parsed), "Invalid cooked material shader representation");
            shaders.push_back({.artifact_name = artifact.name, .package = std::move(parsed).value()});
        }
        const auto realized = arc::render::realize_cooked_material_program(material.value(), shaders);
        require(static_cast<bool>(realized), "Runtime rejected executable material shader package");
        require(!realized.value().passes.empty(), "Material has no executable shader passes");

        const auto passes = realized.value().passes.size();
        const auto artifacts = manifest.artifacts.size();
        std::cout << "Android device smoke: scene=1 mesh=1 textures=" << textured << " passes=" << passes
                  << " verified-artifacts=" << artifacts << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Android device smoke preflight FAILED: " << error.what() << '\n';
        return 1;
    }
}

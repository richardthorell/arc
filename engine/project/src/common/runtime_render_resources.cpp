#include <arc/project/runtime_render_resources.h>

#include <arc/render/cooked_texture.h>
#include <arc/render/mesh_artifact.h>
#include <arc/render/primitives.h>

#include <optional>
#include <set>

namespace arc::project
{
namespace
{
std::optional<render::mesh_data> decode_procedural_primitive(std::string_view name)
{
    if (name == "Plane" || name == "plane") return render::make_plane_mesh(10.0f);
    if (name == "Cube" || name == "cube") return render::make_cube_mesh(1.0f);
    if (name == "Sphere" || name == "sphere") return render::make_uv_sphere_mesh(0.5f, 32, 16);
    if (name == "Cylinder" || name == "cylinder") return render::make_cylinder_mesh(0.5f, 1.0f, 32);
    if (name == "Cone" || name == "cone") return render::make_cone_mesh(0.5f, 1.0f, 32);
    if (name == "Capsule" || name == "capsule") return render::make_capsule_mesh(0.5f, 1.0f, 32, 8);
    return std::nullopt;
}
} // namespace

decoded_scene_render_result decode_scene_render_resources(runtime_catalog_session& catalog,
                                                          const scene::cooked_scene_mesh_bindings& bindings)
{
    decoded_scene_render_resources decoded;
    decoded.reserve(bindings.size());
    for (const auto& binding : bindings)
    {
        const auto owner = assets::to_string(binding.source.guid);
        decoded_scene_render_resource resource;
        resource.entity = binding.entity;
        resource.procedural_primitive = binding.procedural_primitive;
        if (binding.procedural_primitive)
        {
            auto primitive = decode_procedural_primitive(binding.subresource);
            if (!primitive)
                return decoded_scene_render_result::failure("Unsupported authored primitive '" + binding.subresource +
                                                            "'");
            resource.mesh = std::move(*primitive);
        }
        else if (!binding.source.resolved())
            return decoded_scene_render_result::failure("Scene mesh binding has no authored source GUID");
        else
        {
            auto mesh_bytes = catalog.read_active_artifact(binding.source.guid, binding.source.expected_type,
                                                           assets::artifact_schemas::mesh);
            if (!mesh_bytes)
                return decoded_scene_render_result::failure("Mesh " + owner + ": " + mesh_bytes.error().message);
            auto mesh = render::decode_conventional_mesh_artifact(mesh_bytes.value(), binding.subresource);
            if (!mesh) return decoded_scene_render_result::failure("Mesh " + owner + ": " + mesh.error());
            resource.mesh = std::move(mesh).value().mesh;
        }
        if (binding.material.resolved())
        {
            const auto material_guid = binding.material.guid;
            const bool instance_type = binding.material.expected_type == assets::asset_types::material_instance;
            if (!instance_type && binding.material.expected_type != assets::asset_types::material)
                return decoded_scene_render_result::failure("Unsupported cooked scene material asset type");

            auto shader_guid = material_guid;
            if (instance_type)
            {
                auto bytes = catalog.read_active_artifact(material_guid, assets::asset_types::material_instance,
                                                          assets::artifact_schemas::material_instance);
                if (!bytes)
                    return decoded_scene_render_result::failure(
                        "Material Instance " + assets::to_string(material_guid) + ": " + bytes.error().message);
                auto instance = render::decode_cooked_material_instance_values(bytes.value());
                if (!instance)
                    return decoded_scene_render_result::failure(
                        "Material Instance " + assets::to_string(material_guid) + ": " + instance.error());
                shader_guid = instance.value().parent;
                resource.instance = std::move(instance).value();
                resource.material = resource.instance->parent_material;
            }
            else
            {
                auto bytes = catalog.read_active_artifact(material_guid, assets::asset_types::material,
                                                          assets::artifact_schemas::material);
                if (!bytes)
                    return decoded_scene_render_result::failure("Material " + assets::to_string(material_guid) + ": " +
                                                                bytes.error().message);
                auto material = render::decode_cooked_material_metadata(bytes.value());
                if (!material)
                    return decoded_scene_render_result::failure("Material " + assets::to_string(material_guid) + ": " +
                                                                material.error());
                resource.material = std::move(material).value();
            }
            resource.material_guid = material_guid;
            resource.shader_material_guid = shader_guid;

            auto manifest_bytes = catalog.read_active_artifact(shader_guid, assets::asset_types::material,
                                                               assets::artifact_schemas::material_textures);
            if (!manifest_bytes)
                return decoded_scene_render_result::failure("Material texture manifest " +
                                                            assets::to_string(shader_guid) + ": " +
                                                            manifest_bytes.error().message);
            auto manifest = render::decode_cooked_material_texture_manifest(manifest_bytes.value());
            if (!manifest)
                return decoded_scene_render_result::failure("Material texture manifest " +
                                                            assets::to_string(shader_guid) + ": " + manifest.error());
            if (manifest.value().material != resource.material->compiled.package)
                return decoded_scene_render_result::failure(
                    "Cooked material texture manifest has a different program identity");
            resource.material_textures = std::move(manifest).value();

            const auto record = catalog.catalog().database().query(shader_guid, assets::asset_types::material);
            if (!record)
                return decoded_scene_render_result::failure("Cooked material is missing from the active catalog");
            for (const auto& representation : record->representations)
            {
                if (representation.provider_id != record->active_provider) continue;
                const auto locator = io::virtual_path::parse(representation.locator_hint);
                if (!locator) continue;
                const auto address = assets::parse_cooked_artifact_virtual_path(locator.value());
                const bool shader_schema =
                    address && address->asset == shader_guid && address->schema == assets::artifact_schemas::shader;
                if (!shader_schema || address->name != representation.name ||
                    representation.name.ends_with(".coat-neutral"))
                    continue;
                auto bytes = catalog.read_active_artifact(shader_guid, assets::asset_types::material,
                                                          assets::artifact_schemas::shader, representation.name);
                if (!bytes)
                    return decoded_scene_render_result::failure("Material shader " + representation.name + ": " +
                                                                bytes.error().message);
                auto shader = render::deserialize_shader_package(bytes.value());
                if (!shader)
                    return decoded_scene_render_result::failure("Material shader " + representation.name + ": " +
                                                                shader.error().message);
                resource.material_shaders.push_back(
                    {.artifact_name = representation.name, .package = std::move(shader).value()});
            }

            std::set<assets::asset_guid> visited;
            for (const auto& texture_binding : resource.material_textures->textures)
            {
                const auto dependency = texture_binding.asset;
                if (!dependency.valid() || !visited.insert(dependency).second) continue;
                if (!catalog.catalog().database().query(dependency, assets::asset_types::texture_2d))
                    return decoded_scene_render_result::failure(
                        "Cooked material texture GUID is absent from active catalog");
                auto bytes = catalog.read_active_artifact(dependency, assets::asset_types::texture_2d,
                                                          assets::artifact_schemas::texture);
                if (!bytes)
                    return decoded_scene_render_result::failure("Texture " + assets::to_string(dependency) + ": " +
                                                                bytes.error().message);
                auto texture = render::decode_cooked_texture_data(bytes.value());
                if (!texture)
                    return decoded_scene_render_result::failure("Texture " + assets::to_string(dependency) + ": " +
                                                                texture.error());
                resource.textures.emplace_back(dependency, std::move(texture).value());
            }
        }
        decoded.push_back(std::move(resource));
    }
    return decoded_scene_render_result::success(std::move(decoded));
}
} // namespace arc::project

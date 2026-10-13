#include <arc/scene/irradiance_probe_bake.h>

#include <arc/scene/components.h>
#include <arc/scene/irradiance_probe_asset.h>
#include <arc/scene/transforms.h>

#include <algorithm>
#include <cmath>
#include <fstream>

namespace arc::scene
{
namespace
{

bool active(const ecs::world& world, ecs::entity entity)
{
    const auto* state = world.try_get<active_component>(entity);
    return !state || state->active;
}

render::render_mobility mobility(const ecs::world& world, ecs::entity entity)
{
    const auto* value = world.try_get<mobility_component>(entity);
    return value ? value->value : render::render_mobility::movable;
}

bool bake_eligible(const ecs::world& world, ecs::entity entity)
{
    const auto value = mobility(world, entity);
    return active(world, entity) &&
           (value == render::render_mobility::static_object || value == render::render_mobility::stationary);
}

math::vector3f material_diffuse(const render::material_descriptor* material)
{
    if (!material) return math::vector3f::one;
    return {std::max(material->base_color[0], 0.0f), std::max(material->base_color[1], 0.0f),
            std::max(material->base_color[2], 0.0f)};
}

math::vector3f material_emissive(const render::material_descriptor* material)
{
    if (!material) return {};
    const float scale =
        material->emissive_luminance_nits > 0.0f ? material->emissive_luminance_nits : material->emissive_strength;
    return math::mul(material->emissive_factor, std::max(scale, 0.0f));
}

render::irradiance_bake_environment bake_environment(const ecs::world& world)
{
    render::irradiance_bake_environment result;
    for (const auto entity : world.entities())
    {
        if (!active(world, entity)) continue;
        if (const auto* lighting = world.try_get<environment_lighting_component>(entity);
            lighting && lighting->enabled && lighting->source == environment_lighting_source::constant_color)
        {
            const auto value = math::mul(lighting->constant_color, std::max(lighting->diffuse_intensity, 0.0f));
            result.sky_radiance = value;
            result.ground_radiance = math::mul(value, 0.5f);
            return result;
        }
    }
    for (const auto entity : world.entities())
    {
        if (!active(world, entity)) continue;
        if (const auto* environment = world.try_get<world_environment_component>(entity);
            environment && environment->enabled && environment->affect_lighting)
        {
            const auto value = math::mul(environment->solid_color, std::max(environment->radiance_intensity, 0.0f));
            result.sky_radiance = value;
            result.ground_radiance = math::mul(value, 0.5f);
            return result;
        }
    }
    return result;
}

} // namespace

scene_irradiance_probe_bake_result
bake_scene_irradiance_probe_volume(ecs::world& world, render::renderer& renderer, assets::asset_manager& asset_manager,
                                   const scene_irradiance_probe_bake_request& request)
{
    scene_irradiance_probe_bake_result result;
    if (!world.alive(request.volume_entity))
    {
        result.error = "Irradiance probe-volume entity does not exist";
        return result;
    }

    auto* component = world.try_get<irradiance_probe_volume_component>(request.volume_entity);
    const auto* volume_transform = world.try_get<transform_component>(request.volume_entity);
    if (!component || !volume_transform)
    {
        result.error = "Irradiance probe-volume bake requires both volume and transform components";
        return result;
    }
    if (request.output_path.empty())
    {
        result.error = "Irradiance probe-volume bake requires an output .arcprobe path";
        return result;
    }
    if (!std::isfinite(component->probe_spacing) || component->probe_spacing <= 0.0f ||
        !std::isfinite(component->blend_distance) || component->blend_distance < 0.0f)
    {
        result.error = "Irradiance probe-volume bake settings are invalid";
        return result;
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
        if (!std::isfinite(component->bake_extents[axis]) || component->bake_extents[axis] <= 0.0f)
        {
            result.error = "Irradiance probe-volume extents must be finite and positive";
            return result;
        }

    const auto volume_world = volume_transform->dirty ? local_matrix(*volume_transform) : volume_transform->world;
    math::matrix4f world_to_volume{};
    if (!inverse_affine(volume_world, world_to_volume))
    {
        result.error = "Irradiance probe-volume transform is singular";
        return result;
    }

    render::irradiance_probe_volume_descriptor layout;
    const auto minimum = math::mul(component->bake_extents, -1.0f);
    layout.local_bounds = {geometric::point3f{minimum}, geometric::point3f{component->bake_extents}};
    layout.blend_distance = component->blend_distance;
    // Instance priority remains on the component; do not bake it into the asset or runtime resolution would apply it
    // twice.
    layout.priority = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const float extent = component->bake_extents[axis] * 2.0f;
        layout.dimensions[axis] =
            std::max(1u, static_cast<std::uint32_t>(std::floor(extent / component->probe_spacing)) + 1u);
        layout.spacing[axis] =
            layout.dimensions[axis] > 1u ? extent / static_cast<float>(layout.dimensions[axis] - 1u) : extent;
    }

    std::vector<render::irradiance_bake_triangle> triangles;
    world.view<transform_component, mesh_renderer_component>().each(
        [&](ecs::entity entity, const transform_component& transform, const mesh_renderer_component& mesh_renderer)
        {
            if (!bake_eligible(world, entity) || !mesh_renderer.visible || !mesh_renderer.affects_indirect_lighting ||
                !mesh_renderer.mesh.conventional.valid())
                return;
            const auto* mesh = renderer.mesh_data_for(mesh_renderer.mesh.conventional);
            if (!mesh || mesh->vertices.empty() || mesh->indices.size() < 3u)
            {
                result.diagnostics.emplace_back("Skipped a static renderer with unavailable CPU mesh data");
                return;
            }

            const auto object_world = transform.dirty ? local_matrix(transform) : transform.world;
            const auto object_to_volume = math::mul(world_to_volume, object_world);
            const auto* material = renderer.material_data_for(mesh_renderer.material);
            const auto diffuse = material_diffuse(material);
            const auto emissive = material_emissive(material);
            const bool two_sided = material && material->double_sided;

            std::uint32_t emitted{};
            for (std::size_t index = 0; index + 2u < mesh->indices.size(); index += 3u)
            {
                const auto i0 = mesh->indices[index];
                const auto i1 = mesh->indices[index + 1u];
                const auto i2 = mesh->indices[index + 2u];
                if (i0 >= mesh->vertices.size() || i1 >= mesh->vertices.size() || i2 >= mesh->vertices.size()) continue;
                const auto position = [&](std::uint32_t vertex)
                {
                    const auto& value = mesh->vertices[vertex];
                    return math::transform_point(
                        object_to_volume, math::vector3f{value.position[0], value.position[1], value.position[2]});
                };
                const auto a = position(i0);
                const auto b = position(i1);
                const auto c = position(i2);
                const auto cross = math::cross(math::sub(b, a), math::sub(c, a));
                if (math::length_squared(cross) <= 1.0e-10f) continue;
                triangles.push_back({.a = a,
                                     .b = b,
                                     .c = c,
                                     .normal = math::normalize(cross),
                                     .diffuse_albedo = diffuse,
                                     .emissive_radiance = emissive,
                                     .two_sided = two_sided});
                ++emitted;
            }
            if (emitted > 0u)
            {
                ++result.static_meshes;
                result.static_triangles += emitted;
            }
        });

    std::vector<render::irradiance_bake_directional_light> directional;
    world.view<transform_component, directional_light_component>().each(
        [&](ecs::entity entity, const transform_component& transform, const directional_light_component& light)
        {
            if (!bake_eligible(world, entity) || !light.enabled) return;
            auto color = light.color;
            if (light.use_color_temperature)
                color = math::mul(color, render::color_temperature_rgb(light.temperature_kelvin));
            const float intensity = render::light_intensity_scale(light.intensity_unit, light.intensity);
            const auto world_direction = world_forward_direction(transform);
            const auto local_direction = math::normalize(math::transform_vector(world_to_volume, world_direction));
            directional.push_back({.direction = local_direction,
                                   .radiance = math::mul(color, std::max(intensity, 0.0f)),
                                   .enabled = true});
            ++result.static_directional_lights;
        });

    auto options = request.options;
    options.samples_per_probe = component->bake_samples > 0u ? component->bake_samples : options.samples_per_probe;
    const auto baked =
        render::bake_irradiance_probe_volume(layout, triangles, directional, bake_environment(world), options);
    result.statistics = baked.statistics;
    result.diagnostics.insert(result.diagnostics.end(), baked.diagnostics.begin(), baked.diagnostics.end());
    if (baked.volume.probes.empty())
    {
        result.error = "Irradiance probe baker produced no probes";
        return result;
    }

    const auto bytes = render::encode_irradiance_probe_volume(baked.volume);
    if (bytes.empty())
    {
        result.error = "Irradiance probe baker produced an invalid volume";
        return result;
    }

    auto output = request.output_path;
    if (output.extension() != ".arcprobe") output.replace_extension(".arcprobe");
    if (output.is_relative()) output = asset_manager.config().project_root / output;
    std::error_code error;
    std::filesystem::create_directories(output.parent_path(), error);
    if (error)
    {
        result.error = "Could not create irradiance probe output directory: " + error.message();
        return result;
    }

    std::ofstream stream(output, std::ios::binary | std::ios::trunc);
    if (!stream)
    {
        result.error = "Could not open irradiance probe output asset for writing";
        return result;
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream)
    {
        result.error = "Could not write irradiance probe output asset";
        return result;
    }

    const auto scan = asset_manager.scan();
    if (!scan.succeeded())
    {
        result.error =
            scan.error.message.empty() ? "Asset scan failed after irradiance probe bake" : scan.error.message;
        return result;
    }

    auto relative = output.lexically_relative(asset_manager.config().project_root);
    const auto normalized = assets::normalize_asset_path(relative);
    component->asset = asset_manager.resolve(normalized, assets::asset_types::irradiance_probe_volume);
    if (!component->asset.guid.valid()) component->asset.path_hint = normalized;
    component->asset.expected_type = assets::asset_types::irradiance_probe_volume;

    const auto binding = refresh_irradiance_probe_volume_binding(*component, asset_manager);
    if (!binding.succeeded)
    {
        result.error = "Baked asset was written but could not be loaded: " + binding.message;
        result.asset = component->asset;
        return result;
    }

    result.asset = component->asset;
    result.succeeded = true;
    return result;
}

} // namespace arc::scene

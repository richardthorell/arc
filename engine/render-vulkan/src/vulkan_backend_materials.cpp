#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"
#include "vulkan_sky_constants.h"

namespace arc::render::vulkan::backend_detail
{
void vulkan_render_backend::update_light_buffer()
{
    if (light_buffer_.buffer == VK_NULL_HANDLE)
    {
        if (!create_buffer(sizeof(scene_lighting_data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                           light_buffer_))
        {
            arc::diagnostics::warn("render.vulkan", "Failed to allocate scene light buffer");
            return;
        }
    }

    void* mapped{};
    if (vmaMapMemory(allocator_, light_buffer_.allocation, &mapped) != VK_SUCCESS) return;
    std::memcpy(mapped, &frame_lighting_, sizeof(frame_lighting_));
    vmaFlushAllocation(allocator_, light_buffer_.allocation, 0, sizeof(frame_lighting_));
    vmaUnmapMemory(allocator_, light_buffer_.allocation);
}

void vulkan_render_backend::warn_about_skipped_lights(const scene_lighting_data& lighting)
{
    if (lighting.skipped_directional_count > 0)
        arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_directional_count) +
                                                    " directional light(s) over the v1 cap");
    if (lighting.skipped_point_count > 0)
        arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_point_count) +
                                                    " point light(s) over the v1 cap");
    if (lighting.skipped_spot_count > 0)
        arc::diagnostics::warn("render.vulkan", "Skipped " + std::to_string(lighting.skipped_spot_count) +
                                                    " spot light(s) over the v1 cap");
}

math::vector3f vulkan_render_backend::vector_sub(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

math::vector3f vulkan_render_backend::vector_mul(const math::vector3f& value, float scale) noexcept
{
    return {value[0] * scale, value[1] * scale, value[2] * scale};
}

math::vector3f vulkan_render_backend::vector_add(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

float vulkan_render_backend::vector_dot(const math::vector3f& lhs, const math::vector3f& rhs) noexcept
{
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

math::vector3f vulkan_render_backend::vector_normalize(const math::vector3f& value) noexcept
{
    const float length_sq = std::max(vector_dot(value, value), 0.000001f);
    const float inv_length = 1.0f / std::sqrt(length_sq);
    return vector_mul(value, inv_length);
}

vulkan_render_backend::folded_light_constants
vulkan_render_backend::fold_lighting_for_draw(const draw_mesh_event& draw) const noexcept
{
    const math::vector3f origin{draw.model(0, 3), draw.model(1, 3), draw.model(2, 3)};
    math::vector3f color{frame_lighting_.ambient_color_intensity[0] * frame_lighting_.ambient_color_intensity[3],
                         frame_lighting_.ambient_color_intensity[1] * frame_lighting_.ambient_color_intensity[3],
                         frame_lighting_.ambient_color_intensity[2] * frame_lighting_.ambient_color_intensity[3]};
    math::vector3f weighted_direction{};
    float total_weight{};

    for (std::uint32_t index = 0; index < frame_lighting_.directional_count; ++index)
    {
        const auto& light = frame_lighting_.directional_lights[index];
        const float contribution = std::max(light.direction_intensity[3], 0.0f);
        color = vector_add(
            color, vector_mul({light.color_flags[0], light.color_flags[1], light.color_flags[2]}, contribution));
        weighted_direction = vector_add(
            weighted_direction,
            vector_mul({light.direction_intensity[0], light.direction_intensity[1], light.direction_intensity[2]},
                       contribution));
        total_weight += contribution;
    }

    for (std::uint32_t index = 0; index < frame_lighting_.point_count; ++index)
    {
        const auto& light = frame_lighting_.point_lights[index];
        const math::vector3f position{light.position_range[0], light.position_range[1], light.position_range[2]};
        const float range = std::max(light.position_range[3], 0.001f);
        const math::vector3f to_light = vector_sub(position, origin);
        const float distance_sq = std::max(vector_dot(to_light, to_light), 0.000001f);
        const float attenuation = std::max(0.0f, 1.0f - std::sqrt(distance_sq) / range);
        const float contribution = light.color_intensity[3] * attenuation * attenuation;
        color =
            vector_add(color, vector_mul({light.color_intensity[0], light.color_intensity[1], light.color_intensity[2]},
                                         contribution));
        weighted_direction =
            vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
        total_weight += contribution;
    }

    for (std::uint32_t index = 0; index < frame_lighting_.spot_count; ++index)
    {
        const auto& light = frame_lighting_.spot_lights[index];
        const math::vector3f position{light.position_range[0], light.position_range[1], light.position_range[2]};
        const float range = std::max(light.position_range[3], 0.001f);
        const math::vector3f to_light = vector_sub(position, origin);
        const float attenuation = std::max(0.0f, 1.0f - std::sqrt(vector_dot(to_light, to_light)) / range);
        const math::vector3f light_forward = vector_normalize(
            {light.direction_inner_angle[0], light.direction_inner_angle[1], light.direction_inner_angle[2]});
        const float cone_cos = vector_dot(vector_mul(vector_normalize(to_light), -1.0f), light_forward);
        const float inner = std::cos(light.direction_inner_angle[3]);
        const float outer = std::cos(light.params[0]);
        const float cone = outer == inner ? 1.0f : std::clamp((cone_cos - outer) / (inner - outer), 0.0f, 1.0f);
        const float contribution = light.color_intensity[3] * attenuation * attenuation * cone;
        color =
            vector_add(color, vector_mul({light.color_intensity[0], light.color_intensity[1], light.color_intensity[2]},
                                         contribution));
        weighted_direction =
            vector_add(weighted_direction, vector_mul(vector_mul(vector_normalize(to_light), -1.0f), contribution));
        total_weight += contribution;
    }

    folded_light_constants folded;
    folded.color = color;
    folded.intensity = 1.0f;
    folded.direction = total_weight > 0.0001f ? vector_normalize(weighted_direction) : folded.direction;
    return folded;
}

material_alpha_mode vulkan_render_backend::material_alpha_mode_for(const draw_mesh_event& draw) const noexcept
{
    if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
        return material->second.data.alpha_mode;
    return material_alpha_mode::opaque;
}

bool vulkan_render_backend::texture_ready(texture_handle handle) const noexcept
{
    if (!handle.valid()) return false;
    const auto found = textures_.find(resource_key(handle));
    return found != textures_.end() && found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE;
}

bool vulkan_render_backend::material_is_terrain(const draw_mesh_event& draw) const noexcept
{
    const auto material = materials_.find(resource_key(draw.material));
    return material != materials_.end() && material->second.data.domain == material_domain::terrain;
}

bool vulkan_render_backend::material_requires_forward(const draw_mesh_event& draw) const noexcept
{
    const auto material = materials_.find(resource_key(draw.material));
    if (material == materials_.end()) return false;
    return material->second.data.render_path == material_render_path::clustered_forward;
}

mesh_push_constants vulkan_render_backend::build_mesh_constants(const draw_mesh_event& draw) const
{
    const math::matrix4f mvp = math::matmul(draw.view_projection, draw.model);
    mesh_push_constants constants{};
    std::copy(mvp.data(), mvp.data() + 16, constants.model_view_projection);
    std::copy(draw.model.data(), draw.model.data() + 16, constants.model);
    const auto folded_light = fold_lighting_for_draw(draw);
    constants.light_direction_intensity[0] = folded_light.direction[0];
    constants.light_direction_intensity[1] = folded_light.direction[1];
    constants.light_direction_intensity[2] = folded_light.direction[2];
    constants.light_direction_intensity[3] = folded_light.intensity;
    constants.light_color[0] = folded_light.color[0];
    constants.light_color[1] = folded_light.color[1];
    constants.light_color[2] = folded_light.color[2];
    constants.camera_position[0] = frame_camera_.position[0];
    constants.camera_position[1] = frame_camera_.position[1];
    constants.camera_position[2] = frame_camera_.position[2];
    constants.camera_position[3] =
        frame_environment_.enabled ? std::max(frame_environment_.atmosphere.exposure, 0.001f) : 1.0f;
    constants.fog_params[3] = draw.object_id.valid() ? static_cast<float>(draw.object_id.index + 1u) : 0.0f;

    if (frame_environment_.fog.enabled)
    {
        constants.fog_color_density[0] = frame_environment_.fog.color[0];
        constants.fog_color_density[1] = frame_environment_.fog.color[1];
        constants.fog_color_density[2] = frame_environment_.fog.color[2];
        constants.fog_color_density[3] = std::max(0.0f, frame_environment_.fog.density);
        constants.fog_params[0] = std::max(0.0f, frame_environment_.fog.start_distance);
        constants.fog_params[1] = std::max(0.0f, frame_environment_.fog.height_falloff);
        constants.fog_params[2] = std::clamp(frame_environment_.fog.max_opacity, 0.0f, 1.0f);
    }

    if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
    {
        const auto& desc = material->second.data;
        constants.base_color[0] = desc.base_color[0] * draw.base_color_tint[0];
        constants.base_color[1] = desc.base_color[1] * draw.base_color_tint[1];
        constants.base_color[2] = desc.base_color[2] * draw.base_color_tint[2];
        constants.base_color[3] = desc.base_color[3] * draw.base_color_tint[3];
        constants.visualization[1] = desc.metallic;
        constants.visualization[2] = desc.roughness;
        constants.visualization[3] = desc.alpha_cutoff;
        if (desc.domain == material_domain::terrain)
        {
            constants.base_color[0] = draw.base_color_tint[0];
            constants.base_color[1] = draw.base_color_tint[1];
            constants.base_color[2] = draw.base_color_tint[2];
            constants.base_color[3] = draw.base_color_tint[3];
            constants.material_params[0] = desc.terrain_layers[0].world_scale;
            constants.material_params[1] = desc.terrain_layers[1].world_scale;
            constants.material_params[2] = desc.terrain_layers[2].world_scale;
            constants.material_params[3] = desc.terrain_layers[3].world_scale;
            constants.light_color[3] = (texture_ready(desc.terrain_layers[0].base_color_texture) ? 1.0f : 0.0f) +
                                       (texture_ready(desc.terrain_layers[1].base_color_texture) ? 2.0f : 0.0f) +
                                       (texture_ready(desc.terrain_layers[2].base_color_texture) ? 4.0f : 0.0f) +
                                       (texture_ready(desc.terrain_layers[3].base_color_texture) ? 8.0f : 0.0f);
            constants.visualization[1] = (texture_ready(desc.terrain_layers[0].normal_texture) ? 1.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[1].normal_texture) ? 2.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[2].normal_texture) ? 4.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[3].normal_texture) ? 8.0f : 0.0f);
            constants.visualization[2] = (texture_ready(desc.terrain_layers[0].packed_surface_texture) ? 1.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[1].packed_surface_texture) ? 2.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[2].packed_surface_texture) ? 4.0f : 0.0f) +
                                         (texture_ready(desc.terrain_layers[3].packed_surface_texture) ? 8.0f : 0.0f);
        }
        else
        {
            constants.material_params[0] = desc.normal_scale;
            constants.material_params[1] = desc.occlusion_strength;
            constants.material_params[2] = desc.emissive_strength;
            constants.material_params[3] = static_cast<float>(desc.alpha_mode);
            constants.light_color[3] = (texture_ready(desc.base_color_texture) ? 1.0f : 0.0f) +
                                       (texture_ready(desc.metallic_roughness_texture) ? 2.0f : 0.0f) +
                                       (texture_ready(desc.normal_texture) ? 4.0f : 0.0f) +
                                       (texture_ready(desc.occlusion_texture) ? 8.0f : 0.0f) +
                                       (texture_ready(desc.emissive_texture) ? 16.0f : 0.0f);
        }
    }
    else
    {
        constants.base_color[0] = draw.base_color_tint[0];
        constants.base_color[1] = draw.base_color_tint[1];
        constants.base_color[2] = draw.base_color_tint[2];
        constants.base_color[3] = draw.base_color_tint[3];
    }
    constants.visualization[0] = static_cast<float>(draw.visualization);
    return constants;
}

VkDescriptorSet vulkan_render_backend::material_descriptor_set_for(const draw_mesh_event& draw) const noexcept
{
    if (const auto material = materials_.find(resource_key(draw.material)); material != materials_.end())
    {
        const auto slot = current_frame_slot();
        if (slot < material->second.descriptor_sets.size() && material->second.descriptor_sets[slot] != VK_NULL_HANDLE)
            return material->second.descriptor_sets[slot];
    }
    const auto slot = current_frame_slot();
    return slot < white_descriptor_sets_.size() ? white_descriptor_sets_[slot] : VK_NULL_HANDLE;
}

VkDescriptorSet vulkan_render_backend::material_attribute_descriptor_set_for(texture_handle handle)
{
    if (!handle.valid() || material_attribute_descriptor_set_layout_ == VK_NULL_HANDLE ||
        material_attribute_descriptor_pool_ == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    const auto texture = textures_.find(resource_key(handle));
    if (texture == textures_.end() || texture->second.view == VK_NULL_HANDLE ||
        texture->second.sampler == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    const auto key = resource_key(handle);
    auto descriptor = material_attribute_descriptor_sets_.find(key);
    if (descriptor == material_attribute_descriptor_sets_.end())
    {
        VkDescriptorSet set{};
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = material_attribute_descriptor_pool_;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &material_attribute_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
        descriptor = material_attribute_descriptor_sets_.emplace(key, set).first;
    }

    const VkDescriptorImageInfo image{texture->second.sampler, texture->second.view,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptor->second;
    write.dstBinding = 0u;
    write.descriptorCount = 1u;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(device_, 1u, &write, 0u, nullptr);
    return descriptor->second;
}

bool vulkan_render_backend::draw_runtime_material_gbuffer(VkCommandBuffer command_buffer, const draw_mesh_event& draw)
{
    const auto found = materials_.find(resource_key(draw.material));
    if (found == materials_.end() || !found->second.data.runtime_program) return false;
    auto& material = found->second;
    if (!ensure_runtime_gbuffer_pipeline(material) || !update_runtime_material_buffers(material)) return false;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.runtime.gbuffer_pipeline);
    if (material.runtime.descriptor_set_layout != VK_NULL_HANDLE)
    {
        const auto slot = current_frame_slot();
        if (slot >= material.runtime.descriptor_sets.size()) return false;
        const auto descriptor_set = material.runtime.descriptor_sets[slot];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.runtime.pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    }
    draw_indexed_mesh(command_buffer, draw, material.runtime.pipeline_layout,
                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
    return true;
}

bool vulkan_render_backend::draw_runtime_material_gbuffer(VkCommandBuffer command_buffer,
                                                          const virtual_cluster_draw& draw)
{
    const auto found = materials_.find(resource_key(draw.draw.material));
    if (found == materials_.end() || !found->second.data.runtime_program) return false;
    auto& material = found->second;
    if (!ensure_runtime_gbuffer_pipeline(material) || !update_runtime_material_buffers(material)) return false;

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.runtime.gbuffer_pipeline);
    if (material.runtime.descriptor_set_layout != VK_NULL_HANDLE)
    {
        const auto slot = current_frame_slot();
        if (slot >= material.runtime.descriptor_sets.size()) return false;
        const auto descriptor_set = material.runtime.descriptor_sets[slot];
        vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, material.runtime.pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    }
    draw_indexed_virtual_cluster(command_buffer, draw, material.runtime.pipeline_layout,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, true, true);
    return true;
}

} // namespace arc::render::vulkan::backend_detail

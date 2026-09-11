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

void vulkan_render_backend::destroy_mesh_pipeline() noexcept
{
    if (water_surface_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, water_surface_pipeline_, nullptr);
        water_surface_pipeline_ = VK_NULL_HANDLE;
    }
    if (water_surface_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, water_surface_pipeline_layout_, nullptr);
        water_surface_pipeline_layout_ = VK_NULL_HANDLE;
    }
    const auto destroy_debug_pipeline = [&](VkPipeline& pipeline)
    {
        if (pipeline == VK_NULL_HANDLE) return;
        vkDestroyPipeline(device_, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    };
    destroy_debug_pipeline(debug_overlay_line_pipeline_);
    destroy_debug_pipeline(debug_overlay_triangle_pipeline_);
    destroy_debug_pipeline(debug_overlay_output_line_pipeline_);
    destroy_debug_pipeline(debug_overlay_output_triangle_pipeline_);
    if (debug_overlay_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
        debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (deferred_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, deferred_pipeline_, nullptr);
        deferred_pipeline_ = VK_NULL_HANDLE;
    }
    if (deferred_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, deferred_pipeline_layout_, nullptr);
        deferred_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (output_transform_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, output_transform_pipeline_, nullptr);
        output_transform_pipeline_ = VK_NULL_HANDLE;
    }
    if (output_transform_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, output_transform_pipeline_layout_, nullptr);
        output_transform_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (luminance_histogram_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, luminance_histogram_pipeline_, nullptr);
        luminance_histogram_pipeline_ = VK_NULL_HANDLE;
    }
    if (luminance_histogram_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, luminance_histogram_pipeline_layout_, nullptr);
        luminance_histogram_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (exposure_resolve_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, exposure_resolve_pipeline_, nullptr);
        exposure_resolve_pipeline_ = VK_NULL_HANDLE;
    }
    if (exposure_resolve_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, exposure_resolve_pipeline_layout_, nullptr);
        exposure_resolve_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (output_transform_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, output_transform_descriptor_pool_, nullptr);
        output_transform_descriptor_pool_ = VK_NULL_HANDLE;
        output_transform_descriptor_set_ = VK_NULL_HANDLE;
    }
    if (output_transform_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, output_transform_descriptor_set_layout_, nullptr);
        output_transform_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (gbuffer_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, gbuffer_pipeline_, nullptr);
        gbuffer_pipeline_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_gbuffer_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, terrain_surface_gbuffer_pipeline_, nullptr);
        terrain_surface_gbuffer_pipeline_ = VK_NULL_HANDLE;
    }
    if (gbuffer_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, gbuffer_descriptor_pool_, nullptr);
        gbuffer_descriptor_pool_ = VK_NULL_HANDLE;
        gbuffer_descriptor_set_ = VK_NULL_HANDLE;
    }
    if (gbuffer_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, gbuffer_sampler_, nullptr);
        gbuffer_sampler_ = VK_NULL_HANDLE;
    }
    if (gbuffer_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, gbuffer_descriptor_set_layout_, nullptr);
        gbuffer_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (shadow_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, shadow_pipeline_, nullptr);
        shadow_pipeline_ = VK_NULL_HANDLE;
    }
    if (shadow_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, shadow_pipeline_layout_, nullptr);
        shadow_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (mesh_wire_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_wire_pipeline_, nullptr);
        mesh_wire_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_transparent_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_transparent_pipeline_, nullptr);
        mesh_transparent_pipeline_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, terrain_surface_pipeline_, nullptr);
        terrain_surface_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, mesh_pipeline_, nullptr);
        mesh_pipeline_ = VK_NULL_HANDLE;
    }
    if (mesh_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, mesh_pipeline_layout_, nullptr);
        mesh_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (terrain_surface_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, terrain_surface_pipeline_layout_, nullptr);
        terrain_surface_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (material_attribute_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, material_attribute_descriptor_pool_, nullptr);
        material_attribute_descriptor_pool_ = VK_NULL_HANDLE;
        material_attribute_descriptor_sets_.clear();
    }
    if (material_attribute_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, material_attribute_descriptor_set_layout_, nullptr);
        material_attribute_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (sky_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, sky_pipeline_, nullptr);
        sky_pipeline_ = VK_NULL_HANDLE;
    }
    if (sky_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, sky_pipeline_layout_, nullptr);
        sky_pipeline_layout_ = VK_NULL_HANDLE;
    }
}

void vulkan_render_backend::destroy_white_texture() noexcept
{
    for (auto& parameters : white_material_parameter_buffers_)
        destroy_buffer(parameters);
    white_material_parameter_buffers_.clear();
    if (white_descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_, white_descriptor_pool_, nullptr);
        white_descriptor_pool_ = VK_NULL_HANDLE;
        white_descriptor_sets_.clear();
        sky_descriptor_sets_.clear();
        for (auto& [_, material] : materials_)
            material.descriptor_sets.clear();
    }
    if (white_descriptor_set_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_, white_descriptor_set_layout_, nullptr);
        white_descriptor_set_layout_ = VK_NULL_HANDLE;
    }
    if (white_sampler_ != VK_NULL_HANDLE)
    {
        vkDestroySampler(device_, white_sampler_, nullptr);
        white_sampler_ = VK_NULL_HANDLE;
    }
    if (white_view_ != VK_NULL_HANDLE)
    {
        vkDestroyImageView(device_, white_view_, nullptr);
        white_view_ = VK_NULL_HANDLE;
    }
    if (white_image_ != VK_NULL_HANDLE)
    {
        vmaDestroyImage(allocator_, white_image_, white_allocation_);
        white_image_ = VK_NULL_HANDLE;
        white_allocation_ = VK_NULL_HANDLE;
    }
}

VkShaderModule vulkan_render_backend::create_shader_module(const std::uint32_t* code, std::size_t word_count)
{
    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = word_count * sizeof(std::uint32_t);
    info.pCode = code;

    VkShaderModule module{};
    if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS) return VK_NULL_HANDLE;
    return module;
}

VkShaderModule vulkan_render_backend::create_shader_module(const std::vector<std::uint8_t>& bytecode)
{
    if (bytecode.empty() || bytecode.size() % sizeof(std::uint32_t) != 0) return VK_NULL_HANDLE;
    std::vector<std::uint32_t> words(bytecode.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), bytecode.data(), bytecode.size());
    return create_shader_module(words.data(), words.size());
}

void vulkan_render_backend::destroy_material_runtime(gpu_material_runtime& runtime) noexcept
{
    if (runtime.gbuffer_pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device_, runtime.gbuffer_pipeline, nullptr);
    if (runtime.pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device_, runtime.pipeline_layout, nullptr);
    if (runtime.descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, runtime.descriptor_pool, nullptr);
    if (runtime.descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, runtime.descriptor_set_layout, nullptr);
    for (auto& buffer : runtime.parameter_buffers)
        destroy_buffer(buffer);
    for (auto& buffer : runtime.frame_buffers)
        destroy_buffer(buffer);
    runtime = {};
}

bool vulkan_render_backend::reject_runtime_material(gpu_material& material, std::string reason)
{
    const auto generation = material.data.runtime_program ? material.data.runtime_program->generation : 0u;
    destroy_material_runtime(material.runtime);
    material.runtime.generation = generation;
    material.runtime.failed = true;
    arc::diagnostics::warn("render.vulkan", "Compiled Material ABI G-buffer fallback for '" + material.data.name +
                                                "': " + std::move(reason));
    return false;
}

const material_runtime_pass* vulkan_render_backend::runtime_gbuffer_pass(const gpu_material& material) const noexcept
{
    if (!material.data.runtime_program) return nullptr;
    const auto& program = *material.data.runtime_program;
    if (program.contract_version != material_pass_contract_version || program.material_abi != material_abi_version)
        return nullptr;
    const auto found = std::ranges::find(program.passes, material_pass::gbuffer, &material_runtime_pass::pass);
    return found == program.passes.end() ? nullptr : &*found;
}

bool vulkan_render_backend::update_runtime_parameter_buffer(gpu_buffer& buffer, const material_descriptor& material,
                                                            const material_runtime_program& program)
{
    const auto byte_size = std::max<std::size_t>(program.parameter_block_size, 16u);
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(byte_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
        return false;

    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memset(mapped, 0, byte_size);
    if (!program.parameter_defaults.empty())
        std::memcpy(mapped, program.parameter_defaults.data(), std::min(byte_size, program.parameter_defaults.size()));

    const auto copy_override = [&](const shader_parameter_descriptor& parameter, const material_parameter_value& value)
    {
        if (parameter.offset >= byte_size) return;
        const auto destination = static_cast<std::byte*>(mapped) + parameter.offset;
        const auto available = std::min<std::size_t>(parameter.size, byte_size - parameter.offset);
        const auto copy = [&](const void* source, std::size_t size)
        { std::memcpy(destination, source, std::min(size, available)); };
        switch (parameter.type)
        {
            case shader_parameter_type::boolean:
                if (const auto* typed = std::get_if<bool>(&value))
                {
                    const std::uint32_t packed = *typed ? 1u : 0u;
                    copy(&packed, sizeof(packed));
                }
                break;
            case shader_parameter_type::int32:
                if (const auto* typed = std::get_if<std::int32_t>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::uint32:
                if (const auto* typed = std::get_if<std::uint32_t>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::float32:
                if (const auto* typed = std::get_if<float>(&value)) copy(typed, sizeof(*typed));
                break;
            case shader_parameter_type::float2:
                if (const auto* typed = std::get_if<math::vector2f>(&value)) copy(typed->data(), sizeof(float) * 2u);
                break;
            case shader_parameter_type::float3:
                if (const auto* typed = std::get_if<math::vector3f>(&value)) copy(typed->data(), sizeof(float) * 3u);
                break;
            case shader_parameter_type::float4:
                if (const auto* typed = std::get_if<math::vector4f>(&value)) copy(typed->data(), sizeof(float) * 4u);
                break;
            case shader_parameter_type::matrix4x4:
                if (const auto* typed = std::get_if<math::matrix4x4f>(&value)) copy(typed->data(), sizeof(float) * 16u);
                break;
            case shader_parameter_type::texture_2d:
            case shader_parameter_type::texture_cube:
            case shader_parameter_type::sampler:
                break;
        }
    };
    for (const auto& override : material.parameters)
    {
        const auto parameter = std::ranges::find(program.parameters, override.id, &shader_parameter_descriptor::id);
        if (parameter != program.parameters.end()) copy_override(*parameter, override.value);
    }

    vmaFlushAllocation(allocator_, buffer.allocation, 0, byte_size);
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::update_runtime_frame_buffer(gpu_buffer& buffer)
{
    constexpr VkDeviceSize byte_size = sizeof(float) * 4u;
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(byte_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, buffer))
        return false;
    const std::array<float, 4> frame{static_cast<float>(last_profile_.frame_index) / 60.0f, 0.0f, 0.0f, 0.0f};
    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, frame.data(), sizeof(frame));
    vmaFlushAllocation(allocator_, buffer.allocation, 0, sizeof(frame));
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::update_runtime_material_buffers(gpu_material& material)
{
    if (!material.data.runtime_program) return false;
    const auto slot = current_frame_slot();
    if (!material.runtime.parameter_buffers.empty())
    {
        if (slot >= material.runtime.parameter_buffers.size() ||
            !update_runtime_parameter_buffer(material.runtime.parameter_buffers[slot], material.data,
                                             *material.data.runtime_program))
            return false;
    }
    if (!material.runtime.frame_buffers.empty())
    {
        if (slot >= material.runtime.frame_buffers.size() ||
            !update_runtime_frame_buffer(material.runtime.frame_buffers[slot]))
            return false;
    }
    if (material.data.runtime_program->uses_texture_sampling && !update_runtime_texture_descriptors(material, slot))
        return false;
    return true;
}

texture_handle vulkan_render_backend::runtime_texture_handle(const gpu_material& material,
                                                             std::uint32_t slot) const noexcept
{
    texture_handle handle{};
    if (slot < material.data.runtime_textures.size()) handle = material.data.runtime_textures[slot];
    if (!material.data.runtime_program) return handle;
    const auto binding = std::ranges::find(material.data.runtime_program->texture_bindings, slot,
                                           &material_runtime_texture_binding::slot);
    if (binding == material.data.runtime_program->texture_bindings.end() ||
        binding->parameter_id.representation() == 0u)
        return handle;
    const auto override =
        std::ranges::find(material.data.parameters, binding->parameter_id, &material_parameter_override::id);
    if (override == material.data.parameters.end()) return handle;
    if (const auto* resource = std::get_if<resource_handle>(&override->value)) return *resource;
    return handle;
}

VkDescriptorType vulkan_render_backend::runtime_descriptor_type(shader_resource_kind kind) const noexcept
{
    switch (kind)
    {
        case shader_resource_kind::constant_buffer:
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        case shader_resource_kind::sampled_texture:
            return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        case shader_resource_kind::sampler:
            return VK_DESCRIPTOR_TYPE_SAMPLER;
        default:
            return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

bool vulkan_render_backend::update_runtime_texture_descriptors(gpu_material& material, std::uint32_t frame_slot)
{
    if (!material.data.runtime_program || frame_slot >= material.runtime.descriptor_sets.size()) return false;
    const auto* pass = runtime_gbuffer_pass(material);
    if (pass == nullptr) return false;

    const auto& resources = pass->compiled.reflection.resources;
    std::vector<std::vector<VkDescriptorImageInfo>> image_infos(resources.size());
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(resources.size());

    VkSampler graph_sampler = white_sampler_;
    for (std::uint32_t texture_slot = 0;
         texture_slot < static_cast<std::uint32_t>(material.data.runtime_program->texture_bindings.size());
         ++texture_slot)
    {
        const auto handle = runtime_texture_handle(material, texture_slot);
        if (!handle.valid()) continue;
        const auto found = textures_.find(resource_key(handle));
        if (found != textures_.end() && found->second.sampler != VK_NULL_HANDLE)
        {
            graph_sampler = found->second.sampler;
            break;
        }
    }

    for (std::size_t resource_index = 0; resource_index < resources.size(); ++resource_index)
    {
        const auto& resource = resources[resource_index];
        if (resource.kind != shader_resource_kind::sampled_texture && resource.kind != shader_resource_kind::sampler)
            continue;

        auto& infos = image_infos[resource_index];
        infos.resize(resource.kind == shader_resource_kind::sampled_texture ? resource.count : 1u);
        if (resource.kind == shader_resource_kind::sampled_texture)
        {
            for (std::uint32_t texture_slot = 0; texture_slot < resource.count; ++texture_slot)
            {
                VkImageView view = white_view_;
                const auto handle = runtime_texture_handle(material, texture_slot);
                if (handle.valid())
                {
                    const auto found = textures_.find(resource_key(handle));
                    if (found != textures_.end() && found->second.view != VK_NULL_HANDLE) view = found->second.view;
                }
                infos[texture_slot] = {VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            }
        }
        else
            infos[0] = {graph_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = material.runtime.descriptor_sets[frame_slot];
        write.dstBinding = resource.binding;
        write.descriptorCount = static_cast<std::uint32_t>(infos.size());
        write.descriptorType = runtime_descriptor_type(resource.kind);
        write.pImageInfo = infos.data();
        writes.push_back(write);
    }
    if (!writes.empty())
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    return true;
}

bool vulkan_render_backend::create_runtime_material_descriptors(gpu_material& material,
                                                                const material_runtime_pass& pass)
{
    const auto& reflection = pass.compiled.reflection;
    std::vector<const shader_resource_descriptor*> resources;
    resources.reserve(reflection.resources.size());
    for (const auto& resource : reflection.resources)
    {
        if (resource.set != 0u)
            return reject_runtime_material(material, "compiled preview resources must currently use descriptor set 0");
        const auto descriptor_type = runtime_descriptor_type(resource.kind);
        if (descriptor_type == VK_DESCRIPTOR_TYPE_MAX_ENUM)
            return reject_runtime_material(material,
                                           "unsupported reflected Material ABI resource '" + resource.name + "'");
        const bool supported_name =
            (resource.kind == shader_resource_kind::constant_buffer &&
             (resource.name == "arcMaterialParameters" || resource.name == "arcFrame")) ||
            (resource.kind == shader_resource_kind::sampled_texture && resource.name == "arcMaterialTextures") ||
            (resource.kind == shader_resource_kind::sampler && resource.name == "arcMaterialSampler");
        if (!supported_name)
            return reject_runtime_material(material,
                                           "unsupported reflected Material ABI resource '" + resource.name + "'");
        resources.push_back(&resource);
    }
    if (resources.empty()) return true;

    std::ranges::sort(resources, {}, [](const shader_resource_descriptor* resource) { return resource->binding; });
    for (std::size_t index = 1; index < resources.size(); ++index)
        if (resources[index - 1]->binding == resources[index]->binding)
            return reject_runtime_material(material,
                                           "compiled Material ABI reflection contains duplicate descriptor bindings");

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(resources.size());
    for (const auto* resource : resources)
        bindings.push_back({resource->binding, runtime_descriptor_type(resource->kind), resource->count,
                            VK_SHADER_STAGE_FRAGMENT_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &material.runtime.descriptor_set_layout) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create reflected Material ABI descriptor layout");

    const auto frame_count = frame_resource_count();
    std::array<std::uint32_t, 3> descriptor_counts{};
    for (const auto* resource : resources)
    {
        const auto count = resource->count * frame_count;
        switch (runtime_descriptor_type(resource->kind))
        {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                descriptor_counts[0] += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                descriptor_counts[1] += count;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                descriptor_counts[2] += count;
                break;
            default:
                break;
        }
    }
    std::vector<VkDescriptorPoolSize> pool_sizes;
    if (descriptor_counts[0]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_counts[0]});
    if (descriptor_counts[1]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, descriptor_counts[1]});
    if (descriptor_counts[2]) pool_sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, descriptor_counts[2]});
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = frame_count;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &material.runtime.descriptor_pool) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create reflected Material ABI descriptor pool");

    material.runtime.descriptor_sets.resize(frame_count);
    std::vector<VkDescriptorSetLayout> layouts(frame_count, material.runtime.descriptor_set_layout);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = material.runtime.descriptor_pool;
    allocate.descriptorSetCount = frame_count;
    allocate.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device_, &allocate, material.runtime.descriptor_sets.data()) != VK_SUCCESS)
        return reject_runtime_material(material, "failed to allocate reflected Material ABI descriptor sets");

    const bool needs_parameters =
        std::ranges::any_of(resources, [](const auto* resource) { return resource->name == "arcMaterialParameters"; });
    const bool needs_frame =
        std::ranges::any_of(resources, [](const auto* resource) { return resource->name == "arcFrame"; });
    if (needs_parameters) material.runtime.parameter_buffers.resize(frame_count);
    if (needs_frame) material.runtime.frame_buffers.resize(frame_count);

    for (std::uint32_t slot = 0; slot < frame_count; ++slot)
    {
        if (needs_parameters && !update_runtime_parameter_buffer(material.runtime.parameter_buffers[slot],
                                                                 material.data, *material.data.runtime_program))
            return reject_runtime_material(material, "failed to allocate Material ABI parameter buffer");
        if (needs_frame && !update_runtime_frame_buffer(material.runtime.frame_buffers[slot]))
            return reject_runtime_material(material, "failed to allocate Material ABI frame buffer");

        std::vector<VkDescriptorBufferInfo> infos;
        std::vector<VkWriteDescriptorSet> writes;
        infos.reserve(resources.size());
        writes.reserve(resources.size());
        for (const auto* resource : resources)
        {
            if (resource->kind != shader_resource_kind::constant_buffer) continue;
            const bool parameters = resource->name == "arcMaterialParameters";
            const auto& buffer =
                parameters ? material.runtime.parameter_buffers[slot] : material.runtime.frame_buffers[slot];
            const auto range = parameters ? static_cast<VkDeviceSize>(std::max<std::size_t>(
                                                material.data.runtime_program->parameter_block_size, 16u))
                                          : static_cast<VkDeviceSize>(sizeof(float) * 4u);
            infos.push_back({buffer.buffer, 0u, range});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = material.runtime.descriptor_sets[slot];
            write.dstBinding = resource->binding;
            write.descriptorCount = 1u;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &infos.back();
            writes.push_back(write);
        }
        if (!writes.empty())
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        if (!update_runtime_texture_descriptors(material, slot))
            return reject_runtime_material(material, "failed to update Material ABI texture descriptors");
    }
    return true;
}

bool vulkan_render_backend::create_runtime_gbuffer_pipeline(gpu_material& material, const material_runtime_pass& pass)
{
    VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
    VkShaderModule frag = create_shader_module(pass.compiled.bytecode);
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        return reject_runtime_material(material, "failed to create compiled Material ABI shader module");
    }

    VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                             sizeof(mesh_push_constants)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    if (material.runtime.descriptor_set_layout != VK_NULL_HANDLE)
    {
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &material.runtime.descriptor_set_layout;
    }
    layout.pushConstantRangeCount = 1u;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &material.runtime.pipeline_layout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return reject_runtime_material(material, "failed to create compiled Material ABI pipeline layout");
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{0u, sizeof(mesh_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 5> attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)},
        VkVertexInputAttributeDescription{2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)},
        VkVertexInputAttributeDescription{3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)},
        VkVertexInputAttributeDescription{4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    std::array<VkPipelineColorBlendAttachmentState, 6> attachments{};
    for (auto& attachment : attachments)
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    color_blend.pAttachments = attachments.data();
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    const std::array<VkFormat, 6> color_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    rendering.depthAttachmentFormat = depth_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = material.runtime.pipeline_layout;
    const auto result = vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                  &material.runtime.gbuffer_pipeline);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        return reject_runtime_material(material, "failed to create compiled Material ABI G-buffer pipeline: " +
                                                     describe_vk_result(result));
    return true;
}

bool vulkan_render_backend::ensure_runtime_gbuffer_pipeline(gpu_material& material)
{
    const auto* program = material.data.runtime_program.get();
    if (program == nullptr) return false;
    if (material.runtime.failed && material.runtime.generation == program->generation) return false;
    if (material.runtime.gbuffer_pipeline != VK_NULL_HANDLE && material.runtime.generation == program->generation &&
        (material.runtime.descriptor_set_layout == VK_NULL_HANDLE ||
         material.runtime.descriptor_sets.size() == frame_resource_count()))
        return true;

    if (material.runtime.gbuffer_pipeline != VK_NULL_HANDLE || material.runtime.pipeline_layout != VK_NULL_HANDLE ||
        material.runtime.descriptor_pool != VK_NULL_HANDLE ||
        material.runtime.descriptor_set_layout != VK_NULL_HANDLE || !material.runtime.parameter_buffers.empty() ||
        !material.runtime.frame_buffers.empty())
    {
        wait_for_in_flight_frames();
        destroy_material_runtime(material.runtime);
    }
    material.runtime.generation = program->generation;

    if (program->contract_version != 1u || program->material_abi != 1u)
        return reject_runtime_material(material, "unsupported compiled Material ABI contract version");
    if (material.data.alpha_mode != material_alpha_mode::opaque)
        return reject_runtime_material(material, "compiled preview execution currently requires an opaque material");
    const auto* pass = runtime_gbuffer_pass(material);
    if (pass == nullptr || pass->compiled.bytecode.empty())
        return reject_runtime_material(material, "compiled material does not provide an executable G-buffer pass");
    if (!create_runtime_material_descriptors(material, *pass)) return false;
    if (!create_runtime_gbuffer_pipeline(material, *pass)) return false;
    arc::diagnostics::debug("render.vulkan",
                            "Using compiled Material ABI G-buffer pass for '" + material.data.name + "'");
    return true;
}

material_uniform_data
vulkan_render_backend::build_material_parameters(const material_descriptor* material) const noexcept
{
    material_uniform_data parameters{};
    if (material == nullptr || material->domain == material_domain::terrain) return parameters;

    parameters.emissive_factor[0] = material->emissive_factor[0];
    parameters.emissive_factor[1] = material->emissive_factor[1];
    parameters.emissive_factor[2] = material->emissive_factor[2];
    parameters.emissive_factor[3] = material->emissive_luminance_nits > 0.0f
                                        ? material->emissive_luminance_nits / 100.0f
                                        : material->emissive_strength;
    parameters.material_lobes[0] = material->clear_coat_factor;
    parameters.material_lobes[1] = material->clear_coat_roughness;
    parameters.material_lobes[2] = material->anisotropy_factor;
    parameters.material_lobes[3] = material->transmission_factor;
    parameters.volume_params[0] = static_cast<float>(material->shading_model);
    parameters.volume_params[1] = material->index_of_refraction;
    parameters.volume_params[2] = material->thickness_factor;
    parameters.volume_params[3] = material->attenuation_distance;
    parameters.subsurface_color_factor[0] = material->subsurface_color[0];
    parameters.subsurface_color_factor[1] = material->subsurface_color[1];
    parameters.subsurface_color_factor[2] = material->subsurface_color[2];
    parameters.subsurface_color_factor[3] = material->subsurface_factor;
    parameters.attenuation_color[0] = material->attenuation_color[0];
    parameters.attenuation_color[1] = material->attenuation_color[1];
    parameters.attenuation_color[2] = material->attenuation_color[2];
    parameters.attenuation_color[3] = (texture_ready(material->clear_coat_texture) ? 1.0f : 0.0f) +
                                      (texture_ready(material->clear_coat_roughness_texture) ? 2.0f : 0.0f) +
                                      (texture_ready(material->clear_coat_normal_texture) ? 4.0f : 0.0f) +
                                      (texture_ready(material->anisotropy_texture) ? 8.0f : 0.0f) +
                                      (texture_ready(material->subsurface_texture) ? 16.0f : 0.0f) +
                                      (texture_ready(material->thickness_texture) ? 32.0f : 0.0f) +
                                      (texture_ready(material->transmission_texture) ? 64.0f : 0.0f);
    return parameters;
}

bool vulkan_render_backend::update_material_parameter_buffer(gpu_buffer& buffer, const material_descriptor* material)
{
    if (buffer.buffer == VK_NULL_HANDLE &&
        !create_buffer(sizeof(material_uniform_data), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU,
                       buffer))
        return false;

    const auto parameters = build_material_parameters(material);
    void* mapped{};
    if (vmaMapMemory(allocator_, buffer.allocation, &mapped) != VK_SUCCESS) return false;
    std::memcpy(mapped, &parameters, sizeof(parameters));
    vmaFlushAllocation(allocator_, buffer.allocation, 0, sizeof(parameters));
    vmaUnmapMemory(allocator_, buffer.allocation);
    return true;
}

bool vulkan_render_backend::ensure_material_parameter_buffers(std::vector<gpu_buffer>& buffers,
                                                              const material_descriptor* material)
{
    const auto count = frame_resource_count();
    if (buffers.size() != count)
    {
        for (auto& buffer : buffers)
            destroy_buffer(buffer);
        buffers.assign(count, {});
    }
    for (auto& buffer : buffers)
    {
        if (buffer.buffer == VK_NULL_HANDLE && !update_material_parameter_buffer(buffer, material)) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_material_descriptor_sets(gpu_material& material)
{
    if (!ensure_material_parameter_buffers(material.parameter_buffers, &material.data)) return false;
    const auto count = frame_resource_count();
    if (material.descriptor_sets.size() != count) material.descriptor_sets.assign(count, VK_NULL_HANDLE);
    for (auto& set : material.descriptor_sets)
    {
        if (set != VK_NULL_HANDLE) continue;
        set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_white_descriptor_sets()
{
    if (!ensure_material_parameter_buffers(white_material_parameter_buffers_, nullptr)) return false;
    const auto count = frame_resource_count();
    if (white_descriptor_sets_.size() != count) white_descriptor_sets_.assign(count, VK_NULL_HANDLE);
    for (auto& set : white_descriptor_sets_)
    {
        if (set != VK_NULL_HANDLE) continue;
        set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

bool vulkan_render_backend::ensure_sky_descriptor_sets()
{
    if (!ensure_white_texture()) return false;
    const auto count = frame_resource_count();
    if (sky_descriptor_sets_.size() != count) sky_descriptor_sets_.assign(count, VK_NULL_HANDLE);
    for (auto& set : sky_descriptor_sets_)
    {
        if (set == VK_NULL_HANDLE) set = allocate_material_descriptor_set();
        if (set == VK_NULL_HANDLE) return false;
    }
    return true;
}

VkDescriptorSet vulkan_render_backend::update_current_sky_descriptor_set()
{
    if (!ensure_sky_descriptor_sets()) return VK_NULL_HANDLE;
    const auto slot = current_frame_slot();
    if (slot >= sky_descriptor_sets_.size()) return VK_NULL_HANDLE;
    const auto set = sky_descriptor_sets_[slot];
    update_material_descriptor_set(
        set, nullptr,
        slot < white_material_parameter_buffers_.size() ? &white_material_parameter_buffers_[slot] : nullptr, slot);

    VkSampler sampler = white_sampler_;
    VkImageView view = white_view_;
    if (frame_environment_.source == sky_source_mode::hdri && frame_environment_.hdri_texture.valid())
    {
        const auto found = textures_.find(resource_key(frame_environment_.hdri_texture));
        if (found != textures_.end() && found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
        {
            sampler = found->second.sampler;
            view = found->second.view;
        }
        else
        {
            frame_environment_.fallback_reason = "HDRI texture is unavailable; using the visible fallback color";
        }
    }
    VkDescriptorImageInfo image{};
    image.sampler = sampler;
    image.imageView = view;
    image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    return set;
}

void vulkan_render_backend::update_material_descriptor_set(VkDescriptorSet descriptor_set,
                                                           const material_descriptor* material,
                                                           const gpu_buffer* material_parameters,
                                                           std::uint32_t frame_slot)
{
    const auto* shadow_buffer_resource = shadow_uniform_buffer_for_slot(frame_slot);
    if (descriptor_set == VK_NULL_HANDLE || white_view_ == VK_NULL_HANDLE ||
        shadow_atlas_.array_view == VK_NULL_HANDLE || local_shadow_atlas_.view == VK_NULL_HANDLE ||
        shadow_buffer_resource == nullptr || shadow_buffer_resource->buffer == VK_NULL_HANDLE ||
        material_parameters == nullptr || material_parameters->buffer == VK_NULL_HANDLE ||
        light_buffer_.buffer == VK_NULL_HANDLE)
        return;

    const auto resolve_texture = [&](texture_handle handle, VkSampler& sampler, VkImageView& view,
                                     std::optional<texture_semantic> expected = std::nullopt)
    {
        sampler = white_sampler_;
        view = white_view_;
        if (!handle.valid()) return;
        if (const auto found = textures_.find(resource_key(handle)); found != textures_.end())
        {
            if (expected && !valid_texture_color_space(*expected, found->second.data.color_space))
            {
                const auto diagnostic_key = resource_key(handle) ^ (static_cast<std::uint64_t>(*expected) << 56u);
                if (texture_semantic_diagnostics_.insert(diagnostic_key).second)
                {
                    arc::diagnostics::warn("render.vulkan", "Texture '" + found->second.data.name +
                                                                "' has a color space incompatible with its "
                                                                "material slot; binding the explicit fallback");
                }
                return;
            }
            if (found->second.view != VK_NULL_HANDLE && found->second.sampler != VK_NULL_HANDLE)
            {
                sampler = found->second.sampler;
                view = found->second.view;
            }
        }
    };

    std::array<VkDescriptorImageInfo, material_image_bindings.size()> image_infos{};
    VkSampler sampler{};
    VkImageView view{};
    resolve_texture({}, sampler, view);
    for (auto& image : image_infos)
        image = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    if (material != nullptr && material->domain == material_domain::terrain)
    {
        for (std::size_t layer = 0; layer < material->terrain_layers.size(); ++layer)
        {
            resolve_texture(material->terrain_layers[layer].base_color_texture, sampler, view,
                            texture_semantic::base_color);
            image_infos[layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material->terrain_layers[layer].normal_texture, sampler, view, texture_semantic::normal);
            image_infos[6u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            resolve_texture(material->terrain_layers[layer].packed_surface_texture, sampler, view,
                            texture_semantic::metallic_roughness);
            image_infos[10u + layer] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
    }
    else
    {
        resolve_texture(material ? material->base_color_texture : texture_handle{}, sampler, view,
                        texture_semantic::base_color);
        image_infos[0] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->metallic_roughness_texture : texture_handle{}, sampler, view,
                        texture_semantic::metallic_roughness);
        image_infos[1] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->normal_texture : texture_handle{}, sampler, view,
                        texture_semantic::normal);
        image_infos[2] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->occlusion_texture : texture_handle{}, sampler, view,
                        texture_semantic::occlusion);
        image_infos[3] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->emissive_texture : texture_handle{}, sampler, view,
                        texture_semantic::emissive);
        image_infos[4] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_texture : texture_handle{}, sampler, view,
                        texture_semantic::clear_coat);
        image_infos[6] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_roughness_texture : texture_handle{}, sampler, view,
                        texture_semantic::clear_coat);
        image_infos[7] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->clear_coat_normal_texture : texture_handle{}, sampler, view,
                        texture_semantic::normal);
        image_infos[8] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->anisotropy_texture : texture_handle{}, sampler, view,
                        texture_semantic::anisotropy);
        image_infos[9] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->subsurface_texture : texture_handle{}, sampler, view,
                        texture_semantic::thickness);
        image_infos[10] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->thickness_texture : texture_handle{}, sampler, view,
                        texture_semantic::thickness);
        image_infos[11] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        resolve_texture(material ? material->transmission_texture : texture_handle{}, sampler, view,
                        texture_semantic::transmission);
        image_infos[12] = {sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    }

    image_infos[5].sampler = shadow_atlas_.sampler;
    image_infos[5].imageView = shadow_atlas_.array_view;
    image_infos[5].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos.back().sampler = local_shadow_atlas_.sampler;
    image_infos.back().imageView = local_shadow_atlas_.view;
    image_infos.back().imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo shadow_buffer{};
    shadow_buffer.buffer = shadow_buffer_resource->buffer;
    shadow_buffer.offset = 0;
    shadow_buffer.range = sizeof(shadow_uniform_data);

    std::array<VkWriteDescriptorSet, material_image_bindings.size() + 3u> writes{};
    for (std::uint32_t image_index = 0; image_index < image_infos.size(); ++image_index)
    {
        writes[image_index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[image_index].dstSet = descriptor_set;
        writes[image_index].dstBinding = material_image_bindings[image_index];
        writes[image_index].descriptorCount = 1;
        writes[image_index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[image_index].pImageInfo = &image_infos[image_index];
    }
    auto& shadow_write = writes[material_image_bindings.size()];
    shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadow_write.dstSet = descriptor_set;
    shadow_write.dstBinding = material_shadow_data_binding;
    shadow_write.descriptorCount = 1;
    shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadow_write.pBufferInfo = &shadow_buffer;
    VkDescriptorBufferInfo light_buffer{};
    light_buffer.buffer = light_buffer_.buffer;
    light_buffer.range = sizeof(scene_lighting_data);
    auto& light_write = writes[material_image_bindings.size() + 1u];
    light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    light_write.dstSet = descriptor_set;
    light_write.dstBinding = material_light_data_binding;
    light_write.descriptorCount = 1;
    light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    light_write.pBufferInfo = &light_buffer;
    VkDescriptorBufferInfo parameter_buffer{};
    parameter_buffer.buffer = material_parameters->buffer;
    parameter_buffer.range = sizeof(material_uniform_data);
    auto& parameter_write = writes.back();
    parameter_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    parameter_write.dstSet = descriptor_set;
    parameter_write.dstBinding = material_parameters_binding;
    parameter_write.descriptorCount = 1;
    parameter_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    parameter_write.pBufferInfo = &parameter_buffer;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void vulkan_render_backend::update_material_descriptor_sets(gpu_material& material)
{
    if (!ensure_material_descriptor_sets(material)) return;
    for (std::uint32_t frame_slot = 0; frame_slot < material.descriptor_sets.size(); ++frame_slot)
    {
        if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
        update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                       &material.parameter_buffers[frame_slot], frame_slot);
    }
}

void vulkan_render_backend::update_white_descriptor_sets()
{
    if (!ensure_white_descriptor_sets()) return;
    for (std::uint32_t frame_slot = 0; frame_slot < white_descriptor_sets_.size(); ++frame_slot)
        update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                       &white_material_parameter_buffers_[frame_slot], frame_slot);
}

void vulkan_render_backend::update_all_material_descriptor_sets()
{
    update_white_descriptor_sets();
    for (auto& [_, material] : materials_)
        update_material_descriptor_sets(material);
}

void vulkan_render_backend::update_current_material_descriptor_sets()
{
    const auto frame_slot = current_frame_slot();
    if (ensure_white_descriptor_sets() && frame_slot < white_descriptor_sets_.size())
        update_material_descriptor_set(white_descriptor_sets_[frame_slot], nullptr,
                                       &white_material_parameter_buffers_[frame_slot], frame_slot);
    for (auto& [_, material] : materials_)
    {
        if (ensure_material_descriptor_sets(material) && frame_slot < material.descriptor_sets.size())
        {
            if (!update_material_parameter_buffer(material.parameter_buffers[frame_slot], &material.data)) continue;
            update_material_descriptor_set(material.descriptor_sets[frame_slot], &material.data,
                                           &material.parameter_buffers[frame_slot], frame_slot);
        }
    }
}

VkDescriptorSet vulkan_render_backend::allocate_material_descriptor_set()
{
    if (white_descriptor_pool_ == VK_NULL_HANDLE || white_descriptor_set_layout_ == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;

    VkDescriptorSet set{};
    VkDescriptorSetAllocateInfo descriptor_allocate{};
    descriptor_allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate.descriptorPool = white_descriptor_pool_;
    descriptor_allocate.descriptorSetCount = 1;
    descriptor_allocate.pSetLayouts = &white_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &descriptor_allocate, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    return set;
}

bool vulkan_render_backend::ensure_white_texture()
{
    const auto shadow_resolution = shadow_atlas_.resolution == 0 ? 2048u : shadow_atlas_.resolution;
    if (!ensure_shadow_uniform_buffers() ||
        !ensure_shadow_resources({.enabled = false, .resolution = shadow_resolution}) ||
        !ensure_local_shadow_resources())
        return false;

    if (white_descriptor_set_layout_ != VK_NULL_HANDLE && white_descriptor_pool_ != VK_NULL_HANDLE &&
        white_view_ != VK_NULL_HANDLE && white_sampler_ != VK_NULL_HANDLE)
    {
        return ensure_white_descriptor_sets();
    }

    std::array<VkDescriptorSetLayoutBinding, material_binding_count> bindings{};
    for (std::uint32_t binding_index = 0; binding_index < material_binding_count; ++binding_index)
    {
        bindings[binding_index].binding = binding_index;
        bindings[binding_index].descriptorType =
            binding_index == material_shadow_data_binding || binding_index == material_parameters_binding
                ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : binding_index == material_light_data_binding ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                           : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[binding_index].descriptorCount = 1;
        bindings[binding_index].stageFlags = binding_index == material_shadow_data_binding ||
                                                     binding_index == material_light_data_binding ||
                                                     binding_index == material_parameters_binding
                                                 ? VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
                                                 : VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &white_descriptor_set_layout_) != VK_SUCCESS)
        return false;

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_R8G8B8A8_UNORM;
    image.extent = {1, 1, 1};
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VmaAllocationCreateInfo allocation{};
    allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    if (vmaCreateImage(allocator_, &image, &allocation, &white_image_, &white_allocation_, nullptr) != VK_SUCCESS)
        return false;

    VkImageViewCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image = white_image_;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R8G8B8A8_UNORM;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = 1;
    view.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_, &view, nullptr, &white_view_) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sampler{};
    sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler.magFilter = VK_FILTER_NEAREST;
    sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(device_, &sampler, nullptr, &white_sampler_) != VK_SUCCESS) return false;

    const std::uint32_t white = 0xffffffffu;
    gpu_buffer staging;
    if (!create_buffer(sizeof(white), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU, staging))
        return false;
    void* mapped{};
    vmaMapMemory(allocator_, staging.allocation, &mapped);
    std::memcpy(mapped, &white, sizeof(white));
    vmaUnmapMemory(allocator_, staging.allocation);

    VkCommandPool pool{};
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = graphics_queue_family_;
    vkCreateCommandPool(device_, &pool_info, nullptr, &pool);
    VkCommandBuffer command_buffer{};
    VkCommandBufferAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate.commandPool = pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &allocate, &command_buffer);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(command_buffer, &begin);

    VkImageMemoryBarrier to_copy{};
    to_copy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    to_copy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    to_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_copy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_copy.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.image = white_image_;
    to_copy.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    to_copy.subresourceRange.levelCount = 1;
    to_copy.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &to_copy);

    VkBufferImageCopy copy{};
    copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy.imageSubresource.layerCount = 1;
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyBufferToImage(command_buffer, staging.buffer, white_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &copy);

    VkImageMemoryBarrier to_shader = to_copy;
    to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &to_shader);
    vkEndCommandBuffer(command_buffer);

    submit_upload_commands(command_buffer);
    vkDestroyCommandPool(device_, pool, nullptr);
    destroy_buffer(staging);

    std::array<VkDescriptorPoolSize, 3> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount =
        static_cast<std::uint32_t>(material_image_bindings.size()) * material_descriptor_set_capacity;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[1].descriptorCount = material_descriptor_set_capacity * 2u;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = material_descriptor_set_capacity;
    VkDescriptorPoolCreateInfo descriptor_pool{};
    descriptor_pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    descriptor_pool.maxSets = material_descriptor_set_capacity;
    descriptor_pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    descriptor_pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &descriptor_pool, nullptr, &white_descriptor_pool_) != VK_SUCCESS) return false;

    update_all_material_descriptor_sets();
    return true;
}

bool vulkan_render_backend::ensure_mesh_pipeline()
{
    if (mesh_pipeline_ != VK_NULL_HANDLE) return true;
    if (max_push_constant_bytes_ < sizeof(mesh_push_constants))
    {
        if (!push_constant_limit_warning_reported_)
        {
            arc::diagnostics::error("render.vulkan",
                                    "The selected adapter exposes only " + std::to_string(max_push_constant_bytes_) +
                                        " push-constant bytes; ARC's raster mesh path currently requires " +
                                        std::to_string(sizeof(mesh_push_constants)));
            push_constant_limit_warning_reported_ = true;
        }
        return false;
    }
    if (!ensure_white_texture()) return false;

    if (material_attribute_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        const VkDescriptorSetLayoutBinding binding{0u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1u,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        descriptor_layout.bindingCount = 1u;
        descriptor_layout.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr,
                                        &material_attribute_descriptor_set_layout_) != VK_SUCCESS)
            return false;
    }
    if (material_attribute_descriptor_pool_ == VK_NULL_HANDLE)
    {
        const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                             material_attribute_descriptor_set_capacity};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = material_attribute_descriptor_set_capacity;
        pool.poolSizeCount = 1u;
        pool.pPoolSizes = &pool_size;
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &material_attribute_descriptor_pool_) != VK_SUCCESS)
            return false;
    }

    VkShaderModule vert =
        create_shader_module(builtin::default_phong_vert_spv, std::size(builtin::default_phong_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::default_phong_frag_spv, std::size(builtin::default_phong_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(mesh_push_constants);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &white_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &mesh_pipeline_layout_) != VK_SUCCESS) return false;
    const std::array terrain_surface_set_layouts{white_descriptor_set_layout_,
                                                 material_attribute_descriptor_set_layout_};
    layout.setLayoutCount = static_cast<std::uint32_t>(terrain_surface_set_layouts.size());
    layout.pSetLayouts = terrain_surface_set_layouts.data();
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &terrain_surface_pipeline_layout_) != VK_SUCCESS)
        return false;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(mesh_vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 5> attributes{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)};
    attributes[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)};
    attributes[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;

    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &scene_color_format_;
    rendering.depthAttachmentFormat = depth_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = mesh_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_pipeline_);
    if (result == VK_SUCCESS)
    {
        depth.depthWriteEnable = VK_FALSE;
        color_attachment.blendEnable = VK_TRUE;
        color_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        color_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        color_attachment.colorBlendOp = VK_BLEND_OP_ADD;
        color_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        color_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        color_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        const VkResult blend_result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_transparent_pipeline_);
        if (blend_result != VK_SUCCESS)
            arc::diagnostics::warn(
                "render.vulkan",
                "Vulkan transparent mesh pipeline creation failed; blended materials will render opaque");
        color_attachment = {};
        color_attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        depth.depthWriteEnable = VK_TRUE;
    }
    if (result == VK_SUCCESS && capabilities_.fill_mode_non_solid)
    {
        raster.polygonMode = VK_POLYGON_MODE_LINE;
        depth.depthWriteEnable = VK_FALSE;
        const VkResult wire_result =
            vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &mesh_wire_pipeline_);
        if (wire_result != VK_SUCCESS)
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan wireframe pipeline creation failed; shaded rendering will continue");
    }
    else if (result == VK_SUCCESS && !capabilities_.fill_mode_non_solid && !wireframe_warning_reported_)
    {
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan device does not support fillModeNonSolid; wireframe rendering is disabled");
        wireframe_warning_reported_ = true;
    }

    if (result == VK_SUCCESS)
    {
        VkShaderModule terrain_surface_frag = create_shader_module(
            builtin::terrain_surface_forward_frag_spv, std::size(builtin::terrain_surface_forward_frag_spv));
        if (terrain_surface_frag != VK_NULL_HANDLE)
        {
            stages[0].module = vert;
            stages[1].module = terrain_surface_frag;
            pipeline.pVertexInputState = &vertex_input;
            pipeline.layout = terrain_surface_pipeline_layout_;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            depth.depthWriteEnable = VK_TRUE;
            color_attachment = {};
            color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                          &terrain_surface_pipeline_) != VK_SUCCESS)
            {
                terrain_surface_pipeline_ = VK_NULL_HANDLE;
                arc::diagnostics::warn("render.vulkan",
                                       "Vulkan terrain surface pipeline creation failed; using mesh fallback");
            }
            vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
        }
    }

    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_debug_overlay_pipeline()
{
    if (debug_overlay_line_pipeline_ != VK_NULL_HANDLE && debug_overlay_triangle_pipeline_ != VK_NULL_HANDLE &&
        debug_overlay_output_line_pipeline_ != VK_NULL_HANDLE &&
        debug_overlay_output_triangle_pipeline_ != VK_NULL_HANDLE)
        return true;
    VkShaderModule vert =
        create_shader_module(builtin::debug_overlay_vert_spv, std::size(builtin::debug_overlay_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::debug_overlay_frag_spv, std::size(builtin::debug_overlay_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push.size = sizeof(float) * 16u;
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &debug_overlay_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 nullptr,
                 0,
                 VK_SHADER_STAGE_VERTEX_BIT,
                 vert,
                 "main",
                 nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 nullptr,
                 0,
                 VK_SHADER_STAGE_FRAGMENT_BIT,
                 frag,
                 "main",
                 nullptr};
    VkVertexInputBindingDescription binding{0, sizeof(debug_overlay_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array<VkVertexInputAttributeDescription, 2> attributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(debug_overlay_vertex, position)},
        VkVertexInputAttributeDescription{1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(debug_overlay_vertex, color)}};
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &blend;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    const auto create_pipeline =
        [&](VkPrimitiveTopology topology, VkFormat color_format, bool depth_test, VkPipeline& destination)
    {
        input.topology = topology;
        depth.depthTestEnable = depth_test ? VK_TRUE : VK_FALSE;
        VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachmentFormats = &color_format;
        rendering.depthAttachmentFormat = depth_test ? depth_format_ : VK_FORMAT_UNDEFINED;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.pNext = &rendering;
        pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
        pipeline.pStages = stages.data();
        pipeline.pVertexInputState = &vertex_input;
        pipeline.pInputAssemblyState = &input;
        pipeline.pViewportState = &viewport;
        pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample;
        pipeline.pDepthStencilState = &depth;
        pipeline.pColorBlendState = &color_blend;
        pipeline.pDynamicState = &dynamic;
        pipeline.layout = debug_overlay_pipeline_layout_;
        return vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &destination);
    };

    const auto tested_line_result =
        create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, scene_color_format_, true, debug_overlay_line_pipeline_);
    const auto tested_triangle_result = tested_line_result == VK_SUCCESS
                                            ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, scene_color_format_,
                                                              true, debug_overlay_triangle_pipeline_)
                                            : VK_ERROR_INITIALIZATION_FAILED;
    const auto output_line_result = tested_triangle_result == VK_SUCCESS
                                        ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, viewport_format_, false,
                                                          debug_overlay_output_line_pipeline_)
                                        : VK_ERROR_INITIALIZATION_FAILED;
    const auto output_triangle_result = output_line_result == VK_SUCCESS
                                            ? create_pipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, viewport_format_,
                                                              false, debug_overlay_output_triangle_pipeline_)
                                            : VK_ERROR_INITIALIZATION_FAILED;
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (tested_line_result != VK_SUCCESS || tested_triangle_result != VK_SUCCESS || output_line_result != VK_SUCCESS ||
        output_triangle_result != VK_SUCCESS)
    {
        arc::diagnostics::warn("render.vulkan", "Vulkan debug-overlay pipeline creation failed");
        const auto destroy_pipeline = [&](VkPipeline& pipeline)
        {
            if (pipeline == VK_NULL_HANDLE) return;
            vkDestroyPipeline(device_, pipeline, nullptr);
            pipeline = VK_NULL_HANDLE;
        };
        destroy_pipeline(debug_overlay_line_pipeline_);
        destroy_pipeline(debug_overlay_triangle_pipeline_);
        destroy_pipeline(debug_overlay_output_line_pipeline_);
        destroy_pipeline(debug_overlay_output_triangle_pipeline_);
        vkDestroyPipelineLayout(device_, debug_overlay_pipeline_layout_, nullptr);
        debug_overlay_pipeline_layout_ = VK_NULL_HANDLE;
    }
    return tested_line_result == VK_SUCCESS && tested_triangle_result == VK_SUCCESS &&
           output_line_result == VK_SUCCESS && output_triangle_result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gpu_bindless_pipelines()
{
    if (!resolved_config_.features.gpu_visibility_compaction ||
        shared_geometry_buffers_.vertices.buffer == VK_NULL_HANDLE ||
        shared_geometry_buffers_.indices.buffer == VK_NULL_HANDLE ||
        gpu_scene_visibility_buffer_.buffer == VK_NULL_HANDLE)
        return false;
    const auto& material_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::material)];
    const auto& texture_table = gpu_resource_tables_[gpu_table_offset(gpu_resource_table_kind::texture)];
    if (material_table.storage.buffer == VK_NULL_HANDLE) return false;

    if (gpu_bindless_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
        for (std::uint32_t binding = 0u; binding < 4u; ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4u, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, virtual_geometry_bindless_texture_capacity,
                       VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        std::array<VkDescriptorBindingFlags, 5> binding_flags{};
        binding_flags[4] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                           VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
                           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo flags{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flags.bindingCount = static_cast<std::uint32_t>(binding_flags.size());
        flags.pBindingFlags = binding_flags.data();
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.pNext = &flags;
        layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gpu_bindless_descriptor_set_layout_) != VK_SUCCESS)
            return false;
        gpu_bindless_descriptors_dirty_ = true;
    }

    if (gpu_bindless_descriptors_dirty_)
    {
        VkDescriptorPool replacement_pool{};
        VkDescriptorSet replacement_set{};
        const std::array pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4u},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, virtual_geometry_bindless_texture_capacity},
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool.maxSets = 1u;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &replacement_pool) != VK_SUCCESS) return false;
        const std::uint32_t descriptor_count = virtual_geometry_bindless_texture_capacity;
        VkDescriptorSetVariableDescriptorCountAllocateInfo variable_count{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO};
        variable_count.descriptorSetCount = 1u;
        variable_count.pDescriptorCounts = &descriptor_count;
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.pNext = &variable_count;
        allocate.descriptorPool = replacement_pool;
        allocate.descriptorSetCount = 1u;
        allocate.pSetLayouts = &gpu_bindless_descriptor_set_layout_;
        if (vkAllocateDescriptorSets(device_, &allocate, &replacement_set) != VK_SUCCESS)
        {
            vkDestroyDescriptorPool(device_, replacement_pool, nullptr);
            return false;
        }
        const auto texture_buffer = texture_table.storage.buffer != VK_NULL_HANDLE ? texture_table.storage.buffer
                                                                                   : material_table.storage.buffer;
        const std::array buffer_infos{
            VkDescriptorBufferInfo{gpu_scene_visibility_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{gpu_scene_transform_buffer_.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{material_table.storage.buffer, 0u, VK_WHOLE_SIZE},
            VkDescriptorBufferInfo{texture_buffer, 0u, VK_WHOLE_SIZE},
        };
        std::array<VkWriteDescriptorSet, 4> buffer_writes{};
        for (std::uint32_t binding = 0u; binding < buffer_writes.size(); ++binding)
        {
            buffer_writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            buffer_writes[binding].dstSet = replacement_set;
            buffer_writes[binding].dstBinding = binding;
            buffer_writes[binding].descriptorCount = 1u;
            buffer_writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            buffer_writes[binding].pBufferInfo = &buffer_infos[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0u,
                               nullptr);

        std::vector<VkDescriptorImageInfo> texture_infos;
        std::vector<VkWriteDescriptorSet> texture_writes;
        texture_infos.reserve(textures_.size());
        texture_writes.reserve(textures_.size());
        for (const auto& [_, texture] : textures_)
        {
            if (!texture.handle.valid() || texture.handle.index >= virtual_geometry_bindless_texture_capacity ||
                texture.view == VK_NULL_HANDLE || texture.sampler == VK_NULL_HANDLE)
                continue;
            texture_infos.push_back({texture.sampler, texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = replacement_set;
            write.dstBinding = 4u;
            write.dstArrayElement = texture.handle.index;
            write.descriptorCount = 1u;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            texture_writes.push_back(write);
        }
        for (std::size_t index = 0u; index < texture_writes.size(); ++index)
            texture_writes[index].pImageInfo = &texture_infos[index];
        if (!texture_writes.empty())
            vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(texture_writes.size()), texture_writes.data(),
                                   0u, nullptr);
        const auto retired_pool = gpu_bindless_descriptor_pool_;
        gpu_bindless_descriptor_pool_ = replacement_pool;
        gpu_bindless_descriptor_set_ = replacement_set;
        if (retired_pool != VK_NULL_HANDLE)
            deferred_releases_.defer(last_profile_.frame_index + frame_resource_count(), [this, retired_pool]()
                                     { vkDestroyDescriptorPool(device_, retired_pool, nullptr); });
        gpu_bindless_descriptors_dirty_ = false;
    }

    if (gpu_bindless_pipeline_layout_ == VK_NULL_HANDLE)
    {
        VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0u, sizeof(float) * 32u};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1u;
        layout.pSetLayouts = &gpu_bindless_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1u;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &gpu_bindless_pipeline_layout_) != VK_SUCCESS)
            return false;
    }
    if (gpu_bindless_gbuffer_pipeline_ != VK_NULL_HANDLE && gpu_bindless_transparent_pipeline_ != VK_NULL_HANDLE)
        return true;

    const auto vertex_shader =
        create_shader_module(builtin::gpu_scene_bindless_vert_spv, std::size(builtin::gpu_scene_bindless_vert_spv));
    const auto gbuffer_shader = create_shader_module(builtin::gpu_scene_bindless_gbuffer_frag_spv,
                                                     std::size(builtin::gpu_scene_bindless_gbuffer_frag_spv));
    const auto transparent_shader = create_shader_module(builtin::gpu_scene_bindless_transparent_frag_spv,
                                                         std::size(builtin::gpu_scene_bindless_transparent_frag_spv));
    if (vertex_shader == VK_NULL_HANDLE || gbuffer_shader == VK_NULL_HANDLE || transparent_shader == VK_NULL_HANDLE)
    {
        if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vertex_shader, nullptr);
        if (gbuffer_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, gbuffer_shader, nullptr);
        if (transparent_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, transparent_shader, nullptr);
        return false;
    }
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertex_shader;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{0u, sizeof(mesh_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array attributes{
        VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)},
        VkVertexInputAttributeDescription{2u, 0u, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)},
        VkVertexInputAttributeDescription{3u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)},
        VkVertexInputAttributeDescription{4u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)},
    };
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertex_input.vertexBindingDescriptionCount = 1u;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();
    VkPipelineInputAssemblyStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1u;
    viewport.scissorCount = 1u;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    std::array<VkPipelineColorBlendAttachmentState, 6> gbuffer_attachments{};
    for (auto& attachment : gbuffer_attachments)
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = static_cast<std::uint32_t>(gbuffer_attachments.size());
    blend.pAttachments = gbuffer_attachments.data();
    const std::array<VkFormat, 6> gbuffer_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                  VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(gbuffer_formats.size());
    rendering.pColorAttachmentFormats = gbuffer_formats.data();
    rendering.depthAttachmentFormat = depth_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = gpu_bindless_pipeline_layout_;
    stages[1].module = gbuffer_shader;
    const auto gbuffer_result = gpu_bindless_gbuffer_pipeline_ != VK_NULL_HANDLE
                                    ? VK_SUCCESS
                                    : vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                                &gpu_bindless_gbuffer_pipeline_);

    VkPipelineColorBlendAttachmentState transparent_attachment{};
    transparent_attachment.blendEnable = VK_TRUE;
    transparent_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    transparent_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparent_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    transparent_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    transparent_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    transparent_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    transparent_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend.attachmentCount = 1u;
    blend.pAttachments = &transparent_attachment;
    rendering.colorAttachmentCount = 1u;
    rendering.pColorAttachmentFormats = &scene_color_format_;
    stages[1].module = transparent_shader;
    const auto transparent_result = gpu_bindless_transparent_pipeline_ != VK_NULL_HANDLE
                                        ? VK_SUCCESS
                                        : vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr,
                                                                    &gpu_bindless_transparent_pipeline_);
    vkDestroyShaderModule(device_, vertex_shader, nullptr);
    vkDestroyShaderModule(device_, gbuffer_shader, nullptr);
    vkDestroyShaderModule(device_, transparent_shader, nullptr);
    return gbuffer_result == VK_SUCCESS && transparent_result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gbuffer_pipeline()
{
    if (gbuffer_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_mesh_pipeline()) return false;

    VkShaderModule vert = create_shader_module(builtin::gbuffer_vert_spv, std::size(builtin::gbuffer_vert_spv));
    VkShaderModule frag = create_shader_module(builtin::gbuffer_frag_spv, std::size(builtin::gbuffer_frag_spv));
    VkShaderModule terrain_surface_frag = create_shader_module(builtin::terrain_surface_gbuffer_frag_spv,
                                                               std::size(builtin::terrain_surface_gbuffer_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE)
    {
        if (vert != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, frag, nullptr);
        if (terrain_surface_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(mesh_vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::array<VkVertexInputAttributeDescription, 5> attributes{};
    attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)};
    attributes[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)};
    attributes[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)};
    attributes[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)};
    attributes[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    std::array<VkPipelineColorBlendAttachmentState, 6> attachments{};
    for (auto& attachment : attachments)
    {
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    }
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    color_blend.pAttachments = attachments.data();

    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    const std::array<VkFormat, 6> color_formats{VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_FORMAT_R16G16_SFLOAT,       VK_FORMAT_R32_UINT};
    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = static_cast<std::uint32_t>(color_formats.size());
    rendering.pColorAttachmentFormats = color_formats.data();
    rendering.depthAttachmentFormat = depth_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = mesh_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &gbuffer_pipeline_);
    if (result == VK_SUCCESS && terrain_surface_frag != VK_NULL_HANDLE)
    {
        stages[0].module = vert;
        stages[1].module = terrain_surface_frag;
        pipeline.pVertexInputState = &vertex_input;
        pipeline.layout = terrain_surface_pipeline_layout_;
        if (vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr,
                                      &terrain_surface_gbuffer_pipeline_) != VK_SUCCESS)
        {
            terrain_surface_gbuffer_pipeline_ = VK_NULL_HANDLE;
            arc::diagnostics::warn("render.vulkan",
                                   "Vulkan terrain surface G-buffer pipeline creation failed; using mesh fallback");
        }
    }
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (terrain_surface_frag != VK_NULL_HANDLE) vkDestroyShaderModule(device_, terrain_surface_frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan G-buffer pipeline creation failed; falling back to forward rendering");
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_gbuffer_descriptor_set()
{
    if (gbuffer_descriptor_set_ != VK_NULL_HANDLE) return true;
    if (!ensure_white_texture() || gbuffer_albedo_.view == VK_NULL_HANDLE || gbuffer_normal_.view == VK_NULL_HANDLE ||
        gbuffer_material_.view == VK_NULL_HANDLE || gbuffer_emissive_.view == VK_NULL_HANDLE ||
        gbuffer_motion_.view == VK_NULL_HANDLE || gbuffer_object_id_.view == VK_NULL_HANDLE ||
        viewport_depth_view_ == VK_NULL_HANDLE || shadow_atlas_.array_view == VK_NULL_HANDLE ||
        shadow_atlas_.sampler == VK_NULL_HANDLE || local_shadow_atlas_.view == VK_NULL_HANDLE ||
        local_shadow_atlas_.sampler == VK_NULL_HANDLE || current_shadow_uniform_buffer() == nullptr)
        return false;

    if (gbuffer_sampler_ == VK_NULL_HANDLE)
    {
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.magFilter = VK_FILTER_NEAREST;
        sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &sampler, nullptr, &gbuffer_sampler_) != VK_SUCCESS) return false;
    }

    if (gbuffer_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 12> bindings{};
        for (std::uint32_t index = 0; index < 7; ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[index].descriptorCount = 1;
            bindings[index].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        bindings[7].binding = 7;
        bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[7].descriptorCount = 1;
        bindings[7].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[8].binding = 8;
        bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[8].descriptorCount = 1;
        bindings[8].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[9].binding = 9;
        bindings[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[9].descriptorCount = 1;
        bindings[9].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[10].binding = 10;
        bindings[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[10].descriptorCount = 1;
        bindings[10].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[11].binding = 11;
        bindings[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[11].descriptorCount = 1;
        bindings[11].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &layout, nullptr, &gbuffer_descriptor_set_layout_) != VK_SUCCESS)
            return false;
    }

    if (gbuffer_descriptor_pool_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorPoolSize, 3> pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
        VkDescriptorPoolCreateInfo pool{};
        pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool.maxSets = 1;
        pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &gbuffer_descriptor_pool_) != VK_SUCCESS) return false;
    }

    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = gbuffer_descriptor_pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &gbuffer_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &gbuffer_descriptor_set_) != VK_SUCCESS) return false;

    update_gbuffer_descriptor_set();
    return true;
}

void vulkan_render_backend::update_gbuffer_descriptor_set()
{
    if (gbuffer_descriptor_set_ == VK_NULL_HANDLE || light_buffer_.buffer == VK_NULL_HANDLE) return;

    std::array<VkDescriptorImageInfo, 10> images{};
    const VkSampler sampler = gbuffer_sampler_ != VK_NULL_HANDLE ? gbuffer_sampler_ : white_sampler_;
    images[0] = {sampler, gbuffer_albedo_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[1] = {sampler, gbuffer_normal_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[2] = {sampler, gbuffer_material_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[3] = {sampler, gbuffer_emissive_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[4] = {sampler, gbuffer_object_id_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[5] = {sampler, gbuffer_motion_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[6] = {sampler, viewport_depth_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[7] = {white_sampler_, white_view_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    images[8] = {shadow_atlas_.sampler, shadow_atlas_.array_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    images[9] = {local_shadow_atlas_.sampler, local_shadow_atlas_.view,
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    if (const auto* environment = active_environment())
    {
        if (const auto found = textures_.find(resource_key(environment->equirectangular_texture));
            found != textures_.end() && found->second.view != VK_NULL_HANDLE)
        {
            images[7] = {found->second.sampler, found->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
    }

    std::array<VkWriteDescriptorSet, 10> writes{};
    for (std::uint32_t index = 0; index < 7; ++index)
    {
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = gbuffer_descriptor_set_;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[index].pImageInfo = &images[index];
    }
    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = gbuffer_descriptor_set_;
    writes[7].dstBinding = 8;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[7].pImageInfo = &images[7];
    writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet = gbuffer_descriptor_set_;
    writes[8].dstBinding = 9;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[8].pImageInfo = &images[8];
    writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[9].dstSet = gbuffer_descriptor_set_;
    writes[9].dstBinding = 11;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[9].pImageInfo = &images[9];
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    VkDescriptorBufferInfo lights{light_buffer_.buffer, 0, sizeof(scene_lighting_data)};
    VkWriteDescriptorSet light_write{};
    light_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    light_write.dstSet = gbuffer_descriptor_set_;
    light_write.dstBinding = 7;
    light_write.descriptorCount = 1;
    light_write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    light_write.pBufferInfo = &lights;
    const auto* shadow_buffer = current_shadow_uniform_buffer();
    if (shadow_buffer == nullptr) return;
    VkDescriptorBufferInfo shadow{shadow_buffer->buffer, 0, sizeof(shadow_uniform_data)};
    VkWriteDescriptorSet shadow_write{};
    shadow_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadow_write.dstSet = gbuffer_descriptor_set_;
    shadow_write.dstBinding = 10;
    shadow_write.descriptorCount = 1;
    shadow_write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    shadow_write.pBufferInfo = &shadow;
    const std::array buffer_writes{light_write, shadow_write};
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(buffer_writes.size()), buffer_writes.data(), 0, nullptr);
}

bool vulkan_render_backend::ensure_deferred_pipeline()
{
    if (deferred_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_gbuffer_descriptor_set()) return false;

    VkShaderModule vert =
        create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::deferred_lighting_frag_spv, std::size(builtin::deferred_lighting_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(deferred_push_constants);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &gbuffer_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &deferred_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &scene_color_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = deferred_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &deferred_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan",
                               "Vulkan deferred lighting pipeline creation failed; falling back to forward rendering");
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_output_transform_pipeline()
{
    if (exposure_buffer_.buffer == VK_NULL_HANDLE &&
        !create_buffer(exposure_buffer_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_GPU_ONLY, exposure_buffer_))
        return false;

    if (output_transform_pipeline_ != VK_NULL_HANDLE)
    {
        VkDescriptorImageInfo image{viewport_sampler_,
                                    temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = output_transform_descriptor_set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &image;
        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
        return true;
    }
    if (scene_color_.view == VK_NULL_HANDLE || viewport_sampler_ == VK_NULL_HANDLE) return false;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo descriptor_layout{};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    descriptor_layout.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &output_transform_descriptor_set_layout_) !=
        VK_SUCCESS)
        return false;

    std::array<VkDescriptorPoolSize, 2> pool_sizes{VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                                   VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 1;
    pool.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &output_transform_descriptor_pool_) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = output_transform_descriptor_pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &output_transform_descriptor_set_layout_;
    if (vkAllocateDescriptorSets(device_, &allocate, &output_transform_descriptor_set_) != VK_SUCCESS) return false;

    VkDescriptorImageInfo image{viewport_sampler_,
                                temporal_output_view_ != VK_NULL_HANDLE ? temporal_output_view_ : scene_color_.view,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorBufferInfo exposure_buffer_info{exposure_buffer_.buffer, 0, exposure_buffer_bytes};
    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = output_transform_descriptor_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = output_transform_descriptor_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &exposure_buffer_info;
    vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const auto vert =
        create_shader_module(builtin::deferred_lighting_vert_spv, std::size(builtin::deferred_lighting_vert_spv));
    const auto frag =
        create_shader_module(builtin::output_transform_frag_spv, std::size(builtin::output_transform_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.size = sizeof(output_transform_push_constants);
    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &output_transform_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &output_transform_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;
    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &viewport_format_;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.pNext = &rendering;
    pipeline.stageCount = static_cast<std::uint32_t>(stages.size());
    pipeline.pStages = stages.data();
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = output_transform_pipeline_layout_;
    const auto result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &output_transform_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    if (result != VK_SUCCESS)
        arc::diagnostics::warn("render.vulkan", "Vulkan output-transform pipeline creation failed");
    return result == VK_SUCCESS;
}

bool vulkan_render_backend::ensure_exposure_pipelines()
{
    if (luminance_histogram_pipeline_ != VK_NULL_HANDLE && exposure_resolve_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_output_transform_pipeline()) return false;

    const auto create_compute = [&](const std::uint32_t* words, std::size_t word_count, std::uint32_t push_size,
                                    VkPipelineLayout& pipeline_layout, VkPipeline& pipeline)
    {
        const auto shader = create_shader_module(words, word_count);
        if (shader == VK_NULL_HANDLE) return false;
        VkPushConstantRange push{};
        push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push.size = push_size;
        VkPipelineLayoutCreateInfo layout{};
        layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &output_transform_descriptor_set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout) != VK_SUCCESS)
        {
            vkDestroyShaderModule(device_, shader, nullptr);
            return false;
        }
        VkComputePipelineCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        create.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        create.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        create.stage.module = shader;
        create.stage.pName = "main";
        create.layout = pipeline_layout;
        const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1, &create, nullptr, &pipeline);
        vkDestroyShaderModule(device_, shader, nullptr);
        return result == VK_SUCCESS;
    };

    if (!create_compute(builtin::luminance_histogram_comp_spv, std::size(builtin::luminance_histogram_comp_spv),
                        sizeof(histogram_push_constants), luminance_histogram_pipeline_layout_,
                        luminance_histogram_pipeline_))
        return false;
    if (!create_compute(builtin::exposure_resolve_comp_spv, std::size(builtin::exposure_resolve_comp_spv),
                        sizeof(exposure_resolve_push_constants), exposure_resolve_pipeline_layout_,
                        exposure_resolve_pipeline_))
        return false;
    return true;
}

void vulkan_render_backend::dispatch_exposure(VkCommandBuffer command_buffer)
{
    if (!ensure_exposure_pipelines()) return;

    // The exposure state is persistent and shared between frames.
    // Serialize the transfer clear after the previous frame's resolve and
    // output-transform read before reusing the buffer.
    VkBufferMemoryBarrier reuse_barrier{};
    reuse_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    reuse_barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    reuse_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    reuse_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    reuse_barrier.buffer = exposure_buffer_.buffer;
    reuse_barrier.size = exposure_buffer_bytes;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &reuse_barrier, 0, nullptr);

    const VkDeviceSize clear_size = exposure_needs_reset_ ? exposure_buffer_bytes : exposure_histogram_bytes;
    vkCmdFillBuffer(command_buffer, exposure_buffer_.buffer, 0, clear_size, 0u);
    VkBufferMemoryBarrier clear_barrier{};
    clear_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    clear_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clear_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    clear_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_barrier.buffer = exposure_buffer_.buffer;
    clear_barrier.size = exposure_buffer_bytes;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                         nullptr, 1, &clear_barrier, 0, nullptr);

    histogram_push_constants histogram{};
    histogram.log_luminance_extent[2] = static_cast<float>(viewport_width_);
    histogram.log_luminance_extent[3] = static_cast<float>(viewport_height_);
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, luminance_histogram_pipeline_layout_, 0, 1,
                            &output_transform_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, luminance_histogram_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(histogram), &histogram);
    const std::uint32_t sample_width = (viewport_width_ + 3u) / 4u;
    const std::uint32_t sample_height = (viewport_height_ + 3u) / 4u;
    vkCmdDispatch(command_buffer, (sample_width + 15u) / 16u, (sample_height + 15u) / 16u, 1u);

    VkBufferMemoryBarrier histogram_barrier = clear_barrier;
    histogram_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    histogram_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                         0, nullptr, 1, &histogram_barrier, 0, nullptr);

    exposure_resolve_push_constants resolve{};
    resolve.limits_speeds[0] = frame_camera_.exposure.minimum_ev100;
    resolve.limits_speeds[1] = frame_camera_.exposure.maximum_ev100;
    resolve.limits_speeds[2] = frame_camera_.exposure.brighten_speed;
    resolve.limits_speeds[3] = frame_camera_.exposure.darken_speed;
    resolve.timing_mode[1] = frame_camera_.exposure.mode == exposure_mode::automatic ? 1.0f : 0.0f;
    resolve.timing_mode[2] = frame_camera_.exposure.manual_ev100;
    resolve.timing_mode[3] = exposure_needs_reset_ ? 1.0f : 0.0f;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposure_resolve_pipeline_layout_, 0, 1,
                            &output_transform_descriptor_set_, 0, nullptr);
    vkCmdPushConstants(command_buffer, exposure_resolve_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(resolve), &resolve);
    vkCmdDispatch(command_buffer, 1u, 1u, 1u);

    VkBufferMemoryBarrier output_barrier = histogram_barrier;
    output_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         0, nullptr, 1, &output_barrier, 0, nullptr);
    exposure_needs_reset_ = false;
}

bool vulkan_render_backend::ensure_sky_pipeline()
{
    if (sky_pipeline_ != VK_NULL_HANDLE) return true;
    if (!ensure_white_texture()) return false;

    VkShaderModule vert =
        create_shader_module(builtin::sky_atmosphere_vert_spv, std::size(builtin::sky_atmosphere_vert_spv));
    VkShaderModule frag =
        create_shader_module(builtin::sky_atmosphere_frag_spv, std::size(builtin::sky_atmosphere_frag_spv));
    if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) return false;

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push.offset = 0;
    push.size = sizeof(detail::sky_push_constants);

    VkPipelineLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &white_descriptor_set_layout_;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &sky_pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_FALSE;
    depth.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState color_attachment{};
    color_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.attachmentCount = 1;
    color_blend.pAttachments = &color_attachment;

    const std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size());
    dynamic.pDynamicStates = dynamic_states.data();

    VkPipelineRenderingCreateInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachmentFormats = &scene_color_format_;
    rendering.depthAttachmentFormat = depth_format_;

    VkGraphicsPipelineCreateInfo pipeline{};
    pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline.pNext = &rendering;
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex_input;
    pipeline.pInputAssemblyState = &input_assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color_blend;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = sky_pipeline_layout_;
    pipeline.renderPass = VK_NULL_HANDLE;

    const VkResult result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1, &pipeline, nullptr, &sky_pipeline_);
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    return result == VK_SUCCESS;
}

} // namespace arc::render::vulkan::backend_detail

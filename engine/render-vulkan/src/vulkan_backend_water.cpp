#include "vulkan_backend_internal.h"

#include "builtin_shaders.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <ranges>

namespace arc::render::vulkan::backend_detail
{
namespace
{

constexpr VkDeviceSize frequency_sample_size = sizeof(float) * 24u;
constexpr VkDeviceSize surface_sample_size = sizeof(float) * 12u;

struct gpu_complex
{
    float real{};
    float imaginary{};
};

struct alignas(16) gpu_water_surface_metadata
{
    std::uint32_t resolutions[4]{};
    float physical_lengths[4]{};
    float full_detail_distances[4]{};
    float fade_out_distances[4]{};
    std::uint32_t cascade_count{};
    std::uint32_t reserved[3]{};
};
static_assert(sizeof(gpu_water_surface_metadata) == 80u);

struct water_spectrum_push_constants
{
    std::uint32_t resolution{};
    float physical_length{};
    float choppiness{};
    float time_seconds{};
};
static_assert(sizeof(water_spectrum_push_constants) == 16u);

struct water_fft_push_constants
{
    std::uint32_t resolution{};
    std::uint32_t stage{};
    std::uint32_t direction{};
    std::uint32_t reserved{};
};
static_assert(sizeof(water_fft_push_constants) == 16u);

struct water_foam_push_constants
{
    std::uint32_t resolution{};
    float foam_threshold{};
    float foam_retention{};
    std::uint32_t configuration{};
};
static_assert(sizeof(water_foam_push_constants) == 16u);

std::uint64_t water_object_key(render_object_id object) noexcept
{
    return (static_cast<std::uint64_t>(object.generation) << 32u) | object.index;
}

void hash_combine(std::uint64_t& seed, std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    value ^= value >> 31u;
    seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u);
}

std::uint64_t water_settings_signature(const water_render_instance& instance) noexcept
{
    std::uint64_t result{0x4152435741544552ull};
    const auto& simulation = instance.settings.simulation;
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_speed));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_direction[0]));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wind_direction[1]));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.fetch_length));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.wave_amplitude));
    hash_combine(result, std::bit_cast<std::uint32_t>(simulation.choppiness));
    hash_combine(result, simulation.seed);
    hash_combine(result, static_cast<std::uint64_t>(instance.settings.quality));
    hash_combine(result, instance.settings.foam.enabled ? 1u : 0u);
    hash_combine(result, std::bit_cast<std::uint32_t>(instance.settings.foam.threshold));
    hash_combine(result, std::bit_cast<std::uint32_t>(instance.settings.foam.decay));
    return result;
}

void water_compute_barrier(VkCommandBuffer command_buffer)
{
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u,
                         1u, &barrier, 0u, nullptr, 0u, nullptr);
}

} // namespace

bool vulkan_render_backend::ensure_water_compute_resources()
{
    if (water_spectrum_pipeline_ != VK_NULL_HANDLE && water_fft_bit_reverse_pipeline_ != VK_NULL_HANDLE &&
        water_fft_stage_pipeline_ != VK_NULL_HANDLE && water_surface_finalize_pipeline_ != VK_NULL_HANDLE)
        return true;

    if (water_compute_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        const std::array bindings{VkDescriptorSetLayoutBinding{0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                                                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                  VkDescriptorSetLayoutBinding{1u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u,
                                                               VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        const VkDescriptorSetLayoutCreateInfo descriptor_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0u,
            static_cast<std::uint32_t>(bindings.size()), bindings.data()};
        if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &water_compute_descriptor_set_layout_) !=
            VK_SUCCESS)
            return false;
    }

    if (water_surface_descriptor_set_layout_ == VK_NULL_HANDLE)
    {
        std::array<VkDescriptorSetLayoutBinding, 5u> bindings{};
        for (std::uint32_t binding = 0u; binding < bindings.size(); ++binding)
            bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        const VkDescriptorSetLayoutCreateInfo descriptor_layout{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0u,
            static_cast<std::uint32_t>(bindings.size()), bindings.data()};
        if (vkCreateDescriptorSetLayout(device_, &descriptor_layout, nullptr, &water_surface_descriptor_set_layout_) !=
            VK_SUCCESS)
            return false;
    }

    if (water_descriptor_pool_ == VK_NULL_HANDLE)
    {
        constexpr std::uint32_t maximum_water_descriptor_sets = 4096u;
        const std::array pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, maximum_water_descriptor_sets * 5u},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maximum_water_descriptor_sets}};
        const VkDescriptorPoolCreateInfo pool{
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,     nullptr,
            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, maximum_water_descriptor_sets,
            static_cast<std::uint32_t>(pool_sizes.size()),     pool_sizes.data()};
        if (vkCreateDescriptorPool(device_, &pool, nullptr, &water_descriptor_pool_) != VK_SUCCESS) return false;
    }

    if (water_compute_pipeline_layout_ == VK_NULL_HANDLE)
    {
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(water_fft_push_constants)};
        const VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0u,   1u,
                                                &water_compute_descriptor_set_layout_,         1u,      &push};
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &water_compute_pipeline_layout_) != VK_SUCCESS)
            return false;
    }

    const auto create_compute_pipeline =
        [&](const std::uint32_t* words, std::size_t word_count, VkPipeline& destination)
    {
        if (destination != VK_NULL_HANDLE) return true;
        const auto shader = create_shader_module(words, word_count);
        if (shader == VK_NULL_HANDLE) return false;
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                    nullptr,
                                                    0u,
                                                    VK_SHADER_STAGE_COMPUTE_BIT,
                                                    shader,
                                                    "main",
                                                    nullptr};
        const VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                                   nullptr,
                                                   0u,
                                                   stage,
                                                   water_compute_pipeline_layout_,
                                                   VK_NULL_HANDLE,
                                                   0};
        const auto result = vkCreateComputePipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr, &destination);
        vkDestroyShaderModule(device_, shader, nullptr);
        return result == VK_SUCCESS;
    };

    return create_compute_pipeline(builtin::water_spectrum_update_comp_spv,
                                   std::size(builtin::water_spectrum_update_comp_spv), water_spectrum_pipeline_) &&
           create_compute_pipeline(builtin::water_fft_bit_reverse_comp_spv,
                                   std::size(builtin::water_fft_bit_reverse_comp_spv),
                                   water_fft_bit_reverse_pipeline_) &&
           create_compute_pipeline(builtin::water_fft_stage_comp_spv, std::size(builtin::water_fft_stage_comp_spv),
                                   water_fft_stage_pipeline_) &&
           create_compute_pipeline(builtin::water_surface_finalize_comp_spv,
                                   std::size(builtin::water_surface_finalize_comp_spv),
                                   water_surface_finalize_pipeline_);
}

bool vulkan_render_backend::ensure_water_surface_pipeline()
{
    if (water_surface_pipeline_ != VK_NULL_HANDLE) return true;
    if (water_surface_descriptor_set_layout_ == VK_NULL_HANDLE || !ensure_mesh_pipeline()) return false;

    if (water_surface_pipeline_layout_ == VK_NULL_HANDLE)
    {
        const std::array set_layouts{white_descriptor_set_layout_, water_surface_descriptor_set_layout_};
        const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0u,
                                       sizeof(mesh_push_constants)};
        const VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                nullptr,
                                                0u,
                                                static_cast<std::uint32_t>(set_layouts.size()),
                                                set_layouts.data(),
                                                1u,
                                                &push};
        if (vkCreatePipelineLayout(device_, &layout, nullptr, &water_surface_pipeline_layout_) != VK_SUCCESS)
            return false;
    }

    const auto vertex_shader =
        create_shader_module(builtin::water_surface_vert_spv, std::size(builtin::water_surface_vert_spv));
    const auto fragment_shader =
        create_shader_module(builtin::default_phong_frag_spv, std::size(builtin::default_phong_frag_spv));
    if (vertex_shader == VK_NULL_HANDLE || fragment_shader == VK_NULL_HANDLE)
    {
        if (vertex_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, vertex_shader, nullptr);
        if (fragment_shader != VK_NULL_HANDLE) vkDestroyShaderModule(device_, fragment_shader, nullptr);
        return false;
    }
    const std::array stages{
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
                                        VK_SHADER_STAGE_VERTEX_BIT, vertex_shader, "main", nullptr},
        VkPipelineShaderStageCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0u,
                                        VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader, "main", nullptr}};
    const VkVertexInputBindingDescription binding{0u, sizeof(mesh_vertex), VK_VERTEX_INPUT_RATE_VERTEX};
    const std::array attributes{
        VkVertexInputAttributeDescription{0u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, position)},
        VkVertexInputAttributeDescription{1u, 0u, VK_FORMAT_R32G32B32_SFLOAT, offsetof(mesh_vertex, normal)},
        VkVertexInputAttributeDescription{2u, 0u, VK_FORMAT_R32G32_SFLOAT, offsetof(mesh_vertex, texcoord)},
        VkVertexInputAttributeDescription{3u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, color)},
        VkVertexInputAttributeDescription{4u, 0u, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(mesh_vertex, tangent)}};
    const VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                                            nullptr,
                                                            0u,
                                                            1u,
                                                            &binding,
                                                            static_cast<std::uint32_t>(attributes.size()),
                                                            attributes.data()};
    const VkPipelineInputAssemblyStateCreateInfo input_assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, nullptr, 0u, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        VK_FALSE};
    const VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0u, 1u, nullptr, 1u, nullptr};
    const VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                        nullptr,
                                                        0u,
                                                        VK_FALSE,
                                                        VK_FALSE,
                                                        VK_POLYGON_MODE_FILL,
                                                        VK_CULL_MODE_NONE,
                                                        VK_FRONT_FACE_COUNTER_CLOCKWISE,
                                                        VK_FALSE,
                                                        0.0f,
                                                        0.0f,
                                                        0.0f,
                                                        1.0f};
    const VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                           nullptr,
                                                           0u,
                                                           VK_SAMPLE_COUNT_1_BIT,
                                                           VK_FALSE,
                                                           0.0f,
                                                           nullptr,
                                                           VK_FALSE,
                                                           VK_FALSE};
    const VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                      nullptr,
                                                      0u,
                                                      VK_TRUE,
                                                      VK_FALSE,
                                                      VK_COMPARE_OP_LESS_OR_EQUAL,
                                                      VK_FALSE,
                                                      VK_FALSE,
                                                      {},
                                                      {},
                                                      0.0f,
                                                      1.0f};
    const VkPipelineColorBlendAttachmentState color_attachment{VK_TRUE,
                                                               VK_BLEND_FACTOR_SRC_ALPHA,
                                                               VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                               VK_BLEND_OP_ADD,
                                                               VK_BLEND_FACTOR_ONE,
                                                               VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                                               VK_BLEND_OP_ADD,
                                                               VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};
    const VkPipelineColorBlendStateCreateInfo color_blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                          nullptr,
                                                          0u,
                                                          VK_FALSE,
                                                          VK_LOGIC_OP_COPY,
                                                          1u,
                                                          &color_attachment,
                                                          {0.0f, 0.0f, 0.0f, 0.0f}};
    const std::array dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0u,
                                                   static_cast<std::uint32_t>(dynamic_states.size()),
                                                   dynamic_states.data()};
    const VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                  nullptr,
                                                  0u,
                                                  1u,
                                                  &scene_color_format_,
                                                  depth_format_,
                                                  VK_FORMAT_UNDEFINED};
    const VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                                                &rendering,
                                                0u,
                                                static_cast<std::uint32_t>(stages.size()),
                                                stages.data(),
                                                &vertex_input,
                                                &input_assembly,
                                                nullptr,
                                                &viewport,
                                                &raster,
                                                &multisample,
                                                &depth,
                                                &color_blend,
                                                &dynamic,
                                                water_surface_pipeline_layout_,
                                                VK_NULL_HANDLE,
                                                0u,
                                                VK_NULL_HANDLE,
                                                -1};
    const auto result =
        vkCreateGraphicsPipelines(device_, vk_pipeline_cache_, 1u, &pipeline, nullptr, &water_surface_pipeline_);
    vkDestroyShaderModule(device_, vertex_shader, nullptr);
    vkDestroyShaderModule(device_, fragment_shader, nullptr);
    return result == VK_SUCCESS;
}

void vulkan_render_backend::destroy_ocean_simulation(gpu_ocean_simulation& simulation) noexcept
{
    std::vector<VkDescriptorSet> descriptors;
    descriptors.reserve(simulation.profile.cascade_count * 5u + 1u);
    for (auto& cascade : simulation.cascades)
    {
        destroy_buffer(cascade.initial_spectrum);
        destroy_buffer(cascade.frequency_a);
        destroy_buffer(cascade.frequency_b);
        destroy_buffer(cascade.surface);
        for (const auto descriptor :
             {cascade.spectrum_descriptor, cascade.bit_reverse_descriptor, cascade.fft_ab_descriptor,
              cascade.fft_ba_descriptor, cascade.finalize_descriptor})
            if (descriptor != VK_NULL_HANDLE) descriptors.push_back(descriptor);
        cascade = {};
    }
    destroy_buffer(simulation.surface_metadata);
    if (simulation.surface_descriptor != VK_NULL_HANDLE) descriptors.push_back(simulation.surface_descriptor);
    if (water_descriptor_pool_ != VK_NULL_HANDLE && !descriptors.empty())
        vkFreeDescriptorSets(device_, water_descriptor_pool_, static_cast<std::uint32_t>(descriptors.size()),
                             descriptors.data());
    simulation = {};
}

bool vulkan_render_backend::synchronize_water_simulations(std::uint64_t frame_index)
{
    auto& profile = last_profile_.water;
    profile = {};
    if (frame_waters_.empty())
    {
        const bool has_expired = std::ranges::any_of(ocean_simulations_, [frame_index](const auto& entry)
                                                     { return frame_index > entry.second.last_seen_frame + 4u; });
        if (!has_expired) return true;
        wait_for_in_flight_frames();
        for (auto iterator = ocean_simulations_.begin(); iterator != ocean_simulations_.end();)
        {
            if (frame_index <= iterator->second.last_seen_frame + 4u)
            {
                ++iterator;
                continue;
            }
            destroy_ocean_simulation(iterator->second);
            iterator = ocean_simulations_.erase(iterator);
        }
        return true;
    }
    profile.enabled = true;
    profile.deterministic_initial_spectrum = true;
    if (!ensure_water_compute_resources() || !ensure_water_surface_pipeline())
    {
        profile.fallback_reason = "Vulkan spectral Water compute pipelines are unavailable; using the flat W0 surface";
        return false;
    }

    const auto allocate_descriptor = [&](VkDescriptorSetLayout layout, VkDescriptorSet& destination)
    {
        const VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                     water_descriptor_pool_, 1u, &layout};
        return vkAllocateDescriptorSets(device_, &allocation, &destination) == VK_SUCCESS;
    };
    const auto write_pair = [&](VkDescriptorSet descriptor, VkBuffer first, VkBuffer second)
    {
        const std::array infos{VkDescriptorBufferInfo{first, 0u, VK_WHOLE_SIZE},
                               VkDescriptorBufferInfo{second, 0u, VK_WHOLE_SIZE}};
        std::array<VkWriteDescriptorSet, 2u> writes{};
        for (std::uint32_t binding = 0u; binding < writes.size(); ++binding)
        {
            writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[binding].dstSet = descriptor;
            writes[binding].dstBinding = binding;
            writes[binding].descriptorCount = 1u;
            writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[binding].pBufferInfo = &infos[binding];
        }
        vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    };

    bool succeeded = true;
    bool waited_for_rebuild{};
    std::unordered_set<std::uint64_t> active_keys;
    for (const auto& instance : frame_waters_)
    {
        if (instance.type != water::water_body_type::ocean) continue;
        const auto key = water_object_key(instance.object_id);
        active_keys.insert(key);
        const auto signature = water_settings_signature(instance);
        auto [iterator, inserted] = ocean_simulations_.try_emplace(key);
        auto& simulation = iterator->second;
        if (!inserted && simulation.settings_signature != signature)
        {
            if (!waited_for_rebuild)
            {
                wait_for_in_flight_frames();
                waited_for_rebuild = true;
            }
            destroy_ocean_simulation(simulation);
            inserted = true;
        }
        if (inserted)
        {
            simulation.instance = instance;
            simulation.profile = water::ocean_profile(instance.settings.quality);
            simulation.settings_signature = signature;
            bool initialized = true;
            for (std::uint32_t cascade_index = 0u; cascade_index < simulation.profile.cascade_count; ++cascade_index)
            {
                auto& cascade = simulation.cascades[cascade_index];
                const auto& descriptor = simulation.profile.cascades[cascade_index];
                const auto initial = water::initialize_ocean_spectrum(
                    water::make_ocean_spectrum_parameters(instance.settings.simulation), descriptor, cascade_index);
                std::vector<gpu_complex> packed_initial(initial.size());
                std::ranges::transform(initial, packed_initial.begin(),
                                       [](const auto& value) { return gpu_complex{value.real(), value.imag()}; });
                const VkDeviceSize sample_count =
                    static_cast<VkDeviceSize>(descriptor.resolution) * descriptor.resolution;
                const VkDeviceSize frequency_bytes = sample_count * frequency_sample_size;
                const VkDeviceSize surface_bytes = sample_count * surface_sample_size;
                initialized =
                    initialized && !packed_initial.empty() &&
                    upload_buffer(packed_initial.data(), packed_initial.size() * sizeof(gpu_complex),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, cascade.initial_spectrum) &&
                    create_buffer(frequency_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                  cascade.frequency_a) &&
                    create_buffer(frequency_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                  cascade.frequency_b) &&
                    create_buffer(surface_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                                  cascade.surface) &&
                    allocate_descriptor(water_compute_descriptor_set_layout_, cascade.spectrum_descriptor) &&
                    allocate_descriptor(water_compute_descriptor_set_layout_, cascade.bit_reverse_descriptor) &&
                    allocate_descriptor(water_compute_descriptor_set_layout_, cascade.fft_ab_descriptor) &&
                    allocate_descriptor(water_compute_descriptor_set_layout_, cascade.fft_ba_descriptor) &&
                    allocate_descriptor(water_compute_descriptor_set_layout_, cascade.finalize_descriptor);
                if (!initialized) break;
                write_pair(cascade.spectrum_descriptor, cascade.initial_spectrum.buffer, cascade.frequency_a.buffer);
                write_pair(cascade.bit_reverse_descriptor, cascade.frequency_a.buffer, cascade.frequency_b.buffer);
                write_pair(cascade.fft_ab_descriptor, cascade.frequency_a.buffer, cascade.frequency_b.buffer);
                write_pair(cascade.fft_ba_descriptor, cascade.frequency_b.buffer, cascade.frequency_a.buffer);
                write_pair(cascade.finalize_descriptor, cascade.frequency_b.buffer, cascade.surface.buffer);
            }

            gpu_water_surface_metadata metadata{};
            metadata.cascade_count = simulation.profile.cascade_count;
            for (std::uint32_t index = 0u; index < simulation.profile.cascade_count; ++index)
            {
                metadata.resolutions[index] = simulation.profile.cascades[index].resolution;
                metadata.physical_lengths[index] = simulation.profile.cascades[index].physical_length;
                metadata.full_detail_distances[index] = simulation.profile.cascades[index].full_detail_distance;
                metadata.fade_out_distances[index] = simulation.profile.cascades[index].fade_out_distance;
            }
            initialized = initialized &&
                          upload_buffer(&metadata, sizeof(metadata), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        simulation.surface_metadata) &&
                          allocate_descriptor(water_surface_descriptor_set_layout_, simulation.surface_descriptor);
            if (initialized)
            {
                std::array<VkDescriptorBufferInfo, 5u> infos{};
                infos[0] = {simulation.surface_metadata.buffer, 0u, VK_WHOLE_SIZE};
                for (std::uint32_t index = 0u; index < water::maximum_ocean_cascades; ++index)
                {
                    const auto source = index < simulation.profile.cascade_count ? index : 0u;
                    infos[index + 1u] = {simulation.cascades[source].surface.buffer, 0u, VK_WHOLE_SIZE};
                }
                std::array<VkWriteDescriptorSet, 5u> writes{};
                for (std::uint32_t binding = 0u; binding < writes.size(); ++binding)
                {
                    writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    writes[binding].dstSet = simulation.surface_descriptor;
                    writes[binding].dstBinding = binding;
                    writes[binding].descriptorCount = 1u;
                    writes[binding].descriptorType =
                        binding == 0u ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[binding].pBufferInfo = &infos[binding];
                }
                vkUpdateDescriptorSets(device_, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);
                simulation.ready = true;
            }
            else
            {
                destroy_ocean_simulation(simulation);
                simulation.settings_signature = signature ^ std::numeric_limits<std::uint64_t>::max();
                succeeded = false;
                profile.fallback_reason = "Vulkan could not allocate spectral Water fields; using the flat W0 surface";
            }
        }

        simulation.instance = instance;
        simulation.last_seen_frame = frame_index;
        if (!simulation.ready) continue;
        ++profile.active_body_count;
        profile.active_cascade_count += simulation.profile.cascade_count;
        profile.update_interval_frames =
            std::max(profile.update_interval_frames, simulation.profile.update_interval_frames);
        profile.foam_update_interval_frames =
            std::max(profile.foam_update_interval_frames, simulation.profile.foam_update_interval_frames);
        profile.foam_history = profile.foam_history || simulation.instance.settings.foam.enabled;
        profile.simulation_memory_bytes += sizeof(gpu_water_surface_metadata);
        for (std::uint32_t index = 0u; index < simulation.profile.cascade_count; ++index)
        {
            profile.maximum_resolution =
                std::max(profile.maximum_resolution, simulation.profile.cascades[index].resolution);
            const auto resolution = static_cast<std::uint64_t>(simulation.profile.cascades[index].resolution);
            const auto sample_count = resolution * resolution;
            profile.simulation_memory_bytes +=
                sample_count * (sizeof(gpu_complex) + frequency_sample_size * 2u + surface_sample_size);
        }
    }

    for (auto iterator = ocean_simulations_.begin(); iterator != ocean_simulations_.end();)
    {
        if (active_keys.contains(iterator->first) || frame_index <= iterator->second.last_seen_frame + 4u)
        {
            ++iterator;
            continue;
        }
        if (!waited_for_rebuild)
        {
            wait_for_in_flight_frames();
            waited_for_rebuild = true;
        }
        destroy_ocean_simulation(iterator->second);
        iterator = ocean_simulations_.erase(iterator);
    }
    profile.gpu_simulation = profile.active_body_count > 0u;
    return succeeded;
}

void vulkan_render_backend::dispatch_water_spectrum_update(VkCommandBuffer command_buffer)
{
    if (water_spectrum_pipeline_ == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_spectrum_pipeline_);
    bool dispatched{};
    for (auto& [_, simulation] : ocean_simulations_)
    {
        if (!simulation.ready || simulation.last_seen_frame != last_profile_.frame_index) continue;
        const auto interval = std::max(1u, simulation.profile.update_interval_frames);
        if (simulation.last_update_frame != std::numeric_limits<std::uint64_t>::max() &&
            last_profile_.frame_index - simulation.last_update_frame < interval)
            continue;
        for (std::uint32_t index = 0u; index < simulation.profile.cascade_count; ++index)
        {
            const auto& descriptor = simulation.profile.cascades[index];
            const auto& cascade = simulation.cascades[index];
            const water_spectrum_push_constants constants{descriptor.resolution, descriptor.physical_length,
                                                          simulation.instance.settings.simulation.choppiness,
                                                          static_cast<float>(frame_simulation_time_seconds_)};
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_compute_pipeline_layout_, 0u,
                                    1u, &cascade.spectrum_descriptor, 0u, nullptr);
            vkCmdPushConstants(command_buffer, water_compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                               sizeof(constants), &constants);
            vkCmdDispatch(command_buffer, (descriptor.resolution + 7u) / 8u, (descriptor.resolution + 7u) / 8u, 1u);
            ++last_profile_.water.compute_dispatch_count;
            dispatched = true;
        }
    }
    if (dispatched) water_compute_barrier(command_buffer);
}

void vulkan_render_backend::dispatch_water_inverse_fft(VkCommandBuffer command_buffer)
{
    if (water_fft_bit_reverse_pipeline_ == VK_NULL_HANDLE || water_fft_stage_pipeline_ == VK_NULL_HANDLE ||
        water_surface_finalize_pipeline_ == VK_NULL_HANDLE)
        return;
    bool transformed{};
    for (auto& [_, simulation] : ocean_simulations_)
    {
        if (!simulation.ready || simulation.last_seen_frame != last_profile_.frame_index) continue;
        const auto interval = std::max(1u, simulation.profile.update_interval_frames);
        if (simulation.last_update_frame != std::numeric_limits<std::uint64_t>::max() &&
            last_profile_.frame_index - simulation.last_update_frame < interval)
            continue;
        for (std::uint32_t index = 0u; index < simulation.profile.cascade_count; ++index)
        {
            const auto& descriptor = simulation.profile.cascades[index];
            const auto& cascade = simulation.cascades[index];
            water_fft_push_constants constants{.resolution = descriptor.resolution};
            const auto groups = (descriptor.resolution + 7u) / 8u;
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_fft_bit_reverse_pipeline_);
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_compute_pipeline_layout_, 0u,
                                    1u, &cascade.bit_reverse_descriptor, 0u, nullptr);
            vkCmdPushConstants(command_buffer, water_compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                               sizeof(constants), &constants);
            vkCmdDispatch(command_buffer, groups, groups, 1u);
            ++last_profile_.water.compute_dispatch_count;
            water_compute_barrier(command_buffer);

            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_fft_stage_pipeline_);
            bool input_is_b = true;
            const auto stage_count = static_cast<std::uint32_t>(std::countr_zero(descriptor.resolution));
            for (std::uint32_t direction = 0u; direction < 2u; ++direction)
                for (std::uint32_t stage = 0u; stage < stage_count; ++stage)
                {
                    constants.stage = stage;
                    constants.direction = direction;
                    const auto fft_descriptor = input_is_b ? cascade.fft_ba_descriptor : cascade.fft_ab_descriptor;
                    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                            water_compute_pipeline_layout_, 0u, 1u, &fft_descriptor, 0u, nullptr);
                    vkCmdPushConstants(command_buffer, water_compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                                       sizeof(constants), &constants);
                    vkCmdDispatch(command_buffer, groups, groups, 1u);
                    ++last_profile_.water.compute_dispatch_count;
                    water_compute_barrier(command_buffer);
                    input_is_b = !input_is_b;
                }

            transformed = true;
        }
    }
    if (transformed) water_compute_barrier(command_buffer);
}

void vulkan_render_backend::dispatch_water_foam_update(VkCommandBuffer command_buffer)
{
    if (water_surface_finalize_pipeline_ == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_surface_finalize_pipeline_);
    bool finalized{};
    for (auto& [_, simulation] : ocean_simulations_)
    {
        if (!simulation.ready || simulation.last_seen_frame != last_profile_.frame_index) continue;
        const auto simulation_interval = std::max(1u, simulation.profile.update_interval_frames);
        if (simulation.last_update_frame != std::numeric_limits<std::uint64_t>::max() &&
            last_profile_.frame_index - simulation.last_update_frame < simulation_interval)
            continue;

        const auto foam_interval = std::max(1u, simulation.profile.foam_update_interval_frames);
        const bool generate_foam = simulation.last_foam_update_frame == std::numeric_limits<std::uint64_t>::max() ||
                                   last_profile_.frame_index - simulation.last_foam_update_frame >= foam_interval;
        const double elapsed =
            simulation.foam_initialized
                ? std::max(0.0, frame_simulation_time_seconds_ - simulation.last_surface_time_seconds)
                : 0.0;
        const float retention =
            simulation.foam_initialized
                ? std::exp(-std::max(0.0f, simulation.instance.settings.foam.decay) * static_cast<float>(elapsed))
                : 0.0f;
        std::uint32_t configuration = simulation.instance.settings.foam.enabled ? 1u : 0u;
        if (simulation.foam_initialized) configuration |= 2u;
        if (generate_foam) configuration |= 4u;

        for (std::uint32_t index = 0u; index < simulation.profile.cascade_count; ++index)
        {
            const auto& descriptor = simulation.profile.cascades[index];
            const auto& cascade = simulation.cascades[index];
            const water_foam_push_constants constants{
                descriptor.resolution, std::clamp(simulation.instance.settings.foam.threshold, 0.0f, 1.0f), retention,
                configuration};
            vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, water_compute_pipeline_layout_, 0u,
                                    1u, &cascade.finalize_descriptor, 0u, nullptr);
            vkCmdPushConstants(command_buffer, water_compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                               sizeof(constants), &constants);
            vkCmdDispatch(command_buffer, (descriptor.resolution + 7u) / 8u, (descriptor.resolution + 7u) / 8u, 1u);
            ++last_profile_.water.compute_dispatch_count;
            ++last_profile_.water.foam_dispatch_count;
            finalized = true;
        }
        if (generate_foam) simulation.last_foam_update_frame = last_profile_.frame_index;
        simulation.last_update_frame = last_profile_.frame_index;
        simulation.last_surface_time_seconds = frame_simulation_time_seconds_;
        simulation.foam_initialized = true;
    }
    if (finalized)
    {
        const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                      VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0u, 1u, &barrier, 0u, nullptr, 0u, nullptr);
    }
}

const vulkan_render_backend::gpu_ocean_simulation*
vulkan_render_backend::water_simulation_for(render_object_id object) const noexcept
{
    const auto found = ocean_simulations_.find(water_object_key(object));
    return found != ocean_simulations_.end() && found->second.ready ? &found->second : nullptr;
}

void vulkan_render_backend::destroy_water_resources() noexcept
{
    for (auto& [_, simulation] : ocean_simulations_)
        destroy_ocean_simulation(simulation);
    ocean_simulations_.clear();
    if (water_surface_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, water_surface_pipeline_, nullptr);
    if (water_surface_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, water_surface_pipeline_layout_, nullptr);
    if (water_spectrum_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, water_spectrum_pipeline_, nullptr);
    if (water_fft_bit_reverse_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, water_fft_bit_reverse_pipeline_, nullptr);
    if (water_fft_stage_pipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(device_, water_fft_stage_pipeline_, nullptr);
    if (water_surface_finalize_pipeline_ != VK_NULL_HANDLE)
        vkDestroyPipeline(device_, water_surface_finalize_pipeline_, nullptr);
    if (water_compute_pipeline_layout_ != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device_, water_compute_pipeline_layout_, nullptr);
    if (water_descriptor_pool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, water_descriptor_pool_, nullptr);
    if (water_compute_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, water_compute_descriptor_set_layout_, nullptr);
    if (water_surface_descriptor_set_layout_ != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device_, water_surface_descriptor_set_layout_, nullptr);
    water_surface_pipeline_ = VK_NULL_HANDLE;
    water_surface_pipeline_layout_ = VK_NULL_HANDLE;
    water_spectrum_pipeline_ = VK_NULL_HANDLE;
    water_fft_bit_reverse_pipeline_ = VK_NULL_HANDLE;
    water_fft_stage_pipeline_ = VK_NULL_HANDLE;
    water_surface_finalize_pipeline_ = VK_NULL_HANDLE;
    water_compute_pipeline_layout_ = VK_NULL_HANDLE;
    water_descriptor_pool_ = VK_NULL_HANDLE;
    water_compute_descriptor_set_layout_ = VK_NULL_HANDLE;
    water_surface_descriptor_set_layout_ = VK_NULL_HANDLE;
}

} // namespace arc::render::vulkan::backend_detail

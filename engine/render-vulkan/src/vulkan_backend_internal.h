#pragma once

#include <arc/render/vulkan/vulkan_backend.h>

#include <arc/diagnostics/log.h>
#include <arc/render/lighting.h>
#include <arc/render/material_pass.h>
#include <arc/render/render_world.h>
#include <arc/render/resources.h>
#include <arc/render/virtual_shadow.h>

#include "vulkan_swapchain.h"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

#ifndef ARC_VULKAN_SHARED_VIEWPORT
#if defined(_WIN32) && defined(ARC_EDITOR) && ARC_EDITOR
#define ARC_VULKAN_SHARED_VIEWPORT 1
#else
#define ARC_VULKAN_SHARED_VIEWPORT 0
#endif
#endif

#if ARC_VULKAN_SHARED_VIEWPORT
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#endif

namespace arc::render::vulkan::backend_detail
{
constexpr std::uint32_t material_shadow_binding = 5u;
constexpr std::uint32_t material_shadow_data_binding = 6u;
constexpr std::uint32_t terrain_normal_binding = 7u;
constexpr std::uint32_t terrain_surface_binding = 11u;
constexpr std::uint32_t material_light_data_binding = 15u;
constexpr std::uint32_t material_parameters_binding = 16u;
constexpr std::uint32_t material_local_shadow_binding = 17u;
constexpr std::uint32_t material_binding_count = 18u;
constexpr std::uint32_t material_descriptor_set_capacity = 12288u;
constexpr std::uint32_t directional_shadow_layer_count = directional_shadow_cascade_count * 2u;
constexpr VkDeviceSize upload_staging_capacity = 64u * 1024u * 1024u;
constexpr std::array<std::uint32_t, 15> material_image_bindings{0u,
                                                                1u,
                                                                2u,
                                                                3u,
                                                                4u,
                                                                material_shadow_binding,
                                                                terrain_normal_binding + 0u,
                                                                terrain_normal_binding + 1u,
                                                                terrain_normal_binding + 2u,
                                                                terrain_normal_binding + 3u,
                                                                terrain_surface_binding + 0u,
                                                                terrain_surface_binding + 1u,
                                                                terrain_surface_binding + 2u,
                                                                terrain_surface_binding + 3u,
                                                                material_local_shadow_binding};

const char* vk_result_name(VkResult result) noexcept;
std::string describe_vk_result(VkResult result);
void cmd_begin_rendering(VkCommandBuffer command_buffer, const VkRenderingInfo* rendering);
void cmd_end_rendering(VkCommandBuffer command_buffer);
std::uint64_t resource_key(resource_handle handle) noexcept;
std::uint64_t gpu_scene_key(gpu_scene_instance_handle handle) noexcept;
VkDeviceSize buffer_size(std::size_t count, std::size_t stride) noexcept;
math::vector3f matrix_translation(const math::matrix4f& matrix) noexcept;
math::matrix4f look_at_rh(const math::vector3f& eye, const math::vector3f& target, const math::vector3f& up) noexcept;
math::matrix4f perspective_rh_zo(float vertical_fov, float near_plane, float far_plane) noexcept;

struct mesh_push_constants
{
    float model_view_projection[16]{};
    float model[16]{};
    float base_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float light_direction_intensity[4]{0.35f, -0.85f, -0.40f, 1.0f};
    float light_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float camera_position[4]{};
    float visualization[4]{};
    float fog_color_density[4]{};
    float fog_params[4]{};
    float material_params[4]{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(mesh_push_constants) == 256);

struct alignas(16) gpu_scene_transform_record
{
    float model[16]{};
    float previous_model[16]{};
};
static_assert(sizeof(gpu_scene_transform_record) == 128);

struct alignas(16) gpu_scene_visibility_record
{
    float bounds_min[4]{};
    float bounds_max[4]{};
    std::uint32_t geometry[4]{};
    std::uint32_t material_flags[4]{};
    std::uint32_t draw_metadata[4]{};
    float distance_error[2]{};
    std::uint32_t material_attribute[2]{resource_handle::invalid_index, 0u};
};
static_assert(sizeof(gpu_scene_visibility_record) == 96);

struct packed_gpu_scene_instance
{
    gpu_scene_transform_record transform;
    gpu_scene_visibility_record visibility;
};

struct alignas(16) gpu_texture_mip_demand
{
    std::uint32_t slot{};
    std::uint32_t generation{};
    std::uint32_t desired_mip{};
    std::uint32_t coverage{};
};
static_assert(sizeof(gpu_texture_mip_demand) == 16);

struct alignas(16) gpu_texture_mip_slot
{
    std::uint32_t desired_mip{std::numeric_limits<std::uint32_t>::max()};
    std::uint32_t generation{};
    std::uint32_t coverage{};
    std::uint32_t reserved{};
};
static_assert(sizeof(gpu_texture_mip_slot) == 16);

struct texture_feedback_push_constants
{
    std::uint32_t demand_count{};
    std::uint32_t slot_count{};
};
static_assert(sizeof(texture_feedback_push_constants) == 8);

struct alignas(16) gpu_visibility_push_constants
{
    float view_projection[16]{};
    float camera_position_and_error[4]{};
    std::uint32_t instance_capacity{};
    std::uint32_t render_layer_mask{~0u};
    std::uint32_t camera_cut{};
    std::uint32_t reserved{};
    float hzb_parameters[4]{};
};
static_assert(sizeof(gpu_visibility_push_constants) == 112);

inline constexpr VkDeviceSize indexed_indirect_command_stride = sizeof(VkDrawIndexedIndirectCommand);

struct alignas(16) gpu_visibility_counter_data
{
    std::uint32_t visible_count{};
    std::uint32_t frustum_rejected{};
    std::uint32_t distance_rejected{};
    std::uint32_t occlusion_rejected{};
    std::uint32_t candidate_count{};
    std::uint32_t active_bins{};
    std::uint32_t overflow_count{};
    std::uint32_t transparent_count{};
};
static_assert(sizeof(gpu_visibility_counter_data) == 32);

struct alignas(16) virtual_geometry_traversal_counter_data
{
    std::uint32_t visible_count{};
    std::uint32_t request_count{};
    std::uint32_t frustum_rejected{};
    std::uint32_t cone_rejected{};
    std::uint32_t hzb_rejected{};
    std::uint32_t projected_size_rejected{};
    std::uint32_t visible_overflow{};
    std::uint32_t request_overflow{};
    std::uint32_t fallback_instances{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t traversal_overflow{};
    std::uint32_t reserved{};
};
static_assert(sizeof(virtual_geometry_traversal_counter_data) == 48);

struct virtual_geometry_traversal_push_constants
{
    float view_projection[16]{};
    float camera_position_and_error[4]{};
    std::uint32_t capacities[4]{};
    float viewport_hzb[4]{};
    std::uint32_t hzb_generation{};
    std::uint32_t camera_cut{};
};
static_assert(sizeof(virtual_geometry_traversal_push_constants) == 120);

struct alignas(16) virtual_geometry_gpu_raster_bin
{
    std::uint32_t visible_index{};
    std::uint32_t minimum_tile_x{};
    std::uint32_t minimum_tile_y{};
    std::uint32_t maximum_tile_x{};
    std::uint32_t maximum_tile_y{};
    std::uint32_t flags{};
    std::uint32_t reserved[2]{};
};
static_assert(sizeof(virtual_geometry_gpu_raster_bin) == 32);

struct virtual_geometry_raster_push_constants
{
    float view_projection[16]{};
    std::uint32_t viewport_capacities[4]{};
};
static_assert(sizeof(virtual_geometry_raster_push_constants) == 80);

inline constexpr std::uint32_t virtual_geometry_bindless_texture_capacity = 4096u;
inline constexpr std::uint32_t material_attribute_descriptor_set_capacity = 4096u;

struct alignas(16) virtual_geometry_material_frame_data
{
    float view_projection[16]{};
    float previous_view_projection[16]{};
    std::uint32_t viewport_material_texture_debug[4]{};
};
static_assert(sizeof(virtual_geometry_material_frame_data) == 144);

struct material_uniform_data
{
    float emissive_factor[4]{0.0f, 0.0f, 0.0f, 1.0f};
    float material_lobes[4]{};
    float volume_params[4]{};
    float subsurface_color_factor[4]{1.0f, 0.35f, 0.2f, 0.0f};
    float attenuation_color[4]{1.0f, 1.0f, 1.0f, 0.0f};
};
static_assert(sizeof(material_uniform_data) == 80);

struct deferred_push_constants
{
    float inverse_view_projection[16]{};
    float camera_position[4]{};
    float light_direction_intensity[4]{0.35f, -0.85f, -0.40f, 1.0f};
    float light_color[4]{1.0f, 1.0f, 1.0f, 1.0f};
    float ambient_visualization[4]{0.18f, 0.18f, 0.18f, 0.0f};
};
static_assert(sizeof(deferred_push_constants) == 128);

struct output_transform_push_constants
{
    float exposure_output[4]{1.0f, 0.0f, 0.0f, 0.0f};
    float post_process[4]{};
};

struct histogram_push_constants
{
    float log_luminance_extent[4]{-12.0f, 16.0f, 1.0f, 1.0f};
};

struct exposure_resolve_push_constants
{
    float log_range_percentiles[4]{-12.0f, 16.0f, 0.005f, 0.98f};
    float limits_speeds[4]{-8.0f, 20.0f, 3.0f, 1.0f};
    float timing_mode[4]{1.0f / 60.0f, 1.0f, 10.0f, 0.0f};
};

inline constexpr VkDeviceSize exposure_histogram_bytes = sizeof(std::uint32_t) * 256u;
inline constexpr VkDeviceSize exposure_buffer_bytes = exposure_histogram_bytes + sizeof(std::uint32_t) * 4u;

struct shadow_uniform_data
{
    float light_view_projection[directional_shadow_cascade_count][16]{};
    float cascade_splits[4]{};
    float params[4]{};
    float cascade_texel_size[4]{};
    float cascade_blend_starts[4]{};
    float configuration[4]{};
};

struct gpu_scope_record
{
    std::string name;
    std::uint32_t begin_query{};
    std::uint32_t end_query{};
};

struct graph_image
{
    VkImage image{};
    VmaAllocation allocation{};
    VkImageView view{};
    std::vector<VkImageView> mip_views;
    VkFormat format{};
    VkImageAspectFlags aspect{};
    VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t mip_levels{1};
};

struct hzb_reduce_push_constants
{
    std::int32_t destination_width{};
    std::int32_t destination_height{};
    std::int32_t source_width{};
    std::int32_t source_height{};
    std::int32_t source_mip{-1};
};

struct temporal_mask_push_constants
{
    std::int32_t width{};
    std::int32_t height{};
    std::uint32_t history_valid{};
    float disocclusion_threshold{0.01f};
    float reactive_response{1.0f};
};

struct velocity_dilation_push_constants
{
    std::int32_t width{};
    std::int32_t height{};
};

struct temporal_resolve_push_constants
{
    std::int32_t output_width{};
    std::int32_t output_height{};
    float input_width{};
    float input_height{};
    std::uint32_t history_valid{};
    float history_weight{0.9f};
};

struct sharpen_push_constants
{
    std::int32_t output_width{};
    std::int32_t output_height{};
    float strength{0.2f};
    float clamp_strength{0.25f};
};

class vulkan_render_backend final : public render_backend
{
#if ARC_VULKAN_SHARED_VIEWPORT
    struct shared_viewport_slot
    {
        VkImage image{};
        VkDeviceMemory memory{};
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        HANDLE shared_handle{};
        VkCommandPool command_pool{};
        VkCommandBuffer command_buffer{};
        VkFence fence{};
        shared_viewport_frame_state state{shared_viewport_frame_state::available};
        std::uint64_t frame_id{};
        bool initialized{};
    };

    struct shared_viewport_output
    {
        std::string id;
        std::uint64_t generation{};
        std::uint64_t next_frame_id{1};
        std::uint64_t dropped_frames{};
        std::uint32_t width{1};
        std::uint32_t height{1};
        std::uint32_t pending_width{};
        std::uint32_t pending_height{};
        bool visible{true};
        bool destroy_pending{};
        std::array<shared_viewport_slot, 3> slots;
    };
#endif

public:
    vulkan_render_backend(VkInstance instance, VkSurfaceKHR surface, VkPhysicalDevice physical_device, VkDevice device,
                          VkQueue queue, VmaAllocator allocator, std::uint32_t graphics_queue_family,
                          render_capabilities capabilities, viewport_output_type viewport_output);

    ~vulkan_render_backend() override;

    render_backend_type type() const noexcept override;

    const render_capabilities& capabilities() const noexcept override;

    void configure(const resolved_render_config& config) override;

    render_submit_result submit(const render_frame_packet& packet, const compiled_render_graph& graph) override;

    void resize_viewport(std::uint32_t width, std::uint32_t height) override;

    render_viewport_texture viewport_texture() const noexcept override;

    render_backend_frame_profile last_frame_profile() const override;

    texture_feedback_readback take_texture_feedback() override;

    std::vector<texture_stream_upload_result> take_texture_stream_upload_results() override;

    virtual_geometry_feedback_readback take_virtual_geometry_feedback() override;

    std::vector<virtual_geometry_page_upload_result> take_virtual_geometry_page_upload_results() override;

    void request_object_pick(render_object_pick_request request) override;

    render_object_pick_result last_object_pick() const override;

    void request_frame_capture(const render_frame_capture_request& request) override;

    render_frame_capture_result last_frame_capture() const override;

    surface_frame_result present_surface_frame(std::uint32_t width, std::uint32_t height) override;

    bool render_native_viewport_frame(std::uint32_t width, std::uint32_t height, std::string& message);

    void shutdown_surface_presenter() noexcept;

private:
    struct vulkan_context
    {
        VkInstance instance{};
        VkPhysicalDevice physical_device{};
        VkDevice device{};
        VkQueue graphics_queue{};
        std::uint32_t graphics_queue_family{};
        render_capabilities capabilities{};
    };

    std::uint32_t scaled_dimension(std::uint32_t value) const noexcept;

    void wait_for_in_flight_frames() const;

#if ARC_VULKAN_SHARED_VIEWPORT
    void query_shared_viewport_support();

    bool create_shared_d3d_device();

    std::uint32_t shared_memory_type(std::uint32_t type_bits) const noexcept;

    bool create_shared_output_slots(shared_viewport_output& output);

    void poll_shared_output_fences(shared_viewport_output& output);

    void wait_for_shared_output(shared_viewport_output& output);

    surface_frame_result render_shared_viewport_frame(shared_viewport_output& output, shared_viewport_slot& slot);

    void retire_shared_output(shared_viewport_output& output, bool preserve_identity) noexcept;

    void destroy_all_shared_viewports() noexcept;
#endif // ARC_VULKAN_SHARED_VIEWPORT

    struct vulkan_command_context
    {
        VkCommandPool graphics_pool{};
        VkCommandBuffer graphics_buffer{};
        VkFence fence{};
    };

    struct gpu_buffer
    {
        VkBuffer buffer{};
        VmaAllocation allocation{};
    };

    struct gpu_resource_table_buffer
    {
        gpu_buffer storage;
        std::vector<std::byte> mirror;
        std::vector<std::uint32_t> generations;
        std::vector<bool> live;
        std::uint32_t table_generation{};
        std::uint32_t element_stride{};
        std::uint32_t live_entries{};
        bool dirty{};
    };

    struct gpu_shared_geometry_buffers
    {
        gpu_buffer vertices;
        gpu_buffer indices;
        std::vector<std::byte> vertex_mirror;
        std::vector<std::byte> index_mirror;
        std::uint32_t generation{};
        bool vertices_dirty{};
        bool indices_dirty{};
    };

    struct texture_feedback_frame
    {
        gpu_buffer demands;
        gpu_buffer slots;
        VkDescriptorSet descriptor_set{};
        std::uint32_t demand_capacity{};
        std::uint32_t slot_capacity{};
        std::uint32_t submitted_slot_count{};
        std::uint64_t submitted_frame{};
    };

    struct gpu_visibility_feedback_frame
    {
        gpu_buffer counters;
        std::uint64_t submitted_frame{};
    };

    struct virtual_geometry_feedback_frame
    {
        gpu_buffer requests;
        gpu_buffer counters;
        std::uint32_t submitted_request_count{};
        std::uint64_t submitted_frame{};
    };

    struct texture_feedback_slot
    {
        texture_handle resource{};
        std::uint32_t content_generation{};
        std::uint32_t slot_generation{1};
        std::uint32_t mip_count{};
        bool active{};
    };

    struct retired_texture_feedback_slot
    {
        std::uint32_t slot{};
        std::uint64_t reuse_after_frame{};
    };

    struct virtual_texture_cache_slot
    {
        std::uint32_t page{resource_handle::invalid_index};
        std::uint32_t generation{};
        std::uint64_t reusable_after_frame{};
    };

    struct virtual_texture_physical_cache
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView view{};
        VkSampler sampler{};
        VkFormat format{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::vector<virtual_texture_cache_slot> slots;
        std::vector<std::uint32_t> free_slots;
    };

    struct debug_overlay_vertex
    {
        math::vector3f position{};
        math::vector4f color{};
    };

    struct debug_overlay_frame_buffer
    {
        gpu_buffer vertices;
        VkDeviceSize capacity{};
        std::uint32_t tested_line_offset{};
        std::uint32_t tested_line_count{};
        std::uint32_t tested_triangle_offset{};
        std::uint32_t tested_triangle_count{};
        std::uint32_t output_line_offset{};
        std::uint32_t output_line_count{};
        std::uint32_t output_triangle_offset{};
        std::uint32_t output_triangle_count{};
    };

    struct gpu_mesh
    {
        gpu_buffer vertices;
        std::vector<gpu_buffer> dynamic_vertices;
        gpu_buffer skin_vertices;
        gpu_buffer indices;
        std::vector<mesh_vertex> source_vertices;
        std::vector<mesh_skin_vertex> skin_influences;
        std::vector<mesh_vertex> pending_vertices;
        std::vector<std::uint64_t> uploaded_revisions;
        std::uint64_t vertex_revision{};
        std::uint32_t vertex_count{};
        std::uint32_t index_count{};
        bool dynamic{};
    };

    struct gpu_skin_palette
    {
        gpu_buffer current;
        gpu_buffer previous;
        std::vector<math::matrix4f> current_matrices;
        std::vector<math::matrix4f> previous_matrices;
        std::uint32_t joint_count{};
        std::uint64_t content_revision{};
    };

    struct gpu_skinned_instance
    {
        mesh_handle mesh{};
        buffer_handle palette{};
        std::uint32_t vertex_count{};
        std::vector<gpu_buffer> current_vertices;
        std::vector<gpu_buffer> previous_vertices;
        std::vector<VkDescriptorSet> descriptor_sets;
        bool cpu_fallback{};
    };

    struct gpu_virtual_mesh
    {
        gpu_buffer vertices;
        gpu_buffer indices;
        gpu_buffer resources;
        gpu_buffer nodes;
        gpu_buffer clusters_metadata;
        gpu_buffer hierarchy_children;
        gpu_buffer roots;
        gpu_buffer page_table;
        gpu_buffer page_heap;
        std::vector<virtual_mesh_cluster> clusters;
        std::vector<virtual_geometry_gpu_page_record> page_records;
        std::vector<VkDeviceSize> page_offsets;
        std::vector<std::shared_ptr<const std::vector<std::byte>>> resident_page_bytes;
        std::shared_ptr<const virtual_mesh_data> source;
        std::uint32_t resource_generation{};
        std::uint32_t index_count{};
    };

    struct virtual_cluster_draw
    {
        draw_mesh_event draw;
        virtual_mesh_handle mesh{};
        std::uint32_t cluster_index{};
    };

    struct gpu_texture
    {
        texture_handle handle{};
        texture_data data;
        streamed_texture_descriptor streaming;
        std::vector<std::shared_ptr<const std::vector<std::byte>>> streamed_mips;
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView view{};
        VkSampler sampler{};
        VkFormat format{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t mip_count{1};
        std::uint32_t mip_window_base{};
        std::uint32_t feedback_slot{resource_handle::invalid_index};
        std::uint32_t virtual_metadata_index{resource_handle::invalid_index};
        std::uint32_t virtual_page_base{resource_handle::invalid_index};
        std::uint32_t virtual_page_count{};
        bool streamable{};
    };

    struct gpu_environment
    {
        environment_descriptor data;
    };

    struct gpu_material_runtime
    {
        VkPipeline gbuffer_pipeline{};
        VkPipelineLayout pipeline_layout{};
        VkDescriptorSetLayout descriptor_set_layout{};
        VkDescriptorPool descriptor_pool{};
        std::vector<VkDescriptorSet> descriptor_sets;
        std::vector<gpu_buffer> parameter_buffers;
        std::vector<gpu_buffer> frame_buffers;
        std::uint64_t generation{};
        bool failed{};
    };

    struct gpu_material
    {
        material_descriptor data;
        std::vector<gpu_buffer> parameter_buffers;
        std::vector<VkDescriptorSet> descriptor_sets;
        gpu_material_runtime runtime;
    };

    struct folded_light_constants
    {
        math::vector3f direction{0.35f, -0.85f, -0.40f};
        math::vector3f color = math::vector3f::one;
        float intensity{1.0f};
    };

    struct vulkan_shadow_atlas
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView array_view{};
        std::array<VkImageView, directional_shadow_layer_count> cascade_views{};
        VkSampler sampler{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t resolution{};
    };

    struct vulkan_local_shadow_atlas
    {
        VkImage image{};
        VmaAllocation allocation{};
        VkImageView view{};
        VkSampler sampler{};
        VkImageLayout layout{VK_IMAGE_LAYOUT_UNDEFINED};
        std::uint32_t resolution{};
    };

    struct active_local_shadow
    {
        shadow_light_kind kind{shadow_light_kind::spot};
        shadow_atlas_allocation allocation{};
        math::vector3f position{};
        math::vector3f direction{0.0f, -1.0f, 0.0f};
        float range{1.0f};
        float outer_angle{math::pi<float> * 0.25f};
        shadow_settings settings{};
        render_mobility mobility{render_mobility::movable};
        bool redraw{true};
    };

    struct vulkan_shadow_cache
    {
        directional_shadow_cache_key last_directional_key{};
        bool has_directional_key{};
        std::uint64_t static_signature{};
        bool static_layers_valid{};
    };

    struct vulkan_virtual_shadow_resources
    {
        VkImage static_image{};
        VmaAllocation static_allocation{};
        VkImageView static_view{};
        VkImage dynamic_image{};
        VmaAllocation dynamic_allocation{};
        VkImageView dynamic_view{};
        VkSampler sampler{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkImageLayout static_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        VkImageLayout dynamic_layout{VK_IMAGE_LAYOUT_UNDEFINED};
        gpu_buffer page_table;
        gpu_buffer requests;
        gpu_buffer feedback;
        VkDeviceSize page_table_capacity{};
        std::uint32_t atlas_extent{};
        std::uint32_t physical_page_capacity{};
    };

    struct virtual_shadow_light_state
    {
        virtual_shadow_address_space_handle address_space{};
        std::uint64_t last_seen_frame{};
    };

    struct gpu_virtual_shadow_page_mapping
    {
        std::uint32_t address_space_index{};
        std::uint32_t address_space_generation{};
        std::uint32_t physical_page_index{};
        std::uint32_t physical_page_generation{};
        std::uint32_t packed_coordinate{};
        std::uint32_t flags{};
        std::uint32_t content_revision_low{};
        std::uint32_t content_revision_high{};
    };

    static_assert(sizeof(gpu_virtual_shadow_page_mapping) == 32);

    struct object_pick_readback
    {
        render_object_pick_request request{};
        std::uint64_t frame_index{};
        std::uint32_t frame_slot{};
        std::unordered_map<std::uint32_t, render_object_id> objects;
        bool active{};
    };

    struct frame_capture_readback
    {
        render_frame_capture_request request{};
        std::uint64_t frame_index{};
        std::uint32_t frame_slot{};
        render_capture_camera_state camera{};
        std::vector<render_capture_image> images;
        std::vector<VkDeviceSize> offsets;
        std::vector<render_capture_object> objects;
        std::vector<std::string> diagnostics;
        VkDeviceSize byte_size{};
        bool active{};
    };

    static math::vector4f cluster_debug_color(std::uint32_t cluster_index) noexcept;

    void append_render_world(const render_world_event& event);

    void create_support_objects();

    void destroy_support_objects() noexcept;

    void retire_completed_resources();

    void begin_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                           const std::array<float, 4>& color) const;

    void insert_debug_label(VkCommandBuffer command_buffer, std::string_view name,
                            const std::array<float, 4>& color) const;

    void end_debug_label(VkCommandBuffer command_buffer) const;

    void reset_timestamp_queries(VkCommandBuffer command_buffer);

    std::uint32_t begin_gpu_scope(VkCommandBuffer command_buffer, std::string_view name);

    void end_gpu_scope(VkCommandBuffer command_buffer, std::uint32_t end_query);

    void collect_timestamp_results();

    void collect_object_pick_result();

    void collect_frame_capture_result();

    clustered_light_grid_profile make_clustered_light_profile() const noexcept;

    static std::uint64_t light_shadow_key(render_object_id object) noexcept;

    void update_shadow_profile(std::uint64_t frame_index);

    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memory_usage, gpu_buffer& out);

    bool submit_upload_commands(VkCommandBuffer command_buffer);

    void destroy_upload_objects() noexcept;

    bool begin_upload_batch();

    upload_allocation reserve_upload(VkDeviceSize size, std::size_t alignment);

    bool flush_upload_batch();

    void destroy_buffer(gpu_buffer& value) noexcept;

    bool ensure_pick_readback_buffer();

    bool ensure_capture_readback_buffer(VkDeviceSize required_size);

    static bool capture_channel_requested(const render_frame_capture_request& request, render_capture_channel channel);

    static VkDeviceSize align_capture_offset(VkDeviceSize value) noexcept;

    static std::optional<std::pair<render_capture_format, std::uint32_t>> capture_format_for(VkFormat format);

    void record_frame_capture(VkCommandBuffer command_buffer);

    void destroy_texture(gpu_texture& value) noexcept;

    void destroy_meshes() noexcept;

    std::optional<VkFormat> vulkan_texture_format(texture_format format) const noexcept;

    bool texture_format_supported(VkFormat format) const noexcept;

    bool upload_buffer(const void* source, VkDeviceSize size, VkBufferUsageFlags usage, gpu_buffer& destination);

    bool upload_buffer_region(const void* source, VkDeviceSize size, gpu_buffer& destination, VkDeviceSize offset);

    bool upload_texture_image(const texture_data& data, gpu_texture& destination);

    void upload_mesh(const mesh_upload_event& event);

    void retire_mesh(mesh_handle handle);

    void destroy_skin_palette_buffers(gpu_skin_palette& palette) noexcept;

    void upload_skin_palette(const skin_palette_upload_event& event);

    void retire_skin_palette(buffer_handle handle);

    surface_frame_result create_viewport_output(const viewport_output_descriptor& descriptor) override;

    surface_frame_result resize_viewport_output(std::string_view viewport_id, std::uint32_t width,
                                                std::uint32_t height) override;

    surface_frame_result present_viewport_output(std::string_view viewport_id) override;

    shared_viewport_frame_result poll_viewport_output(std::string_view viewport_id) override;

    void release_viewport_frame(std::string_view viewport_id, std::uint64_t generation,
                                std::uint64_t frame_id) override;

    void set_viewport_output_visible(std::string_view viewport_id, bool visible) override;

    void destroy_viewport_output(std::string_view viewport_id) override;
    VkBuffer mesh_vertex_buffer(const gpu_mesh& mesh, gpu_scene_instance_handle instance = {}) const noexcept;

    void update_dynamic_mesh_vertices();

    void destroy_virtual_mesh_buffers(gpu_virtual_mesh& mesh) noexcept;

    void upload_virtual_mesh(const virtual_mesh_upload_event& event);

    void upload_virtual_geometry_page(const virtual_geometry_page_upload_event& event);

    void evict_virtual_geometry_page(const virtual_geometry_page_evict_event& event);

    void retire_virtual_mesh(virtual_mesh_handle handle);

    void defer_texture_release(gpu_texture texture);

    void collect_texture_feedback_slots();

    std::uint32_t allocate_texture_feedback_slot(texture_handle resource, std::uint32_t content_generation,
                                                 std::uint32_t mip_count);

    void retire_texture_feedback_slot(std::uint32_t slot);

    void retire_texture(texture_handle handle);

    void register_streamed_texture(const texture_stream_register_event& event);

    void update_gpu_texture_table_window(const gpu_texture& texture);

    bool rebuild_streamed_mip_window(gpu_texture& texture);

    void upload_streamed_texture(const texture_stream_upload_event& event);

    void evict_streamed_texture(const texture_stream_evict_event& event);

    bool update_host_visible_buffer(gpu_buffer& buffer, const void* data, VkDeviceSize bytes);

    bool ensure_virtual_texture_table_capacity();

    void register_virtual_texture(gpu_texture& texture);

    void retire_virtual_texture(gpu_texture& texture);

    bool ensure_virtual_texture_cache(texture_format source_format, std::uint32_t& cache_index);

    std::optional<std::uint32_t> allocate_virtual_texture_cache_slot(virtual_texture_physical_cache& cache);

    bool upload_virtual_texture_page(gpu_texture& texture, const texture_stream_upload& upload,
                                     texture_stream_upload_result& result);

    void destroy_virtual_texture_resources() noexcept;

    void destroy_texture_feedback_resources() noexcept;

    bool ensure_texture_feedback_pipeline();

    bool ensure_texture_feedback_frame(texture_feedback_frame& frame, std::uint32_t demand_count,
                                       std::uint32_t slot_count);

    float projected_texture_extent(const geometric::box3f& bounds) const noexcept;

    std::vector<gpu_texture_mip_demand> build_texture_mip_demands() const;

    void dispatch_texture_mip_feedback(VkCommandBuffer command_buffer);

    void collect_texture_mip_feedback(std::uint32_t frame_index);

    void upload_texture(const texture_upload_event& event);

    void upload_material(const material_upload_event& event);

    void upload_environment(const environment_upload_event& event);

    const environment_descriptor* active_environment() const noexcept;

    void update_environment_profile(const environment_descriptor* lighting_environment);

    packed_gpu_scene_instance pack_gpu_scene_instance(const gpu_scene_instance& source) const;

    static std::size_t gpu_table_offset(gpu_resource_table_kind table) noexcept;

    void apply_gpu_resource_table_update(const gpu_resource_table_update_event& event);

    bool replace_gpu_mirror_buffer(gpu_buffer& destination, std::span<const std::byte> mirror,
                                   VkBufferUsageFlags usage);

    bool flush_gpu_resource_tables();

    void destroy_gpu_resource_tables() noexcept;

    bool ensure_gpu_scene_buffer(std::uint32_t required_capacity);

    void apply_gpu_scene_update(const gpu_scene_update_event& event);

    void destroy_gpu_skinned_instance(gpu_skinned_instance& instance) noexcept;

    bool ensure_gpu_skinning_pipeline();

    bool ensure_gpu_skinned_instance(gpu_scene_instance_handle handle, mesh_handle mesh_handle_value,
                                     buffer_handle palette_handle, std::uint32_t vertex_count,
                                     gpu_skinned_instance*& result);

    bool ensure_cpu_skinned_instance(gpu_scene_instance_handle handle, mesh_handle mesh_handle_value,
                                     buffer_handle palette_handle, std::uint32_t vertex_count,
                                     gpu_skinned_instance*& result);

    void update_cpu_skinned_vertices();

    void dispatch_gpu_skinning(VkCommandBuffer command_buffer);

    void destroy_gpu_visibility_resources();

    bool rebuild_virtual_geometry_tables();

    void destroy_virtual_geometry_traversal_resources();

    bool ensure_gpu_visibility_resources();

    bool ensure_gpu_visibility_feedback_frame(gpu_visibility_feedback_frame& frame);

    void apply_gpu_visibility_statistics(const gpu_visibility_statistics& statistics);

    void collect_gpu_visibility_feedback(std::uint32_t frame_index);

    void dispatch_gpu_visibility(VkCommandBuffer command_buffer);

    bool draw_gpu_visibility_command(VkCommandBuffer command_buffer, gpu_scene_instance_handle handle) const;

    bool gpu_bindless_draw_compatible(const draw_mesh_event& draw, bool transparent) const;

    bool draw_gpu_bindless_batch(VkCommandBuffer command_buffer, bool transparent);

    bool ensure_virtual_geometry_raster_resources();

    void dispatch_virtual_geometry_raster(VkCommandBuffer command_buffer);

    bool ensure_virtual_geometry_material_resources();

    bool dispatch_virtual_geometry_material_resolve(VkCommandBuffer command_buffer);

    bool ensure_virtual_geometry_traversal_resources();

    bool ensure_virtual_geometry_feedback_frame(virtual_geometry_feedback_frame& frame);

    void dispatch_virtual_geometry_traversal(VkCommandBuffer command_buffer);

    void collect_virtual_geometry_feedback(std::uint32_t frame_index);

    void update_light_buffer();

    void warn_about_skipped_lights(const scene_lighting_data& lighting);

    static math::vector3f vector_sub(const math::vector3f& lhs, const math::vector3f& rhs) noexcept;

    static math::vector3f vector_mul(const math::vector3f& value, float scale) noexcept;

    static math::vector3f vector_add(const math::vector3f& lhs, const math::vector3f& rhs) noexcept;

    static float vector_dot(const math::vector3f& lhs, const math::vector3f& rhs) noexcept;

    static math::vector3f vector_normalize(const math::vector3f& value) noexcept;

    folded_light_constants fold_lighting_for_draw(const draw_mesh_event& draw) const noexcept;

    material_alpha_mode material_alpha_mode_for(const draw_mesh_event& draw) const noexcept;

    bool texture_ready(texture_handle handle) const noexcept;

    bool material_is_terrain(const draw_mesh_event& draw) const noexcept;

    bool material_requires_forward(const draw_mesh_event& draw) const noexcept;

    mesh_push_constants build_mesh_constants(const draw_mesh_event& draw) const;

    VkDescriptorSet material_descriptor_set_for(const draw_mesh_event& draw) const noexcept;

    VkDescriptorSet material_attribute_descriptor_set_for(texture_handle handle);

    bool draw_runtime_material_gbuffer(VkCommandBuffer command_buffer, const draw_mesh_event& draw);

    bool draw_runtime_material_gbuffer(VkCommandBuffer command_buffer, const virtual_cluster_draw& draw);

    void destroy_mesh_pipeline() noexcept;

    void destroy_white_texture() noexcept;

    VkShaderModule create_shader_module(const std::uint32_t* code, std::size_t word_count);

    VkShaderModule create_shader_module(const std::vector<std::uint8_t>& bytecode);

    void destroy_material_runtime(gpu_material_runtime& runtime) noexcept;

    bool reject_runtime_material(gpu_material& material, std::string reason);

    const material_runtime_pass* runtime_gbuffer_pass(const gpu_material& material) const noexcept;

    bool update_runtime_parameter_buffer(gpu_buffer& buffer, const material_descriptor& material,
                                         const material_runtime_program& program);

    bool update_runtime_frame_buffer(gpu_buffer& buffer);

    bool update_runtime_material_buffers(gpu_material& material);

    texture_handle runtime_texture_handle(const gpu_material& material, std::uint32_t slot) const noexcept;

    VkDescriptorType runtime_descriptor_type(shader_resource_kind kind) const noexcept;

    bool update_runtime_texture_descriptors(gpu_material& material, std::uint32_t frame_slot);

    bool create_runtime_material_descriptors(gpu_material& material, const material_runtime_pass& pass);

    bool create_runtime_gbuffer_pipeline(gpu_material& material, const material_runtime_pass& pass);

    bool ensure_runtime_gbuffer_pipeline(gpu_material& material);

    void destroy_virtual_shadow_resources(vulkan_virtual_shadow_resources& resources) noexcept;

    void retire_virtual_shadow_resources();

    bool ensure_virtual_shadow_resources();

    static std::uint64_t virtual_shadow_light_key(shadow_light_kind kind, render_object_id object) noexcept;

    void prepare_virtual_shadow_cache(std::uint64_t frame_index);

    void transition_virtual_shadow_image(VkCommandBuffer command_buffer, VkImage image, VkImageLayout& current_layout,
                                         VkImageLayout new_layout);

    void clear_virtual_shadow_render_pages(VkCommandBuffer command_buffer, virtual_shadow_page_layer layer);

    void publish_virtual_shadow_pages(VkCommandBuffer command_buffer);

    void destroy_shadow_resources() noexcept;

    void destroy_local_shadow_resources() noexcept;

    bool ensure_local_shadow_resources();

    std::uint32_t frame_resource_count() const noexcept;

    std::uint32_t current_frame_slot() const noexcept;

    bool ensure_shadow_uniform_buffers();

    gpu_buffer* current_shadow_uniform_buffer() noexcept;

    bool update_debug_overlay_buffer();

    const gpu_buffer* shadow_uniform_buffer_for_slot(std::uint32_t slot) const noexcept;

    bool ensure_shadow_resources(const shadow_settings& settings);

    material_uniform_data build_material_parameters(const material_descriptor* material) const noexcept;

    bool update_material_parameter_buffer(gpu_buffer& buffer, const material_descriptor* material);

    bool ensure_material_parameter_buffers(std::vector<gpu_buffer>& buffers, const material_descriptor* material);

    bool ensure_material_descriptor_sets(gpu_material& material);

    bool ensure_white_descriptor_sets();

    bool ensure_sky_descriptor_sets();

    VkDescriptorSet update_current_sky_descriptor_set();

    void update_material_descriptor_set(VkDescriptorSet descriptor_set, const material_descriptor* material,
                                        const gpu_buffer* material_parameters, std::uint32_t frame_slot);

    void update_material_descriptor_sets(gpu_material& material);

    void update_white_descriptor_sets();

    void update_all_material_descriptor_sets();

    void update_current_material_descriptor_sets();

    VkDescriptorSet allocate_material_descriptor_set();

    bool ensure_white_texture();

    bool ensure_mesh_pipeline();

    bool ensure_debug_overlay_pipeline();

    bool ensure_gpu_bindless_pipelines();

    bool ensure_gbuffer_pipeline();

    bool ensure_gbuffer_descriptor_set();

    void update_gbuffer_descriptor_set();

    bool ensure_deferred_pipeline();

    bool ensure_output_transform_pipeline();

    bool ensure_exposure_pipelines();

    void dispatch_exposure(VkCommandBuffer command_buffer);

    bool ensure_sky_pipeline();

    void destroy_graph_image(graph_image& image) noexcept;

    bool ensure_graph_image(graph_image& target, std::uint32_t width, std::uint32_t height, VkFormat format,
                            VkImageUsageFlags usage, VkImageAspectFlags aspect, std::uint32_t mip_levels = 1);

    void transition_graph_image(VkCommandBuffer command_buffer, graph_image& image, VkImageLayout new_layout);

    bool ensure_deferred_targets(std::uint32_t width, std::uint32_t height);

    void destroy_hzb_resources() noexcept;

    bool ensure_hzb_resources(std::uint32_t width, std::uint32_t height);

    void dispatch_hzb(VkCommandBuffer command_buffer);

    void destroy_temporal_resources() noexcept;

    bool create_temporal_pipeline(const std::uint32_t* code, std::size_t code_words, VkDescriptorSetLayout set_layout,
                                  std::uint32_t push_size, VkPipelineLayout& pipeline_layout, VkPipeline& pipeline);

    bool ensure_temporal_pipelines();

    bool ensure_temporal_resources(std::uint32_t input_width, std::uint32_t input_height, std::uint32_t output_width,
                                   std::uint32_t output_height);

    void update_temporal_descriptors(std::uint32_t generation);

    void prepare_temporal_images(VkCommandBuffer command_buffer, std::uint32_t generation);

    void dispatch_velocity_dilation(VkCommandBuffer command_buffer);

    void dispatch_temporal_masks(VkCommandBuffer command_buffer);

    void dispatch_temporal_resolve(VkCommandBuffer command_buffer);

    void dispatch_temporal_sharpen(VkCommandBuffer command_buffer);

    void ensure_viewport(std::uint32_t width, std::uint32_t height);

    void destroy_viewport() noexcept;

    void transition_viewport(VkCommandBuffer command_buffer, VkImageLayout new_layout);

    void transition_depth(VkCommandBuffer command_buffer, VkImageLayout new_layout);

    const directional_light_event* active_directional_shadow_light() const noexcept;

    void execute_compiled_graph(VkCommandBuffer command_buffer);

    void prepare_frame_gpu_resources();

    void transition_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout);

    void transition_local_shadow_atlas(VkCommandBuffer command_buffer, VkImageLayout new_layout);

    shadow_uniform_data build_shadow_uniform(const directional_light_event* light) const noexcept;

    void update_shadow_uniform(const shadow_uniform_data& data);

    bool ensure_shadow_pipeline();

    void render_shadow_maps(VkCommandBuffer command_buffer);

    void render_local_shadow_maps(VkCommandBuffer command_buffer, shadow_light_kind requested_kind);

    void set_viewport_and_scissor(VkCommandBuffer command_buffer) const;

    void draw_debug_overlay(VkCommandBuffer command_buffer, debug_overlay_depth_mode mode);

    void draw_indexed_mesh(VkCommandBuffer command_buffer, const draw_mesh_event& draw, VkPipelineLayout layout,
                           VkShaderStageFlags stages, bool gpu_culled = false, bool write_motion = false);

    void draw_indexed_virtual_cluster(VkCommandBuffer command_buffer, const virtual_cluster_draw& draw,
                                      VkPipelineLayout layout, VkShaderStageFlags stages, bool gpu_culled = false,
                                      bool write_motion = false);

    bool render_deferred_scene(VkCommandBuffer command_buffer);

    void render_viewport(VkCommandBuffer command_buffer, bool render_scene, bool render_output);

    VkInstance instance_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physical_device_{};
    VkDevice device_{};
    VkQueue queue_{};
    VmaAllocator allocator_{};
    std::uint32_t graphics_queue_family_{};
    std::uint32_t max_indirect_draw_count_{1u};
    render_capabilities capabilities_{};
    resolved_render_config resolved_config_{};
    vulkan_context context_{};
    vulkan_command_context command_context_{};
    descriptor_slot_pool descriptor_slots_;
    deferred_resource_releaser deferred_releases_;
    frame_allocator frame_arena_{256u * 1024u};
    pipeline_handle_cache pipeline_handles_;
    VkPipelineCache vk_pipeline_cache_{};
    gpu_buffer upload_staging_;
    void* upload_staging_mapped_{};
    std::unique_ptr<gpu_upload_arena> upload_arena_;
    VkCommandPool upload_command_pool_{};
    VkCommandBuffer upload_command_buffer_{};
    VkFence upload_fence_{};
    VkSemaphore upload_timeline_{};
    std::uint64_t upload_timeline_value_{};
    bool upload_timeline_enabled_{};
    std::uint64_t upload_frame_{};
    bool upload_batch_active_{};
    bool upload_batch_has_work_{};
    bool upload_batch_failed_{};
    static constexpr std::uint32_t max_timestamp_queries_{64};
    VkQueryPool timestamp_query_pool_{};
    float timestamp_period_{1.0f};
    std::uint32_t max_push_constant_bytes_{};
    bool push_constant_limit_warning_reported_{};
    bool timestamps_supported_{};
    std::uint32_t next_timestamp_query_{};
    std::vector<gpu_scope_record> timestamp_scopes_;
    render_backend_frame_profile last_profile_;
    std::uint64_t last_completed_frame_{};
    std::optional<render_object_pick_request> pending_pick_request_;
    render_object_pick_result last_pick_result_{};
    gpu_buffer pick_readback_buffer_;
    object_pick_readback in_flight_pick_;
    std::optional<render_frame_capture_request> pending_capture_request_;
    render_frame_capture_result last_capture_result_{};
    gpu_buffer capture_readback_buffer_;
    VkDeviceSize capture_readback_capacity_{};
    frame_capture_readback in_flight_capture_;
    std::vector<std::string> pending_debug_markers_;
    std::unordered_map<std::uint64_t, gpu_mesh> meshes_;
    std::unordered_map<std::uint64_t, gpu_virtual_mesh> virtual_meshes_;
    std::unordered_map<std::uint64_t, gpu_texture> textures_;
    texture_feedback_readback completed_texture_feedback_;
    std::vector<texture_stream_upload_result> frame_texture_upload_results_;
    std::vector<texture_stream_upload_result> completed_texture_upload_results_;
    virtual_geometry_feedback_readback completed_virtual_geometry_feedback_;
    std::vector<virtual_geometry_page_upload_result> frame_virtual_geometry_upload_results_;
    std::vector<virtual_geometry_page_upload_result> completed_virtual_geometry_upload_results_;
    std::vector<texture_feedback_slot> texture_feedback_slots_;
    std::vector<std::uint32_t> free_texture_feedback_slots_;
    std::vector<retired_texture_feedback_slot> retired_texture_feedback_slots_;
    std::vector<texture_feedback_frame> texture_feedback_frames_;
    VkDescriptorSetLayout texture_feedback_descriptor_set_layout_{};
    VkDescriptorPool texture_feedback_descriptor_pool_{};
    VkPipelineLayout texture_feedback_pipeline_layout_{};
    VkPipeline texture_feedback_pipeline_{};
    std::vector<virtual_texture_gpu_metadata> virtual_texture_metadata_;
    std::vector<virtual_texture_page_table_entry> virtual_texture_pages_;
    gpu_buffer virtual_texture_metadata_buffer_;
    gpu_buffer virtual_texture_page_table_buffer_;
    std::uint32_t virtual_texture_metadata_capacity_{};
    std::uint32_t virtual_texture_page_capacity_{};
    std::vector<virtual_texture_physical_cache> virtual_texture_caches_;
    std::unordered_map<std::uint32_t, std::uint32_t> virtual_texture_cache_lookup_;
    bool virtual_texture_descriptors_dirty_{true};
    std::unordered_map<std::uint64_t, gpu_material> materials_;
    std::unordered_map<std::uint64_t, gpu_skin_palette> skin_palettes_;
    std::unordered_map<std::uint64_t, gpu_skinned_instance> gpu_skinned_instances_;
    std::unordered_map<std::uint64_t, gpu_environment> environments_;
    std::unordered_set<std::uint64_t> texture_semantic_diagnostics_;
    std::vector<draw_mesh_event> frame_draws_;
    std::vector<virtual_cluster_draw> frame_virtual_draws_;
    std::vector<draw_mesh_event> frame_shadow_draws_;
    std::vector<virtual_cluster_draw> frame_virtual_shadow_draws_;
    std::vector<directional_light_event> frame_directional_lights_;
    std::vector<point_light_event> frame_point_lights_;
    std::vector<spot_light_event> frame_spot_lights_;
    std::vector<area_light_event> frame_area_lights_;
    const std::vector<area_light_event> empty_area_lights_{};
    std::vector<debug_overlay_line> frame_debug_overlay_lines_;
    std::vector<debug_overlay_triangle> frame_debug_overlay_triangles_;
    scene_lighting_data frame_lighting_;
    world_environment_data frame_environment_;
    render_camera frame_camera_;
    bool frame_camera_valid_{};
    bool frame_shadows_enabled_{true};
    bool frame_fxaa_enabled_{};
    gpu_buffer light_buffer_;
    gpu_buffer gpu_scene_visibility_buffer_;
    gpu_buffer gpu_scene_transform_buffer_;
    std::array<gpu_resource_table_buffer, 7> gpu_resource_tables_;
    gpu_shared_geometry_buffers shared_geometry_buffers_;
    std::vector<gpu_scene_visibility_record> gpu_scene_visibility_mirror_;
    std::vector<gpu_scene_transform_record> gpu_scene_transform_mirror_;
    std::uint32_t gpu_scene_capacity_{};
    gpu_buffer gpu_visibility_commands_;
    gpu_buffer gpu_visibility_counters_;
    std::uint32_t gpu_visibility_capacity_{};
    std::vector<gpu_visibility_feedback_frame> gpu_visibility_feedback_frames_;
    gpu_visibility_statistics completed_gpu_visibility_statistics_{};
    VkDescriptorSetLayout gpu_visibility_descriptor_set_layout_{};
    VkDescriptorPool gpu_visibility_descriptor_pool_{};
    VkDescriptorSet gpu_visibility_descriptor_set_{};
    VkPipelineLayout gpu_visibility_pipeline_layout_{};
    VkPipeline gpu_visibility_pipeline_{};
    VkPipeline gpu_transparent_sort_pipeline_{};
    VkDescriptorSetLayout gpu_skinning_descriptor_set_layout_{};
    VkDescriptorPool gpu_skinning_descriptor_pool_{};
    VkPipelineLayout gpu_skinning_pipeline_layout_{};
    VkPipeline gpu_skinning_pipeline_{};
    VkDescriptorSetLayout gpu_bindless_descriptor_set_layout_{};
    VkDescriptorPool gpu_bindless_descriptor_pool_{};
    VkDescriptorSet gpu_bindless_descriptor_set_{};
    VkPipelineLayout gpu_bindless_pipeline_layout_{};
    VkPipeline gpu_bindless_gbuffer_pipeline_{};
    VkPipeline gpu_bindless_transparent_pipeline_{};
    bool gpu_visibility_active_{};
    bool gpu_visibility_descriptors_dirty_{true};
    bool gpu_bindless_descriptors_dirty_{true};
    std::vector<virtual_geometry_gpu_resource_record> virtual_geometry_resource_mirror_;
    std::vector<virtual_geometry_gpu_node_record> virtual_geometry_node_mirror_;
    std::vector<virtual_geometry_gpu_cluster_record> virtual_geometry_cluster_mirror_;
    std::vector<std::uint32_t> virtual_geometry_child_mirror_;
    std::vector<std::uint32_t> virtual_geometry_root_mirror_;
    std::vector<virtual_geometry_gpu_page_record> virtual_geometry_page_mirror_;
    std::vector<std::byte> virtual_geometry_page_heap_mirror_;
    gpu_buffer virtual_geometry_resource_buffer_;
    gpu_buffer virtual_geometry_node_buffer_;
    gpu_buffer virtual_geometry_cluster_buffer_;
    gpu_buffer virtual_geometry_child_buffer_;
    gpu_buffer virtual_geometry_root_buffer_;
    gpu_buffer virtual_geometry_page_buffer_;
    gpu_buffer virtual_geometry_page_heap_buffer_;
    gpu_buffer virtual_geometry_visible_buffer_;
    gpu_buffer virtual_geometry_request_buffer_;
    gpu_buffer virtual_geometry_counter_buffer_;
    gpu_buffer virtual_geometry_raster_bin_buffer_;
    gpu_buffer virtual_geometry_material_frame_buffer_;
    graph_image virtual_geometry_encoded_depth_;
    graph_image virtual_geometry_visibility_ids_;
    std::uint32_t virtual_geometry_visible_capacity_{};
    std::uint32_t virtual_geometry_request_capacity_{};
    std::uint32_t virtual_geometry_raster_bin_capacity_{};
    std::vector<virtual_geometry_feedback_frame> virtual_geometry_feedback_frames_;
    VkDescriptorSetLayout virtual_geometry_traversal_descriptor_set_layout_{};
    VkDescriptorPool virtual_geometry_traversal_descriptor_pool_{};
    VkDescriptorSet virtual_geometry_traversal_descriptor_set_{};
    VkPipelineLayout virtual_geometry_traversal_pipeline_layout_{};
    VkPipeline virtual_geometry_traversal_pipeline_{};
    VkDescriptorSetLayout virtual_geometry_raster_descriptor_set_layout_{};
    VkDescriptorPool virtual_geometry_raster_descriptor_pool_{};
    VkDescriptorSet virtual_geometry_raster_descriptor_set_{};
    VkPipelineLayout virtual_geometry_raster_pipeline_layout_{};
    std::array<VkPipeline, 3> virtual_geometry_raster_pipelines_{};
    VkDescriptorSetLayout virtual_geometry_material_descriptor_set_layout_{};
    VkDescriptorPool virtual_geometry_material_descriptor_pool_{};
    VkDescriptorSet virtual_geometry_material_descriptor_set_{};
    VkPipelineLayout virtual_geometry_material_pipeline_layout_{};
    VkPipeline virtual_geometry_material_pipeline_{};
    bool virtual_geometry_tables_dirty_{true};
    bool virtual_geometry_traversal_descriptors_dirty_{true};
    bool virtual_geometry_raster_descriptors_dirty_{true};
    bool virtual_geometry_material_descriptors_dirty_{true};
    std::vector<gpu_buffer> shadow_uniform_buffers_;
    std::vector<debug_overlay_frame_buffer> debug_overlay_buffers_;
    std::uint32_t active_frame_index_{};
    environment_handle active_environment_;
    vulkan_shadow_atlas shadow_atlas_;
    vulkan_local_shadow_atlas local_shadow_atlas_;
    vulkan_shadow_cache shadow_cache_;
    std::unique_ptr<shadow_atlas_allocator> local_shadow_allocator_;
    vulkan_virtual_shadow_resources virtual_shadow_resources_;
    std::unique_ptr<virtual_shadow_cache> virtual_shadow_cache_;
    std::unordered_map<std::uint64_t, virtual_shadow_light_state> virtual_shadow_lights_;
    std::vector<virtual_shadow_page_mapping> pending_virtual_shadow_pages_;
    std::vector<active_local_shadow> active_local_shadows_;
    std::unordered_map<std::uint64_t, std::uint64_t> local_shadow_static_signatures_;
    std::unordered_map<std::uint64_t, std::uint64_t> static_shadow_transform_hashes_;
    std::unordered_set<std::uint64_t> reported_moved_static_objects_;
    std::uint64_t shadow_resource_revision_{1};
    bool last_static_shadow_cache_hit_{};

    VkDescriptorSetLayout white_descriptor_set_layout_{};
    VkDescriptorPool white_descriptor_pool_{};
    std::vector<VkDescriptorSet> white_descriptor_sets_;
    std::vector<gpu_buffer> white_material_parameter_buffers_;
    std::vector<VkDescriptorSet> sky_descriptor_sets_;
    VkImage white_image_{};
    VmaAllocation white_allocation_{};
    VkImageView white_view_{};
    VkSampler white_sampler_{};
    VkPipelineLayout mesh_pipeline_layout_{};
    VkDescriptorSetLayout material_attribute_descriptor_set_layout_{};
    VkDescriptorPool material_attribute_descriptor_pool_{};
    std::unordered_map<std::uint64_t, VkDescriptorSet> material_attribute_descriptor_sets_;
    VkPipelineLayout terrain_surface_pipeline_layout_{};
    VkPipeline mesh_pipeline_{};
    VkPipeline mesh_transparent_pipeline_{};
    VkPipeline mesh_wire_pipeline_{};
    VkPipeline terrain_surface_pipeline_{};
    VkPipeline gbuffer_pipeline_{};
    VkPipeline terrain_surface_gbuffer_pipeline_{};
    VkDescriptorSetLayout gbuffer_descriptor_set_layout_{};
    VkDescriptorPool gbuffer_descriptor_pool_{};
    VkDescriptorSet gbuffer_descriptor_set_{};
    VkSampler gbuffer_sampler_{};
    VkPipelineLayout deferred_pipeline_layout_{};
    VkPipeline deferred_pipeline_{};
    VkDescriptorSetLayout output_transform_descriptor_set_layout_{};
    VkDescriptorPool output_transform_descriptor_pool_{};
    VkDescriptorSet output_transform_descriptor_set_{};
    VkPipelineLayout output_transform_pipeline_layout_{};
    VkPipeline output_transform_pipeline_{};
    gpu_buffer exposure_buffer_;
    VkPipelineLayout luminance_histogram_pipeline_layout_{};
    VkPipeline luminance_histogram_pipeline_{};
    VkPipelineLayout exposure_resolve_pipeline_layout_{};
    VkPipeline exposure_resolve_pipeline_{};
    bool exposure_needs_reset_{true};
    VkPipelineLayout sky_pipeline_layout_{};
    VkPipeline sky_pipeline_{};
    VkPipelineLayout shadow_pipeline_layout_{};
    VkPipeline shadow_pipeline_{};
    VkPipelineLayout debug_overlay_pipeline_layout_{};
    VkPipeline debug_overlay_line_pipeline_{};
    VkPipeline debug_overlay_triangle_pipeline_{};
    VkPipeline debug_overlay_output_line_pipeline_{};
    VkPipeline debug_overlay_output_triangle_pipeline_{};
    bool wireframe_warning_reported_{};
    viewport_output_type configured_viewport_output_{viewport_output_type::native_window};

#if ARC_VULKAN_SHARED_VIEWPORT
    bool shared_viewport_supported_{};
    std::string shared_viewport_failure_;
    PFN_vkGetMemoryWin32HandlePropertiesKHR get_memory_win32_handle_properties_{};
    Microsoft::WRL::ComPtr<ID3D11Device> shared_d3d_device_;
    std::unordered_map<std::string, std::uint64_t> shared_viewport_generations_;
    std::unordered_map<std::string, shared_viewport_output> shared_viewports_;
#endif

    detail::vulkan_swapchain swapchain_{};
    bool native_swapchain_initialized_{};
    bool swapchain_rebuild_{};
    bool device_lost_{};
    std::uint32_t min_image_count_{2};
    VkFormat viewport_format_{VK_FORMAT_R16G16B16A16_SFLOAT};
    VkFormat scene_color_format_{VK_FORMAT_R16G16B16A16_SFLOAT};
    VkFormat depth_format_{VK_FORMAT_D32_SFLOAT};
    VkImage viewport_image_{};
    VmaAllocation viewport_allocation_{};
    VkImageView viewport_view_{};
    VkSampler viewport_sampler_{};
    VkImageLayout viewport_layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage viewport_depth_image_{};
    VmaAllocation viewport_depth_allocation_{};
    VkImageView viewport_depth_view_{};
    VkImageLayout viewport_depth_layout_{VK_IMAGE_LAYOUT_UNDEFINED};
    graph_image scene_color_{};
    graph_image gbuffer_albedo_{};
    graph_image gbuffer_normal_{};
    graph_image gbuffer_material_{};
    graph_image gbuffer_emissive_{};
    graph_image gbuffer_motion_{};
    graph_image gbuffer_object_id_{};
    graph_image selection_mask_{};
    std::array<graph_image, 2> hzb_history_{};
    VkSampler hzb_sampler_{};
    VkDescriptorSetLayout hzb_descriptor_set_layout_{};
    VkDescriptorPool hzb_descriptor_pool_{};
    std::vector<VkDescriptorSet> hzb_descriptor_sets_;
    VkPipelineLayout hzb_pipeline_layout_{};
    VkPipeline hzb_pipeline_{};
    std::uint32_t hzb_mip_count_{};
    bool hzb_history_valid_{};
    std::array<graph_image, 2> temporal_dilated_motion_{};
    std::array<graph_image, 2> temporal_reactive_{};
    std::array<graph_image, 2> temporal_disocclusion_{};
    std::array<graph_image, 2> temporal_color_history_{};
    std::array<graph_image, 2> temporal_depth_history_{};
    std::array<graph_image, 2> temporal_moments_history_{};
    std::array<graph_image, 2> temporal_confidence_history_{};
    std::array<graph_image, 2> temporal_sharpened_{};
    VkDescriptorSetLayout temporal_velocity_descriptor_layout_{};
    VkDescriptorSetLayout temporal_mask_descriptor_layout_{};
    VkDescriptorSetLayout temporal_resolve_descriptor_layout_{};
    VkDescriptorSetLayout temporal_sharpen_descriptor_layout_{};
    VkDescriptorPool temporal_descriptor_pool_{};
    std::array<VkDescriptorSet, 2> temporal_velocity_sets_{};
    std::array<VkDescriptorSet, 2> temporal_mask_sets_{};
    std::array<VkDescriptorSet, 2> temporal_resolve_sets_{};
    std::array<VkDescriptorSet, 2> temporal_sharpen_sets_{};
    VkPipelineLayout temporal_velocity_pipeline_layout_{};
    VkPipelineLayout temporal_mask_pipeline_layout_{};
    VkPipelineLayout temporal_resolve_pipeline_layout_{};
    VkPipelineLayout temporal_sharpen_pipeline_layout_{};
    VkPipeline temporal_velocity_pipeline_{};
    VkPipeline temporal_mask_pipeline_{};
    VkPipeline temporal_resolve_pipeline_{};
    VkPipeline temporal_sharpen_pipeline_{};
    std::uint32_t temporal_input_width_{};
    std::uint32_t temporal_input_height_{};
    std::uint32_t temporal_output_width_{};
    std::uint32_t temporal_output_height_{};
    bool temporal_history_valid_{};
    bool temporal_resources_initialized_{};
    VkImageView temporal_output_view_{};
    std::uint32_t viewport_width_{};
    std::uint32_t viewport_height_{};
    std::uint32_t output_viewport_width_{};
    std::uint32_t output_viewport_height_{};
};

} // namespace arc::render::vulkan::backend_detail

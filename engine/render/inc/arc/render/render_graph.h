#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

class command_encoder;
using render_pass_record_fn = void (*)(command_encoder& encoder, void* user_data);

/**
 * @brief Queue class requested by a render graph pass.
 */
enum class render_queue_type : std::uint8_t
{
    graphics,
    compute,
    transfer
};

/**
 * @brief High-level pass behavior.
 */
enum class render_pass_kind : std::uint8_t
{
    clear,
    depth_prepass,
    gbuffer,
    lighting,
    post_process,
    imgui,
    present,
    custom
};

/**
 * @brief High-level graph resource category.
 */
enum class render_resource_kind : std::uint8_t
{
    unknown,
    color_texture,
    depth_texture,
    buffer,
    swapchain_image
};

/**
 * @brief Backend-neutral formats used by render-graph resources.
 */
enum class render_format : std::uint8_t
{
    unknown,
    rgba8_unorm,
    rgba8_srgb,
    rgba16_float,
    rg16_float,
    r8_unorm,
    r32_uint,
    d24_unorm_s8_uint,
    d32_float
};

/**
 * @brief Stable reference to one logical resource in a render graph.
 */
struct render_graph_resource_handle
{
    static constexpr std::uint32_t invalid_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t index{ invalid_index };

    constexpr bool valid() const noexcept { return index != invalid_index; }
    friend constexpr bool operator==(render_graph_resource_handle, render_graph_resource_handle) noexcept = default;
};

/**
 * @brief How a graph resource extent is resolved for a view.
 */
enum class render_extent_mode : std::uint8_t
{
    absolute,
    relative_to_view
};

/**
 * @brief Backend-neutral resource usage requested by a graph pass.
 */
enum class render_resource_usage : std::uint8_t
{
    unknown,
    color_attachment,
    depth_attachment,
    sampled,
    storage,
    transfer_src,
    transfer_dst,
    vertex_buffer,
    index_buffer,
    uniform_buffer,
    storage_buffer,
    indirect_buffer,
    present
};

/**
 * @brief Attachment load behavior requested by a graph pass.
 */
enum class render_load_op : std::uint8_t
{
    load,
    clear,
    dont_care
};

/**
 * @brief Attachment store behavior requested by a graph pass.
 */
enum class render_store_op : std::uint8_t
{
    store,
    dont_care
};

/**
 * @brief Texture dimensions used by typed graph resources.
 */
struct render_extent
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{ 1 };
};

/**
 * @brief Logical graph resource declaration.
 */
struct render_graph_resource
{
    std::string name;
    render_resource_kind kind{ render_resource_kind::unknown };
    render_extent extent{};
    render_extent_mode extent_mode{ render_extent_mode::relative_to_view };
    float width_scale{ 1.0f };
    float height_scale{ 1.0f };
    render_format format{ render_format::unknown };
    std::uint32_t mip_levels{ 1 };
    std::uint32_t array_layers{ 1 };
    std::uint32_t sample_count{ 1 };
    bool imported{};
    bool persistent{};
};

/**
 * @brief Logical resource access declared by a render graph pass.
 */
struct render_resource_access
{
    render_graph_resource_handle handle{};
    // Transitional label-based access for external graph producers. Compiled
    // accesses always contain both the strong handle and canonical name.
    std::string resource;
    render_resource_kind kind{ render_resource_kind::unknown };
    render_resource_usage usage{ render_resource_usage::unknown };
    bool write{};
    render_load_op load_op{ render_load_op::load };
    render_store_op store_op{ render_store_op::store };
    float clear_color[4]{};
    float clear_depth{ 1.0f };
};

/**
 * @brief One pass declaration in the render graph.
 */
struct render_graph_pass
{
    std::string name;
    render_queue_type queue{ render_queue_type::graphics };
    render_pass_kind kind{ render_pass_kind::custom };
    std::vector<render_resource_access> reads;
    std::vector<render_resource_access> writes;
    render_pass_record_fn record{};
    void* user_data{};
};

/**
 * @brief Compiled pass metadata.
 */
struct compiled_render_pass
{
    std::uint32_t source_index{};
    std::string name;
    render_queue_type queue{ render_queue_type::graphics };
    render_pass_kind kind{ render_pass_kind::custom };
    std::vector<render_resource_access> reads;
    std::vector<render_resource_access> writes;
    render_pass_record_fn record{};
    void* user_data{};
};

/**
 * @brief Backend-neutral resource transition emitted by graph compilation.
 */
struct render_resource_transition
{
    render_graph_resource_handle handle{};
    std::string resource;
    render_resource_usage before{ render_resource_usage::unknown };
    render_resource_usage after{ render_resource_usage::unknown };
    std::uint32_t before_pass{};
    std::uint32_t after_pass{};
    render_queue_type before_queue{ render_queue_type::graphics };
    render_queue_type after_queue{ render_queue_type::graphics };
};

/**
 * @brief Lifetime and physical-allocation assignment for one logical resource.
 */
struct render_resource_lifetime
{
    render_graph_resource_handle handle{};
    std::uint32_t first_pass{ std::numeric_limits<std::uint32_t>::max() };
    std::uint32_t last_pass{};
    std::uint32_t physical_resource{};
    std::uint64_t estimated_bytes{};
};

/**
 * @brief Result of render graph compilation.
 */
struct compiled_render_graph
{
    std::vector<compiled_render_pass> passes;
    std::vector<render_graph_resource> resources;
    std::vector<render_resource_transition> transitions;
    std::vector<render_resource_lifetime> lifetimes;
};

/**
 * @brief Minimal render graph that orders passes by declared resource dependencies.
 */
class render_graph
{
public:
    /**
     * @brief Declare a graph resource and return its index.
     */
    render_graph_resource_handle add_resource(render_graph_resource resource);

    /**
     * @brief Find a declared resource by name.
     */
    const render_graph_resource* find_resource(std::string_view name) const noexcept;

    /**
     * @brief Return a declared resource by strong graph handle.
     */
    const render_graph_resource* find_resource(render_graph_resource_handle handle) const noexcept;

    /**
     * @brief Add a pass declaration and return its index.
     */
    std::uint32_t add_pass(render_graph_pass pass);

    /**
     * @brief Compile pass order and validate dependencies.
     */
    compiled_render_graph compile() const;

    /**
     * @brief Remove all pass declarations.
     */
    void clear();

    /**
     * @brief Return declared passes.
     */
    const std::vector<render_graph_pass>& passes() const noexcept;

    /**
     * @brief Return declared resources.
     */
    const std::vector<render_graph_resource>& resources() const noexcept;

private:
    std::vector<render_graph_resource> resources_;
    std::vector<render_graph_pass> passes_;
};

/**
 * @brief Build the initial clear/present render graph for viewport bring-up.
 */
render_graph make_clear_present_graph(std::string_view target_name);

/**
 * @brief Return a stable display label for a typed format.
 */
std::string_view render_format_name(render_format format) noexcept;

} // namespace arc::render

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

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
    std::string format;
    bool imported{};
    bool persistent{};
};

/**
 * @brief Logical resource access declared by a render graph pass.
 */
struct render_resource_access
{
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
};

/**
 * @brief Backend-neutral resource transition emitted by graph compilation.
 */
struct render_resource_transition
{
    std::string resource;
    render_resource_usage before{ render_resource_usage::unknown };
    render_resource_usage after{ render_resource_usage::unknown };
    std::uint32_t before_pass{};
    std::uint32_t after_pass{};
};

/**
 * @brief Result of render graph compilation.
 */
struct compiled_render_graph
{
    std::vector<compiled_render_pass> passes;
    std::vector<render_graph_resource> resources;
    std::vector<render_resource_transition> transitions;
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
    std::uint32_t add_resource(render_graph_resource resource);

    /**
     * @brief Find a declared resource by name.
     */
    const render_graph_resource* find_resource(std::string_view name) const noexcept;

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

} // namespace arc::render

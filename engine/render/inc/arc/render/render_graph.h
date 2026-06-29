#pragma once

#include <cstdint>
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
    imgui,
    present,
    custom
};

/**
 * @brief Logical resource access declared by a render graph pass.
 */
struct render_resource_access
{
    std::string resource;
    bool write{};
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
};

/**
 * @brief Result of render graph compilation.
 */
struct compiled_render_graph
{
    std::vector<compiled_render_pass> passes;
};

/**
 * @brief Minimal render graph that orders passes by declared resource dependencies.
 */
class render_graph
{
public:
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

private:
    std::vector<render_graph_pass> passes_;
};

/**
 * @brief Build the initial clear/present render graph for viewport bring-up.
 */
render_graph make_clear_present_graph(std::string_view target_name);

} // namespace arc::render

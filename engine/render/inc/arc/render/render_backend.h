#pragma once

#include <arc/render/events.h>
#include <arc/render/render_graph.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace arc::render
{

/**
 * @brief Render API backend family.
 */
enum class render_backend_type : std::uint8_t
{
    vulkan,
    d3d12,
    metal
};

/**
 * @brief Optional backend features exposed through capability queries.
 */
struct render_capabilities
{
    render_backend_type backend{ render_backend_type::vulkan };
    std::uint32_t api_major{ 1 };
    std::uint32_t api_minor{ 3 };
    bool synchronization2{};
    bool timeline_semaphores{};
    bool dynamic_rendering{};
    bool descriptor_indexing{};
    bool descriptor_buffer{};
    bool mesh_shaders{};
    bool ray_tracing{};
    bool variable_rate_shading{};
    bool fill_mode_non_solid{};
};

/**
 * @brief Abstract renderer device.
 */
class render_device
{
public:
    virtual ~render_device() = default;
};

/**
 * @brief Abstract swapchain owned by a renderer backend.
 */
class render_swapchain
{
public:
    virtual ~render_swapchain() = default;
};

/**
 * @brief Abstract GPU queue.
 */
class render_queue
{
public:
    virtual ~render_queue() = default;
};

/**
 * @brief Abstract command encoder.
 */
class command_encoder
{
public:
    virtual ~command_encoder() = default;
};

/**
 * @brief Abstract resource allocator.
 */
class resource_allocator
{
public:
    virtual ~resource_allocator() = default;
};

/**
 * @brief Abstract pipeline cache.
 */
class pipeline_cache
{
public:
    virtual ~pipeline_cache() = default;
};

/**
 * @brief Abstract shader library.
 */
class shader_library
{
public:
    virtual ~shader_library() = default;
};

/**
 * @brief Result from submitting a frame packet to a backend.
 */
struct render_submit_result
{
    bool submitted{};
    std::string message;
};

/**
 * @brief One named GPU/backend timing sample in milliseconds.
 */
struct render_pass_timing
{
    std::string name;
    double milliseconds{};
};

/**
 * @brief Backend frame profile data exposed to tools such as the editor profiler.
 */
struct render_backend_frame_profile
{
    std::uint64_t frame_index{};
    std::vector<render_pass_timing> pass_timings;
    std::string summary;
};

/**
 * @brief Opaque UI-facing texture exported by a backend.
 */
struct render_viewport_texture
{
    std::uint64_t id{};
    std::uint32_t width{};
    std::uint32_t height{};

    /**
     * @brief Return whether this texture can be shown by an editor UI.
     */
    bool valid() const noexcept
    {
        return id != 0 && width > 0 && height > 0;
    }
};

/**
 * @brief Backend-neutral render API implementation.
 */
class render_backend
{
public:
    virtual ~render_backend() = default;

    /**
     * @brief Return the backend family.
     */
    virtual render_backend_type type() const noexcept = 0;

    /**
     * @brief Return optional feature support.
     */
    virtual const render_capabilities& capabilities() const noexcept = 0;

    /**
     * @brief Submit one immutable frame packet and compiled graph.
     */
    virtual render_submit_result submit(const render_frame_packet& packet, const compiled_render_graph& graph) = 0;

    /**
     * @brief Resize the backend-owned editor/game viewport target.
     */
    virtual void resize_viewport(std::uint32_t width, std::uint32_t height);

    /**
     * @brief Return an opaque texture identifier for editor display.
     */
    virtual render_viewport_texture viewport_texture() const noexcept;

    /**
     * @brief Return the most recent backend frame profile.
     */
    virtual render_backend_frame_profile last_frame_profile() const;
};

/**
 * @brief Factory result for backend creation.
 */
struct render_backend_create_result
{
    std::unique_ptr<render_backend> backend;
    std::string message;

    /**
     * @brief Return whether a backend was created.
     */
    bool succeeded() const noexcept
    {
        return static_cast<bool>(backend);
    }
};

} // namespace arc::render

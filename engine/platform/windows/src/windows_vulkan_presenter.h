#pragma once

#include <arc/project/runtime_session.h>
#include <arc/render/renderer.h>
#include <arc/scene/cooked_material_hydration.h>
#include <arc/scene/cooked_mesh_hydration.h>
#include <arc/scene/cooked_texture_hydration.h>

#include <cstdint>
#include <memory>
#include <string>

namespace arc::platform::windows
{
/** Owns renderer-generation resources for one packaged Win32 client window. */
class windows_vulkan_presenter final
{
public:
    windows_vulkan_presenter() = default;
    ~windows_vulkan_presenter() noexcept;

    windows_vulkan_presenter(const windows_vulkan_presenter&) = delete;
    windows_vulkan_presenter& operator=(const windows_vulkan_presenter&) = delete;

    [[nodiscard]] std::string initialize(void* native_window, project::packaged_runtime_session& session);
    [[nodiscard]] std::string present(std::uint32_t width, std::uint32_t height, float delta_seconds);
    void shutdown() noexcept;

private:
    [[nodiscard]] std::string hydrate_scene();

    project::packaged_runtime_session* session_{};
    framework::runtime_world* world_{};
    std::unique_ptr<render::renderer> renderer_;
    scene::cooked_mesh_residencies mesh_residency_;
    scene::cooked_material_residencies material_residency_;
    scene::cooked_texture_residencies texture_residency_;
    render::texture_handle primitive_texture_{};
    render::material_handle primitive_material_{};
    std::uint64_t frame_index_{};
};
} // namespace arc::platform::windows

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#define VK_USE_PLATFORM_WIN32_KHR

#include "windows_vulkan_presenter.h"

#include <arc/project/runtime_render_resources.h>
#include <arc/render/cooked_material.h>
#include <arc/render/cooked_material_instance.h>
#include <arc/render/cooked_material_texture_bindings.h>
#include <arc/render/render_graph.h>
#include <arc/render/vulkan/vulkan_backend.h>
#include <arc/scene/render_scene.h>

#include <windows.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>
#include <vector>

namespace arc::platform::windows
{
namespace
{
void write_render_smoke_stage(std::string_view stage)
{
    const DWORD length = GetEnvironmentVariableW(L"ARC_CI_SMOKE_LOG", nullptr, 0);
    if (length <= 1 || length > 32768) return;
    std::wstring path(length, L'\0');
    const DWORD actual = GetEnvironmentVariableW(L"ARC_CI_SMOKE_LOG", path.data(), length);
    if (actual == 0 || actual >= length) return;
    path.resize(actual);
    std::ofstream output(std::filesystem::path(path), std::ios::binary | std::ios::trunc);
    if (output) output << stage << '\n';
}

bool create_win32_surface(VkInstance instance, PFN_vkGetInstanceProcAddr get_proc, VkSurfaceKHR* surface,
                          void* user_data)
{
    if (!get_proc || !user_data) return false;
    const auto create_surface =
        reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(get_proc(instance, "vkCreateWin32SurfaceKHR"));
    if (!create_surface) return false;
    VkWin32SurfaceCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    info.hinstance = GetModuleHandleW(nullptr);
    info.hwnd = static_cast<HWND>(user_data);
    return create_surface(instance, &info, nullptr, surface) == VK_SUCCESS;
}
} // namespace

windows_vulkan_presenter::~windows_vulkan_presenter() noexcept
{
    shutdown();
}

std::string windows_vulkan_presenter::initialize(void* native_window, project::packaged_runtime_session& session)
{
    shutdown();
    if (!native_window) return "Windows Vulkan presenter requires a native window";
    auto* runtime = session.runtime();
    if (!runtime || runtime->worlds().ordered_worlds().empty()) return "Packaged runtime has no startup world";
    world_ = runtime->worlds().find(runtime->worlds().ordered_worlds().front());
    if (!world_) return "Packaged runtime startup world is unavailable";

    render::vulkan::vulkan_backend_config backend_config;
    backend_config.instance_extensions = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    backend_config.create_surface = create_win32_surface;
    backend_config.surface_user_data = native_window;
    backend_config.viewport_output = render::viewport_output_type::native_window;
    auto created = render::vulkan::create_vulkan_backend(backend_config);
    if (!created) return created.error().message;

    render::renderer_config config;
    renderer_ = std::make_unique<render::renderer>(config);
    renderer_->set_platform_capabilities({.family = framework::platform_family::windows,
                                          .form_factor = framework::device_form_factor::desktop,
                                          .window_system = true,
                                          .high_dpi = true});
    renderer_->set_backend(std::move(created).value());
    session_ = &session;
    return hydrate_scene();
}

std::string windows_vulkan_presenter::hydrate_scene()
{
    if (!renderer_ || !session_ || !world_) return "Windows Vulkan presenter is not initialized";
    if (session_->scenes().empty() || session_->scenes().front().render_bindings.empty()) return {};
    auto* catalog = session_->catalog();
    if (!catalog) return "Packaged runtime catalog is unavailable";
    auto decoded = project::decode_scene_render_resources(*catalog, session_->scenes().front().render_bindings);
    if (!decoded) return decoded.error();

    std::vector<scene::cooked_mesh_upload> meshes;
    std::vector<scene::cooked_texture_upload> textures;
    std::vector<scene::cooked_material_upload> materials;
    for (auto& resource : decoded.value())
    {
        meshes.push_back({.entity = resource.entity, .mesh = std::move(resource.mesh)});
        for (auto& [guid, texture] : resource.textures)
            textures.push_back({.asset = guid, .texture = std::move(texture)});
    }
    if (auto error = scene::hydrate_cooked_scene_textures(*renderer_, textures, texture_residency_); !error.empty())
        return error;

    for (auto& resource : decoded.value())
    {
        if (!resource.material) continue;
        auto program = render::realize_cooked_material_program(*resource.material, resource.material_shaders);
        if (!program) return program.error();
        auto descriptor = render::make_cooked_material_descriptor(
            *resource.material, std::make_shared<render::material_runtime_program>(std::move(program).value()));
        if (!descriptor) return descriptor.error();
        if (!resource.material_textures) return "Cooked material has no texture binding manifest";
        std::vector<std::pair<assets::asset_guid, render::texture_handle>> resident;
        resident.reserve(texture_residency_.size());
        for (const auto& texture : texture_residency_)
            resident.emplace_back(texture.asset, texture.handle);
        auto bound =
            render::bind_cooked_material_textures(std::move(descriptor).value(), *resource.material_textures, resident);
        if (!bound) return bound.error();
        if (resource.instance)
        {
            auto applied = render::apply_cooked_material_instance_values(std::move(bound).value(), *resource.instance);
            if (!applied) return applied.error();
            bound = std::move(applied);
        }
        else
            bound.value().name = assets::to_string(resource.material_guid);
        materials.push_back({.entity = resource.entity,
                             .material = resource.material_guid,
                             .program_asset = resource.shader_material_guid,
                             .descriptor = std::move(bound).value()});
    }

    if (auto error =
            scene::hydrate_cooked_scene_materials(world_->entities(), *renderer_, materials, material_residency_);
        !error.empty())
        return error;
    if (auto error = scene::hydrate_cooked_scene_meshes(world_->entities(), *renderer_, meshes, mesh_residency_);
        !error.empty())
        return error;

    if (std::any_of(decoded.value().begin(), decoded.value().end(),
                    [](const auto& resource) { return resource.procedural_primitive && !resource.material; }))
    {
        render::texture_data checker;
        checker.name = "Default Runtime Checker";
        checker.width = 8;
        checker.height = 8;
        checker.pixels.resize(checker.width * checker.height * 4u);
        for (std::uint32_t y = 0; y < checker.height; ++y)
            for (std::uint32_t x = 0; x < checker.width; ++x)
            {
                const auto value = static_cast<std::byte>(((x + y) & 1u) != 0u ? 176u : 74u);
                const auto offset = (y * checker.width + x) * 4u;
                checker.pixels[offset] = value;
                checker.pixels[offset + 1u] = value;
                checker.pixels[offset + 2u] = value;
                checker.pixels[offset + 3u] = std::byte{255};
            }
        primitive_texture_ = renderer_->create_texture(std::move(checker));
        if (!primitive_texture_.valid()) return "Cannot create the runtime primitive checker texture";
        render::material_descriptor material;
        material.name = "Default Runtime Primitive Material";
        material.base_color = math::vector4f::one;
        material.base_color_texture = primitive_texture_;
        material.roughness = 0.62f;
        primitive_material_ = renderer_->create_material(std::move(material));
        if (!primitive_material_.valid()) return "Cannot create the runtime primitive material";
        for (const auto& resource : decoded.value())
        {
            if (!resource.procedural_primitive || resource.material) continue;
            for (const auto entity : world_->entities().entities())
            {
                const auto* id = world_->entities().try_get<ecs::persistent_id_component>(entity);
                if (!id || id->value != resource.entity) continue;
                auto* mesh = world_->entities().try_get<scene::mesh_renderer_component>(entity);
                if (mesh) mesh->material = primitive_material_;
                break;
            }
        }
    }
    return {};
}

std::string windows_vulkan_presenter::present(std::uint32_t width, std::uint32_t height, float delta_seconds)
{
    if (!renderer_ || !world_ || width == 0 || height == 0) return {};
    write_render_smoke_stage("extracting packaged scene frame");
    const auto extracted = scene::render_scene(world_->entities(), *renderer_, width, height,
                                               render::render_mode::shaded, render::mesh_visualization_mode::standard,
                                               render::editor_overlay_mode::none, true, {}, delta_seconds);
    write_render_smoke_stage("submitting packaged scene frame");
    if (extracted.camera_found)
    {
        const auto graph =
            render::make_scene_draw_graph("runtime", renderer_->resolved_config(), false, extracted.environment);
        auto submitted = renderer_->render_frame(++frame_index_, graph);
        if (!submitted) return submitted.error().message;
    }
    else
    {
        auto graph = render::make_clear_present_graph("runtime");
        render::render_graph_compile_options options;
        options.output_extent = {.width = width, .height = height};
        options.render_extent = options.output_extent;
        options.frame_index = frame_index_;
        auto compiled = graph.compile(options);
        if (!compiled) return compiled.error().message;
        render::render_frame_packet packet;
        packet.frame_index = ++frame_index_;
        auto submitted = renderer_->backend()->submit(packet, compiled.value());
        if (!submitted) return submitted.error().message;
    }
    write_render_smoke_stage("presenting packaged surface frame");
    auto presented = renderer_->backend()->present_surface_frame(width, height);
    if (!presented && presented.error().code != render::surface_frame_error_code::out_of_date)
        return presented.error().message;
    write_render_smoke_stage("packaged surface frame presented");
    return {};
}

void windows_vulkan_presenter::shutdown() noexcept
{
    if (world_)
    {
        scene::unbind_cooked_scene_materials(world_->entities(), material_residency_);
        if (renderer_) scene::release_cooked_scene_meshes(world_->entities(), *renderer_, mesh_residency_);
    }
    if (renderer_) scene::release_cooked_scene_textures(*renderer_, texture_residency_);
    renderer_.reset();
    material_residency_.clear();
    mesh_residency_.clear();
    texture_residency_.clear();
    primitive_texture_ = {};
    primitive_material_ = {};
    world_ = nullptr;
    session_ = nullptr;
}
} // namespace arc::platform::windows

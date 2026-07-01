#define SDL_MAIN_HANDLED

#include <arc/editor/editor_console.h>
#include <arc/editor/asset_drag_drop.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/editor_ui_state.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/panels/console_panel.h>
#include <arc/editor/panels/content_browser_panel.h>
#include <arc/editor/panels/inspector_panel.h>
#include <arc/editor/panels/material_editor_panel.h>
#include <arc/editor/panels/profiler_panel.h>
#include <arc/editor/panels/render_graph_panel.h>
#include <arc/editor/panels/scene_hierarchy_panel.h>
#include <arc/editor/panels/shader_graph_panel.h>
#include <arc/editor/panels/status_bar.h>
#include <arc/editor/panels/toolbar_panel.h>
#include <arc/editor/sdl_events.h>
#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/framework/framework.h>
#include <arc/input/input.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
#include <arc/render/vulkan/vulkan_backend.h>
#endif

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
#include <SDL3/SDL_vulkan.h>
#endif
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <ImGuizmo.h>
#if !defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
#include <imgui_impl_sdlrenderer3.h>
#endif
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace
{

using arc::editor::editor_ui_state;
using arc::editor::editor_asset_state;
using arc::editor::editor_build_state;
using arc::editor::editor_metrics_state;
using arc::editor::editor_project_state;
using arc::editor::editor_scene_state;
using arc::editor::editor_source_control_state;
using arc::editor::viewport_camera_label;
using arc::editor::viewport_camera_mode;
using arc::editor::viewport_shading_label;
using arc::editor::viewport_shading_mode;
using arc::editor::overlay_for_shading;
using arc::editor::render_mode_for_shading;
using arc::editor::visualization_for_shading;
using arc::editor::ui::scaled;

#if !defined(ARC_EDITOR_ASSET_ROOT)
#define ARC_EDITOR_ASSET_ROOT "assets"
#endif

struct scene_dialog_request
{
    bool pending{};
    bool replace{};
    std::filesystem::path path;
};

void SDLCALL scene_dialog_callback(void* userdata, const char* const* filelist, int)
{
    auto* request = static_cast<scene_dialog_request*>(userdata);
    if (!request || !filelist || !filelist[0])
        return;
    request->path = filelist[0];
    request->pending = true;
}

void show_scene_file_dialog(SDL_Window* window, scene_dialog_request& request, bool replace)
{
    request.replace = replace;
    static constexpr SDL_DialogFileFilter filters[] = {
        { "Scene assets", "fbx;glb;gltf" }
    };
    SDL_ShowOpenFileDialog(scene_dialog_callback, &request, window, filters, 1, nullptr, false);
}

void draw_scene_import_modal(arc::editor::editor_scene_import_state& import_state)
{
    using arc::editor::editor_scene_import_status;
    if (!import_state.modal_open && import_state.status == editor_scene_import_status::idle)
        return;

    if (import_state.modal_open)
        ImGui::OpenPopup("Importing Scene");

    const float modal_width = scaled(560.0f);
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(modal_width, 0.0f),
        ImVec2(modal_width, scaled(720.0f)));
    if (!ImGui::BeginPopupModal("Importing Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        return;

    float progress{};
    arc::render::scene_import_stage stage{ arc::render::scene_import_stage::loading };
    std::string message;
    std::vector<std::string> diagnostics;
    if (import_state.shared)
    {
        std::scoped_lock lock(import_state.shared->mutex);
        progress = import_state.shared->progress;
        stage = import_state.shared->stage;
        message = import_state.shared->message;
        diagnostics = import_state.shared->diagnostics;
    }
    if (message.empty())
        message = import_state.result.message.empty() ? "Preparing import" : import_state.result.message;

    const float content_width = std::max(scaled(240.0f), ImGui::GetContentRegionAvail().x);
    const auto source_name = import_state.source_path.filename().string();
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + content_width);
    ImGui::TextUnformatted(source_name.c_str());
    ImGui::PopTextWrapPos();
    arc::editor::ui::muted_text(arc::editor::scene_import_stage_label(stage));
    ImGui::ProgressBar(progress, ImVec2(content_width, scaled(18.0f)));
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + content_width);
    ImGui::TextWrapped("%s", message.c_str());
    ImGui::PopTextWrapPos();

    if (!diagnostics.empty())
    {
        ImGui::Spacing();
        arc::editor::ui::muted_text("Diagnostics");
        ImGui::BeginChild("import-diagnostics", ImVec2(content_width, scaled(96.0f)), ImGuiChildFlags_Borders);
        for (const auto& diagnostic : diagnostics)
            ImGui::TextWrapped("%s", diagnostic.c_str());
        ImGui::EndChild();
    }

    ImGui::Spacing();
    if (import_state.status == editor_scene_import_status::running)
    {
        if (ImGui::Button("Cancel", ImVec2(scaled(96.0f), 0.0f)) && import_state.shared)
            import_state.shared->cancel_requested = true;
    }
    else
    {
        if (ImGui::Button("Close", ImVec2(scaled(96.0f), 0.0f)))
        {
            arc::editor::reset_scene_import(import_state);
            ImGui::CloseCurrentPopup();
        }
    }

    ImGui::EndPopup();
}

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
bool create_sdl_vulkan_surface(VkInstance instance, VkSurfaceKHR* surface, void* user_data)
{
    return SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(user_data), instance, nullptr, surface);
}
#endif

struct editor_mouse_state
{
    float x{};
    float y{};
    float delta_x{};
    float delta_y{};
    float wheel_delta{};
    float select_drag_distance{};
    bool has_position{};
    bool shift_down{};

    void begin_frame() noexcept
    {
        delta_x = 0.0f;
        delta_y = 0.0f;
        wheel_delta = 0.0f;
    }
};

struct editor_window_chrome
{
    int title_height{ 42 };
    int resize_border{ 7 };
    int menu_right{ 0 };
    int project_left{ 0 };
    int controls_left{ 0 };
    int width{ 0 };
    int height{ 0 };
    bool resizable{ true };
};

std::filesystem::path editor_asset_root()
{
    std::filesystem::path root = ARC_EDITOR_ASSET_ROOT;
    if (std::filesystem::exists(root))
        return root;

    const std::filesystem::path cwd_root = std::filesystem::current_path() / "assets";
    if (std::filesystem::exists(cwd_root))
        return cwd_root;

    return root;
}

editor_asset_state load_default_editor_assets()
{
    editor_asset_state assets{};
    assets.root = editor_asset_root();
    assets.default_mesh_path = assets.root / "models" / "UAL2_Standard.glb";
    assets.default_vertex_shader_path = assets.root / "shaders" / "default_phong.vert";
    assets.default_fragment_shader_path = assets.root / "shaders" / "default_phong.frag";

    auto mesh_result = arc::render::load_gltf_mesh(assets.default_mesh_path);
    assets.default_mesh_message = mesh_result.message;
    if (mesh_result.succeeded())
    {
        assets.default_mesh = std::move(mesh_result.mesh);
        assets.default_textures = std::move(mesh_result.textures);
        assets.default_materials = std::move(mesh_result.materials);
        assets.default_mesh_loaded = true;
        arc::info(
            "editor.assets",
            "Loaded default mesh '" + assets.default_mesh_path.filename().string() + "' with " +
                std::to_string(assets.default_mesh.vertices.size()) + " vertices and " +
                std::to_string(assets.default_mesh.indices.size()) + " indices, " +
                std::to_string(assets.default_materials.size()) + " material(s)");
    }
    else
    {
        arc::warn(
            "editor.assets",
            "Default mesh '" + assets.default_mesh_path.string() + "' was not loaded: " + assets.default_mesh_message);
    }

    if (std::filesystem::exists(assets.default_vertex_shader_path) &&
        std::filesystem::exists(assets.default_fragment_shader_path))
    {
        arc::info("editor.assets", "Default PBR shader sources are available");
    }
    else
    {
        arc::warn("editor.assets", "Default PBR shader source files are missing");
    }

    return assets;
}

editor_scene_state create_default_editor_scene(
    const editor_asset_state& assets,
    arc::render::renderer* renderer)
{
    editor_scene_state state;

    arc::math::vector3f center{};
    arc::math::vector3f local_min{ -0.5f, -0.5f, -0.5f };
    arc::math::vector3f local_max{ 0.5f, 0.5f, 0.5f };
    float radius = 1.0f;
    if (assets.default_mesh_loaded && !assets.default_mesh.vertices.empty())
    {
        local_min = arc::math::vector3f{
            assets.default_mesh.vertices[0].position[0],
            assets.default_mesh.vertices[0].position[1],
            assets.default_mesh.vertices[0].position[2]
        };
        local_max = local_min;
        for (const auto& vertex : assets.default_mesh.vertices)
        {
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                local_min[axis] = std::min(local_min[axis], vertex.position[axis]);
                local_max[axis] = std::max(local_max[axis], vertex.position[axis]);
            }
        }

        center = arc::math::mul(arc::math::add(local_min, local_max), 0.5f);
        const auto span = arc::math::sub(local_max, local_min);
        radius = std::max({ span[0], span[1], span[2], 1.0f }) * 0.5f;
    }

    if (renderer && assets.default_mesh_loaded)
    {
        state.default_textures.reserve(assets.default_textures.size());
        for (const auto& texture : assets.default_textures)
            state.default_textures.push_back(renderer->create_texture(texture));

        arc::render::material_desc material;
        material.name = assets.default_mesh.name + " Material";
        if (!assets.default_materials.empty())
        {
            const auto material_index = assets.default_mesh.material_index < assets.default_materials.size()
                ? assets.default_mesh.material_index
                : std::size_t{ 0 };
            const auto& imported = assets.default_materials[material_index];
            material = imported.material;

            const auto assign_texture = [&](std::size_t index, arc::render::texture_handle& handle) {
                if (index != arc::render::material_texture_indices::invalid && index < state.default_textures.size())
                    handle = state.default_textures[index];
            };

            assign_texture(imported.textures.base_color, material.base_color_texture);
            assign_texture(imported.textures.metallic_roughness, material.metallic_roughness_texture);
            assign_texture(imported.textures.normal, material.normal_texture);
            assign_texture(imported.textures.occlusion, material.occlusion_texture);
            assign_texture(imported.textures.emissive, material.emissive_texture);
        }
        state.default_material = renderer->create_material(material);
        state.default_mesh = renderer->create_mesh(assets.default_mesh);
        state.mesh_uploaded = state.default_mesh.valid();
    }

    const auto camera = state.scene.create();
    state.camera_entity = camera;
    arc::scene::transform_component camera_transform;
    camera_transform.position = arc::math::vector3f{ 0.0f, 0.0f, 4.0f };
    state.scene.emplace<arc::scene::name_component>(camera, "Editor Camera");
    state.scene.emplace<arc::scene::tag_component>(camera, "Editor");
    state.scene.emplace<arc::scene::active_component>(camera);
    state.scene.emplace<arc::scene::transform_component>(camera, camera_transform);
    state.scene.emplace<arc::scene::camera_component>(camera);
    state.camera_created = true;

    const auto sun = state.scene.create();
    state.sun_entity = sun;
    arc::scene::transform_component sun_transform;
    sun_transform.rotation = arc::editor::quaternion_from_euler_degrees({ -50.0f, -35.0f, 0.0f });
    state.scene.emplace<arc::scene::name_component>(sun, "Sun");
    state.scene.emplace<arc::scene::tag_component>(sun, "Light");
    state.scene.emplace<arc::scene::active_component>(sun);
    state.scene.emplace<arc::scene::transform_component>(sun, sun_transform);
    state.scene.emplace<arc::scene::directional_light_component>(
        sun,
        arc::math::vector3f{ 1.0f, 1.0f, 1.0f },
        1.8f,
        true);
    auto& sun_light = state.scene.get<arc::scene::directional_light_component>(sun);
    sun_light.shadow.resolution = 4096;
    sun_light.shadow.filter = arc::render::shadow_filter::pcf_3x3;
    sun_light.shadow.bias = 0.0008f;
    sun_light.shadow.normal_bias = 0.003f;

    arc::editor::add_world_environment_to_scene(state);
    if (renderer)
        arc::editor::add_terrain_to_scene(state, *renderer);

    if (state.default_mesh.valid())
    {
        const auto mesh = state.scene.create();
        state.mesh_entity = mesh;
        const float scale = 1.8f / radius;
        arc::scene::transform_component mesh_transform;
        mesh_transform.position = arc::math::mul(center, -scale);
        mesh_transform.scale = arc::math::vector3f{ scale, scale, scale };
        state.scene.emplace<arc::scene::name_component>(mesh, assets.default_mesh.name);
        state.scene.emplace<arc::scene::tag_component>(mesh, "Mesh");
        state.scene.emplace<arc::scene::active_component>(mesh);
        state.scene.emplace<arc::scene::selection_component>(mesh, true);
        state.scene.emplace<arc::scene::bounds_component>(
            mesh,
            arc::geometric::box3f{ arc::geometric::point3f(local_min), arc::geometric::point3f(local_max) },
            arc::geometric::box3f{ arc::geometric::point3f(local_min), arc::geometric::point3f(local_max) },
            true);
        state.scene.emplace<arc::scene::transform_component>(mesh, mesh_transform);
        state.scene.emplace<arc::scene::mesh_renderer_component>(
            mesh,
            state.default_mesh,
            state.default_material,
            true);
        state.selected_entity = mesh;
    }

    return state;
}

editor_project_state make_project_state(const editor_asset_state& assets)
{
    editor_project_state state;
    state.root = assets.root.empty() ? std::filesystem::current_path() : assets.root.parent_path();
    state.name = state.root.filename().string();
    if (state.name.empty())
        state.name = "Arc Project";
    return state;
}

editor_build_state make_build_state()
{
    editor_build_state state;
#if defined(_WIN32)
    state.platform = "Windows (x64)";
#elif defined(__APPLE__)
    state.platform = "macOS";
#elif defined(__linux__)
    state.platform = "Linux";
#else
    state.platform = "Unknown";
#endif
#if defined(NDEBUG)
    state.configuration = "Release";
#else
    state.configuration = "Development";
#endif
    return state;
}

editor_source_control_state make_source_control_state()
{
    return {};
}

editor_metrics_state make_metrics_state(const arc::frame_time& time, const editor_scene_state& scene)
{
    editor_metrics_state state;
    state.frame_ms = static_cast<float>(time.delta_seconds * 1000.0);
    state.fps = time.delta_seconds > 0.0 ? static_cast<float>(1.0 / time.delta_seconds) : 0.0f;
    state.draw_calls = scene.last_render.submitted_draw_count;
    return state;
}

void update_mouse_state(editor_mouse_state& mouse, const arc::event& event) noexcept
{
    switch (event.type)
    {
    case arc::event_type::mouse_moved:
    case arc::event_type::mouse_button_down:
    case arc::event_type::mouse_button_up:
        if (mouse.has_position)
        {
            mouse.delta_x += static_cast<float>(event.x) - mouse.x;
            mouse.delta_y += static_cast<float>(event.y) - mouse.y;
        }
        mouse.x = static_cast<float>(event.x);
        mouse.y = static_cast<float>(event.y);
        mouse.has_position = true;
        break;
    case arc::event_type::mouse_wheel:
        mouse.wheel_delta += event.wheel_delta;
        mouse.x = static_cast<float>(event.x);
        mouse.y = static_cast<float>(event.y);
        mouse.has_position = true;
        break;
    default:
        break;
    }
}

ImVec4 color_for_level(arc::log_level level)
{
    switch (level)
    {
    case arc::log_level::trace:
        return ImVec4(0.62f, 0.67f, 0.72f, 1.0f);
    case arc::log_level::debug:
        return ImVec4(0.39f, 0.67f, 1.0f, 1.0f);
    case arc::log_level::info:
        return ImVec4(0.45f, 0.75f, 1.0f, 1.0f);
    case arc::log_level::warn:
        return ImVec4(1.0f, 0.78f, 0.28f, 1.0f);
    case arc::log_level::error:
    case arc::log_level::fatal:
        return ImVec4(1.0f, 0.32f, 0.32f, 1.0f);
    }

    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
}

ImU32 color_for_level_u32(arc::log_level level, float alpha = 1.0f)
{
    ImVec4 color = color_for_level(level);
    color.w = alpha;
    return ImGui::ColorConvertFloat4ToU32(color);
}

SDL_HitTestResult SDLCALL editor_window_hit_test(SDL_Window*, const SDL_Point* area, void* data)
{
    const auto* chrome = static_cast<const editor_window_chrome*>(data);
    if (!chrome || !area)
        return SDL_HITTEST_NORMAL;

    if (chrome->resizable)
    {
        const bool left = area->x < chrome->resize_border;
        const bool right = area->x >= chrome->width - chrome->resize_border;
        const bool top = area->y < chrome->resize_border;
        const bool bottom = area->y >= chrome->height - chrome->resize_border;

        if (top && left)
            return SDL_HITTEST_RESIZE_TOPLEFT;
        if (top && right)
            return SDL_HITTEST_RESIZE_TOPRIGHT;
        if (bottom && left)
            return SDL_HITTEST_RESIZE_BOTTOMLEFT;
        if (bottom && right)
            return SDL_HITTEST_RESIZE_BOTTOMRIGHT;
        if (top)
            return SDL_HITTEST_RESIZE_TOP;
        if (bottom)
            return SDL_HITTEST_RESIZE_BOTTOM;
        if (left)
            return SDL_HITTEST_RESIZE_LEFT;
        if (right)
            return SDL_HITTEST_RESIZE_RIGHT;
    }

    if (area->y < chrome->title_height)
    {
        const bool over_menu = area->x <= chrome->menu_right;
        const bool over_project = chrome->project_left > 0 && area->x >= chrome->project_left;
        const bool over_controls = chrome->controls_left > 0 && area->x >= chrome->controls_left;
        if (!over_menu && !over_project && !over_controls)
            return SDL_HITTEST_DRAGGABLE;
    }

    return SDL_HITTEST_NORMAL;
}

std::string_view display_level(arc::log_level level)
{
    switch (level)
    {
    case arc::log_level::trace:
        return "Trace";
    case arc::log_level::debug:
        return "Debug";
    case arc::log_level::info:
        return "Info";
    case arc::log_level::warn:
        return "Warn";
    case arc::log_level::error:
        return "Error";
    case arc::log_level::fatal:
        return "Fatal";
    }

    return "Log";
}

std::string format_timestamp(std::chrono::system_clock::time_point timestamp)
{
    if (timestamp == std::chrono::system_clock::time_point{})
        timestamp = std::chrono::system_clock::now();

    const std::time_t value = std::chrono::system_clock::to_time_t(timestamp);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &value);
#else
    localtime_r(&value, &local_time);
#endif

    char buffer[16]{};
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &local_time);
    return buffer;
}

bool level_visible(const editor_ui_state& state, arc::log_level level)
{
    switch (level)
    {
    case arc::log_level::trace:
        return state.console_show_trace;
    case arc::log_level::debug:
        return state.console_show_debug;
    case arc::log_level::info:
        return state.console_show_info;
    case arc::log_level::warn:
        return state.console_show_warn;
    case arc::log_level::error:
    case arc::log_level::fatal:
        return state.console_show_error;
    }

    return true;
}

bool text_matches_filter(const arc::editor::console_log_entry& entry, std::string_view filter)
{
    if (filter.empty())
        return true;

    return entry.message.find(filter) != std::string::npos ||
        entry.category.find(filter) != std::string::npos ||
        display_level(entry.level).find(filter) != std::string_view::npos;
}

void draw_level_badge(arc::log_level level, int count)
{
    ImGui::PushStyleColor(ImGuiCol_Button, color_for_level(level));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color_for_level(level));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, color_for_level(level));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.04f, 0.05f, 0.06f, 1.0f));
    ImGui::Button(display_level(level).data());
    ImGui::PopStyleColor(4);
    ImGui::SameLine(0.0f, 4.0f);
    ImGui::Text("%d", count);
}

void draw_brand_mark(ImVec2 center)
{
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const float radius = scaled(9.0f);
    const ImU32 color = IM_COL32(42, 144, 255, 255);
    std::array<ImVec2, 6> points{};
    for (std::size_t index = 0; index < points.size(); ++index)
    {
        const float angle = -1.5708f + static_cast<float>(index) * 1.0472f;
        points[index] = ImVec2(center.x + std::cos(angle) * radius, center.y + std::sin(angle) * radius);
    }
    draw_list->AddPolyline(points.data(), static_cast<int>(points.size()), color, ImDrawFlags_Closed, scaled(2.0f));
    draw_list->AddCircleFilled(center, scaled(2.4f), color);
}

void draw_title_button(SDL_Window* window, const char* label, const ImVec2& size, bool close_button, bool& exit_requested)
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, close_button ? ImVec4(0.76f, 0.16f, 0.16f, 1.0f) : ImVec4(0.13f, 0.16f, 0.19f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, close_button ? ImVec4(0.62f, 0.10f, 0.10f, 1.0f) : ImVec4(0.16f, 0.22f, 0.28f, 1.0f));
    if (ImGui::Button(label, size))
    {
        if (close_button)
            exit_requested = true;
        else if (label[0] == '-')
            SDL_MinimizeWindow(window);
        else if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED) != 0)
            SDL_RestoreWindow(window);
        else
            SDL_MaximizeWindow(window);
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(3);
}

void draw_main_menu(
    SDL_Window* window,
    bool& exit_requested,
    editor_ui_state& state,
    const editor_project_state& project,
    editor_window_chrome& chrome,
    scene_dialog_request& scene_dialog)
{
    if (!ImGui::BeginMenuBar())
        return;

    const ImVec2 bar_min = ImGui::GetWindowPos();
    const float title_height = scaled(40.0f);
    const float controls_width = scaled(156.0f);
    chrome.title_height = static_cast<int>(std::ceil(title_height));
    chrome.resize_border = static_cast<int>(std::ceil(scaled(7.0f)));
    chrome.width = static_cast<int>(ImGui::GetWindowWidth());

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scaled(1.0f));
    draw_brand_mark(ImVec2(bar_min.x + scaled(22.0f), bar_min.y + title_height * 0.5f));
    ImGui::SetCursorPosX(scaled(42.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.88f, 0.93f, 0.96f, 1.0f));
    ImGui::TextUnformatted("arc editor");
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, scaled(22.0f));

    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Open Scene..."))
            show_scene_file_dialog(window, scene_dialog, true);
        if (ImGui::MenuItem("Import Scene Into Current..."))
            show_scene_file_dialog(window, scene_dialog, false);
        ImGui::Separator();
        if (ImGui::MenuItem("Exit"))
            exit_requested = true;
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Edit"))
    {
        ImGui::MenuItem("Undo", "Ctrl+Z", false, false);
        ImGui::MenuItem("Redo", "Ctrl+Y", false, false);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View"))
    {
        ImGui::MenuItem("Scene", nullptr, true, false);
        ImGui::MenuItem("Game", nullptr, false, false);
        ImGui::MenuItem("Lighting", nullptr, false, false);
        ImGui::Separator();
        ImGui::MenuItem("World Grid", nullptr, &state.show_world_grid);
        ImGui::MenuItem("Sky", nullptr, &state.show_sky);
        ImGui::MenuItem("Height Fog", nullptr, &state.show_fog);
        ImGui::MenuItem("Terrain", nullptr, &state.show_terrain);
        ImGui::MenuItem("Water", nullptr, &state.show_water);
        ImGui::MenuItem("Vegetation", nullptr, &state.show_vegetation);
        ImGui::MenuItem("Decals", nullptr, &state.show_decals);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools"))
    {
        if (ImGui::MenuItem("Command Palette", "Ctrl+P"))
            state.command_palette_open = true;
        ImGui::MenuItem("Project Settings", nullptr, false, false);
        ImGui::MenuItem("Preferences", nullptr, false, false);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Build"))
    {
        ImGui::MenuItem("Build Game", nullptr, false, false);
        ImGui::MenuItem("Package Project", nullptr, false, false);
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Window"))
    {
        ImGui::MenuItem("Viewport", nullptr, true, false);
        ImGui::MenuItem("Scene", nullptr, true, false);
        ImGui::MenuItem("Inspector", nullptr, true, false);
        ImGui::MenuItem("Console", nullptr, true, false);
        ImGui::MenuItem("Stats", nullptr, true, false);
        ImGui::MenuItem("Content Browser", nullptr, true, false);
        ImGui::MenuItem("Render Graph", nullptr, true, false);
        ImGui::Separator();
        if (ImGui::MenuItem("Reset Layout"))
            state.reset_layout = true;
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Help"))
    {
        ImGui::MenuItem("About ARC", nullptr, false, false);
        ImGui::EndMenu();
    }

    chrome.menu_right = static_cast<int>(std::ceil(ImGui::GetCursorScreenPos().x - bar_min.x + scaled(8.0f)));

    const float project_width = scaled(210.0f);
    const float project_left = ImGui::GetWindowWidth() - controls_width - project_width - scaled(12.0f);
    if (ImGui::GetCursorPosX() < project_left)
        ImGui::SetCursorPosX(project_left);
    chrome.project_left = static_cast<int>(std::floor(project_left));
    ImGui::TextDisabled("Project:");
    ImGui::SameLine();
    if (ImGui::BeginMenu(project.name.c_str()))
    {
        ImGui::MenuItem("Open Project", nullptr, false, false);
        ImGui::MenuItem("Project Settings", nullptr, false, false);
        ImGui::EndMenu();
    }

    const float controls_left = ImGui::GetWindowWidth() - controls_width;
    if (ImGui::GetCursorPosX() < controls_left)
        ImGui::SetCursorPosX(controls_left);
    chrome.controls_left = static_cast<int>(std::floor(controls_left));

    const ImVec2 control_size(scaled(46.0f), title_height);
    draw_title_button(window, "-", control_size, false, exit_requested);
    ImGui::SameLine(0.0f, 0.0f);
    const bool maximized = (SDL_GetWindowFlags(window) & SDL_WINDOW_MAXIMIZED) != 0;
    draw_title_button(window, maximized ? "[]" : "[ ]", control_size, false, exit_requested);
    ImGui::SameLine(0.0f, 0.0f);
    draw_title_button(window, "X", control_size, true, exit_requested);

    ImGui::EndMenuBar();
}

void build_default_dock_layout(
    ImGuiID dockspace_id,
    ImVec2 size,
    arc::editor::editor_layout_preset preset,
    bool show_bottom_panels)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID main_id = dockspace_id;
    const ImGuiID left_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Left, 0.18f, nullptr, &main_id);
    const ImGuiID right_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Right, 0.22f, nullptr, &main_id);
    ImGuiID bottom_id{};
    if (show_bottom_panels)
        bottom_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Down, preset == arc::editor::editor_layout_preset::profiling ? 0.32f : 0.26f, nullptr, &main_id);

    ImGui::DockBuilderDockWindow("Scene##Hierarchy", left_id);
    ImGui::DockBuilderDockWindow("Scene##Viewport", main_id);
    ImGui::DockBuilderDockWindow("Inspector", right_id);
    if (show_bottom_panels)
    {
        if (preset == arc::editor::editor_layout_preset::profiling)
            ImGui::DockBuilderDockWindow("Profiler", bottom_id);
        else
            ImGui::DockBuilderDockWindow("Content Browser", bottom_id);
        ImGui::DockBuilderDockWindow("Console", bottom_id);
        if (preset != arc::editor::editor_layout_preset::profiling)
            ImGui::DockBuilderDockWindow("Profiler", bottom_id);
        ImGui::DockBuilderDockWindow("Render Graph", bottom_id);
        ImGui::DockBuilderDockWindow("Shader Graph", bottom_id);
        ImGui::DockBuilderDockWindow("Stats", bottom_id);
    }
    ImGui::DockBuilderFinish(dockspace_id);
}

void draw_dockspace(
    SDL_Window* window,
    bool& exit_requested,
    editor_ui_state& state,
    const editor_project_state& project,
    editor_window_chrome& chrome,
    scene_dialog_request& scene_dialog)
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float status_height = scaled(30.0f);
    const ImVec2 dockspace_size(viewport->WorkSize.x, std::max(1.0f, viewport->WorkSize.y - status_height));
    chrome.height = static_cast<int>(viewport->WorkSize.y);
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(dockspace_size);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(8.0f), scaled(10.0f)));
    ImGui::Begin("ARC Editor Dockspace", nullptr, flags);

    draw_main_menu(window, exit_requested, state, project, chrome, scene_dialog);
    ImGui::PopStyleVar(3);
    arc::editor::draw_toolbar(state);
    const ImGuiID dockspace_id = ImGui::GetID("ArcEditorDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    if (state.reset_layout)
    {
        build_default_dock_layout(dockspace_id, dockspace_size, state.layout_preset, state.show_bottom_panels);
        state.reset_layout = false;
    }
    ImGui::End();
}

void draw_command_palette(
    editor_ui_state& state,
    editor_scene_state& scene,
    arc::editor::editor_camera_controller& camera)
{
    if (state.command_palette_open)
    {
        ImGui::OpenPopup("Command Palette");
        state.command_palette_open = false;
    }

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f, viewport->WorkPos.y + scaled(90.0f)),
        ImGuiCond_Appearing,
        ImVec2(0.5f, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(scaled(520.0f), 0.0f), ImGuiCond_Appearing);
    if (ImGui::BeginPopupModal("Command Palette", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::SetNextItemWidth(-1.0f);
        arc::editor::ui::search_box("##command-search", "Search commands...", state.command_palette_filter, sizeof(state.command_palette_filter));
        ImGui::Separator();

        const auto command = [&](const char* label, const auto& action) {
            if (state.command_palette_filter[0] != '\0' && std::string(label).find(state.command_palette_filter) == std::string::npos)
                return;
            if (ImGui::Selectable(label))
            {
                action();
                ImGui::CloseCurrentPopup();
            }
        };

        command("Focus Selected", [&] {
            if (!arc::editor::focus_selected_entity(scene.scene, scene.selected_entity, camera))
                arc::info("editor.commands", "No selected entity to focus");
        });
        command("Reset Layout", [&] {
            state.layout_preset = arc::editor::editor_layout_preset::default_layout;
            state.show_bottom_panels = true;
            state.reset_layout = true;
        });
        command("Toggle Grid", [&] { state.show_world_grid = !state.show_world_grid; });
        command("Open Project Settings", [] { arc::info("editor.commands", "Project settings are not implemented yet"); });

        if (ImGui::Button("Close"))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
}

void draw_viewport_overlay_controls(
    const ImVec2& origin,
    const ImVec2& end,
    editor_ui_state& state)
{
    constexpr viewport_shading_mode shading_modes[]{
        viewport_shading_mode::wireframe,
        viewport_shading_mode::standard,
        viewport_shading_mode::albedo,
        viewport_shading_mode::opacity,
        viewport_shading_mode::world_normal,
        viewport_shading_mode::specularity,
        viewport_shading_mode::gloss,
        viewport_shading_mode::metalness,
        viewport_shading_mode::ao,
        viewport_shading_mode::emission,
        viewport_shading_mode::lighting,
        viewport_shading_mode::uv0,
        viewport_shading_mode::cascade_debug,
        viewport_shading_mode::shadow_mask,
        viewport_shading_mode::light_complexity
    };

    const float button_height = scaled(34.0f);
    const float shading_width = scaled(142.0f);
    const float camera_width = scaled(152.0f);
    const float gap = scaled(6.0f);
    const ImVec2 pos(
        std::max(origin.x + scaled(12.0f), end.x - shading_width - camera_width - gap - scaled(12.0f)),
        origin.y + scaled(12.0f));

    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, scaled(5.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(12.0f), scaled(7.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.060f, 0.070f, 0.080f, 0.88f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.105f, 0.135f, 0.165f, 0.94f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.130f, 0.190f, 0.250f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.060f, 0.075f, 0.085f, 0.98f));

    ImGui::SetNextItemWidth(shading_width);
    if (ImGui::Button(viewport_shading_label(state.viewport_shading), ImVec2(shading_width, button_height)))
        ImGui::OpenPopup("viewport-shading-popup");
    if (ImGui::BeginPopup("viewport-shading-popup"))
    {
        for (const viewport_shading_mode mode : shading_modes)
        {
            const bool selected = state.viewport_shading == mode;
            if (ImGui::RadioButton(viewport_shading_label(mode), selected))
                state.viewport_shading = mode;
        }
        ImGui::EndPopup();
    }

    ImGui::SameLine(0.0f, gap);
    if (ImGui::Button(viewport_camera_label(state.viewport_camera), ImVec2(camera_width, button_height)))
        ImGui::OpenPopup("viewport-camera-popup");
    if (ImGui::BeginPopup("viewport-camera-popup"))
    {
        if (ImGui::RadioButton("Perspective", state.viewport_camera == viewport_camera_mode::perspective))
            state.viewport_camera = viewport_camera_mode::perspective;
        if (ImGui::RadioButton("Orthographic", state.viewport_camera == viewport_camera_mode::orthographic))
            state.viewport_camera = viewport_camera_mode::orthographic;
        ImGui::EndPopup();
    }

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
}

void draw_viewport_panel(
    arc::editor::editor_viewport& viewport,
    editor_ui_state& state,
    const editor_metrics_state& metrics,
    editor_scene_state& editor_scene,
    const editor_asset_state& editor_assets,
    arc::render::renderer* renderer
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    ,
    arc::render::vulkan::vulkan_backend* vulkan_backend
#endif
)
{
    ImGui::Begin("Scene##Viewport");

    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 size(std::max(available.x, 1.0f), std::max(available.y, 1.0f));
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 end(origin.x + size.x, origin.y + size.y);

    viewport.set_screen_rect(origin.x, origin.y, size.x, size.y);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    bool drew_engine_viewport = false;
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    if (vulkan_backend)
    {
        vulkan_backend->resize_viewport(static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y));
        const auto texture = vulkan_backend->viewport_texture();
        if (texture.valid())
        {
            ImGui::Image(static_cast<ImTextureID>(texture.id), size);
            drew_engine_viewport = true;
        }
    }
#endif

    if (!drew_engine_viewport)
    {
        ImGui::InvisibleButton("viewport-canvas", size);
        draw_list->AddRectFilled(origin, end, IM_COL32(31, 41, 44, 255));

        const ImU32 sky_top = IM_COL32(92, 118, 128, 255);
        const ImU32 ground = IM_COL32(48, 59, 46, 255);
        draw_list->AddRectFilledMultiColor(
            origin,
            end,
            sky_top,
            sky_top,
            ground,
            ground);
        draw_list->AddRectFilled(
            ImVec2(origin.x, origin.y + size.y * 0.58f),
            end,
            IM_COL32(42, 50, 38, 180));
    }
    viewport.set_focused(ImGui::IsItemFocused());
    viewport.set_hovered(ImGui::IsItemHovered());
    arc::editor::editor_asset_payload payload;
    if (arc::editor::accept_asset_drag_drop(payload))
    {
        if (payload.kind != arc::editor::editor_asset_kind::material)
        {
            arc::warn("editor.materials", "Only material assets can be dropped onto the scene viewport");
        }
        else if (!renderer)
        {
            arc::warn("editor.materials", "No renderer is available for viewport material assignment");
        }
        else
        {
            const auto* camera_transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.camera_entity);
            const auto* camera = editor_scene.scene.try_get<arc::scene::camera_component>(editor_scene.camera_entity);
            const ImVec2 mouse = ImGui::GetMousePos();
            if (camera_transform && camera && viewport.contains_screen_point(mouse.x, mouse.y))
            {
                const auto ray = arc::editor::screen_ray_from_camera(
                    *camera,
                    *camera_transform,
                    viewport,
                    viewport.local_x(mouse.x),
                    viewport.local_y(mouse.y));
                std::string message;
                const auto hit = arc::editor::apply_material_asset_to_viewport_hit(
                    editor_scene.material_library,
                    *renderer,
                    editor_assets.root,
                    payload.relative_path,
                    editor_scene.scene,
                    ray,
                    editor_scene.selected_entity,
                    &message);
                if (hit.valid())
                    arc::info("editor.materials", message);
                else
                    arc::warn("editor.materials", message.empty() ? "Material drop missed all renderable entities" : message);
            }
        }
    }

    draw_list->AddRect(origin, end, IM_COL32(48, 56, 64, 130));

    const ImVec2 overlay_min(origin.x + scaled(14.0f), origin.y + scaled(14.0f));
    const ImVec2 overlay_max(origin.x + scaled(190.0f), origin.y + scaled(104.0f));
    draw_list->AddRectFilled(overlay_min, overlay_max, IM_COL32(12, 15, 18, 132), scaled(6.0f));
    draw_list->AddRect(overlay_min, overlay_max, IM_COL32(95, 115, 130, 72), scaled(6.0f));
    const ImU32 stats_text = IM_COL32(192, 202, 211, 220);
    char line[64]{};
    std::snprintf(line, sizeof(line), "FPS       %.1f", metrics.fps);
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(9.0f)), stats_text, line);
    std::snprintf(line, sizeof(line), "Frame     %.2f ms", metrics.frame_ms);
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(29.0f)), stats_text, line);
    std::snprintf(line, sizeof(line), "Draws     %zu", metrics.draw_calls);
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(49.0f)), stats_text, line);
    std::snprintf(line, sizeof(line), "GPU       %s", metrics.gpu_time_available ? "ready" : "pending");
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(69.0f)), stats_text, line);

    if (!drew_engine_viewport)
    {
        const ImVec2 center(origin.x + size.x * 0.52f, origin.y + size.y * 0.56f);
        draw_list->AddRect(
            ImVec2(center.x - 38.0f, center.y - 38.0f),
            ImVec2(center.x + 38.0f, center.y + 38.0f),
            IM_COL32(255, 214, 73, 255));
        draw_list->AddLine(center, ImVec2(center.x + 110.0f, center.y), IM_COL32(255, 65, 65, 255), 3.0f);
        draw_list->AddLine(center, ImVec2(center.x, center.y - 110.0f), IM_COL32(102, 220, 86, 255), 3.0f);
        draw_list->AddLine(center, ImVec2(center.x - 84.0f, center.y + 56.0f), IM_COL32(62, 124, 255, 255), 3.0f);

        const char* text = "Renderer viewport placeholder";
        const ImVec2 text_size = ImGui::CalcTextSize(text);
        draw_list->AddText(
            ImVec2(origin.x + (size.x - text_size.x) * 0.5f, origin.y + size.y - text_size.y - 18.0f),
            IM_COL32(214, 220, 229, 180),
            text);
    }

    draw_viewport_overlay_controls(origin, end, state);

    ImGui::End();
}

ImGuizmo::OPERATION gizmo_operation_for_tool(arc::editor::editor_tool tool) noexcept
{
    switch (tool)
    {
    case arc::editor::editor_tool::translate:
        return ImGuizmo::TRANSLATE;
    case arc::editor::editor_tool::rotate:
        return ImGuizmo::ROTATE;
    case arc::editor::editor_tool::scale:
        return ImGuizmo::SCALE;
    case arc::editor::editor_tool::select:
        break;
    }
    return ImGuizmo::TRANSLATE;
}

float matrix_value(const std::array<float, 16>& matrix, std::size_t row, std::size_t col) noexcept
{
    return matrix[col * 4 + row];
}

arc::math::vector3f column_vector(const std::array<float, 16>& matrix, std::size_t col) noexcept
{
    return {
        matrix_value(matrix, 0, col),
        matrix_value(matrix, 1, col),
        matrix_value(matrix, 2, col)
    };
}

arc::math::quatf quaternion_from_transform_matrix(const std::array<float, 16>& matrix, const arc::math::vector3f& scale) noexcept
{
    const auto inv_scale = [](float value) noexcept {
        return std::abs(value) <= 0.000001f ? 1.0f : 1.0f / value;
    };

    const auto x_axis = arc::math::mul(column_vector(matrix, 0), inv_scale(scale[0]));
    const auto y_axis = arc::math::mul(column_vector(matrix, 1), inv_scale(scale[1]));
    const auto z_axis = arc::math::mul(column_vector(matrix, 2), inv_scale(scale[2]));

    const float m00 = x_axis[0];
    const float m01 = y_axis[0];
    const float m02 = z_axis[0];
    const float m10 = x_axis[1];
    const float m11 = y_axis[1];
    const float m12 = z_axis[1];
    const float m20 = x_axis[2];
    const float m21 = y_axis[2];
    const float m22 = z_axis[2];

    const float trace = m00 + m11 + m22;
    arc::math::quatf result;
    if (trace > 0.0f)
    {
        const float s = std::sqrt(trace + 1.0f) * 2.0f;
        result = { (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s };
    }
    else if (m00 > m11 && m00 > m22)
    {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
        result = { 0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s };
    }
    else if (m11 > m22)
    {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
        result = { (m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s };
    }
    else
    {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
        result = { (m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s };
    }
    return arc::math::normalize(result);
}

void draw_transform_gizmo(
    editor_scene_state& editor_scene,
    const arc::editor::editor_viewport& viewport,
    arc::editor::editor_tool tool)
{
    if (!viewport.valid() || !editor_scene.scene.alive(editor_scene.selected_entity))
        return;

    auto* transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.selected_entity);
    const auto* camera_transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.camera_entity);
    const auto* camera = editor_scene.scene.try_get<arc::scene::camera_component>(editor_scene.camera_entity);
    if (!transform || !camera_transform || !camera)
        return;

    const float aspect = viewport.height() == 0 ? 1.0f : static_cast<float>(viewport.width()) / static_cast<float>(viewport.height());
    const auto view = arc::scene::view_matrix(*camera_transform);
    const auto projection = camera->projection == arc::scene::camera_projection::orthographic
        ? arc::scene::orthographic_rh_zo(camera->orthographic_height, aspect, camera->near_plane, camera->far_plane)
        : arc::scene::perspective_rh_zo(camera->fov_y_radians, aspect, camera->near_plane, camera->far_plane);
    auto model = arc::scene::local_matrix(*transform);
    std::array<float, 16> model_data{};
    std::copy(model.data(), model.data() + model_data.size(), model_data.begin());

    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(camera->projection == arc::scene::camera_projection::orthographic);
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    ImGuizmo::SetDrawlist(draw_list);
    ImGuizmo::SetRect(
        viewport.screen_x(),
        viewport.screen_y(),
        static_cast<float>(viewport.width()),
        static_cast<float>(viewport.height()));
    draw_list->PushClipRect(
        ImVec2(viewport.screen_x(), viewport.screen_y()),
        ImVec2(
            viewport.screen_x() + static_cast<float>(viewport.width()),
            viewport.screen_y() + static_cast<float>(viewport.height())),
        true);

    if (ImGuizmo::Manipulate(
            view.data(),
            projection.data(),
            gizmo_operation_for_tool(tool),
            ImGuizmo::WORLD,
            model_data.data()))
    {
        const arc::math::vector3f scale{
            arc::math::length(column_vector(model_data, 0)),
            arc::math::length(column_vector(model_data, 1)),
            arc::math::length(column_vector(model_data, 2))
        };
        transform->set_position({ model_data[12], model_data[13], model_data[14] });
        transform->set_rotation(quaternion_from_transform_matrix(model_data, scale));
        transform->set_scale(scale);
    }
    draw_list->PopClipRect();
}

void update_editor_camera_controls(
    editor_scene_state& editor_scene,
    arc::editor::editor_camera_controller& camera,
    const arc::editor::editor_viewport& viewport,
    const arc::input::input_manager& input,
    const editor_mouse_state& mouse)
{
    if (input.pressed("viewport.focus"))
        arc::editor::focus_selected_entity(editor_scene.scene, editor_scene.selected_entity, camera);

    if (viewport.hovered())
    {
        const bool gizmo_capturing_mouse = ImGuizmo::IsUsing() || ImGuizmo::IsOver();
        if (!gizmo_capturing_mouse && (input.down("viewport.pan") || (input.down("viewport.orbit") && mouse.shift_down)))
            camera.pan(mouse.delta_x, mouse.delta_y);
        else if (!gizmo_capturing_mouse && input.down("viewport.orbit"))
            camera.orbit(mouse.delta_x, mouse.delta_y);
        if (mouse.wheel_delta != 0.0f)
            camera.zoom(mouse.wheel_delta);
    }

    if (auto* camera_transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.camera_entity))
        camera.apply_to(*camera_transform);
}

void handle_viewport_selection(
    editor_scene_state& editor_scene,
    const arc::editor::editor_viewport& viewport,
    arc::render::renderer& renderer,
    const arc::input::input_manager& input,
    const editor_mouse_state& mouse)
{
    if (!input.released("viewport.select") || mouse.select_drag_distance > 4.0f || !viewport.contains_screen_point(mouse.x, mouse.y))
        return;
    if (ImGuizmo::IsUsing() || ImGuizmo::IsOver())
        return;

    const auto* camera_transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.camera_entity);
    const auto* camera = editor_scene.scene.try_get<arc::scene::camera_component>(editor_scene.camera_entity);
    if (!camera_transform || !camera)
        return;

    const auto local_x = static_cast<std::uint32_t>(std::max(0.0f, viewport.local_x(mouse.x)));
    const auto local_y = static_cast<std::uint32_t>(std::max(0.0f, viewport.local_y(mouse.y)));
    renderer.request_object_pick(local_x, local_y);
    const auto gpu_pick = renderer.last_object_pick();
    if (gpu_pick.available && gpu_pick.x == local_x && gpu_pick.y == local_y)
    {
        if (gpu_pick.hit)
            arc::editor::select_entity(
                editor_scene.scene,
                arc::scene::entity{ .index = gpu_pick.object.index, .generation = gpu_pick.object.generation },
                editor_scene.selected_entity);
        else
            arc::editor::clear_selection(editor_scene.scene, editor_scene.selected_entity);
        return;
    }

    const auto ray = arc::editor::screen_ray_from_camera(
        *camera,
        *camera_transform,
        viewport,
        viewport.local_x(mouse.x),
        viewport.local_y(mouse.y));
    const auto picked = arc::editor::pick_bounded_entity(editor_scene.scene, ray);
    if (picked.valid())
        arc::editor::select_entity(editor_scene.scene, picked, editor_scene.selected_entity);
    else
        arc::editor::clear_selection(editor_scene.scene, editor_scene.selected_entity);
}

bool project_world_point(
    const arc::math::matrix4f& view_projection,
    const arc::editor::editor_viewport& viewport,
    const arc::math::vector3f& point,
    ImVec2& screen) noexcept
{
    const float x =
        view_projection(0, 0) * point[0] +
        view_projection(0, 1) * point[1] +
        view_projection(0, 2) * point[2] +
        view_projection(0, 3);
    const float y =
        view_projection(1, 0) * point[0] +
        view_projection(1, 1) * point[1] +
        view_projection(1, 2) * point[2] +
        view_projection(1, 3);
    const float w =
        view_projection(3, 0) * point[0] +
        view_projection(3, 1) * point[1] +
        view_projection(3, 2) * point[2] +
        view_projection(3, 3);
    if (w <= 0.0001f)
        return false;

    const float ndc_x = x / w;
    const float ndc_y = y / w;
    screen.x = viewport.screen_x() + (ndc_x * 0.5f + 0.5f) * static_cast<float>(viewport.width());
    screen.y = viewport.screen_y() + (1.0f - (ndc_y * 0.5f + 0.5f)) * static_cast<float>(viewport.height());
    return true;
}

bool active_view_projection(
    const editor_scene_state& editor_scene,
    const arc::editor::editor_viewport& viewport,
    arc::math::matrix4f& view_projection) noexcept
{
    const auto* camera_transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.camera_entity);
    const auto* camera = editor_scene.scene.try_get<arc::scene::camera_component>(editor_scene.camera_entity);
    if (!camera_transform || !camera || !viewport.valid())
        return false;

    const float aspect = viewport.height() == 0 ? 1.0f : static_cast<float>(viewport.width()) / static_cast<float>(viewport.height());
    view_projection = arc::scene::view_projection(*camera, *camera_transform, aspect);
    return true;
}

void draw_projected_line(
    ImDrawList* draw_list,
    const arc::math::matrix4f& view_projection,
    const arc::editor::editor_viewport& viewport,
    const arc::math::vector3f& a,
    const arc::math::vector3f& b,
    ImU32 color,
    float thickness)
{
    ImVec2 screen_a{};
    ImVec2 screen_b{};
    if (project_world_point(view_projection, viewport, a, screen_a) &&
        project_world_point(view_projection, viewport, b, screen_b))
    {
        draw_list->AddLine(screen_a, screen_b, color, thickness);
    }
}

void draw_world_grid(
    const editor_scene_state& editor_scene,
    const arc::editor::editor_viewport& viewport,
    bool enabled)
{
    if (!enabled)
        return;

    arc::math::matrix4f view_projection{};
    if (!active_view_projection(editor_scene, viewport, view_projection))
        return;

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    draw_list->PushClipRect(
        ImVec2(viewport.screen_x(), viewport.screen_y()),
        ImVec2(
            viewport.screen_x() + static_cast<float>(viewport.width()),
            viewport.screen_y() + static_cast<float>(viewport.height())),
        true);
    const ImU32 major = IM_COL32(120, 135, 150, 90);
    const ImU32 minor = IM_COL32(120, 135, 150, 35);
    for (int line = -4; line <= 4; ++line)
    {
        const ImU32 color = line == 0 ? major : minor;
        const float thickness = line == 0 ? 1.5f : 1.0f;
        draw_projected_line(
            draw_list,
            view_projection,
            viewport,
            arc::math::vector3f{ static_cast<float>(line), 0.0f, -4.0f },
            arc::math::vector3f{ static_cast<float>(line), 0.0f, 4.0f },
            color,
            thickness);
        draw_projected_line(
            draw_list,
            view_projection,
            viewport,
            arc::math::vector3f{ -4.0f, 0.0f, static_cast<float>(line) },
            arc::math::vector3f{ 4.0f, 0.0f, static_cast<float>(line) },
            color,
            thickness);
    }
    draw_list->PopClipRect();
}

void draw_selected_bounds(
    const editor_scene_state& editor_scene,
    const arc::editor::editor_viewport& viewport)
{
    if (!editor_scene.scene.alive(editor_scene.selected_entity))
        return;

    const auto* transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.selected_entity);
    const auto* bounds = editor_scene.scene.try_get<arc::scene::bounds_component>(editor_scene.selected_entity);
    if (!transform || !bounds)
        return;

    arc::math::matrix4f view_projection{};
    if (!active_view_projection(editor_scene, viewport, view_projection))
        return;

    const auto world = arc::editor::transformed_bounds(bounds->local_bounds, *transform);
    const std::array<arc::math::vector3f, 8> corners{
        arc::math::vector3f{ world.min[0], world.min[1], world.min[2] },
        arc::math::vector3f{ world.max[0], world.min[1], world.min[2] },
        arc::math::vector3f{ world.min[0], world.max[1], world.min[2] },
        arc::math::vector3f{ world.max[0], world.max[1], world.min[2] },
        arc::math::vector3f{ world.min[0], world.min[1], world.max[2] },
        arc::math::vector3f{ world.max[0], world.min[1], world.max[2] },
        arc::math::vector3f{ world.min[0], world.max[1], world.max[2] },
        arc::math::vector3f{ world.max[0], world.max[1], world.max[2] }
    };
    constexpr std::array<std::array<int, 2>, 12> edges{
        std::array<int, 2>{ 0, 1 },
        std::array<int, 2>{ 1, 3 },
        std::array<int, 2>{ 3, 2 },
        std::array<int, 2>{ 2, 0 },
        std::array<int, 2>{ 4, 5 },
        std::array<int, 2>{ 5, 7 },
        std::array<int, 2>{ 7, 6 },
        std::array<int, 2>{ 6, 4 },
        std::array<int, 2>{ 0, 4 },
        std::array<int, 2>{ 1, 5 },
        std::array<int, 2>{ 2, 6 },
        std::array<int, 2>{ 3, 7 }
    };

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();
    draw_list->PushClipRect(
        ImVec2(viewport.screen_x(), viewport.screen_y()),
        ImVec2(
            viewport.screen_x() + static_cast<float>(viewport.width()),
            viewport.screen_y() + static_cast<float>(viewport.height())),
        true);
    for (const auto& edge : edges)
    {
        draw_projected_line(
            draw_list,
            view_projection,
            viewport,
            corners[static_cast<std::size_t>(edge[0])],
            corners[static_cast<std::size_t>(edge[1])],
            IM_COL32(56, 152, 255, 80),
            3.0f);
        draw_projected_line(
            draw_list,
            view_projection,
            viewport,
            corners[static_cast<std::size_t>(edge[0])],
            corners[static_cast<std::size_t>(edge[1])],
            IM_COL32(92, 182, 255, 210),
            1.25f);
    }
    draw_list->PopClipRect();
}

void draw_stats_panel(
    arc::runtime& runtime,
    const arc::frame_time& time,
    const arc::editor::editor_viewport& viewport,
    const editor_asset_state& editor_assets,
    const editor_scene_state& editor_scene,
    const arc::render::renderer& renderer,
    arc::render::render_mode render_mode,
    viewport_shading_mode shading_mode,
    arc::editor::editor_tool active_tool)
{
    const arc::memory_stats memory = arc::default_memory_stats();
    const auto profile = renderer.last_frame_profile();

    ImGui::Begin("Stats");
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(time.frame_index));
    ImGui::Text("Delta: %.3f ms", time.delta_seconds * 1000.0);
    ImGui::Text("Total: %.2f s", time.total_seconds);
    ImGui::Separator();
    ImGui::Text("Workers: %zu", runtime.jobs().worker_count());
    ImGui::Text("Viewport: %u x %u", viewport.width(), viewport.height());
    ImGui::Text("Render Mode: %s", render_mode == arc::render::render_mode::wireframe ? "Wireframe" : "Shaded");
    ImGui::Text("Shading: %s", viewport_shading_label(shading_mode));
    ImGui::Text("Tool: %s", arc::editor::editor_tool_label(active_tool));
    ImGui::Text("Selected: %s", selected_entity_name(editor_scene));
    ImGui::Separator();
    ImGui::Text("Default Mesh: %s", editor_assets.default_mesh_loaded ? "Loaded" : "Missing");
    ImGui::Text("Vertices: %zu", editor_assets.default_mesh.vertices.size());
    ImGui::Text("Indices: %zu", editor_assets.default_mesh.indices.size());
    ImGui::Text("Scene Entities: %zu", editor_scene.scene.live_count());
    ImGui::Text("Active Camera: %s", editor_scene.last_render.camera_found ? "Yes" : "No");
    ImGui::Text("Renderables: %zu", editor_scene.last_render.renderable_count);
    ImGui::Text("Draws: %zu", editor_scene.last_render.submitted_draw_count);
    ImGui::Text("Selected: %zu", editor_scene.last_render.selected_count);
    ImGui::Text(
        "Lights: %zu dir / %zu point / %zu spot",
        editor_scene.last_render.directional_light_count,
        editor_scene.last_render.point_light_count,
        editor_scene.last_render.spot_light_count);
    ImGui::Text(
        "Skipped Lights: %zu dir / %zu point / %zu spot",
        editor_scene.last_render.skipped_directional_light_count,
        editor_scene.last_render.skipped_point_light_count,
        editor_scene.last_render.skipped_spot_light_count);
    if (profile.clustered_lights.available)
    {
        ImGui::Text(
            "Clusters: %u x %u x %u (%u)",
            profile.clustered_lights.tiles_x,
            profile.clustered_lights.tiles_y,
            profile.clustered_lights.depth_slices,
            profile.clustered_lights.cluster_count);
        ImGui::Text(
            "Cluster refs: %u point / %u spot / %u overflow",
            profile.clustered_lights.point_light_references,
            profile.clustered_lights.spot_light_references,
            profile.clustered_lights.overflow_count);
    }
    ImGui::Text(
        "Probes: %zu reflection / %zu irradiance",
        editor_scene.last_render.reflection_probe_count,
        editor_scene.last_render.irradiance_probe_count);
    ImGui::Text(
        "Environment: %zu sky / %zu fog",
        editor_scene.last_render.sky_atmosphere_count,
        editor_scene.last_render.height_fog_count);
    ImGui::Text(
        "Terrain/Water: %zu / %zu",
        editor_scene.last_render.terrain_count,
        editor_scene.last_render.water_count);
    ImGui::Text(
        "Vegetation: %zu patch(es), %zu instance(s)",
        editor_scene.last_render.vegetation_count,
        editor_scene.last_render.vegetation_instance_count);
    ImGui::Text("Decals: %zu", editor_scene.last_render.decal_count);
    ImGui::Text("Mesh Upload: %s", editor_scene.mesh_uploaded ? "Queued" : "Missing");
    ImGui::Separator();
    ImGui::Text("Allocations: %zu", memory.allocation_count);
    ImGui::Text("Outstanding: %zu bytes", memory.bytes_outstanding);
    ImGui::Text("Peak: %zu bytes", memory.peak_bytes_outstanding);
    ImGui::End();
}

void setup_editor_fonts(ImGuiIO& io)
{
    const float font_size = 15.0f * arc::editor::ui::ui_scale;
    const std::array<std::filesystem::path, 8> candidates{
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/SegoeUI.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/inter/Inter-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf"
    };

    for (const auto& candidate : candidates)
    {
        if (!std::filesystem::exists(candidate))
            continue;
        if (io.Fonts->AddFontFromFileTTF(candidate.string().c_str(), font_size))
            return;
    }

    io.Fonts->AddFontDefault();
}

} // namespace

int main(int, char**)
{
    if (!SDL_Init(SDL_INIT_VIDEO))
    {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    std::unique_ptr<arc::application> app = arc::create_application();
    if (!app)
    {
        SDL_Quit();
        return 2;
    }

    arc::runtime runtime(*app);
    const arc::application_config& config = runtime.config();

    SDL_WindowFlags window_flags = 0;
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    window_flags |= SDL_WINDOW_VULKAN;
#endif
    if (config.resizable)
        window_flags |= SDL_WINDOW_RESIZABLE;
    window_flags |= SDL_WINDOW_BORDERLESS;
    if (config.maximized)
        window_flags |= SDL_WINDOW_MAXIMIZED;
    if (!config.visible)
        window_flags |= SDL_WINDOW_HIDDEN;

    SDL_Window* window = SDL_CreateWindow(
        config.title.c_str(),
        static_cast<int>(config.initial_width),
        static_cast<int>(config.initial_height),
        window_flags);
    if (!window)
    {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 3;
    }
    editor_window_chrome window_chrome{};
    window_chrome.resizable = config.resizable;
    if (!SDL_SetWindowHitTest(window, editor_window_hit_test, &window_chrome))
        arc::warn("editor", std::string("SDL_SetWindowHitTest failed: ") + SDL_GetError());

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    arc::render::renderer editor_renderer;
    arc::render::renderer* editor_renderer_for_ui = &editor_renderer;
    arc::render::vulkan::vulkan_backend* vulkan_backend{};
    {
        std::uint32_t extension_count = 0;
        const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&extension_count);
        if (!extensions || extension_count == 0)
        {
            std::fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions failed: %s\n", SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 4;
        }

        arc::render::vulkan::vulkan_backend_config backend_config{};
        backend_config.instance_extensions.reserve(extension_count);
        for (std::uint32_t index = 0; index < extension_count; ++index)
            backend_config.instance_extensions.emplace_back(extensions[index]);
        backend_config.create_surface = create_sdl_vulkan_surface;
        backend_config.surface_user_data = window;

        auto backend_result = arc::render::vulkan::create_vulkan_backend(backend_config);
        if (!backend_result.succeeded())
        {
            std::fprintf(stderr, "Vulkan backend creation failed: %s\n", backend_result.message.c_str());
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 5;
        }

        editor_renderer.set_backend(std::move(backend_result.backend));
        vulkan_backend = arc::render::vulkan::as_vulkan_backend(editor_renderer.backend());
        if (!vulkan_backend)
        {
            std::fprintf(stderr, "Vulkan backend interface is unavailable\n");
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 6;
        }
    }
#else
    arc::render::renderer editor_renderer_fallback;
    arc::render::renderer* editor_renderer_for_ui = &editor_renderer_fallback;
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer)
    {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 4;
    }

    SDL_SetRenderVSync(renderer, 1);
#endif

    auto console_sink = std::make_shared<arc::editor::editor_console_sink>();
    arc::add_log_sink(console_sink);
    const editor_asset_state editor_assets = load_default_editor_assets();
    const editor_project_state project_state = make_project_state(editor_assets);
    const editor_build_state build_state = make_build_state();
    const editor_source_control_state source_control_state = make_source_control_state();
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    editor_scene_state editor_scene = create_default_editor_scene(editor_assets, &editor_renderer);
#else
    editor_scene_state editor_scene = create_default_editor_scene(editor_assets, nullptr);
#endif

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.FontGlobalScale = 1.0f;
    setup_editor_fonts(io);
    arc::editor::ui::apply_dark_theme();

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    ImGui_ImplSDL3_InitForVulkan(window);
    {
        int width = 1;
        int height = 1;
        SDL_GetWindowSize(window, &width, &height);
        std::string message;
        if (!vulkan_backend->initialize_imgui(
                static_cast<std::uint32_t>(std::max(1, width)),
                static_cast<std::uint32_t>(std::max(1, height)),
                message))
        {
            std::fprintf(stderr, "Vulkan editor presentation failed: %s\n", message.c_str());
            ImGui_ImplSDL3_Shutdown();
            ImGui::DestroyContext();
            SDL_DestroyWindow(window);
            SDL_Quit();
            return 7;
        }
        arc::info("editor", message);
    }
#else
    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);
#endif

    arc::editor::editor_viewport editor_viewport;
    editor_ui_state ui_state;
    editor_mouse_state mouse_state;
    scene_dialog_request scene_dialog;
    arc::editor::editor_scene_import_state scene_import;
    arc::editor::editor_camera_controller editor_camera;
    arc::editor::focus_selected_entity(editor_scene.scene, editor_scene.selected_entity, editor_camera);
    arc::input::input_manager input;
    input.bind_action(
        "viewport.orbit",
        { .device = arc::input::input_device_type::mouse, .code = static_cast<int>(arc::mouse_button::left) });
    input.bind_action(
        "viewport.pan",
        { .device = arc::input::input_device_type::mouse, .code = static_cast<int>(arc::mouse_button::middle) });
    input.bind_action(
        "viewport.select",
        { .device = arc::input::input_device_type::mouse, .code = static_cast<int>(arc::mouse_button::left) });
    input.bind_action(
        "viewport.focus",
        { .device = arc::input::input_device_type::keyboard, .code = SDLK_F });
    input.bind_action(
        "tool.select",
        { .device = arc::input::input_device_type::keyboard, .code = SDLK_Q });
    input.bind_action(
        "tool.translate",
        { .device = arc::input::input_device_type::keyboard, .code = SDLK_W });
    input.bind_action(
        "tool.rotate",
        { .device = arc::input::input_device_type::keyboard, .code = SDLK_E });
    input.bind_action(
        "tool.scale",
        { .device = arc::input::input_device_type::keyboard, .code = SDLK_R });
    arc::frame_time last_time{};
    runtime.start();

    while (runtime.running())
    {
        input.begin_frame();
        mouse_state.begin_frame();
        bool exit_requested = false;
        SDL_Event sdl_event{};
        while (SDL_PollEvent(&sdl_event))
        {
            ImGui_ImplSDL3_ProcessEvent(&sdl_event);
            mouse_state.shift_down = (SDL_GetModState() & SDL_KMOD_SHIFT) != 0;

            arc::event arc_event{};
            if (arc::editor::translate_sdl_event(sdl_event, arc_event))
            {
                update_mouse_state(mouse_state, arc_event);
                input.process_event(arc_event);
                runtime.dispatch(arc_event);
            }
        }
        if (input.pressed("viewport.select"))
            mouse_state.select_drag_distance = 0.0f;
        if (input.down("viewport.select"))
            mouse_state.select_drag_distance += std::sqrt(mouse_state.delta_x * mouse_state.delta_x + mouse_state.delta_y * mouse_state.delta_y);

        if (!runtime.running())
            break;

        last_time = runtime.tick();

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
        vulkan_backend->new_imgui_frame();
#else
        ImGui_ImplSDLRenderer3_NewFrame();
#endif
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        if (!ImGui::GetIO().WantTextInput)
        {
            arc::editor::apply_tool_shortcuts(input, ui_state.active_tool);
            const ImGuiIO& imgui_io = ImGui::GetIO();
            if (imgui_io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_P))
                ui_state.command_palette_open = true;
        }

        const editor_metrics_state metrics = make_metrics_state(last_time, editor_scene);

        if (scene_dialog.pending)
        {
            const auto mode = scene_dialog.replace
                ? arc::editor::editor_scene_open_mode::replace
                : arc::editor::editor_scene_open_mode::append;
            if (!arc::editor::start_scene_import(scene_import, editor_assets.root, scene_dialog.path, mode))
                arc::warn("editor.assets", "A scene import is already running");
            scene_dialog.pending = false;
            scene_dialog.path.clear();
        }

        if (arc::editor::poll_scene_import(scene_import))
        {
            for (const auto& diagnostic : scene_import.result.diagnostics)
                arc::warn("editor.assets", diagnostic);
            if (scene_import.status == arc::editor::editor_scene_import_status::succeeded)
            {
                const auto source_path = scene_import.source_path;
                const auto mode = scene_import.mode;
                auto imported = std::move(scene_import.result);
                const auto result = arc::editor::apply_scene_import_result_to_editor(
                    editor_scene,
                    *editor_renderer_for_ui,
                    source_path,
                    std::move(imported),
                    mode);
                if (!result.succeeded)
                    arc::error("editor.assets", result.message);
                arc::editor::reset_scene_import(scene_import);
            }
            else if (scene_import.status == arc::editor::editor_scene_import_status::failed)
            {
                arc::error("editor.assets", scene_import.result.message);
            }
            else if (scene_import.status == arc::editor::editor_scene_import_status::cancelled)
            {
                arc::warn("editor.assets", scene_import.result.message.empty() ? "Scene import cancelled" : scene_import.result.message);
            }
        }

        draw_dockspace(window, exit_requested, ui_state, project_state, window_chrome, scene_dialog);
        draw_scene_import_modal(scene_import);
        draw_viewport_panel(
            editor_viewport,
            ui_state,
            metrics,
            editor_scene,
            editor_assets,
            editor_renderer_for_ui
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
            ,
            vulkan_backend
#endif
        );
        if (auto* camera = editor_scene.scene.try_get<arc::scene::camera_component>(editor_scene.camera_entity))
        {
            camera->projection = ui_state.viewport_camera == viewport_camera_mode::orthographic
                ? arc::scene::camera_projection::orthographic
                : arc::scene::camera_projection::perspective;
        }
        const bool import_modal_active = scene_import.modal_open ||
            scene_import.status == arc::editor::editor_scene_import_status::running ||
            scene_import.status == arc::editor::editor_scene_import_status::failed ||
            scene_import.status == arc::editor::editor_scene_import_status::cancelled;
        if (!import_modal_active)
        {
            draw_world_grid(editor_scene, editor_viewport, ui_state.show_world_grid);
            draw_selected_bounds(editor_scene, editor_viewport);
            draw_transform_gizmo(editor_scene, editor_viewport, ui_state.active_tool);
            update_editor_camera_controls(editor_scene, editor_camera, editor_viewport, input, mouse_state);
            handle_viewport_selection(editor_scene, editor_viewport, *editor_renderer_for_ui, input, mouse_state);
        }
        if (editor_scene.focus_imported_scene_requested)
        {
            arc::editor::focus_selected_entity(editor_scene.scene, editor_scene.selected_entity, editor_camera);
            editor_scene.focus_imported_scene_requested = false;
        }
        draw_command_palette(ui_state, editor_scene, editor_camera);
        arc::editor::draw_scene_hierarchy_panel(
            editor_scene,
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
            editor_renderer
#else
            editor_renderer_fallback
#endif
        );
        arc::editor::draw_inspector_panel(editor_scene, editor_assets, editor_renderer_for_ui);
        arc::editor::draw_material_editor_panel(editor_scene, editor_assets, editor_renderer_for_ui);
        if (ui_state.show_bottom_panels)
        {
            arc::editor::draw_content_browser_panel(editor_assets, editor_scene, editor_renderer_for_ui, &scene_import);
            arc::editor::draw_console_panel(*console_sink, ui_state);
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
            arc::editor::draw_profiler_panel(editor_renderer);
#else
            arc::editor::draw_profiler_panel(editor_renderer_fallback);
#endif
            arc::editor::draw_render_graph_panel(*editor_renderer_for_ui);
            arc::editor::draw_shader_graph_panel();
            draw_stats_panel(
                runtime,
                last_time,
                editor_viewport,
                editor_assets,
                editor_scene,
                *editor_renderer_for_ui,
                render_mode_for_shading(ui_state.viewport_shading),
                ui_state.viewport_shading,
                ui_state.active_tool);
        }
        arc::editor::draw_status_bar(runtime, build_state, source_control_state);

        if (exit_requested)
            runtime.request_stop();

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
        editor_scene.last_render = arc::scene::render_scene(
            editor_scene.scene,
            editor_renderer,
            editor_viewport.width(),
            editor_viewport.height(),
            render_mode_for_shading(ui_state.viewport_shading),
            visualization_for_shading(ui_state.viewport_shading),
            overlay_for_shading(ui_state.viewport_shading),
            ui_state.show_shadows,
            arc::scene::render_environment_visibility{
                .sky = ui_state.show_sky,
                .fog = ui_state.show_fog,
                .terrain = ui_state.show_terrain,
                .water = ui_state.show_water,
                .vegetation = ui_state.show_vegetation,
                .decals = ui_state.show_decals });
        const auto submit_result = editor_renderer.render_frame(
            last_time.frame_index,
            arc::render::make_scene_draw_graph("viewport"));
        if (!submit_result.submitted && !submit_result.message.empty())
            arc::error("editor", submit_result.message);
#endif

        ImGui::Render();
#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
        int width = 1;
        int height = 1;
        SDL_GetWindowSize(window, &width, &height);
        std::string render_message;
        if (!vulkan_backend->render_imgui_frame(
                ImGui::GetDrawData(),
                static_cast<std::uint32_t>(std::max(1, width)),
                static_cast<std::uint32_t>(std::max(1, height)),
                render_message))
        {
            arc::error("editor", render_message);
            runtime.request_stop();
        }
#else
        SDL_SetRenderDrawColor(renderer, 18, 20, 24, 255);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
#endif
    }

    runtime.shutdown();

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    if (vulkan_backend)
        vulkan_backend->shutdown_imgui();
#else
    ImGui_ImplSDLRenderer3_Shutdown();
#endif
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

#if !defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
    SDL_DestroyRenderer(renderer);
#else
    editor_renderer.set_backend(nullptr);
#endif
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

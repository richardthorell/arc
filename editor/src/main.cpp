#define SDL_MAIN_HANDLED

#include <arc/editor/editor_console.h>
#include <arc/editor/editor_interaction.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/sdl_events.h>
#include <arc/framework/framework.h>
#include <arc/input/input.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
#include <arc/render/vulkan/vulkan_backend.h>
#endif

#include <SDL3/SDL.h>
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

constexpr float editor_ui_scale = 1.28f;

#if !defined(ARC_EDITOR_ASSET_ROOT)
#define ARC_EDITOR_ASSET_ROOT "assets"
#endif

#if defined(ARC_EDITOR_ENABLE_VULKAN_RENDER)
bool create_sdl_vulkan_surface(VkInstance instance, VkSurfaceKHR* surface, void* user_data)
{
    return SDL_Vulkan_CreateSurface(static_cast<SDL_Window*>(user_data), instance, nullptr, surface);
}
#endif

struct editor_asset_state
{
    std::filesystem::path root;
    std::filesystem::path default_mesh_path;
    std::filesystem::path default_vertex_shader_path;
    std::filesystem::path default_fragment_shader_path;
    arc::render::mesh_data default_mesh;
    std::string default_mesh_message;
    bool default_mesh_loaded{};
};

struct editor_scene_state
{
    arc::scene::registry scene;
    arc::render::mesh_handle default_mesh;
    arc::scene::entity camera_entity;
    arc::scene::entity sun_entity;
    arc::scene::entity mesh_entity;
    arc::scene::entity selected_entity;
    arc::scene::render_scene_result last_render;
    bool mesh_uploaded{};
    bool camera_created{};
};

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
    assets.default_vertex_shader_path = assets.root / "shaders" / "default_unlit.vert";
    assets.default_fragment_shader_path = assets.root / "shaders" / "default_unlit.frag";

    const auto mesh_result = arc::render::load_gltf_mesh(assets.default_mesh_path);
    assets.default_mesh_message = mesh_result.message;
    if (mesh_result.succeeded())
    {
        assets.default_mesh = std::move(mesh_result.mesh);
        assets.default_mesh_loaded = true;
        arc::info(
            "editor.assets",
            "Loaded default mesh '" + assets.default_mesh_path.filename().string() + "' with " +
                std::to_string(assets.default_mesh.vertices.size()) + " vertices and " +
                std::to_string(assets.default_mesh.indices.size()) + " indices");
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
        arc::info("editor.assets", "Default Phong shader sources are available");
    }
    else
    {
        arc::warn("editor.assets", "Default Phong shader source files are missing");
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
    sun_transform.rotation = arc::math::from_axis_angle(arc::math::vector3f{ 1.0f, 0.0f, 0.0f }, -0.72f);
    state.scene.emplace<arc::scene::name_component>(sun, "Sun");
    state.scene.emplace<arc::scene::tag_component>(sun, "Light");
    state.scene.emplace<arc::scene::active_component>(sun);
    state.scene.emplace<arc::scene::transform_component>(sun, sun_transform);
    state.scene.emplace<arc::scene::directional_light_component>(
        sun,
        arc::math::vector3f{ 1.0f, 0.94f, 0.82f },
        2.8f,
        true);

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
            arc::render::material_handle{},
            true);
        state.selected_entity = mesh;
    }

    return state;
}

float scaled(float value)
{
    return value * editor_ui_scale;
}

enum class viewport_shading_mode : std::uint8_t
{
    wireframe,
    standard,
    albedo,
    opacity,
    world_normal,
    specularity,
    gloss,
    metalness,
    ao,
    emission,
    lighting,
    uv0
};

enum class viewport_camera_mode : std::uint8_t
{
    perspective,
    orthographic
};

const char* viewport_shading_label(viewport_shading_mode mode) noexcept
{
    switch (mode)
    {
    case viewport_shading_mode::wireframe:
        return "Wireframe";
    case viewport_shading_mode::standard:
        return "Standard";
    case viewport_shading_mode::albedo:
        return "Albedo";
    case viewport_shading_mode::opacity:
        return "Opacity";
    case viewport_shading_mode::world_normal:
        return "World Normal";
    case viewport_shading_mode::specularity:
        return "Specularity";
    case viewport_shading_mode::gloss:
        return "Gloss";
    case viewport_shading_mode::metalness:
        return "Metalness";
    case viewport_shading_mode::ao:
        return "AO";
    case viewport_shading_mode::emission:
        return "Emission";
    case viewport_shading_mode::lighting:
        return "Lighting";
    case viewport_shading_mode::uv0:
        return "UV0";
    }
    return "Standard";
}

const char* viewport_camera_label(viewport_camera_mode mode) noexcept
{
    return mode == viewport_camera_mode::orthographic ? "Orthographic" : "Perspective";
}

arc::render::render_mode render_mode_for_shading(viewport_shading_mode mode) noexcept
{
    return mode == viewport_shading_mode::wireframe
        ? arc::render::render_mode::wireframe
        : arc::render::render_mode::shaded;
}

arc::render::mesh_visualization_mode visualization_for_shading(viewport_shading_mode mode) noexcept
{
    switch (mode)
    {
    case viewport_shading_mode::wireframe:
    case viewport_shading_mode::standard:
        return arc::render::mesh_visualization_mode::standard;
    case viewport_shading_mode::albedo:
        return arc::render::mesh_visualization_mode::albedo;
    case viewport_shading_mode::opacity:
        return arc::render::mesh_visualization_mode::opacity;
    case viewport_shading_mode::world_normal:
        return arc::render::mesh_visualization_mode::world_normal;
    case viewport_shading_mode::specularity:
        return arc::render::mesh_visualization_mode::specularity;
    case viewport_shading_mode::gloss:
        return arc::render::mesh_visualization_mode::gloss;
    case viewport_shading_mode::metalness:
        return arc::render::mesh_visualization_mode::metalness;
    case viewport_shading_mode::ao:
        return arc::render::mesh_visualization_mode::ao;
    case viewport_shading_mode::emission:
        return arc::render::mesh_visualization_mode::emission;
    case viewport_shading_mode::lighting:
        return arc::render::mesh_visualization_mode::lighting;
    case viewport_shading_mode::uv0:
        return arc::render::mesh_visualization_mode::uv0;
    }
    return arc::render::mesh_visualization_mode::standard;
}

arc::render::editor_overlay_mode overlay_for_shading(viewport_shading_mode mode) noexcept
{
    return mode == viewport_shading_mode::wireframe
        ? arc::render::editor_overlay_mode::none
        : arc::render::editor_overlay_mode::selected_wireframe;
}

struct editor_ui_state
{
    viewport_shading_mode viewport_shading{ viewport_shading_mode::standard };
    viewport_camera_mode viewport_camera{ viewport_camera_mode::perspective };
    arc::editor::editor_tool active_tool{ arc::editor::editor_tool::translate };
    bool show_world_grid{ true };
    bool console_collapse{ false };
    bool console_auto_scroll{ true };
    bool console_show_trace{ true };
    bool console_show_debug{ true };
    bool console_show_info{ true };
    bool console_show_warn{ true };
    bool console_show_error{ true };
    bool reset_layout{ true };
    char console_filter[128]{};
    char console_command[256]{};
};

const char* selected_entity_name(const editor_scene_state& scene, const char* fallback = "None")
{
    if (!scene.scene.alive(scene.selected_entity))
        return fallback;

    const auto* name = scene.scene.try_get<arc::scene::name_component>(scene.selected_entity);
    return name ? name->value.c_str() : fallback;
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

ImVec4 panel_bg()
{
    return ImVec4(0.055f, 0.071f, 0.086f, 1.0f);
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
    editor_window_chrome& chrome)
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
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Tools"))
    {
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
    if (ImGui::BeginMenu("Verdant Valley"))
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

void toolbar_separator()
{
    ImGui::SameLine(0.0f, scaled(8.0f));
    ImGui::TextDisabled("|");
    ImGui::SameLine(0.0f, scaled(8.0f));
}

bool toolbar_button(const char* label, bool active = false, ImVec2 size = ImVec2(0.0f, 0.0f))
{
    if (active)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.26f, 0.47f, 1.0f));
    const bool clicked = ImGui::Button(label, size);
    if (active)
        ImGui::PopStyleColor();
    return clicked;
}

void draw_toolbar(editor_ui_state& state)
{
    const ImVec2 band_min = ImGui::GetCursorScreenPos();
    const float toolbar_height = scaled(48.0f);
    ImGui::GetWindowDrawList()->AddRectFilled(
        band_min,
        ImVec2(band_min.x + ImGui::GetContentRegionAvail().x, band_min.y + toolbar_height),
        IM_COL32(16, 22, 27, 255));

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, scaled(3.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(9.0f), scaled(6.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.075f, 0.095f, 0.115f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.14f, 0.18f, 0.23f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.34f, 0.55f, 1.0f));

    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + scaled(7.0f));

    const auto tool_button = [&](arc::editor::editor_tool tool, const char* label) {
        const bool active = state.active_tool == tool;
        if (toolbar_button(label, active, ImVec2(scaled(38.0f), scaled(30.0f))))
            state.active_tool = tool;
        ImGui::SameLine(0.0f, scaled(6.0f));
    };
    toolbar_button("P", false, ImVec2(scaled(30.0f), scaled(30.0f)));
    ImGui::SameLine(0.0f, scaled(6.0f));
    toolbar_button("=", false, ImVec2(scaled(30.0f), scaled(30.0f)));
    toolbar_separator();
    tool_button(arc::editor::editor_tool::select, "S");
    tool_button(arc::editor::editor_tool::translate, "M");
    tool_button(arc::editor::editor_tool::rotate, "R");
    tool_button(arc::editor::editor_tool::scale, "T");
    toolbar_separator();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.32f, 0.18f, 1.0f));
    ImGui::Button("Play", ImVec2(scaled(54.0f), scaled(30.0f)));
    ImGui::PopStyleColor();
    ImGui::SameLine(0.0f, scaled(6.0f));
    ImGui::Button("Pause", ImVec2(scaled(62.0f), scaled(30.0f)));
    ImGui::SameLine(0.0f, scaled(6.0f));
    ImGui::Button("Step", ImVec2(scaled(54.0f), scaled(30.0f)));
    toolbar_separator();

    static const char* apis[]{ "Vulkan" };
    static int selected_api{};
    ImGui::SetNextItemWidth(scaled(128.0f));
    ImGui::Combo("##api", &selected_api, apis, 1);
    ImGui::SameLine(0.0f, scaled(8.0f));
    static const char* modes[]{ "Debug" };
    static int selected_mode{};
    ImGui::SetNextItemWidth(scaled(116.0f));
    ImGui::Combo("##mode", &selected_mode, modes, 1);
    toolbar_separator();
    toolbar_button("Grid", true, ImVec2(scaled(46.0f), scaled(30.0f)));
    ImGui::SameLine(0.0f, scaled(6.0f));
    toolbar_button("Snap", false, ImVec2(scaled(52.0f), scaled(30.0f)));
    toolbar_separator();
    ImGui::TextDisabled("Camera");
    ImGui::SameLine(0.0f, scaled(8.0f));
    static const char* camera_modes[]{ "Perspective" };
    static int selected_camera{};
    ImGui::SetNextItemWidth(scaled(148.0f));
    ImGui::Combo("##toolbar-camera", &selected_camera, camera_modes, 1);
    ImGui::SameLine(0.0f, scaled(8.0f));
    toolbar_button("Layout", false, ImVec2(scaled(72.0f), scaled(30.0f)));
    if (ImGui::IsItemClicked())
        state.reset_layout = true;

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    ImGui::Dummy(ImVec2(1.0f, scaled(7.0f)));
}

void build_default_dock_layout(ImGuiID dockspace_id, ImVec2 size)
{
    ImGui::DockBuilderRemoveNode(dockspace_id);
    ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dockspace_id, size);

    ImGuiID main_id = dockspace_id;
    const ImGuiID left_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Left, 0.17f, nullptr, &main_id);
    const ImGuiID right_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Right, 0.25f, nullptr, &main_id);
    const ImGuiID bottom_id = ImGui::DockBuilderSplitNode(main_id, ImGuiDir_Down, 0.34f, nullptr, &main_id);

    ImGuiID bottom_center_id = bottom_id;
    const ImGuiID bottom_left_id = ImGui::DockBuilderSplitNode(bottom_center_id, ImGuiDir_Left, 0.42f, nullptr, &bottom_center_id);
    const ImGuiID bottom_right_id = ImGui::DockBuilderSplitNode(bottom_center_id, ImGuiDir_Right, 0.40f, nullptr, &bottom_center_id);

    ImGui::DockBuilderDockWindow("Scene##Hierarchy", left_id);
    ImGui::DockBuilderDockWindow("Scene##Viewport", main_id);
    ImGui::DockBuilderDockWindow("Inspector", right_id);
    ImGui::DockBuilderDockWindow("Content Browser", bottom_left_id);
    ImGui::DockBuilderDockWindow("Asset Browser", bottom_left_id);
    ImGui::DockBuilderDockWindow("Console", bottom_center_id);
    ImGui::DockBuilderDockWindow("Profiler", bottom_right_id);
    ImGui::DockBuilderDockWindow("Render Graph", bottom_right_id);
    ImGui::DockBuilderDockWindow("Shader Graph", bottom_right_id);
    ImGui::DockBuilderDockWindow("Stats", bottom_right_id);
    ImGui::DockBuilderFinish(dockspace_id);
}

void draw_dockspace(SDL_Window* window, bool& exit_requested, editor_ui_state& state, editor_window_chrome& chrome)
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

    draw_main_menu(window, exit_requested, state, chrome);
    ImGui::PopStyleVar(3);
    draw_toolbar(state);
    const ImGuiID dockspace_id = ImGui::GetID("ArcEditorDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    if (state.reset_layout)
    {
        build_default_dock_layout(dockspace_id, dockspace_size);
        state.reset_layout = false;
    }
    ImGui::End();
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
        viewport_shading_mode::uv0
    };

    const float button_height = scaled(34.0f);
    const float shading_width = scaled(142.0f);
    const float camera_width = scaled(152.0f);
    const float gap = scaled(6.0f);
    const ImVec2 pos(
        std::max(origin.x + scaled(12.0f), end.x - shading_width - camera_width - gap - scaled(12.0f)),
        origin.y + scaled(12.0f));

    ImGui::SetCursorScreenPos(pos);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, scaled(2.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(scaled(12.0f), scaled(7.0f)));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.10f, 0.13f, 0.14f, 0.94f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.15f, 0.19f, 0.21f, 0.96f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.18f, 0.24f, 0.28f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.10f, 0.15f, 0.16f, 0.98f));

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
    editor_ui_state& state
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

    draw_list->AddRect(origin, end, IM_COL32(76, 88, 104, 255));

    const ImVec2 overlay_min(origin.x + scaled(14.0f), origin.y + scaled(14.0f));
    const ImVec2 overlay_max(origin.x + scaled(214.0f), origin.y + scaled(128.0f));
    draw_list->AddRectFilled(overlay_min, overlay_max, IM_COL32(8, 11, 14, 178), scaled(4.0f));
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(10.0f)), IM_COL32(212, 219, 226, 255), "FPS        120.3");
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(32.0f)), IM_COL32(212, 219, 226, 255), "Frame Time  8.31 ms");
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(54.0f)), IM_COL32(212, 219, 226, 255), "Draw Calls  0");
    draw_list->AddText(ImVec2(overlay_min.x + scaled(10.0f), overlay_min.y + scaled(76.0f)), IM_COL32(212, 219, 226, 255), "GPU Time    pending");

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
        std::array<float, 3> translation{};
        std::array<float, 3> rotation{};
        std::array<float, 3> scale{};
        ImGuizmo::DecomposeMatrixToComponents(model_data.data(), translation.data(), rotation.data(), scale.data());
        transform->set_position({ translation[0], translation[1], translation[2] });
        transform->set_rotation(arc::editor::quaternion_from_euler_degrees({ rotation[0], rotation[1], rotation[2] }));
        transform->set_scale({ scale[0], scale[1], scale[2] });
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
    const ImU32 major = IM_COL32(255, 255, 255, 150);
    const ImU32 minor = IM_COL32(255, 255, 255, 75);
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
            IM_COL32(255, 255, 255, 220),
            1.5f);
    }
    draw_list->PopClipRect();
}

void draw_scene_panel(editor_scene_state& editor_scene)
{
    ImGui::Begin("Scene##Hierarchy");
    static char scene_search[128]{};
    ImGui::SetNextItemWidth(-1.0f);
    ImGui::InputTextWithHint("##scene-search", "Search...", scene_search, sizeof(scene_search));
    ImGui::Separator();

    if (ImGui::TreeNodeEx("World", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const auto show_entity = [&](arc::scene::entity value, const char* fallback, bool selected = false) {
            const auto* name = editor_scene.scene.try_get<arc::scene::name_component>(value);
            if (ImGui::Selectable(name ? name->value.c_str() : fallback, selected))
                arc::editor::select_entity(editor_scene.scene, value, editor_scene.selected_entity);
        };

        show_entity(editor_scene.sun_entity, "Sun");
        show_entity(editor_scene.camera_entity, "Editor Camera");
        if (editor_scene.mesh_entity.valid())
        {
            const auto* selection = editor_scene.scene.try_get<arc::scene::selection_component>(editor_scene.mesh_entity);
            show_entity(editor_scene.mesh_entity, "Default Mesh", selection && selection->selected);
        }
        ImGui::TreePop();
    }

    ImGui::Separator();
    if (ImGui::BeginTabBar("scene-tabs"))
    {
        if (ImGui::BeginTabItem("Scene"))
            ImGui::EndTabItem();
        if (ImGui::BeginTabItem("Layers"))
            ImGui::EndTabItem();
        ImGui::EndTabBar();
    }
    ImGui::End();
}

void draw_inspector_panel(editor_scene_state& editor_scene)
{
    ImGui::Begin("Inspector");
    if (!editor_scene.scene.alive(editor_scene.selected_entity))
    {
        ImGui::TextDisabled("No entity selected");
        ImGui::End();
        return;
    }

    const auto* name = editor_scene.scene.try_get<arc::scene::name_component>(editor_scene.selected_entity);
    ImGui::TextUnformatted(name ? name->value.c_str() : "Unnamed Entity");
    ImGui::SameLine(ImGui::GetWindowWidth() - 112.0f);
    bool active = true;
    if (auto* active_component = editor_scene.scene.try_get<arc::scene::active_component>(editor_scene.selected_entity))
        active = active_component->active;
    if (ImGui::Checkbox("Active", &active))
        editor_scene.scene.emplace<arc::scene::active_component>(editor_scene.selected_entity, active);
    ImGui::Separator();

    std::array<char, 96> tag_text{};
    if (const auto* tag = editor_scene.scene.try_get<arc::scene::tag_component>(editor_scene.selected_entity))
        std::snprintf(tag_text.data(), tag_text.size(), "%s", tag->value.c_str());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputText("Tag", tag_text.data(), tag_text.size()))
        editor_scene.scene.emplace<arc::scene::tag_component>(editor_scene.selected_entity, std::string{ tag_text.data() });

    if (auto* transform = editor_scene.scene.try_get<arc::scene::transform_component>(editor_scene.selected_entity))
    {
        if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen))
        {
            std::array<float, 3> position{ transform->position[0], transform->position[1], transform->position[2] };
            auto euler = arc::editor::euler_degrees_from_quaternion(transform->rotation);
            std::array<float, 3> rotation{ euler[0], euler[1], euler[2] };
            std::array<float, 3> scale{ transform->scale[0], transform->scale[1], transform->scale[2] };
            if (ImGui::DragFloat3("Position", position.data(), 0.01f))
                transform->set_position({ position[0], position[1], position[2] });
            if (ImGui::DragFloat3("Rotation", rotation.data(), 0.1f))
                transform->set_rotation(arc::editor::quaternion_from_euler_degrees({ rotation[0], rotation[1], rotation[2] }));
            if (ImGui::DragFloat3("Scale", scale.data(), 0.01f, 0.001f, 1000.0f))
                transform->set_scale({ scale[0], scale[1], scale[2] });
        }
    }

    if (editor_scene.scene.has<arc::scene::mesh_renderer_component>(editor_scene.selected_entity) &&
        ImGui::CollapsingHeader("Mesh Renderer", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextUnformatted("Static mesh renderer");
        ImGui::Text("Mesh handle: %u:%u", editor_scene.default_mesh.index, editor_scene.default_mesh.generation);
    }

    if (auto* light = editor_scene.scene.try_get<arc::scene::directional_light_component>(editor_scene.selected_entity))
    {
        if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen))
        {
            std::array<float, 3> color{ light->color[0], light->color[1], light->color[2] };
            ImGui::ColorEdit3("Color", color.data());
            light->color = { color[0], color[1], color[2] };
            ImGui::DragFloat("Intensity", &light->intensity, 0.05f, 0.0f, 100.0f);
            ImGui::Checkbox("Cast Shadows", &light->casts_shadows);
        }
    }

    ImGui::End();
}

void draw_asset_tile(const char* label, const ImVec4& color)
{
    ImGui::PushID(label);
    ImGui::BeginGroup();

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 size(scaled(82.0f), scaled(82.0f));

    ImGui::InvisibleButton("asset-tile", size);
    draw_list->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), ImGui::ColorConvertFloat4ToU32(color), 4.0f);
    draw_list->AddRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), IM_COL32(46, 56, 66, 255), 4.0f);

    ImGui::SetNextItemWidth(size.x);
    ImGui::TextUnformatted(label);

    ImGui::EndGroup();
    ImGui::PopID();
}

void draw_content_browser_panel(const editor_asset_state& editor_assets)
{
    ImGui::Begin("Content Browser");
    ImGui::BeginChild("asset-tree", ImVec2(130.0f, 0.0f), true);
    ImGui::TextDisabled("Favorites");
    ImGui::Separator();
    if (ImGui::TreeNodeEx("Assets", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TreeNodeEx("_Core", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::TreeNodeEx("Animations", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::TreeNodeEx("Materials", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::TreeNodeEx("Meshes", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::Selectable("Scenes", true);
        ImGui::TreeNodeEx("Shaders", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::TreeNodeEx("Textures", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen);
        ImGui::TreePop();
    }
    ImGui::Separator();
    ImGui::TextDisabled("Packages");
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("asset-grid", ImVec2(0.0f, 0.0f), true);
    ImGui::TextDisabled("<  >   Assets");
    ImGui::SameLine(ImGui::GetWindowWidth() - 176.0f);
    static char asset_search[128]{};
    ImGui::SetNextItemWidth(150.0f);
    ImGui::InputTextWithHint("##asset-search", "Search...", asset_search, sizeof(asset_search));
    ImGui::Separator();

    const std::array<std::pair<const char*, ImVec4>, 13> assets{ {
        { "DemoScene", ImVec4(0.55f, 0.40f, 0.20f, 1.0f) },
        { "Environment", ImVec4(0.62f, 0.44f, 0.22f, 1.0f) },
        { "SampleScene", ImVec4(0.65f, 0.49f, 0.26f, 1.0f) },
        { "Menu.scene", ImVec4(0.20f, 0.24f, 0.30f, 1.0f) },
        { "Sandbox", ImVec4(0.20f, 0.24f, 0.30f, 1.0f) },
        { "UAL2_Standard.glb", ImVec4(0.31f, 0.42f, 0.56f, 1.0f) },
        { "default_unlit.vert", ImVec4(0.30f, 0.50f, 0.42f, 1.0f) },
        { "default_unlit.frag", ImVec4(0.30f, 0.50f, 0.42f, 1.0f) },
        { "Forest", ImVec4(0.18f, 0.36f, 0.20f, 1.0f) },
        { "Rock", ImVec4(0.37f, 0.37f, 0.35f, 1.0f) },
        { "Crate", ImVec4(0.46f, 0.31f, 0.18f, 1.0f) },
        { "Lamp", ImVec4(0.72f, 0.55f, 0.24f, 1.0f) },
        { "Player", ImVec4(0.34f, 0.42f, 0.52f, 1.0f) },
    } };

    const float tile_width = scaled(104.0f);
    const int columns = std::max(1, static_cast<int>(ImGui::GetContentRegionAvail().x / tile_width));
    ImGui::Columns(columns, nullptr, false);
    for (const auto& asset : assets)
    {
        draw_asset_tile(asset.first, asset.second);
        ImGui::NextColumn();
    }
    ImGui::Columns(1);
    ImGui::EndChild();

    if (editor_assets.default_mesh_loaded && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows))
    {
        ImGui::SetTooltip(
            "Default mesh loaded: %zu vertices, %zu indices",
            editor_assets.default_mesh.vertices.size(),
            editor_assets.default_mesh.indices.size());
    }

    ImGui::End();
}

void draw_asset_browser_panel()
{
    ImGui::Begin("Asset Browser");
    ImGui::TextDisabled("No indexed asset database yet.");
    ImGui::Separator();
    ImGui::TextUnformatted("Recent imports and dependency views will live here.");
    ImGui::End();
}

void draw_console_panel(arc::editor::editor_console_sink& sink, editor_ui_state& state)
{
    ImGui::Begin("Console");
    const auto entries = sink.entries();

    std::array<int, 5> counts{};
    for (const auto& entry : entries)
    {
        switch (entry.level)
        {
        case arc::log_level::trace:
        case arc::log_level::debug:
            ++counts[0];
            break;
        case arc::log_level::info:
            ++counts[1];
            break;
        case arc::log_level::warn:
            ++counts[2];
            break;
        case arc::log_level::error:
        case arc::log_level::fatal:
            ++counts[3];
            break;
        }
    }

    if (ImGui::Button("Clear"))
        sink.clear();
    ImGui::SameLine();
    ImGui::Checkbox("Collapse", &state.console_collapse);
    ImGui::SameLine();
    ImGui::Checkbox("Auto-scroll", &state.console_auto_scroll);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ImGui::InputTextWithHint("##console-filter", "Search logs...", state.console_filter, sizeof(state.console_filter));

    ImGui::Checkbox("Trace", &state.console_show_trace);
    ImGui::SameLine();
    ImGui::Checkbox("Debug", &state.console_show_debug);
    ImGui::SameLine();
    ImGui::Checkbox("Info", &state.console_show_info);
    ImGui::SameLine();
    ImGui::Checkbox("Warn", &state.console_show_warn);
    ImGui::SameLine();
    ImGui::Checkbox("Error", &state.console_show_error);
    ImGui::Separator();

    draw_level_badge(arc::log_level::debug, counts[0]);
    ImGui::SameLine(0.0f, 12.0f);
    draw_level_badge(arc::log_level::info, counts[1]);
    ImGui::SameLine(0.0f, 12.0f);
    draw_level_badge(arc::log_level::warn, counts[2]);
    ImGui::SameLine(0.0f, 12.0f);
    draw_level_badge(arc::log_level::error, counts[3]);
    ImGui::Separator();

    ImGui::BeginChild("console-scroll", ImVec2(0.0f, -ImGui::GetFrameHeightWithSpacing() - 4.0f), true);
    std::string previous_key;
    for (const auto& entry : entries)
    {
        if (!level_visible(state, entry.level) || !text_matches_filter(entry, state.console_filter))
            continue;

        const std::string key = std::string(display_level(entry.level)) + entry.category + entry.message;
        if (state.console_collapse && key == previous_key)
            continue;
        previous_key = key;

        const std::string time = format_timestamp(entry.timestamp);
        ImGui::TextDisabled("[%s]", time.c_str());
        ImGui::SameLine(74.0f);

        const ImVec2 badge_min = ImGui::GetCursorScreenPos();
        const ImVec2 label_size = ImGui::CalcTextSize(display_level(entry.level).data());
        const ImVec2 badge_max(badge_min.x + label_size.x + 14.0f, badge_min.y + label_size.y + 4.0f);
        ImGui::GetWindowDrawList()->AddRectFilled(badge_min, badge_max, color_for_level_u32(entry.level, 0.90f), 3.0f);
        ImGui::GetWindowDrawList()->AddText(
            ImVec2(badge_min.x + 7.0f, badge_min.y + 2.0f),
            IM_COL32(5, 7, 9, 255),
            display_level(entry.level).data());
        ImGui::Dummy(ImVec2(label_size.x + 14.0f, label_size.y + 4.0f));
        ImGui::SameLine(128.0f);

        if (!entry.category.empty())
        {
            ImGui::TextColored(ImVec4(0.50f, 0.68f, 0.86f, 1.0f), "%s", entry.category.c_str());
            ImGui::SameLine(220.0f);
        }

        ImGui::PushStyleColor(ImGuiCol_Text, color_for_level(entry.level));
        ImGui::TextWrapped("%s", entry.message.c_str());
        ImGui::PopStyleColor();
    }

    if (state.console_auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f)
        ImGui::SetScrollHereY(1.0f);

    ImGui::EndChild();
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##console-command", "> Enter console command...", state.console_command, sizeof(state.console_command), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        if (state.console_command[0] != '\0')
        {
            arc::info("console", state.console_command);
            state.console_command[0] = '\0';
        }
    }
    ImGui::End();
}

void draw_stats_panel(
    arc::runtime& runtime,
    const arc::frame_time& time,
    const arc::editor::editor_viewport& viewport,
    const editor_asset_state& editor_assets,
    const editor_scene_state& editor_scene,
    arc::render::render_mode render_mode,
    viewport_shading_mode shading_mode,
    arc::editor::editor_tool active_tool)
{
    const arc::memory_stats memory = arc::default_memory_stats();

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
    ImGui::Text("Mesh Upload: %s", editor_scene.mesh_uploaded ? "Queued" : "Missing");
    ImGui::Separator();
    ImGui::Text("Allocations: %zu", memory.allocation_count);
    ImGui::Text("Outstanding: %zu bytes", memory.bytes_outstanding);
    ImGui::Text("Peak: %zu bytes", memory.peak_bytes_outstanding);
    ImGui::End();
}

void draw_profiler_panel(const arc::render::renderer& renderer)
{
    ImGui::Begin("Profiler");
    const auto profile = renderer.last_frame_profile();
    if (profile.summary.empty())
    {
        ImGui::TextDisabled("No GPU captures yet.");
        ImGui::Text("CPU frame: waiting for renderer timings");
        ImGui::Text("GPU frame: waiting for backend timing queries");
        ImGui::Text("Frame graph compile: pending real scene passes");
    }
    else
    {
        ImGui::Text("Frame: %llu", static_cast<unsigned long long>(profile.frame_index));
        ImGui::TextWrapped("%s", profile.summary.c_str());
        ImGui::Separator();
        if (profile.pass_timings.empty())
        {
            ImGui::TextDisabled("GPU timestamp results pending.");
        }
        else
        {
            for (const auto& timing : profile.pass_timings)
                ImGui::Text("%s: %.3f ms", timing.name.c_str(), timing.milliseconds);
        }
    }
    ImGui::End();
}

void draw_render_graph_panel()
{
    ImGui::Begin("Render Graph");
    ImGui::TextUnformatted("clear_pass -> imgui_pass -> present_pass");
    ImGui::Separator();
    ImGui::TextDisabled("Resource lifetime and barrier views will land with real scene passes.");
    ImGui::End();
}

void draw_shader_graph_panel()
{
    ImGui::Begin("Shader Graph");
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 a(origin.x + 18.0f, origin.y + 30.0f);
    const ImVec2 b(origin.x + 172.0f, origin.y + 64.0f);
    const ImVec2 c(origin.x + 308.0f, origin.y + 48.0f);
    draw_list->AddRectFilled(a, ImVec2(a.x + 104.0f, a.y + 72.0f), IM_COL32(45, 94, 72, 255), 4.0f);
    draw_list->AddText(ImVec2(a.x + 10.0f, a.y + 10.0f), IM_COL32(224, 236, 230, 255), "Texture2D");
    draw_list->AddRectFilled(b, ImVec2(b.x + 100.0f, b.y + 58.0f), IM_COL32(67, 88, 74, 255), 4.0f);
    draw_list->AddText(ImVec2(b.x + 10.0f, b.y + 10.0f), IM_COL32(224, 236, 230, 255), "Fresnel");
    draw_list->AddRectFilled(c, ImVec2(c.x + 112.0f, c.y + 92.0f), IM_COL32(58, 66, 74, 255), 4.0f);
    draw_list->AddText(ImVec2(c.x + 10.0f, c.y + 10.0f), IM_COL32(224, 236, 230, 255), "PBR Output");
    draw_list->AddBezierCubic(ImVec2(a.x + 104.0f, a.y + 34.0f), ImVec2(a.x + 146.0f, a.y + 34.0f), ImVec2(b.x - 42.0f, b.y + 28.0f), ImVec2(b.x, b.y + 28.0f), IM_COL32(70, 190, 190, 255), 2.0f);
    draw_list->AddBezierCubic(ImVec2(b.x + 100.0f, b.y + 28.0f), ImVec2(b.x + 138.0f, b.y + 28.0f), ImVec2(c.x - 38.0f, c.y + 42.0f), ImVec2(c.x, c.y + 42.0f), IM_COL32(218, 193, 69, 255), 2.0f);
    ImGui::Dummy(ImVec2(440.0f, 170.0f));
    ImGui::End();
}

void draw_status_bar(arc::runtime& runtime)
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float height = scaled(30.0f);
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x, viewport->WorkPos.y + viewport->WorkSize.y - height));
    ImGui::SetNextWindowSize(ImVec2(viewport->WorkSize.x, height));
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("ARC Editor Status", nullptr, flags);
    ImGui::PopStyleVar(2);
    ImGui::TextUnformatted("Ready");
    ImGui::SameLine(scaled(220.0f));
    ImGui::TextDisabled("Branch: main");
    ImGui::SameLine(scaled(390.0f));
    ImGui::TextDisabled("No Changes");
    ImGui::SameLine(scaled(610.0f));
    ImGui::TextDisabled("Platform:");
    ImGui::SameLine();
    ImGui::TextUnformatted("Windows (x64)");
    ImGui::SameLine(scaled(860.0f));
    ImGui::TextDisabled("Build:");
    ImGui::SameLine();
    ImGui::TextUnformatted("Development");
    ImGui::SameLine(ImGui::GetWindowWidth() - scaled(330.0f));
    ImGui::Text("Workers: %zu", runtime.jobs().worker_count());
    ImGui::SameLine(ImGui::GetWindowWidth() - scaled(155.0f));
    ImGui::ProgressBar(0.38f, ImVec2(scaled(96.0f), scaled(18.0f)), "38%");
    ImGui::End();
}

void setup_editor_fonts(ImGuiIO& io)
{
    const float font_size = 15.0f * editor_ui_scale;
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

void apply_editor_style()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 2.0f;
    style.FrameRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.GrabRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.ScrollbarRounding = 2.0f;
    style.WindowPadding = ImVec2(8.0f, 7.0f);
    style.FramePadding = ImVec2(8.0f, 4.0f);
    style.ItemSpacing = ImVec2(7.0f, 5.0f);
    style.ScaleAllSizes(editor_ui_scale);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_Text] = ImVec4(0.82f, 0.86f, 0.90f, 1.0f);
    colors[ImGuiCol_TextDisabled] = ImVec4(0.46f, 0.51f, 0.57f, 1.0f);
    colors[ImGuiCol_WindowBg] = panel_bg();
    colors[ImGuiCol_ChildBg] = ImVec4(0.045f, 0.058f, 0.071f, 1.0f);
    colors[ImGuiCol_PopupBg] = ImVec4(0.055f, 0.069f, 0.084f, 1.0f);
    colors[ImGuiCol_Border] = ImVec4(0.16f, 0.20f, 0.24f, 1.0f);
    colors[ImGuiCol_FrameBg] = ImVec4(0.075f, 0.090f, 0.108f, 1.0f);
    colors[ImGuiCol_FrameBgHovered] = ImVec4(0.12f, 0.16f, 0.20f, 1.0f);
    colors[ImGuiCol_FrameBgActive] = ImVec4(0.15f, 0.22f, 0.30f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.070f, 0.086f, 0.102f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.085f, 0.105f, 0.125f, 1.0f);
    colors[ImGuiCol_MenuBarBg] = ImVec4(0.050f, 0.061f, 0.073f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.12f, 0.25f, 0.42f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.16f, 0.32f, 0.52f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.18f, 0.38f, 0.62f, 1.0f);
    colors[ImGuiCol_Button] = ImVec4(0.085f, 0.105f, 0.125f, 1.0f);
    colors[ImGuiCol_ButtonHovered] = ImVec4(0.13f, 0.18f, 0.24f, 1.0f);
    colors[ImGuiCol_ButtonActive] = ImVec4(0.16f, 0.26f, 0.38f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.075f, 0.092f, 0.110f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.16f, 0.27f, 0.42f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.12f, 0.20f, 0.32f, 1.0f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.16f, 0.42f, 0.78f, 0.70f);
    colors[ImGuiCol_Separator] = ImVec4(0.14f, 0.17f, 0.20f, 1.0f);
    colors[ImGuiCol_CheckMark] = ImVec4(0.34f, 0.62f, 1.0f, 1.0f);
    colors[ImGuiCol_SliderGrab] = ImVec4(0.31f, 0.53f, 0.92f, 1.0f);
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
    apply_editor_style();

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
            arc::editor::apply_tool_shortcuts(input, ui_state.active_tool);

        draw_dockspace(window, exit_requested, ui_state, window_chrome);
        draw_viewport_panel(
            editor_viewport,
            ui_state
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
        draw_world_grid(editor_scene, editor_viewport, ui_state.show_world_grid);
        draw_selected_bounds(editor_scene, editor_viewport);
        draw_transform_gizmo(editor_scene, editor_viewport, ui_state.active_tool);
        update_editor_camera_controls(editor_scene, editor_camera, editor_viewport, input, mouse_state);
        handle_viewport_selection(editor_scene, editor_viewport, input, mouse_state);
        draw_scene_panel(editor_scene);
        draw_inspector_panel(editor_scene);
        draw_content_browser_panel(editor_assets);
        draw_asset_browser_panel();
        draw_console_panel(*console_sink, ui_state);
        draw_profiler_panel(editor_renderer);
        draw_render_graph_panel();
        draw_shader_graph_panel();
        draw_stats_panel(
            runtime,
            last_time,
            editor_viewport,
            editor_assets,
            editor_scene,
            render_mode_for_shading(ui_state.viewport_shading),
            ui_state.viewport_shading,
            ui_state.active_tool);
        draw_status_bar(runtime);

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
            overlay_for_shading(ui_state.viewport_shading));
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

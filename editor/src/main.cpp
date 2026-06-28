#define SDL_MAIN_HANDLED

#include <arc/editor/editor_console.h>
#include <arc/editor/editor_viewport.h>
#include <arc/editor/sdl_events.h>
#include <arc/framework.h>

#include <SDL3/SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string_view>

namespace
{

ImVec4 color_for_level(arc::log_level level)
{
    switch (level)
    {
    case arc::log_level::trace:
        return ImVec4(0.62f, 0.65f, 0.70f, 1.0f);
    case arc::log_level::debug:
        return ImVec4(0.52f, 0.74f, 1.0f, 1.0f);
    case arc::log_level::info:
        return ImVec4(0.84f, 0.86f, 0.90f, 1.0f);
    case arc::log_level::warn:
        return ImVec4(1.0f, 0.78f, 0.35f, 1.0f);
    case arc::log_level::error:
    case arc::log_level::fatal:
        return ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
    }

    return ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
}

void draw_main_menu(bool& exit_requested)
{
    if (!ImGui::BeginMenuBar())
        return;

    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("Exit"))
            exit_requested = true;
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Window"))
    {
        ImGui::MenuItem("Viewport", nullptr, true, false);
        ImGui::MenuItem("Scene", nullptr, true, false);
        ImGui::MenuItem("Inspector", nullptr, true, false);
        ImGui::MenuItem("Console", nullptr, true, false);
        ImGui::MenuItem("Stats", nullptr, true, false);
        ImGui::EndMenu();
    }

    ImGui::EndMenuBar();
}

void draw_dockspace(bool& exit_requested)
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
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
    ImGui::Begin("ARC Editor Dockspace", nullptr, flags);
    ImGui::PopStyleVar(2);

    draw_main_menu(exit_requested);
    ImGui::DockSpace(ImGui::GetID("ArcEditorDockspace"), ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();
}

void draw_viewport_panel(arc::editor::editor_viewport& viewport)
{
    ImGui::Begin("Viewport");
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 size(std::max(available.x, 1.0f), std::max(available.y, 1.0f));
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const ImVec2 end(origin.x + size.x, origin.y + size.y);

    ImGui::InvisibleButton("viewport-canvas", size);
    viewport.set_size(size.x, size.y);
    viewport.set_focused(ImGui::IsItemFocused());
    viewport.set_hovered(ImGui::IsItemHovered());

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->AddRectFilled(origin, end, IM_COL32(26, 30, 36, 255));
    draw_list->AddRect(origin, end, IM_COL32(74, 84, 100, 255));

    const char* text = "Renderer viewport";
    const ImVec2 text_size = ImGui::CalcTextSize(text);
    draw_list->AddText(
        ImVec2(origin.x + (size.x - text_size.x) * 0.5f, origin.y + (size.y - text_size.y) * 0.5f),
        IM_COL32(180, 188, 200, 255),
        text);

    ImGui::End();
}

void draw_scene_panel()
{
    ImGui::Begin("Scene");
    ImGui::TextUnformatted("Untitled Scene");
    ImGui::Separator();
    ImGui::TextUnformatted("Camera");
    ImGui::TextUnformatted("Directional Light");
    ImGui::End();
}

void draw_inspector_panel()
{
    ImGui::Begin("Inspector");
    ImGui::TextUnformatted("No entity selected");
    ImGui::End();
}

void draw_console_panel(const arc::editor::editor_console_sink& sink)
{
    ImGui::Begin("Console");
    const auto entries = sink.entries();

    if (ImGui::Button("Clear"))
        const_cast<arc::editor::editor_console_sink&>(sink).clear();
    ImGui::Separator();

    for (const auto& entry : entries)
    {
        ImGui::PushStyleColor(ImGuiCol_Text, color_for_level(entry.level));
        if (entry.category.empty())
            ImGui::TextUnformatted(entry.message.c_str());
        else
            ImGui::Text("[%s] %s", entry.category.c_str(), entry.message.c_str());
        ImGui::PopStyleColor();
    }

    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
        ImGui::SetScrollHereY(1.0f);

    ImGui::End();
}

void draw_stats_panel(arc::runtime& runtime, const arc::frame_time& time, const arc::editor::editor_viewport& viewport)
{
    const arc::memory_stats memory = arc::default_memory_stats();

    ImGui::Begin("Stats");
    ImGui::Text("Frame: %llu", static_cast<unsigned long long>(time.frame_index));
    ImGui::Text("Delta: %.3f ms", time.delta_seconds * 1000.0);
    ImGui::Text("Total: %.2f s", time.total_seconds);
    ImGui::Separator();
    ImGui::Text("Workers: %zu", runtime.jobs().worker_count());
    ImGui::Text("Viewport: %u x %u", viewport.width(), viewport.height());
    ImGui::Separator();
    ImGui::Text("Allocations: %zu", memory.allocation_count);
    ImGui::Text("Outstanding: %zu bytes", memory.bytes_outstanding);
    ImGui::Text("Peak: %zu bytes", memory.peak_bytes_outstanding);
    ImGui::End();
}

void apply_editor_style()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 3.0f;
    style.FrameRounding = 3.0f;
    style.TabRounding = 3.0f;
    style.GrabRounding = 3.0f;

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg] = ImVec4(0.11f, 0.12f, 0.14f, 1.0f);
    colors[ImGuiCol_TitleBg] = ImVec4(0.13f, 0.14f, 0.16f, 1.0f);
    colors[ImGuiCol_TitleBgActive] = ImVec4(0.17f, 0.19f, 0.22f, 1.0f);
    colors[ImGuiCol_Header] = ImVec4(0.23f, 0.30f, 0.37f, 1.0f);
    colors[ImGuiCol_HeaderHovered] = ImVec4(0.28f, 0.37f, 0.45f, 1.0f);
    colors[ImGuiCol_HeaderActive] = ImVec4(0.32f, 0.42f, 0.50f, 1.0f);
    colors[ImGuiCol_Tab] = ImVec4(0.15f, 0.17f, 0.20f, 1.0f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.27f, 0.36f, 0.44f, 1.0f);
    colors[ImGuiCol_TabSelected] = ImVec4(0.21f, 0.27f, 0.33f, 1.0f);
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

    SDL_Window* window = SDL_CreateWindow(
        config.title.c_str(),
        static_cast<int>(config.initial_width),
        static_cast<int>(config.initial_height),
        SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 3;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer)
    {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 4;
    }

    SDL_SetRenderVSync(renderer, 1);

    auto console_sink = std::make_shared<arc::editor::editor_console_sink>();
    arc::add_log_sink(console_sink);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    apply_editor_style();

    ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer3_Init(renderer);

    arc::editor::editor_viewport editor_viewport;
    arc::frame_time last_time{};
    runtime.start();

    while (runtime.running())
    {
        bool exit_requested = false;
        SDL_Event sdl_event{};
        while (SDL_PollEvent(&sdl_event))
        {
            ImGui_ImplSDL3_ProcessEvent(&sdl_event);

            arc::event arc_event{};
            if (arc::editor::translate_sdl_event(sdl_event, arc_event))
                runtime.dispatch(arc_event);
        }

        if (!runtime.running())
            break;

        last_time = runtime.tick();

        ImGui_ImplSDLRenderer3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        draw_dockspace(exit_requested);
        draw_viewport_panel(editor_viewport);
        draw_scene_panel();
        draw_inspector_panel();
        draw_console_panel(*console_sink);
        draw_stats_panel(runtime, last_time, editor_viewport);

        if (exit_requested)
            runtime.request_stop();

        ImGui::Render();
        SDL_SetRenderDrawColor(renderer, 18, 20, 24, 255);
        SDL_RenderClear(renderer);
        ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }

    runtime.shutdown();

    ImGui_ImplSDLRenderer3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

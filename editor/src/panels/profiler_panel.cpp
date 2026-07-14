#include <arc/editor/panels/profiler_panel.h>

#include <arc/editor/ui/widgets.h>

#include <imgui.h>

namespace arc::editor
{

namespace
{

const char* quality_label(render::render_quality_tier quality) noexcept
{
    switch (quality)
    {
    case render::render_quality_tier::low: return "Low";
    case render::render_quality_tier::medium: return "Standard";
    case render::render_quality_tier::high: return "High";
    default: return "Auto";
    }
}

const char* path_label(render::render_path path) noexcept
{
    switch (path)
    {
    case render::render_path::forward_plus: return "Forward+";
    case render::render_path::deferred: return "Deferred";
    default: return "Auto";
    }
}

const char* enabled_label(bool enabled) noexcept
{
    return enabled ? "on" : "off";
}

} // namespace

void draw_profiler_panel(const render::renderer& renderer)
{
    if (!ui::begin_panel("Profiler"))
    {
        ui::end_panel();
        return;
    }

    const auto profile = renderer.last_frame_profile();
    const auto* backend = renderer.backend();
    if (backend && ui::section_header("Adapter & Render Path", true))
    {
        const auto& capabilities = backend->capabilities();
        const auto& configuration = renderer.resolved_config();
        ImGui::Text("%s", capabilities.adapter_name.empty() ? "Unknown adapter" : capabilities.adapter_name.c_str());
        ImGui::Text("Vulkan %u.%u  |  %s / %s",
            capabilities.api_major,
            capabilities.api_minor,
            quality_label(configuration.quality),
            path_label(configuration.path));
        ImGui::Text("Memory: %llu MiB budget, %llu MiB used",
            static_cast<unsigned long long>(capabilities.memory_budget / (1024ull * 1024ull)),
            static_cast<unsigned long long>(capabilities.memory_usage / (1024ull * 1024ull)));
        ImGui::TextWrapped("Dynamic rendering %s | Sync2 %s | Timeline %s | Descriptor indexing %s",
            enabled_label(configuration.features.dynamic_rendering),
            enabled_label(configuration.features.synchronization2),
            enabled_label(configuration.features.timeline_semaphores),
            enabled_label(configuration.features.descriptor_indexing));
        ImGui::Text("Render scale: %.0f%% (minimum %.0f%%)  |  target %.2f ms",
            configuration.render_scale * 100.0f,
            configuration.minimum_render_scale * 100.0f,
            configuration.target_frame_time_ms);
        for (const auto& reason : configuration.fallback_reasons)
            ui::muted_text(reason.c_str());
        ImGui::TreePop();
    }
    if (profile.summary.empty())
    {
        ui::empty_state("No GPU capture yet", "Run a frame with GPU timing enabled to inspect pass timings.");
    }
    else
    {
        if (ui::section_header("Frame", true))
        {
            ImGui::Text("Frame: %llu", static_cast<unsigned long long>(profile.frame_index));
            ImGui::TextWrapped("%s", profile.summary.c_str());
            ImGui::TreePop();
        }
        if (ui::section_header("World Environment", true))
        {
            const auto& environment = profile.environment;
            ImGui::Text("Source: %s", environment.source.empty() ? "None" : environment.source.c_str());
            ImGui::Text("Path: %s", environment.quality_path.empty() ? "Unavailable" : environment.quality_path.c_str());
            ImGui::Text("Visible: %s  |  Lighting: %s",
                enabled_label(environment.sky_visible),
                enabled_label(environment.affects_lighting));
            ImGui::TextWrapped("Atmosphere LUTs: %s",
                environment.atmosphere_lut_state.empty() ? "Unavailable" : environment.atmosphere_lut_state.c_str());
            ImGui::TextWrapped("Environment lighting: %s",
                environment.environment_lighting_state.empty() ? "Unavailable" : environment.environment_lighting_state.c_str());
            if (environment.cloud_shadow_resolution > 0)
                ImGui::Text("Cloud shadow: %ux%u", environment.cloud_shadow_resolution, environment.cloud_shadow_resolution);
            else
                ui::muted_text("Cloud shadow: inactive");
            if (!environment.fallback_reason.empty())
                ImGui::TextWrapped("Fallback: %s", environment.fallback_reason.c_str());
            ImGui::TreePop();
        }
        if (ui::section_header("Pass Timings", true))
        {
            if (profile.pass_timings.empty())
                ui::muted_text("GPU timestamp results pending.");
            else
            {
                for (const auto& timing : profile.pass_timings)
                    ImGui::Text("%s: %.3f ms", timing.name.c_str(), timing.milliseconds);
            }
            ImGui::TreePop();
        }
    }

    ui::end_panel();
}

} // namespace arc::editor

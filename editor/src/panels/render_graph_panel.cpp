#include <arc/editor/panels/render_graph_panel.h>

#include <arc/editor/ui/widgets.h>
#include <arc/render/renderer.h>

#include <imgui.h>

namespace arc::editor
{

namespace
{

const char* resource_kind_label(render::render_resource_kind kind) noexcept
{
    switch (kind)
    {
    case render::render_resource_kind::color_texture:
        return "Color";
    case render::render_resource_kind::depth_texture:
        return "Depth";
    case render::render_resource_kind::buffer:
        return "Buffer";
    case render::render_resource_kind::swapchain_image:
        return "Swapchain";
    default:
        return "Unknown";
    }
}

const char* pass_kind_label(render::render_pass_kind kind) noexcept
{
    switch (kind)
    {
    case render::render_pass_kind::clear:
        return "Clear";
    case render::render_pass_kind::depth_prepass:
        return "Depth";
    case render::render_pass_kind::gbuffer:
        return "G-buffer";
    case render::render_pass_kind::lighting:
        return "Lighting";
    case render::render_pass_kind::post_process:
        return "Post";
    case render::render_pass_kind::imgui:
        return "ImGui";
    case render::render_pass_kind::present:
        return "Present";
    default:
        return "Custom";
    }
}

} // namespace

void draw_render_graph_panel(const render::renderer& renderer)
{
    if (!ui::begin_panel("Render Graph"))
    {
        ui::end_panel();
        return;
    }

    const auto profile = renderer.last_frame_profile();
    if (profile.graph.passes.empty())
    {
        ui::empty_state("No render graph yet", "Run a frame with a renderer backend to inspect compiled passes and resources.");
        ui::end_panel();
        return;
    }

    if (profile.clustered_lights.available)
    {
        ui::section_header("Clustered Lighting");
        ImGui::Text(
            "%u x %u tiles, %u depth slices, %u clusters",
            profile.clustered_lights.tiles_x,
            profile.clustered_lights.tiles_y,
            profile.clustered_lights.depth_slices,
            profile.clustered_lights.cluster_count);
        ui::muted_text("CPU-built diagnostics path; GPU clustered culling can replace this buffer later.");
    }

    ui::section_header("Passes");
    if (ImGui::BeginTable("render-graph-passes", 4, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
    {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 36.0f);
        ImGui::TableSetupColumn("Pass");
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Resources");
        ImGui::TableHeadersRow();
        for (std::size_t index = 0; index < profile.graph.passes.size(); ++index)
        {
            const auto& pass = profile.graph.passes[index];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%zu", index);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(pass.name.c_str());
            ImGui::TableNextColumn();
            ui::muted_text(pass_kind_label(pass.kind));
            ImGui::TableNextColumn();
            ImGui::Text("%zu read / %zu write", pass.reads.size(), pass.writes.size());
        }
        ImGui::EndTable();
    }

    ui::section_header("Resources");
    if (ImGui::BeginTable("render-graph-resources", 5, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
    {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Format", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Lifetime", ImGuiTableColumnFlags_WidthFixed, 90.0f);
        ImGui::TableSetupColumn("Physical", ImGuiTableColumnFlags_WidthFixed, 70.0f);
        ImGui::TableHeadersRow();
        for (std::size_t index = 0; index < profile.graph.resources.size(); ++index)
        {
            const auto& resource = profile.graph.resources[index];
            const auto* lifetime = index < profile.graph.lifetimes.size() ? &profile.graph.lifetimes[index] : nullptr;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(resource.name.c_str());
            ImGui::TableNextColumn();
            ui::muted_text(resource_kind_label(resource.kind));
            ImGui::TableNextColumn();
            const auto format = render::render_format_name(resource.format);
            ui::muted_text(format.data());
            ImGui::TableNextColumn();
            if (resource.persistent)
                ui::muted_text("Persistent");
            else if (lifetime && lifetime->first_pass != render::render_graph_resource_handle::invalid_index)
                ImGui::Text("%u-%u", lifetime->first_pass, lifetime->last_pass);
            else
                ui::muted_text("Unused");
            ImGui::TableNextColumn();
            if (lifetime)
                ImGui::Text("#%u", lifetime->physical_resource);
            else
                ui::muted_text("-");
        }
        ImGui::EndTable();
    }

    ui::section_header("Transitions");
    if (profile.graph.transitions.empty())
        ui::muted_text("No resource transitions recorded.");
    else if (ImGui::BeginTable("render-graph-transitions", 3, ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
    {
        ImGui::TableSetupColumn("Resource");
        ImGui::TableSetupColumn("From", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableSetupColumn("To", ImGuiTableColumnFlags_WidthFixed, 80.0f);
        ImGui::TableHeadersRow();
        for (const auto& transition : profile.graph.transitions)
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(transition.resource.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%u", static_cast<unsigned>(transition.before));
            ImGui::TableNextColumn();
            ImGui::Text("%u", static_cast<unsigned>(transition.after));
        }
        ImGui::EndTable();
    }

    ui::end_panel();
}

} // namespace arc::editor

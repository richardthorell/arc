#pragma once

#include <arc/editor/editor_interaction.h>
#include <arc/render/events.h>

#include <cstdint>

namespace arc::editor
{

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
    uv0,
    cascade_debug,
    shadow_mask,
    light_complexity
};

enum class viewport_camera_mode : std::uint8_t
{
    perspective,
    orthographic
};

enum class editor_layout_preset : std::uint8_t
{
    default_layout,
    scene_editing,
    profiling,
    minimal
};

struct editor_ui_state
{
    viewport_shading_mode viewport_shading{ viewport_shading_mode::standard };
    viewport_camera_mode viewport_camera{ viewport_camera_mode::perspective };
    editor_tool active_tool{ editor_tool::translate };
    bool show_world_grid{ true };
    bool show_shadows{ true };
    bool show_sky{ true };
    bool show_fog{ true };
    bool show_terrain{ true };
    bool show_water{ true };
    bool show_vegetation{ false };
    bool show_decals{ true };
    bool console_collapse{ false };
    bool console_auto_scroll{ true };
    bool console_show_trace{ true };
    bool console_show_debug{ true };
    bool console_show_info{ true };
    bool console_show_warn{ true };
    bool console_show_error{ true };
    bool reset_layout{ true };
    bool show_bottom_panels{ true };
    editor_layout_preset layout_preset{ editor_layout_preset::default_layout };
    bool command_palette_open{ false };
    char console_filter[128]{};
    char console_command[256]{};
    char command_palette_filter[128]{};
};

inline const char* viewport_shading_label(viewport_shading_mode mode) noexcept
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
    case viewport_shading_mode::cascade_debug:
        return "Cascade Debug";
    case viewport_shading_mode::shadow_mask:
        return "Shadow Mask";
    case viewport_shading_mode::light_complexity:
        return "Light Complexity";
    }
    return "Standard";
}

inline const char* viewport_camera_label(viewport_camera_mode mode) noexcept
{
    return mode == viewport_camera_mode::orthographic ? "Orthographic" : "Perspective";
}

inline render::render_mode render_mode_for_shading(viewport_shading_mode mode) noexcept
{
    return mode == viewport_shading_mode::wireframe
        ? render::render_mode::wireframe
        : render::render_mode::shaded;
}

inline render::mesh_visualization_mode visualization_for_shading(viewport_shading_mode mode) noexcept
{
    switch (mode)
    {
    case viewport_shading_mode::wireframe:
    case viewport_shading_mode::standard:
        return render::mesh_visualization_mode::standard;
    case viewport_shading_mode::albedo:
        return render::mesh_visualization_mode::albedo;
    case viewport_shading_mode::opacity:
        return render::mesh_visualization_mode::opacity;
    case viewport_shading_mode::world_normal:
        return render::mesh_visualization_mode::world_normal;
    case viewport_shading_mode::specularity:
        return render::mesh_visualization_mode::specularity;
    case viewport_shading_mode::gloss:
        return render::mesh_visualization_mode::gloss;
    case viewport_shading_mode::metalness:
        return render::mesh_visualization_mode::metalness;
    case viewport_shading_mode::ao:
        return render::mesh_visualization_mode::ao;
    case viewport_shading_mode::emission:
        return render::mesh_visualization_mode::emission;
    case viewport_shading_mode::lighting:
        return render::mesh_visualization_mode::lighting;
    case viewport_shading_mode::uv0:
        return render::mesh_visualization_mode::uv0;
    case viewport_shading_mode::cascade_debug:
        return render::mesh_visualization_mode::cascade_debug;
    case viewport_shading_mode::shadow_mask:
        return render::mesh_visualization_mode::shadow_mask;
    case viewport_shading_mode::light_complexity:
        return render::mesh_visualization_mode::light_complexity;
    }
    return render::mesh_visualization_mode::standard;
}

inline render::editor_overlay_mode overlay_for_shading(viewport_shading_mode mode) noexcept
{
    return mode == viewport_shading_mode::wireframe
        ? render::editor_overlay_mode::none
        : render::editor_overlay_mode::selected_wireframe;
}

} // namespace arc::editor

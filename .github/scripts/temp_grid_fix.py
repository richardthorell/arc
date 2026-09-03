from pathlib import Path

protocol = Path('editor/native/inc/arc/editor/host_protocol_base.h')
text = protocol.read_text()
if text.count('bool grid{false};') < 2:
    raise SystemExit('expected native grid defaults to be false')
protocol.write_text(text.replace('bool grid{false};', 'bool grid{true};'))

viewport = Path('editor/src/renderer/src/viewport/ViewportPanel.tsx')
text = viewport.read_text()
if 'grid: false,' not in text or 'useState(false)' not in text:
    raise SystemExit('viewport grid defaults not found')
text = text.replace('grid: false,', 'grid: true,', 1)
text = text.replace('const [localGridVisible, setLocalGridVisible] = useState(false);',
                    'const [localGridVisible, setLocalGridVisible] = useState(true);', 1)
viewport.write_text(text)

gizmo = Path('editor/native/src/editor_gizmo.cpp')
text = gizmo.read_text()
text = text.replace('constexpr int grid_half_line_count = 50;\n',
                    'constexpr int grid_min_half_line_count = 50;\nconstexpr int grid_max_half_line_count = 512;\n', 1)
old = '''    const float center_x = std::floor(camera_position[0] / spacing) * spacing;
    const float center_z = std::floor(camera_position[2] / spacing) * spacing;
    const float extent = spacing * static_cast<float>(grid_half_line_count);

    constexpr math::vector4f minor_color{0.30f, 0.33f, 0.37f, 0.42f};
    constexpr math::vector4f major_color{0.42f, 0.45f, 0.50f, 0.62f};
    constexpr math::vector4f x_axis_color{0.76f, 0.22f, 0.18f, 0.90f};
    constexpr math::vector4f z_axis_color{0.18f, 0.40f, 0.78f, 0.90f};
    const auto grid_color = [&](float coordinate, float center, const math::vector4f& axis_color)
    {
        if (std::abs(coordinate) <= spacing * 0.25f) return axis_color;
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        auto color = world_line % grid_major_interval == 0 ? major_color : minor_color;
        const float edge = std::clamp(std::abs(coordinate - center) / extent, 0.0f, 1.0f);
        color[3] *= 1.0f - edge * edge;
        return color;
    };

    for (int line = -grid_half_line_count; line <= grid_half_line_count; ++line)
'''
new = '''    const float center_x = std::floor(camera_position[0] / spacing) * spacing;
    const float center_z = std::floor(camera_position[2] / spacing) * spacing;
    const float minimum_extent = spacing * static_cast<float>(grid_min_half_line_count);
    const float requested_extent = std::max(150.0f, visible_height * 16.0f);
    const float maximum_extent = spacing * static_cast<float>(grid_max_half_line_count);
    const float extent = std::clamp(requested_extent, minimum_extent, maximum_extent);
    const int half_line_count = std::clamp(static_cast<int>(std::ceil(extent / spacing)), grid_min_half_line_count,
                                           grid_max_half_line_count);
    const float rendered_extent = spacing * static_cast<float>(half_line_count);

    constexpr math::vector4f minor_color{1.0f, 1.0f, 1.0f, 0.20f};
    constexpr math::vector4f major_color{1.0f, 1.0f, 1.0f, 0.38f};
    constexpr math::vector4f axis_color{1.0f, 1.0f, 1.0f, 0.55f};
    const auto grid_color = [&](float coordinate)
    {
        if (std::abs(coordinate) <= spacing * 0.25f) return axis_color;
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        return world_line % grid_major_interval == 0 ? major_color : minor_color;
    };

    for (int line = -half_line_count; line <= half_line_count; ++line)
'''
if old not in text:
    raise SystemExit('grid implementation marker not found')
text = text.replace(old, new, 1)
text = text.replace('''        stream.lines.push_back({.start = {center_x - extent, grid_height, z},
                                .end = {center_x + extent, grid_height, z},
                                .color = grid_color(z, center_z, x_axis_color),
                                .depth = render::debug_overlay_depth_mode::tested});
        stream.lines.push_back({.start = {x, grid_height, center_z - extent},
                                .end = {x, grid_height, center_z + extent},
                                .color = grid_color(x, center_x, z_axis_color),
                                .depth = render::debug_overlay_depth_mode::tested});
''', '''        stream.lines.push_back({.start = {center_x - rendered_extent, grid_height, z},
                                .end = {center_x + rendered_extent, grid_height, z},
                                .color = grid_color(z),
                                .depth = render::debug_overlay_depth_mode::tested});
        stream.lines.push_back({.start = {x, grid_height, center_z - rendered_extent},
                                .end = {x, grid_height, center_z + rendered_extent},
                                .color = grid_color(x),
                                .depth = render::debug_overlay_depth_mode::tested});
''', 1)
gizmo.write_text(text)

multi = Path('editor/native/tests/multi_viewport_surface_tests.cpp')
text = multi.read_text()
text = text.replace('.viewport_id = "material-preview", .grid = true, .camera_speed = 9.0f',
                    '.viewport_id = "material-preview", .grid = false, .camera_speed = 9.0f', 1)
text = text.replace('CHECK(scene.at("renderOptions").at("grid") == false);',
                    'CHECK(scene.at("renderOptions").at("grid") == true);', 1)
text = text.replace('CHECK(preview.at("renderOptions").at("grid") == true);',
                    'CHECK(preview.at("renderOptions").at("grid") == false);', 1)
multi.write_text(text)

editor_tests = Path('editor/native/tests/editor_tests.cpp')
text = editor_tests.read_text()
marker = 'TEST_CASE("editor gizmos keep constant screen size and hit test colored axes")\n'
if marker not in text:
    raise SystemExit('editor gizmo test marker missing')
test = r'''TEST_CASE("editor grid is white and extends beyond the default floor")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 5.0f, 8.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    arc::scene::update_world_transforms(registry);

    arc::render::debug_overlay_stream overlay;
    arc::editor::append_editor_grid_overlay(overlay, registry.get<arc::scene::camera_component>(camera_entity),
                                            registry.get<arc::scene::transform_component>(camera_entity), 720u);
    REQUIRE_FALSE(overlay.lines.empty());

    float maximum_extent = 0.0f;
    for (const auto& line : overlay.lines)
    {
        CHECK(line.color[0] == Catch::Approx(line.color[1]));
        CHECK(line.color[1] == Catch::Approx(line.color[2]));
        maximum_extent = std::max(maximum_extent, std::abs(line.start[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.start[2]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[2]));
    }
    CHECK(maximum_extent >= 50.0f);
}

'''
editor_tests.write_text(text.replace(marker, test + marker, 1))

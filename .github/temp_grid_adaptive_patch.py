from pathlib import Path


# The existing grid-color patch wires settings/protocol support. Finish the viewport
# behavior here so the validation workflow can apply and test the complete change.
gizmo = Path('editor/native/src/editor_gizmo.cpp')
text = gizmo.read_text()
start_marker = '''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform, std::uint32_t viewport_height,
                                const math::vector3f& grid_color)
'''
end_marker = '''float editor_gizmo_world_scale(const scene::camera_component& camera,
'''
start = text.find(start_marker)
if start < 0:
    raise SystemExit('patched grid function start not found')
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit('grid function end not found')

replacement = r'''void append_editor_grid_overlay(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                                const scene::transform_component& camera_transform, std::uint32_t viewport_height,
                                const math::vector3f& grid_color)
{
    constexpr float minimum_spacing = 0.1f;
    constexpr float maximum_spacing = 1000.0f;
    constexpr float minor_alpha = 0.45f;
    constexpr float major_alpha = 0.60f;
    constexpr float axis_alpha = 0.85f;
    constexpr float orthographic_opacity_scale = 1.25f;

    const float height = static_cast<float>(std::max(1u, viewport_height));
    const auto camera_position = scene::world_position(camera_transform);

    // Perspective viewports keep ARC's Y-up XZ world-reference grid. Orthographic
    // views automatically rotate the grid onto the camera plane so top/front/side
    // editing all retain the same spatial reference behavior.
    std::size_t normal_axis = 1u;
    if (camera.projection == scene::camera_projection::orthographic)
    {
        const auto forward = math::mul(matrix_axis(camera_transform.world, 2), -1.0f);
        const float x = std::abs(forward[0]);
        const float y = std::abs(forward[1]);
        const float z = std::abs(forward[2]);
        normal_axis = y >= x && y >= z ? 1u : z >= x ? 2u : 0u;
    }
    const std::size_t first_axis = normal_axis == 0u ? 1u : 0u;
    const std::size_t second_axis = normal_axis == 2u ? 1u : 2u;

    const float visible_height =
        camera.projection == scene::camera_projection::orthographic
            ? camera.orthographic_height
            : 2.0f * std::max(std::abs(camera_position[1]), 1.0f) * std::tan(camera.fov_y_radians * 0.5f);
    const float desired_spacing =
        std::clamp(visible_height * grid_target_pixel_spacing / height, minimum_spacing, maximum_spacing);

    // Visual grid scale is independent of transform snapping. Pick decade levels
    // (0.1 m, 1 m, 10 m, 100 m, 1 km) and cross-fade their hierarchy instead of
    // popping when a camera zoom crosses a threshold.
    const float level = std::floor(std::log10(desired_spacing));
    const float spacing = std::clamp(std::pow(10.0f, level), minimum_spacing, maximum_spacing);
    float level_blend{};
    if (spacing < maximum_spacing)
    {
        const float raw_blend = std::clamp(std::log10(desired_spacing / spacing), 0.0f, 1.0f);
        level_blend = raw_blend * raw_blend * (3.0f - 2.0f * raw_blend);
    }

    const float center_first = std::floor(camera_position[first_axis] / spacing) * spacing;
    const float center_second = std::floor(camera_position[second_axis] / spacing) * spacing;
    const float minimum_extent = spacing * static_cast<float>(grid_min_half_line_count);
    const float requested_extent = std::max(150.0f, visible_height * 16.0f);
    const float maximum_extent = spacing * static_cast<float>(grid_max_half_line_count);
    const float extent = std::clamp(requested_extent, minimum_extent, maximum_extent);
    const int half_line_count =
        std::clamp(static_cast<int>(std::ceil(extent / spacing)), grid_min_half_line_count, grid_max_half_line_count);
    const float rendered_extent = spacing * static_cast<float>(half_line_count);
    const float opacity_scale =
        camera.projection == scene::camera_projection::orthographic ? orthographic_opacity_scale : 1.0f;

    const auto with_alpha = [&](const math::vector3f& color, float multiplier, float alpha)
    {
        return math::vector4f{std::clamp(color[0] * multiplier, 0.0f, 1.0f),
                              std::clamp(color[1] * multiplier, 0.0f, 1.0f),
                              std::clamp(color[2] * multiplier, 0.0f, 1.0f),
                              std::clamp(alpha * opacity_scale, 0.0f, 1.0f)};
    };
    const auto minor_color = with_alpha(grid_color, 1.0f, minor_alpha);
    const auto major_color = with_alpha(grid_color, 1.45f, major_alpha);
    const auto faded_minor = math::vector4f{minor_color[0], minor_color[1], minor_color[2],
                                             minor_color[3] * (1.0f - level_blend)};
    const auto lerp_color = [](const math::vector4f& first, const math::vector4f& second, float amount)
    {
        return math::vector4f{first[0] + (second[0] - first[0]) * amount,
                              first[1] + (second[1] - first[1]) * amount,
                              first[2] + (second[2] - first[2]) * amount,
                              first[3] + (second[3] - first[3]) * amount};
    };
    const auto transitioning_major = lerp_color(major_color, minor_color, level_blend);

    constexpr std::array<math::vector3f, 3> grid_axis_colors{
        math::vector3f{0.7882353f, 0.3372549f, 0.2980392f},
        math::vector3f{0.4117647f, 0.7098039f, 0.3568627f},
        math::vector3f{0.2980392f, 0.4980392f, 0.8196079f}};
    const auto axis_color = [&](std::size_t axis)
    { return with_alpha(grid_axis_colors[axis], 1.0f, axis_alpha); };
    const auto regular_color = [&](float coordinate)
    {
        const auto world_line = static_cast<long long>(std::llround(coordinate / spacing));
        if (world_line % (grid_major_interval * grid_major_interval) == 0) return major_color;
        if (world_line % grid_major_interval == 0) return transitioning_major;
        return faded_minor;
    };
    const auto line_color = [&](float coordinate, std::size_t direction_axis)
    {
        if (std::abs(coordinate) <= spacing * 0.25f) return axis_color(direction_axis);
        return regular_color(coordinate);
    };
    const float normal_coordinate = camera_position[normal_axis] < 0.0f ? -grid_height : grid_height;
    const auto point = [&](float first, float second)
    {
        math::vector3f result{};
        result[first_axis] = first;
        result[second_axis] = second;
        result[normal_axis] = normal_coordinate;
        return result;
    };
    const auto append_line = [&](const math::vector3f& first, const math::vector3f& second,
                                 const math::vector4f& color)
    {
        if (color[3] <= 0.001f) return;
        stream.lines.push_back(
            {.start = first, .end = second, .color = color, .depth = render::debug_overlay_depth_mode::tested});
    };

    for (int line = -half_line_count; line <= half_line_count; ++line)
    {
        const float first = center_first + static_cast<float>(line) * spacing;
        const float second = center_second + static_cast<float>(line) * spacing;
        append_line(point(center_first - rendered_extent, second), point(center_first + rendered_extent, second),
                    line_color(second, first_axis));
        append_line(point(first, center_second - rendered_extent), point(first, center_second + rendered_extent),
                    line_color(first, second_axis));
    }
}

'''
text = text[:start] + replacement + text[end:]
gizmo.write_text(text)

# Replace the old all-white-grid regression with coverage for semantic axes,
# configurable neutral colors, adaptive cross-fade, and orthographic plane orientation.
test = Path('editor/native/tests/editor_tests.cpp')
text = test.read_text()
start_marker = 'TEST_CASE("editor grid is white and extends beyond the default floor")\n'
end_marker = 'TEST_CASE("editor gizmos keep constant screen size and hit test colored axes")\n'
start = text.find(start_marker)
if start < 0:
    raise SystemExit('legacy grid test start not found')
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit('legacy grid test end not found')

tests = r'''TEST_CASE("editor perspective grid uses adaptive hierarchy and semantic XZ axes")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 5.0f, 8.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    registry.emplace<arc::scene::camera_component>(camera_entity);
    arc::scene::update_world_transforms(registry);

    arc::render::debug_overlay_stream overlay;
    const arc::math::vector3f grid_color{0.2f, 0.21568628f, 0.23921569f};
    arc::editor::append_editor_grid_overlay(overlay, registry.get<arc::scene::camera_component>(camera_entity),
                                            registry.get<arc::scene::transform_component>(camera_entity), 720u,
                                            grid_color);
    REQUIRE_FALSE(overlay.lines.empty());

    float maximum_extent = 0.0f;
    bool saw_minor{};
    bool saw_major{};
    bool saw_x_axis{};
    bool saw_z_axis{};
    for (const auto& line : overlay.lines)
    {
        CHECK(line.depth == arc::render::debug_overlay_depth_mode::tested);
        maximum_extent = std::max(maximum_extent, std::abs(line.start[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.start[2]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[0]));
        maximum_extent = std::max(maximum_extent, std::abs(line.end[2]));
        if (line.color[0] == Catch::Approx(0.7882353f).margin(0.0001f) &&
            line.color[1] == Catch::Approx(0.3372549f).margin(0.0001f))
            saw_x_axis = true;
        else if (line.color[0] == Catch::Approx(0.2980392f).margin(0.0001f) &&
                 line.color[2] == Catch::Approx(0.8196079f).margin(0.0001f))
            saw_z_axis = true;
        else
        {
            saw_minor = saw_minor || line.color[3] < 0.45f;
            saw_major = saw_major || line.color[3] > 0.50f;
        }
    }
    CHECK(maximum_extent >= 50.0f);
    CHECK(saw_minor);
    CHECK(saw_major);
    CHECK(saw_x_axis);
    CHECK(saw_z_axis);
}

TEST_CASE("editor orthographic grid rotates onto the camera plane and emphasizes its axes")
{
    arc::ecs::world registry;
    const auto camera_entity = registry.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 0.0f, 10.0f};
    registry.emplace<arc::scene::transform_component>(camera_entity, camera_transform);
    arc::scene::camera_component camera;
    camera.projection = arc::scene::camera_projection::orthographic;
    camera.orthographic_height = 67.5f;
    registry.emplace<arc::scene::camera_component>(camera_entity, camera);
    arc::scene::update_world_transforms(registry);

    arc::render::debug_overlay_stream overlay;
    arc::editor::append_editor_grid_overlay(overlay, registry.get<arc::scene::camera_component>(camera_entity),
                                            registry.get<arc::scene::transform_component>(camera_entity), 720u);
    REQUIRE_FALSE(overlay.lines.empty());

    bool saw_x_axis{};
    bool saw_y_axis{};
    float minimum_grid_alpha = 1.0f;
    float maximum_grid_alpha = 0.0f;
    for (const auto& line : overlay.lines)
    {
        CHECK(std::abs(line.start[2]) < 0.01f);
        CHECK(std::abs(line.end[2]) < 0.01f);
        const bool x_axis = line.color[0] == Catch::Approx(0.7882353f).margin(0.0001f) &&
                            line.color[1] == Catch::Approx(0.3372549f).margin(0.0001f);
        const bool y_axis = line.color[0] == Catch::Approx(0.4117647f).margin(0.0001f) &&
                            line.color[1] == Catch::Approx(0.7098039f).margin(0.0001f);
        saw_x_axis = saw_x_axis || x_axis;
        saw_y_axis = saw_y_axis || y_axis;
        if (!x_axis && !y_axis)
        {
            minimum_grid_alpha = std::min(minimum_grid_alpha, line.color[3]);
            maximum_grid_alpha = std::max(maximum_grid_alpha, line.color[3]);
        }
    }
    CHECK(saw_x_axis);
    CHECK(saw_y_axis);
    CHECK(minimum_grid_alpha < 0.45f);
    CHECK(maximum_grid_alpha > minimum_grid_alpha + 0.1f);
}

'''
text = text[:start] + tests + text[end:]
test.write_text(text)

print('Applied adaptive editor grid behavior and regressions')

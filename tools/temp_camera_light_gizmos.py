from pathlib import Path

p = Path('editor/native/src/editor_gizmo.cpp')
text = p.read_text()
marker = '''void append_bounds(render::debug_overlay_stream& stream, const geometric::box3f& bounds)\n{'''
if marker not in text:
    raise SystemExit('append_bounds marker not found')
insert = r'''void append_overlay_line(render::debug_overlay_stream& stream, const math::vector3f& start,
                         const math::vector3f& end, const math::vector4f& color,
                         render::debug_overlay_depth_mode depth = render::debug_overlay_depth_mode::always)
{
    stream.lines.push_back({.start = start, .end = end, .color = color, .depth = depth});
}

void append_billboard_camera_icon(render::debug_overlay_stream& stream, const scene::transform_component& transform,
                                  const scene::camera_component& view_camera,
                                  const scene::transform_component& view_transform, std::uint32_t viewport_height,
                                  const math::vector4f& color)
{
    const auto origin = scene::world_position(transform);
    const float scale = editor_gizmo_world_scale(view_camera, view_transform, origin, viewport_height) * 0.28f;
    const auto right = matrix_axis(view_transform.world, 0);
    const auto up = matrix_axis(view_transform.world, 1);
    const auto point = [&](float x, float y)
    { return math::add(origin, math::add(math::mul(right, x * scale), math::mul(up, y * scale))); };

    const auto p0 = point(-0.55f, -0.34f);
    const auto p1 = point(0.30f, -0.34f);
    const auto p2 = point(0.30f, 0.34f);
    const auto p3 = point(-0.55f, 0.34f);
    append_overlay_line(stream, p0, p1, color);
    append_overlay_line(stream, p1, p2, color);
    append_overlay_line(stream, p2, p3, color);
    append_overlay_line(stream, p3, p0, color);
    append_overlay_line(stream, point(0.30f, -0.22f), point(0.72f, -0.45f), color);
    append_overlay_line(stream, point(0.72f, -0.45f), point(0.72f, 0.45f), color);
    append_overlay_line(stream, point(0.72f, 0.45f), point(0.30f, 0.22f), color);
}

void append_billboard_light_icon(render::debug_overlay_stream& stream, const scene::transform_component& transform,
                                 const scene::camera_component& view_camera,
                                 const scene::transform_component& view_transform, std::uint32_t viewport_height,
                                 const math::vector4f& color)
{
    constexpr std::uint32_t segments = 16;
    const auto origin = scene::world_position(transform);
    const float scale = editor_gizmo_world_scale(view_camera, view_transform, origin, viewport_height) * 0.22f;
    const auto right = matrix_axis(view_transform.world, 0);
    const auto up = matrix_axis(view_transform.world, 1);
    const auto point = [&](float angle, float radius)
    {
        return math::add(origin, math::mul(math::add(math::mul(right, std::cos(angle)), math::mul(up, std::sin(angle))),
                                           radius * scale));
    };
    for (std::uint32_t segment = 0; segment < segments; ++segment)
    {
        const float first = math::tau<float> * static_cast<float>(segment) / static_cast<float>(segments);
        const float second = math::tau<float> * static_cast<float>(segment + 1u) / static_cast<float>(segments);
        append_overlay_line(stream, point(first, 0.52f), point(second, 0.52f), color);
    }
    for (std::uint32_t ray = 0; ray < 8; ++ray)
    {
        const float angle = math::tau<float> * static_cast<float>(ray) / 8.0f;
        append_overlay_line(stream, point(angle, 0.68f), point(angle, 0.95f), color);
    }
}

void append_camera_frustum(render::debug_overlay_stream& stream, const scene::camera_component& camera,
                           const scene::transform_component& transform, float aspect)
{
    constexpr math::vector4f frustum_color{1.0f, 0.64f, 0.12f, 0.9f};
    const auto origin = scene::world_position(transform);
    const auto right = matrix_axis(transform.world, 0);
    const auto up = matrix_axis(transform.world, 1);
    const auto forward = math::mul(matrix_axis(transform.world, 2), -1.0f);
    const float near_distance = std::max(camera.near_plane, 0.01f);
    const float far_distance = std::min(camera.far_plane, std::max(near_distance + 0.5f, 6.0f));
    if (!(far_distance > near_distance)) return;

    const auto plane = [&](float distance)
    {
        float half_height{};
        if (camera.projection == scene::camera_projection::orthographic)
            half_height = std::max(0.01f, camera.orthographic_height * 0.5f);
        else
            half_height = std::tan(camera.fov_y_radians * 0.5f) * distance;
        const float half_width = half_height * std::max(aspect, 0.01f);
        const auto center = math::add(origin, math::mul(forward, distance));
        return std::array<math::vector3f, 4>{
            math::add(center, math::add(math::mul(right, -half_width), math::mul(up, -half_height))),
            math::add(center, math::add(math::mul(right, half_width), math::mul(up, -half_height))),
            math::add(center, math::add(math::mul(right, half_width), math::mul(up, half_height))),
            math::add(center, math::add(math::mul(right, -half_width), math::mul(up, half_height)))};
    };

    const auto near_corners = plane(near_distance);
    const auto far_corners = plane(far_distance);
    for (std::size_t corner = 0; corner < 4; ++corner)
    {
        append_overlay_line(stream, near_corners[corner], near_corners[(corner + 1u) % 4u], frustum_color,
                            render::debug_overlay_depth_mode::tested);
        append_overlay_line(stream, far_corners[corner], far_corners[(corner + 1u) % 4u], frustum_color,
                            render::debug_overlay_depth_mode::tested);
        append_overlay_line(stream, near_corners[corner], far_corners[corner], frustum_color,
                            render::debug_overlay_depth_mode::tested);
    }
}

void append_scene_component_icons(render::debug_overlay_stream& stream, const ecs::world& registry,
                                  ecs::entity selected, ecs::entity view_camera_entity,
                                  const scene::camera_component& view_camera,
                                  const scene::transform_component& view_transform,
                                  const editor_gizmo_context& context)
{
    constexpr math::vector4f camera_color{0.35f, 0.76f, 1.0f, 1.0f};
    constexpr math::vector4f light_color{1.0f, 0.82f, 0.24f, 1.0f};
    constexpr math::vector4f selected_color{1.0f, 0.55f, 0.08f, 1.0f};
    const auto color_for = [&](ecs::entity entity, const math::vector4f& normal)
    { return entity == selected ? selected_color : normal; };

    registry.view<scene::transform_component, scene::camera_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform, const scene::camera_component& camera)
        {
            if (entity == view_camera_entity || !camera.active) return;
            append_billboard_camera_icon(stream, transform, view_camera, view_transform, context.viewport_height,
                                         color_for(entity, camera_color));
        });

    const auto append_light = [&](ecs::entity entity, const scene::transform_component& transform, bool enabled)
    {
        if (!enabled) return;
        append_billboard_light_icon(stream, transform, view_camera, view_transform, context.viewport_height,
                                    color_for(entity, light_color));
    };
    registry.view<scene::transform_component, scene::directional_light_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform,
            const scene::directional_light_component& light) { append_light(entity, transform, light.enabled); });
    registry.view<scene::transform_component, scene::point_light_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform, const scene::point_light_component& light)
        { append_light(entity, transform, light.enabled); });
    registry.view<scene::transform_component, scene::spot_light_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform, const scene::spot_light_component& light)
        { append_light(entity, transform, light.enabled); });
    registry.view<scene::transform_component, scene::area_light_component>().each(
        [&](ecs::entity entity, const scene::transform_component& transform, const scene::area_light_component& light)
        { append_light(entity, transform, light.enabled); });
}

'''
text = text.replace(marker, insert + marker, 1)

old = '''    render::debug_overlay_stream stream;\n    const auto* transform = registry.try_get<scene::transform_component>(selected);\n    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);\n    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);\n    if (!transform || !camera || !camera_transform) return stream;'''
new = '''    render::debug_overlay_stream stream;\n    const auto* camera = registry.try_get<scene::camera_component>(camera_entity);\n    const auto* camera_transform = registry.try_get<scene::transform_component>(camera_entity);\n    if (!camera || !camera_transform) return stream;\n    append_scene_component_icons(stream, registry, selected, camera_entity, *camera, *camera_transform, context);\n    const auto* transform = registry.try_get<scene::transform_component>(selected);\n    if (!transform) return stream;\n    if (const auto* selected_camera = registry.try_get<scene::camera_component>(selected))\n    {\n        const float aspect = static_cast<float>(std::max(1u, context.viewport_width)) /\n                             static_cast<float>(std::max(1u, context.viewport_height));\n        append_camera_frustum(stream, *selected_camera, *transform, aspect);\n    }'''
if old not in text:
    raise SystemExit('build overlay preamble not found')
text = text.replace(old, new, 1)
p.write_text(text)

tests = Path('editor/native/tests/editor_tests.cpp')
test_text = tests.read_text()
test_marker = '''TEST_CASE("editor grid is adaptive and remains anchored to world axes")'''
if test_marker not in test_text:
    raise SystemExit('test insertion marker missing')
test_case = r'''TEST_CASE("editor viewport shows camera and light icons plus selected camera frustum")
{
    arc::ecs::world registry;
    const auto view_camera = registry.create();
    arc::scene::transform_component view_transform;
    view_transform.position = {0.0f, 2.0f, 8.0f};
    registry.emplace<arc::scene::transform_component>(view_camera, view_transform);
    registry.emplace<arc::scene::camera_component>(view_camera);

    const auto scene_camera = registry.create();
    arc::scene::transform_component scene_camera_transform;
    scene_camera_transform.position = {0.0f, 1.0f, 0.0f};
    registry.emplace<arc::scene::transform_component>(scene_camera, scene_camera_transform);
    registry.emplace<arc::scene::camera_component>(scene_camera);

    const auto light = registry.create();
    arc::scene::transform_component light_transform;
    light_transform.position = {2.0f, 3.0f, 1.0f};
    registry.emplace<arc::scene::transform_component>(light, light_transform);
    registry.emplace<arc::scene::point_light_component>(light);
    arc::scene::update_world_transforms(registry);

    const auto overlay = arc::editor::build_editor_gizmo_overlay(
        registry, scene_camera, view_camera,
        {.tool = arc::editor::editor_tool::select, .viewport_width = 1280, .viewport_height = 720});
    REQUIRE(overlay.lines.size() >= 40);
    const auto frustum_lines = std::count_if(overlay.lines.begin(), overlay.lines.end(), [](const auto& line) {
        return line.depth == arc::render::debug_overlay_depth_mode::tested && line.color[0] > 0.9f &&
               line.color[1] > 0.5f && line.color[1] < 0.8f;
    });
    REQUIRE(frustum_lines == 12);
}

'''
test_text = test_text.replace(test_marker, test_case + test_marker, 1)
tests.write_text(test_text)

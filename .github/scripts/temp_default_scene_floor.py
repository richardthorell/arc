from pathlib import Path

protocol = Path("editor/native/inc/arc/editor/host_protocol_base.h")
text = protocol.read_text()
count = text.count("bool grid{true};")
if count < 2:
    raise SystemExit(f"expected at least two native grid defaults, found {count}")
protocol.write_text(text.replace("bool grid{true};", "bool grid{false};"))

viewport = Path("editor/src/renderer/src/viewport/ViewportPanel.tsx")
text = viewport.read_text()
if "grid: true," not in text:
    raise SystemExit("viewport grid default marker missing")
text = text.replace("grid: true,", "grid: false,", 1)
if "useState(true);" not in text:
    raise SystemExit("local grid state marker missing")
text = text.replace("const [localGridVisible, setLocalGridVisible] = useState(true);", "const [localGridVisible, setLocalGridVisible] = useState(false);", 1)
viewport.write_text(text)

state_h = Path("editor/native/inc/arc/editor/editor_state.h")
text = state_h.read_text()
marker = "    render::material_handle primitive_material;\n"
if marker not in text:
    raise SystemExit("scene material marker missing")
text = text.replace(marker, marker + "    render::material_handle floor_material;\n", 1)
declaration = "ecs::entity add_primitive_to_scene(editor_scene_state& scene, render::renderer& renderer, editor_primitive_type type);\n"
if declaration not in text:
    raise SystemExit("primitive declaration marker missing")
text = text.replace(declaration, declaration + "\necs::entity add_default_floor_to_scene(editor_scene_state& scene, render::renderer& renderer);\n", 1)
state_h.write_text(text)

state_cpp = Path("editor/native/src/editor_state.cpp")
text = state_cpp.read_text()
insertion_marker = "render::material_handle ensure_terrain_material(editor_scene_state& scene, render::renderer& renderer)\n{"
if insertion_marker not in text:
    raise SystemExit("terrain material marker missing")
floor_material = r'''render::material_handle ensure_default_floor_material(editor_scene_state& scene, render::renderer& renderer)
{
    if (scene.floor_material.valid()) return scene.floor_material;

    constexpr std::uint32_t texture_size = 256u;
    constexpr std::uint32_t checker_size = 4u;
    render::texture_data checker;
    checker.name = "Default Checker Floor";
    checker.width = texture_size;
    checker.height = texture_size;
    checker.format = render::texture_format::rgba8_srgb;
    checker.color_space = render::texture_color_space::srgb;
    checker.semantic = render::texture_semantic::base_color;
    checker.pixels.resize(static_cast<std::size_t>(texture_size) * texture_size * 4u);

    constexpr std::array<std::uint8_t, 4> light{108u, 113u, 119u, 255u};
    constexpr std::array<std::uint8_t, 4> dark{66u, 71u, 77u, 255u};
    for (std::uint32_t y = 0; y < texture_size; ++y)
    {
        for (std::uint32_t x = 0; x < texture_size; ++x)
        {
            const auto& color = ((x / checker_size) + (y / checker_size)) % 2u == 0u ? light : dark;
            const auto offset = (static_cast<std::size_t>(y) * texture_size + x) * 4u;
            for (std::size_t channel = 0; channel < color.size(); ++channel)
                checker.pixels[offset + channel] = static_cast<std::byte>(color[channel]);
        }
    }

    const auto checker_texture = renderer.create_texture(std::move(checker));
    if (checker_texture.valid()) scene.default_textures.push_back(checker_texture);

    render::material_descriptor material;
    material.name = "Default Checker Floor Material";
    material.base_color = math::vector4f::one;
    material.roughness = 0.82f;
    material.base_color_texture = checker_texture;
    scene.floor_material = renderer.create_material(material);
    return scene.floor_material.valid() ? scene.floor_material : ensure_default_material(scene, renderer);
}

'''
text = text.replace(insertion_marker, floor_material + insertion_marker, 1)

primitive_end_marker = '''    select_entity(scene.scene, entity, scene.selected_entity);
    return entity;
}

ecs::entity add_world_environment_to_scene'''
if primitive_end_marker not in text:
    raise SystemExit("primitive function end marker missing")
floor_function = r'''    select_entity(scene.scene, entity, scene.selected_entity);
    return entity;
}

ecs::entity add_default_floor_to_scene(editor_scene_state& scene, render::renderer& renderer)
{
    constexpr float floor_size = 100.0f;
    auto mesh = render::make_plane_mesh(floor_size);
    mesh.name = "Default Checker Floor";
    const auto local_bounds = bounds_for_mesh(mesh);
    const auto mesh_handle = renderer.create_mesh(mesh);
    if (!mesh_handle.valid()) return {};

    const auto entity = scene.scene.create();
    scene::transform_component transform;
    transform.position = math::vector3f{0.0f, -0.01f, 0.0f};
    scene.scene.emplace<scene::name_component>(entity, "Floor");
    scene.scene.emplace<scene::tag_component>(entity, "Environment");
    scene.scene.emplace<scene::active_component>(entity);
    scene.scene.emplace<scene::selection_component>(entity, true);
    scene.scene.emplace<scene::bounds_component>(entity, local_bounds, local_bounds, true);
    scene.scene.emplace<scene::transform_component>(entity, transform);
    scene::mesh_renderer_component renderer_component;
    renderer_component.mesh = mesh_handle;
    renderer_component.material = ensure_default_floor_material(scene, renderer);
    scene.scene.emplace<scene::mesh_renderer_component>(entity, renderer_component);
    scene.scene.emplace<scene::persistent_id_component>(entity, ecs::generate_entity_guid());
    scene.scene.emplace<scene::hierarchy_component>(entity);
    scene.asset_bindings.push_back({.entity = scene.scene.get<scene::persistent_id_component>(entity).value,
                                    .source_kind = "builtin",
                                    .subresource = "Checker Floor"});
    scene.primitive_entities.push_back(entity);
    select_entity(scene.scene, entity, scene.selected_entity);
    return entity;
}

ecs::entity add_world_environment_to_scene'''
text = text.replace(primitive_end_marker, floor_function, 1)
state_cpp.write_text(text)

host = Path("editor/native/src/arc_host_base.inc")
text = host.read_text()
old = "const auto floor = add_primitive_to_scene(state, renderer, editor_primitive_type::plane);"
if old not in text:
    raise SystemExit("default scene floor call missing")
host.write_text(text.replace(old, "const auto floor = add_default_floor_to_scene(state, renderer);", 1))

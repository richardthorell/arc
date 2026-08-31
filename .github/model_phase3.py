from pathlib import Path


def rep(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"missing pattern in {path}: {old[:120]}")
    file.write_text(text.replace(old, new, 1))


# Fix raw model drops onto the viewport so they are not gated by texture-picker logic.
p = "editor/src/preload/preload.ts"
old = """        if (!pickerTarget || !pickerImport) return;

        const asset = await waitForImportedAsset(pickerImport.path, 'texture');
        // Asset publication emits asset.changed; give React one refresh turn so the
        // existing AssetPicker candidate list contains the newly imported texture.
        await sleep(75);
        replayImportedAssetDrop(pickerTarget, asset);

        if (viewportTarget) {
          for (const file of modelFiles) {
            const imported = await importDroppedModel(file);
            const modelAsset = await waitForImportedAsset(imported.path, 'model');
            await sleep(75);
            replayImportedAssetDrop(viewportTarget, modelAsset, 'mesh', dropCoordinates);
          }
        }
"""
new = """        if (pickerTarget && pickerImport) {
          const asset = await waitForImportedAsset(pickerImport.path, 'texture');
          // Asset publication emits asset.changed; give React one refresh turn so the
          // existing AssetPicker candidate list contains the newly imported texture.
          await sleep(75);
          replayImportedAssetDrop(pickerTarget, asset);
        }

        if (viewportTarget) {
          for (const file of modelFiles) {
            const imported = await importDroppedModel(file);
            const modelAsset = await waitForImportedAsset(imported.path, 'model');
            await sleep(75);
            replayImportedAssetDrop(viewportTarget, modelAsset, 'mesh', dropCoordinates);
          }
        }
"""
rep(p, old, new)
rep(p, "console.error('[ARC] External texture import failed', error);", "console.error('[ARC] External asset import failed', error);")

# Register OBJ as an imported-scene source.
p = "engine/assets/inc/arc/assets/assets.h"
rep(
    p,
    "inline constexpr asset_importer_id material_instance{0xa7ca55e700000002ull, 0x000000000000000eull};\n",
    "inline constexpr asset_importer_id material_instance{0xa7ca55e700000002ull, 0x000000000000000eull};\n"
    "inline constexpr asset_importer_id obj{0xa7ca55e700000002ull, 0x000000000000000full};\n",
)

p = "engine/assets/src/common/asset_metadata.cpp"
rep(
    p,
    '    if (extension == ".fbx")\n        return asset_metadata{.type = asset_types::imported_scene, .importer = importer_ids::fbx};\n',
    '    if (extension == ".fbx")\n        return asset_metadata{.type = asset_types::imported_scene, .importer = importer_ids::fbx};\n'
    '    if (extension == ".obj")\n        return asset_metadata{.type = asset_types::imported_scene, .importer = importer_ids::obj};\n',
)

p = "engine/assets/src/common/asset_manager.cpp"
rep(
    p,
    '        source_blob_importer(importer_ids::fbx, asset_types::imported_scene, "FBX", {".fbx"}),\n',
    '        source_blob_importer(importer_ids::fbx, asset_types::imported_scene, "FBX", {".fbx"}),\n'
    '        source_blob_importer(importer_ids::obj, asset_types::imported_scene, "Wavefront OBJ", {".obj"}),\n',
)

# Add the native Wavefront OBJ geometry parser.
p = "engine/render/src/common/mesh.cpp"
rep(p, "#include <algorithm>\n", "#include <algorithm>\n#include <array>\n")
marker = "mesh_load_result load_gltf_mesh(const std::filesystem::path& path)\n{"
file = Path(p)
text = file.read_text()
if marker not in text:
    raise RuntimeError("load_gltf_mesh marker missing")
obj_loader = r'''mesh_load_result load_obj_mesh(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    if (!stream) return {.message = "failed to open OBJ file"};

    std::vector<std::array<float, 3>> positions;
    std::vector<std::array<float, 3>> normals;
    std::vector<std::array<float, 2>> texcoords;
    mesh_data mesh;
    mesh.name = path.stem().string();

    const auto resolve_index = [](int value, std::size_t count) -> std::optional<std::size_t>
    {
        if (value > 0 && static_cast<std::size_t>(value) <= count) return static_cast<std::size_t>(value - 1);
        if (value < 0 && static_cast<std::size_t>(-value) <= count)
            return count - static_cast<std::size_t>(-value);
        return std::nullopt;
    };

    struct face_ref
    {
        int position{};
        int texcoord{};
        int normal{};
        bool has_texcoord{};
        bool has_normal{};
    };

    const auto parse_ref = [](std::string_view token) -> std::optional<face_ref>
    {
        face_ref result;
        try
        {
            const auto first = token.find('/');
            if (first == std::string_view::npos)
            {
                result.position = std::stoi(std::string{token});
                return result;
            }
            if (first == 0) return std::nullopt;
            result.position = std::stoi(std::string{token.substr(0, first)});
            const auto second = token.find('/', first + 1);
            const auto uv = token.substr(first + 1, second == std::string_view::npos ? std::string_view::npos
                                                                                     : second - first - 1);
            if (!uv.empty())
            {
                result.texcoord = std::stoi(std::string{uv});
                result.has_texcoord = true;
            }
            if (second != std::string_view::npos && second + 1 < token.size())
            {
                result.normal = std::stoi(std::string{token.substr(second + 1)});
                result.has_normal = true;
            }
            return result;
        }
        catch (...)
        {
            return std::nullopt;
        }
    };

    const auto make_vertex = [&](const face_ref& ref) -> std::optional<mesh_vertex>
    {
        const auto position_index = resolve_index(ref.position, positions.size());
        if (!position_index) return std::nullopt;
        mesh_vertex vertex{};
        const auto& position = positions[*position_index];
        std::copy(position.begin(), position.end(), vertex.position);
        if (ref.has_texcoord)
        {
            const auto texcoord_index = resolve_index(ref.texcoord, texcoords.size());
            if (!texcoord_index) return std::nullopt;
            vertex.texcoord[0] = texcoords[*texcoord_index][0];
            vertex.texcoord[1] = texcoords[*texcoord_index][1];
        }
        if (ref.has_normal)
        {
            const auto normal_index = resolve_index(ref.normal, normals.size());
            if (!normal_index) return std::nullopt;
            const auto normal = safe_normalize(
                {normals[*normal_index][0], normals[*normal_index][1], normals[*normal_index][2]},
                {0.0f, 1.0f, 0.0f});
            vertex.normal[0] = normal[0];
            vertex.normal[1] = normal[1];
            vertex.normal[2] = normal[2];
        }
        return vertex;
    };

    std::string line;
    std::size_t line_number{};
    while (std::getline(stream, line))
    {
        ++line_number;
        std::istringstream input(line);
        std::string kind;
        input >> kind;
        if (kind.empty() || kind.starts_with('#')) continue;
        if (kind == "v")
        {
            std::array<float, 3> value{};
            if (!(input >> value[0] >> value[1] >> value[2]))
                return {.message = "invalid OBJ position at line " + std::to_string(line_number)};
            positions.push_back(value);
        }
        else if (kind == "vn")
        {
            std::array<float, 3> value{};
            if (!(input >> value[0] >> value[1] >> value[2]))
                return {.message = "invalid OBJ normal at line " + std::to_string(line_number)};
            normals.push_back(value);
        }
        else if (kind == "vt")
        {
            std::array<float, 2> value{};
            if (!(input >> value[0] >> value[1]))
                return {.message = "invalid OBJ texcoord at line " + std::to_string(line_number)};
            texcoords.push_back(value);
        }
        else if (kind == "f")
        {
            std::vector<face_ref> face;
            std::string token;
            while (input >> token)
            {
                const auto ref = parse_ref(token);
                if (!ref) return {.message = "invalid OBJ face at line " + std::to_string(line_number)};
                face.push_back(*ref);
            }
            if (face.size() < 3)
                return {.message = "OBJ face has fewer than three vertices at line " + std::to_string(line_number)};

            for (std::size_t corner = 1; corner + 1 < face.size(); ++corner)
            {
                const std::array<face_ref, 3> triangle{face[0], face[corner], face[corner + 1]};
                if (mesh.vertices.size() > std::numeric_limits<std::uint32_t>::max() - 3u)
                    return {.message = "OBJ mesh exceeds 32-bit index capacity"};
                const auto base = static_cast<std::uint32_t>(mesh.vertices.size());
                std::array<bool, 3> missing_normal{};
                for (std::size_t index = 0; index < triangle.size(); ++index)
                {
                    const auto vertex = make_vertex(triangle[index]);
                    if (!vertex) return {.message = "OBJ face index is out of range at line " + std::to_string(line_number)};
                    missing_normal[index] = !triangle[index].has_normal;
                    mesh.vertices.push_back(*vertex);
                    mesh.indices.push_back(base + static_cast<std::uint32_t>(index));
                }

                if (missing_normal[0] || missing_normal[1] || missing_normal[2])
                {
                    const auto p0 = vertex_position(mesh.vertices[base]);
                    const auto p1 = vertex_position(mesh.vertices[base + 1u]);
                    const auto p2 = vertex_position(mesh.vertices[base + 2u]);
                    const auto face_normal = safe_normalize(math::cross(math::sub(p1, p0), math::sub(p2, p0)),
                                                            {0.0f, 1.0f, 0.0f});
                    for (std::size_t index = 0; index < missing_normal.size(); ++index)
                    {
                        if (!missing_normal[index]) continue;
                        auto& vertex = mesh.vertices[base + static_cast<std::uint32_t>(index)];
                        vertex.normal[0] = face_normal[0];
                        vertex.normal[1] = face_normal[1];
                        vertex.normal[2] = face_normal[2];
                    }
                }
            }
        }
    }

    if (mesh.vertices.empty() || mesh.indices.empty()) return {.message = "OBJ contains no renderable faces"};
    generate_tangents(mesh);
    return {.mesh = std::move(mesh), .message = "loaded OBJ mesh"};
}

'''
file.write_text(text.replace(marker, obj_loader + marker, 1))

# Route OBJ through the scene importer as a one-node static scene.
p = "engine/render/src/common/mesh.cpp"
old = '''    if (extension == ".fbx")
    {
'''
new = '''    if (extension == ".obj")
    {
        if (!report_progress(progress, scene_import_stage::loading, 0.05f, "Loading OBJ"))
            return {.message = "OBJ import cancelled"};
        auto mesh_result = load_obj_mesh(path);
        scene_import_result result;
        result.import_directory = default_import_directory(path, options);
        result.manifest_path = result.import_directory / "import.json";
        result.message = mesh_result.message;
        if (!mesh_result.succeeded()) return result;
        result.meshes.push_back(std::move(mesh_result.mesh));
        result.nodes.push_back({.name = result.meshes.front().name.empty() ? path.stem().string() : result.meshes.front().name,
                                .mesh_index = 0,
                                .material_index = result.meshes.front().material_index});
        report_progress(progress, scene_import_stage::finalizing, 1.0f, "Import complete");
        result.message = "loaded OBJ scene";
        return result;
    }

    if (extension == ".fbx")
    {
'''
rep(p, old, new)

# Add a focused native regression test for quad triangulation and negative indices.
p = Path("engine/render/tests/render_tests.cpp")
text = p.read_text()
text += r'''

TEST_CASE("OBJ scene import triangulates polygons and supports negative indices", "[render][mesh][obj]")
{
    const auto path = std::filesystem::temp_directory_path() / "arc-render-obj-import-test.obj";
    {
        std::ofstream output(path, std::ios::trunc);
        REQUIRE(output.good());
        output << "v 0 0 0\n"
               << "v 1 0 0\n"
               << "v 1 1 0\n"
               << "v 0 1 0\n"
               << "vt 0 0\n"
               << "vt 1 0\n"
               << "vt 1 1\n"
               << "vt 0 1\n"
               << "f -4/-4 -3/-3 -2/-2 -1/-1\n";
    }

    arc::render::scene_import_options options;
    const auto imported = arc::render::load_scene_asset(path, options);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    REQUIRE(imported.succeeded());
    REQUIRE(imported.meshes.size() == 1);
    REQUIRE(imported.nodes.size() == 1);
    CHECK(imported.meshes.front().indices.size() == 6);
    CHECK(imported.meshes.front().vertices.size() == 6);
    CHECK(imported.nodes.front().mesh_index == 0);
}
'''
p.write_text(text)

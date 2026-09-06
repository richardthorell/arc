from pathlib import Path

path = Path('engine/render/src/common/mesh.cpp')
text = path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global text
    if old not in text:
        raise SystemExit(f'{label} pattern not found')
    text = text.replace(old, new, 1)


replace_once(
    '''std::size_t component_count(std::string_view type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    return 0;
}
''',
    '''std::size_t component_count(std::string_view type)
{
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT2") return 4;
    if (type == "MAT3") return 9;
    if (type == "MAT4") return 16;
    return 0;
}
''',
    'component_count',
)

marker = '''std::optional<std::uint32_t> read_index(const accessor& value, const std::vector<buffer_view>& views,
                                        std::span<const std::byte> bin, std::size_t index)
'''
helpers = r'''struct gltf_node_record
{
    std::string name;
    std::vector<std::size_t> children;
    std::int32_t parent{-1};
    math::vector3f translation{};
    math::quatf rotation{};
    math::vector3f scale = math::vector3f::one;
    std::size_t mesh_index{material_texture_indices::invalid};
    std::size_t skin_index{material_texture_indices::invalid};
};

std::vector<std::size_t> parse_index_array(std::string_view object, std::string_view key)
{
    std::vector<std::size_t> result;
    const auto key_pos = object.find('"' + std::string(key) + '"');
    if (key_pos == std::string_view::npos) return result;
    const auto begin = object.find('[', key_pos);
    const auto end = object.find(']', begin);
    if (begin == std::string_view::npos || end == std::string_view::npos || end <= begin) return result;
    const std::string body(object.substr(begin + 1, end - begin - 1));
    const std::regex number("\\d+");
    for (std::sregex_iterator it(body.begin(), body.end(), number), last; it != last; ++it)
        result.push_back(static_cast<std::size_t>(std::stoull((*it)[0].str())));
    return result;
}

std::vector<gltf_node_record> parse_gltf_nodes(std::string_view json)
{
    std::vector<gltf_node_record> result;
    const auto array = extract_array(json, "nodes");
    if (!array) return result;
    for (const auto& object : extract_objects(*array))
    {
        gltf_node_record node;
        node.name = parse_string(object, "name").value_or("");
        node.children = parse_index_array(object, "children");
        node.mesh_index = parse_size(object, "mesh").value_or(material_texture_indices::invalid);
        node.skin_index = parse_size(object, "skin").value_or(material_texture_indices::invalid);
        if (const auto translation = parse_float_array(object, "translation"); translation && translation->size() >= 3)
            node.translation = {(*translation)[0], (*translation)[1], (*translation)[2]};
        if (const auto rotation = parse_float_array(object, "rotation"); rotation && rotation->size() >= 4)
            node.rotation = {(*rotation)[0], (*rotation)[1], (*rotation)[2], (*rotation)[3]};
        if (const auto scale = parse_float_array(object, "scale"); scale && scale->size() >= 3)
            node.scale = {(*scale)[0], (*scale)[1], (*scale)[2]};
        result.push_back(std::move(node));
    }
    for (std::size_t parent = 0; parent < result.size(); ++parent)
        for (const auto child : result[parent].children)
            if (child < result.size()) result[child].parent = static_cast<std::int32_t>(parent);
    return result;
}

bool read_matrix4_f32(const accessor& value, const std::vector<buffer_view>& views, std::span<const std::byte> bin,
                      std::size_t index, math::matrix4f& out)
{
    if (value.type != "MAT4") return false;
    std::array<float, 16> values{};
    if (!read_vec_f32(value, views, bin, index, values.data(), values.size())) return false;
    for (std::size_t column = 0; column < 4; ++column)
        for (std::size_t row = 0; row < 4; ++row)
            out(row, column) = values[column * 4 + row];
    return true;
}

std::vector<skeleton_asset> parse_gltf_skeletons(std::string_view json, const std::vector<accessor>& accessors,
                                                 const std::vector<buffer_view>& views,
                                                 std::span<const std::byte> bin)
{
    std::vector<skeleton_asset> result;
    const auto skins = extract_array(json, "skins");
    if (!skins) return result;
    const auto nodes = parse_gltf_nodes(json);
    const auto skin_objects = extract_objects(*skins);
    result.reserve(skin_objects.size());
    for (std::size_t skin_index = 0; skin_index < skin_objects.size(); ++skin_index)
    {
        const auto& object = skin_objects[skin_index];
        const auto joint_nodes = parse_index_array(object, "joints");
        if (joint_nodes.empty()) continue;
        skeleton_asset skeleton;
        skeleton.name = parse_string(object, "name").value_or("Skeleton " + std::to_string(skin_index));
        skeleton.joints.reserve(joint_nodes.size());
        std::unordered_map<std::size_t, std::uint32_t> node_to_joint;
        for (std::uint32_t joint = 0; joint < joint_nodes.size(); ++joint)
            node_to_joint[joint_nodes[joint]] = joint;
        const auto inverse_bind_accessor = parse_size(object, "inverseBindMatrices");
        for (std::uint32_t joint = 0; joint < joint_nodes.size(); ++joint)
        {
            const auto node_index = joint_nodes[joint];
            skeleton_joint imported;
            if (node_index < nodes.size())
            {
                const auto& node = nodes[node_index];
                imported.name = node.name.empty() ? "Joint " + std::to_string(joint) : node.name;
                imported.bind_position = node.translation;
                imported.bind_rotation = node.rotation;
                imported.bind_scale = node.scale;
                if (node.parent >= 0)
                {
                    const auto parent = node_to_joint.find(static_cast<std::size_t>(node.parent));
                    if (parent != node_to_joint.end()) imported.parent = static_cast<std::int32_t>(parent->second);
                }
            }
            else
            {
                imported.name = "Joint " + std::to_string(joint);
            }
            if (inverse_bind_accessor && *inverse_bind_accessor < accessors.size())
            {
                const auto& matrices = accessors[*inverse_bind_accessor];
                if (matrices.count != joint_nodes.size() ||
                    !read_matrix4_f32(matrices, views, bin, joint, imported.inverse_bind_matrix))
                    return {};
            }
            skeleton.joints.push_back(std::move(imported));
        }
        if (const auto root_node = parse_size(object, "skeleton"))
        {
            const auto found = node_to_joint.find(*root_node);
            if (found != node_to_joint.end()) skeleton.root_joint = found->second;
        }
        if (skeleton.root_joint == skeleton_asset::invalid_joint)
        {
            for (std::uint32_t joint = 0; joint < skeleton.joints.size(); ++joint)
                if (skeleton.joints[joint].parent < 0)
                {
                    skeleton.root_joint = joint;
                    break;
                }
        }
        result.push_back(std::move(skeleton));
    }
    return result;
}

std::size_t first_mesh_skin_index(std::string_view json)
{
    const auto nodes = parse_gltf_nodes(json);
    for (const auto& node : nodes)
        if (node.mesh_index == 0 && node.skin_index != material_texture_indices::invalid) return node.skin_index;
    return material_texture_indices::invalid;
}

'''
replace_once(marker, helpers + marker, 'gltf helpers')

marker = '''void normalize3(float* values)
{
'''
fbx_helpers = r'''math::matrix4f to_matrix4(const ufbx_matrix& value)
{
    auto result = math::identity<float, 4>();
    result(0, 0) = static_cast<float>(value.m00);
    result(1, 0) = static_cast<float>(value.m10);
    result(2, 0) = static_cast<float>(value.m20);
    result(0, 1) = static_cast<float>(value.m01);
    result(1, 1) = static_cast<float>(value.m11);
    result(2, 1) = static_cast<float>(value.m21);
    result(0, 2) = static_cast<float>(value.m02);
    result(1, 2) = static_cast<float>(value.m12);
    result(2, 2) = static_cast<float>(value.m22);
    result(0, 3) = static_cast<float>(value.m03);
    result(1, 3) = static_cast<float>(value.m13);
    result(2, 3) = static_cast<float>(value.m23);
    return result;
}

skeleton_asset import_fbx_skeleton(const ufbx_skin_deformer& skin)
{
    skeleton_asset result;
    result.name = to_string(skin.name);
    if (result.name.empty()) result.name = "Skeleton";
    result.joints.reserve(skin.clusters.count);
    std::unordered_map<const ufbx_node*, std::uint32_t> bone_to_joint;
    for (std::size_t joint = 0; joint < skin.clusters.count; ++joint)
    {
        const auto* cluster = skin.clusters.data[joint];
        if (cluster && cluster->bone_node) bone_to_joint[cluster->bone_node] = static_cast<std::uint32_t>(joint);
    }
    for (std::size_t joint = 0; joint < skin.clusters.count; ++joint)
    {
        const auto* cluster = skin.clusters.data[joint];
        skeleton_joint imported;
        imported.name = cluster && cluster->bone_node ? to_string(cluster->bone_node->name)
                                                       : "Joint " + std::to_string(joint);
        if (imported.name.empty()) imported.name = "Joint " + std::to_string(joint);
        if (cluster)
        {
            imported.inverse_bind_matrix = to_matrix4(cluster->geometry_to_bone);
            if (cluster->bone_node)
            {
                const auto& transform = cluster->bone_node->local_transform;
                imported.bind_position = to_vec3(transform.translation);
                imported.bind_rotation = to_quat(transform.rotation);
                imported.bind_scale = to_vec3(transform.scale);
                if (cluster->bone_node->parent)
                {
                    const auto parent = bone_to_joint.find(cluster->bone_node->parent);
                    if (parent != bone_to_joint.end()) imported.parent = static_cast<std::int32_t>(parent->second);
                }
            }
        }
        result.joints.push_back(std::move(imported));
    }
    for (std::uint32_t joint = 0; joint < result.joints.size(); ++joint)
        if (result.joints[joint].parent < 0)
        {
            result.root_joint = joint;
            break;
        }
    return result;
}

mesh_skin_vertex import_fbx_skin_vertex(const ufbx_mesh& mesh, const ufbx_skin_deformer& skin,
                                        std::uint32_t polygon_vertex)
{
    mesh_skin_vertex result;
    for (auto& weight : result.joint_weights)
        weight = 0.0f;
    if (polygon_vertex >= mesh.vertex_indices.count) return result;
    const auto vertex = mesh.vertex_indices.data[polygon_vertex];
    if (vertex >= skin.vertices.count) return result;
    const auto influences = skin.vertices.data[vertex];
    const auto count = std::min<std::size_t>(4u, influences.num_weights);
    float total_weight{};
    for (std::size_t influence = 0; influence < count; ++influence)
    {
        const auto weight = skin.weights.data[influences.weight_begin + influence];
        if (weight.cluster_index >= skin.clusters.count) continue;
        result.joint_indices[influence] = weight.cluster_index;
        result.joint_weights[influence] = std::max(0.0f, static_cast<float>(weight.weight));
        total_weight += result.joint_weights[influence];
    }
    if (total_weight > 1.0e-6f)
        for (auto& weight : result.joint_weights)
            weight /= total_weight;
    return result;
}

'''
replace_once(marker, fbx_helpers + marker, 'fbx helpers')

replace_once(
    '''    std::unordered_map<const ufbx_material*, std::size_t> material_indices;
    std::unordered_map<const ufbx_texture*, std::size_t> texture_indices;
    const auto source_folder = path.parent_path();
''',
    '''    std::unordered_map<const ufbx_material*, std::size_t> material_indices;
    std::unordered_map<const ufbx_texture*, std::size_t> texture_indices;
    std::unordered_map<const ufbx_skin_deformer*, std::size_t> skin_indices;
    const auto source_folder = path.parent_path();
''',
    'fbx maps',
)

replace_once(
    '        const ufbx_mesh* mesh = node->mesh;\n',
    '''        const ufbx_mesh* mesh = node->mesh;
        const ufbx_skin_deformer* skin = mesh->skin_deformers.count != 0 ? mesh->skin_deformers.data[0] : nullptr;
        std::size_t skin_index = material_texture_indices::invalid;
        if (skin)
        {
            const auto found = skin_indices.find(skin);
            if (found != skin_indices.end())
            {
                skin_index = found->second;
            }
            else
            {
                auto skeleton = import_fbx_skeleton(*skin);
                if (skeleton.valid())
                {
                    skin_index = result.skeletons.size();
                    result.skeletons.push_back(std::move(skeleton));
                    skin_indices.emplace(skin, skin_index);
                }
            }
            if (skin->max_weights_per_vertex > 4)
                result.diagnostics.push_back("FBX skin '" + to_string(skin->name) +
                                             "' has more than four influences per vertex; keeping the four strongest");
            if (mesh->skin_deformers.count > 1)
                result.diagnostics.push_back("FBX mesh '" + to_string(mesh->name) +
                                             "' has multiple skin deformers; importing the first skin only");
        }
''',
    'fbx mesh',
)

replace_once(
    '''            imported_mesh.vertices.reserve(part.num_triangles * 3);
            imported_mesh.indices.reserve(part.num_triangles * 3);
            const bool has_imported_tangents = mesh->vertex_tangent.exists;
''',
    '''            imported_mesh.vertices.reserve(part.num_triangles * 3);
            imported_mesh.indices.reserve(part.num_triangles * 3);
            if (skin) imported_mesh.skin_vertices.reserve(part.num_triangles * 3);
            const bool has_imported_tangents = mesh->vertex_tangent.exists;
''',
    'fbx reserve',
)

replace_once(
    '''                    imported_mesh.indices.push_back(static_cast<std::uint32_t>(imported_mesh.vertices.size()));
                    imported_mesh.vertices.push_back(vertex);
''',
    '''                    imported_mesh.indices.push_back(static_cast<std::uint32_t>(imported_mesh.vertices.size()));
                    imported_mesh.vertices.push_back(vertex);
                    if (skin) imported_mesh.skin_vertices.push_back(import_fbx_skin_vertex(*mesh, *skin, ix));
''',
    'fbx skin vertex',
)

replace_once(
    '''            result.nodes.push_back({.name = result.meshes.back().name,
                                    .mesh_index = mesh_index,
                                    .material_index = material_index,
                                    .position = to_vec3(node_transform.translation),
                                    .rotation = to_quat(node_transform.rotation),
                                    .scale = to_vec3(node_transform.scale)});
''',
    '''            result.nodes.push_back({.name = result.meshes.back().name,
                                    .mesh_index = mesh_index,
                                    .material_index = material_index,
                                    .skin_index = skin_index,
                                    .position = to_vec3(node_transform.translation),
                                    .rotation = to_quat(node_transform.rotation),
                                    .scale = to_vec3(node_transform.scale)});
''',
    'fbx node',
)

replace_once(
    '        manifest << "  \\"textures\\": " << result.textures.size() << ",\\n";\n',
    '        manifest << "  \\"textures\\": " << result.textures.size() << ",\\n";\n'
    '        manifest << "  \\"skeletons\\": " << result.skeletons.size() << ",\\n";\n',
    'manifest',
)

replace_once(
    '''    const auto views = parse_buffer_views(json);
    const auto accessors = parse_accessors(json);
    const auto primitive = parse_first_primitive(json);
''',
    '''    const auto views = parse_buffer_views(json);
    const auto accessors = parse_accessors(json);
    auto skeletons = parse_gltf_skeletons(json, accessors, views, bin);
    const auto primitive = parse_first_primitive(json);
''',
    'gltf parse',
)

replace_once(
    '''    if (primitive->has_skin)
    {
        if (primitive->joints_accessor >= accessors.size() || primitive->weights_accessor >= accessors.size() ||
            accessors[primitive->joints_accessor].count != positions.count ||
            accessors[primitive->weights_accessor].count != positions.count)
            return {.message = "GLB skin accessors must match POSITION count"};
        mesh.skin_vertices.resize(positions.count);
    }
''',
    '''    std::size_t skin_index = material_texture_indices::invalid;
    if (primitive->has_skin)
    {
        if (primitive->joints_accessor >= accessors.size() || primitive->weights_accessor >= accessors.size() ||
            accessors[primitive->joints_accessor].count != positions.count ||
            accessors[primitive->weights_accessor].count != positions.count)
            return {.message = "GLB skin accessors must match POSITION count"};
        skin_index = first_mesh_skin_index(json);
        if (skin_index == material_texture_indices::invalid && skeletons.size() == 1) skin_index = 0;
        if (skin_index >= skeletons.size()) return {.message = "GLB skinned mesh references a missing skin"};
        mesh.skin_vertices.resize(positions.count);
    }
''',
    'gltf skin validation',
)

replace_once(
    '''    return {.mesh = std::move(mesh),
            .textures = std::move(textures),
            .materials = std::move(materials),
            .message = "loaded GLB mesh"};
''',
    '''    return {.mesh = std::move(mesh),
            .textures = std::move(textures),
            .materials = std::move(materials),
            .skeletons = std::move(skeletons),
            .skin_index = skin_index,
            .message = "loaded GLB mesh"};
''',
    'gltf return',
)

replace_once(
    '''        result.meshes.push_back(std::move(mesh_result.mesh));
        result.textures = std::move(mesh_result.textures);
        result.materials = std::move(mesh_result.materials);
        result.nodes.push_back(
            {.name = result.meshes.front().name.empty() ? path.stem().string() : result.meshes.front().name,
             .mesh_index = 0,
             .material_index = result.meshes.front().material_index});
''',
    '''        result.meshes.push_back(std::move(mesh_result.mesh));
        result.textures = std::move(mesh_result.textures);
        result.materials = std::move(mesh_result.materials);
        result.skeletons = std::move(mesh_result.skeletons);
        result.nodes.push_back(
            {.name = result.meshes.front().name.empty() ? path.stem().string() : result.meshes.front().name,
             .mesh_index = 0,
             .material_index = result.meshes.front().material_index,
             .skin_index = mesh_result.skin_index});
''',
    'gltf scene wrapping',
)

path.write_text(text)

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:80]!r}")
    write(path, text.replace(old, new, 1))


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    text = read(path)
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} occurrences, found {count}: {old[:80]!r}")
    write(path, text.replace(old, new))


# Generic GPU Scene identity: one entity can own multiple render resources without
# changing its picking/object identity.
replace_count(
    "engine/render/inc/arc/render/render_world.h",
    "    std::uint64_t sort_key{};\n    render_object_id object_id{};",
    "    std::uint64_t sort_key{};\n"
    "    /** Stable per-object render-instance discriminator. Zero identifies the primary instance. */\n"
    "    std::uint64_t instance_id{};\n"
    "    render_object_id object_id{};",
    2,
)
replace_once(
    "engine/render/inc/arc/render/gpu_scene.h",
    "        gpu_scene_geometry_kind geometry_kind{gpu_scene_geometry_kind::mesh};\n"
    "        std::uint32_t submesh_or_cluster{};\n\n"
    "        friend bool operator==(const instance_key&, const instance_key&) noexcept = default;",
    "        gpu_scene_geometry_kind geometry_kind{gpu_scene_geometry_kind::mesh};\n"
    "        std::uint32_t submesh_or_cluster{};\n"
    "        std::uint64_t instance_id{};\n\n"
    "        friend bool operator==(const instance_key&, const instance_key&) noexcept = default;",
)
replace_once(
    "engine/render/src/common/gpu_scene.cpp",
    "    hash_combine(seed, value.submesh_or_cluster);\n    return seed;",
    "    hash_combine(seed, value.submesh_or_cluster);\n"
    "    hash_combine(seed, value.instance_id);\n"
    "    return seed;",
)
replace_once(
    "engine/render/src/common/gpu_scene.cpp",
    "                               .geometry_kind = geometry_kind,\n"
    "                               .submesh_or_cluster = item.submesh};",
    "                               .geometry_kind = geometry_kind,\n"
    "                               .submesh_or_cluster = item.submesh,\n"
    "                               .instance_id = item.instance_id};",
)
replace_once(
    "engine/render/src/common/gpu_scene.cpp",
    "                               .geometry_kind = gpu_scene_geometry_kind::virtual_mesh,\n"
    "                               .submesh_or_cluster = item.root_node};",
    "                               .geometry_kind = gpu_scene_geometry_kind::virtual_mesh,\n"
    "                               .submesh_or_cluster = item.root_node,\n"
    "                               .instance_id = item.instance_id};",
)

# Region IRs use authored local bounds as canonical X/Z placement.
replace_once(
    "engine/scene/src/common/terrain_surface_ir.cpp",
    "    const float half_width = heightfield.width * 0.5f;\n"
    "    const float half_depth = heightfield.depth * 0.5f;\n"
    "    const float x_denominator = static_cast<float>(heightfield.sample_width - 1u);\n"
    "    const float z_denominator = static_cast<float>(heightfield.sample_height - 1u);",
    "    const float minimum_x = static_cast<float>(surface.local_bounds.min_x);\n"
    "    const float minimum_z = static_cast<float>(surface.local_bounds.min_z);\n"
    "    const float extent_x = static_cast<float>(surface.local_bounds.max_x - surface.local_bounds.min_x);\n"
    "    const float extent_z = static_cast<float>(surface.local_bounds.max_z - surface.local_bounds.min_z);\n"
    "    const float x_denominator = static_cast<float>(heightfield.sample_width - 1u);\n"
    "    const float z_denominator = static_cast<float>(heightfield.sample_height - 1u);",
)
replace_once(
    "engine/scene/src/common/terrain_surface_ir.cpp",
    "            const float local_x = -half_width + heightfield.width * static_cast<float>(x) / x_denominator;\n"
    "            const float local_z = -half_depth + heightfield.depth * static_cast<float>(z) / z_denominator;",
    "            const float local_x = minimum_x + extent_x * static_cast<float>(x) / x_denominator;\n"
    "            const float local_z = minimum_z + extent_z * static_cast<float>(z) / z_denominator;",
)

write(
    "engine/scene/inc/arc/scene/terrain_render_regions.h",
    r'''#pragma once

#include <arc/scene/terrain_asset.h>
#include <arc/scene/terrain_surface_ir.h>

#include <cstdint>
#include <vector>

namespace arc::scene
{

/** Default world-space edge length used to group terrain render artifacts. */
inline constexpr double default_terrain_render_region_size = 256.0;

/**
 * @brief One independently compilable terrain render region.
 *
 * Region geometry and material attributes are owning so asynchronous/cached compilation can
 * outlive the source view. vertex_normals contains normals evaluated from the complete source
 * surface so duplicated boundary vertices remain shading-compatible across neighboring regions.
 */
struct terrain_render_region
{
    terrain_region_id id{};
    terrain_evaluated_surface surface{};
    std::vector<math::vector3f> vertex_normals;
    std::uint64_t geometry_fingerprint{};
    std::uint64_t attribute_fingerprint{};
};

/**
 * @brief Partition an evaluated terrain surface into stable sample-aligned render regions.
 *
 * Heightfields are split only on source quad boundaries and duplicate the shared boundary row/
 * column, guaranteeing identical seam positions. Mesh surfaces remain a single region until the
 * mesh-native terrain milestone introduces topology-aware spatial partitioning.
 */
[[nodiscard]] std::vector<terrain_render_region>
build_terrain_render_regions(const terrain_surface_ir& surface,
                             double target_region_size = default_terrain_render_region_size);

/** Stable non-zero render-instance discriminator for one terrain region. */
[[nodiscard]] std::uint64_t terrain_render_region_instance_id(terrain_region_id id) noexcept;

} // namespace arc::scene
''',
)

write(
    "engine/scene/src/common/terrain_render_regions.cpp",
    r'''#include <arc/scene/terrain_render_regions.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace arc::scene
{
namespace
{

constexpr float normal_epsilon = 1.0e-12f;

class stable_hash64
{
public:
    void byte(std::uint8_t value) noexcept
    {
        value_ ^= value;
        value_ *= 1099511628211ull;
    }

    void u32(std::uint32_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 32u; shift += 8u)
            byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void u64(std::uint64_t value) noexcept
    {
        for (std::uint32_t shift = 0; shift < 64u; shift += 8u)
            byte(static_cast<std::uint8_t>((value >> shift) & 0xffu));
    }

    void f32(float value) noexcept
    {
        u32(std::bit_cast<std::uint32_t>(value));
    }

    void f64(double value) noexcept
    {
        u64(std::bit_cast<std::uint64_t>(value));
    }

    [[nodiscard]] std::uint64_t value() const noexcept
    {
        return value_ == 0u ? 1u : value_;
    }

private:
    std::uint64_t value_{14695981039346656037ull};
};

void append_region_identity(stable_hash64& hash, terrain_region_id id, const terrain_world_bounds& bounds) noexcept
{
    hash.u64(static_cast<std::uint64_t>(id.x));
    hash.u64(static_cast<std::uint64_t>(id.z));
    hash.f64(bounds.min_x);
    hash.f64(bounds.min_z);
    hash.f64(bounds.max_x);
    hash.f64(bounds.max_z);
}

std::uint32_t region_quad_span(float extent, std::uint32_t total_quads, double target_size) noexcept
{
    const double spacing = static_cast<double>(extent) / static_cast<double>(total_quads);
    if (!std::isfinite(spacing) || spacing <= 0.0) return total_quads;
    const auto desired = static_cast<std::uint64_t>(std::floor(target_size / spacing));
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(desired, 1u, total_quads));
}

std::vector<math::vector3f> source_normals(const terrain_surface_ir& surface)
{
    const auto canonical = canonicalize_terrain_surface_geometry(surface);
    if (!canonical) return {};

    std::vector<math::vector3f> normals(canonical->positions.size());
    for (std::size_t index = 0; index + 2u < canonical->indices.size(); index += 3u)
    {
        const auto i0 = canonical->indices[index + 0u];
        const auto i1 = canonical->indices[index + 1u];
        const auto i2 = canonical->indices[index + 2u];
        const auto edge0 = math::sub(canonical->positions[i1], canonical->positions[i0]);
        const auto edge1 = math::sub(canonical->positions[i2], canonical->positions[i0]);
        const auto face = math::cross(edge0, edge1);
        if (math::length_squared(face) <= normal_epsilon) continue;
        normals[i0] = math::add(normals[i0], face);
        normals[i1] = math::add(normals[i1], face);
        normals[i2] = math::add(normals[i2], face);
    }
    for (auto& normal : normals)
        normal = math::length_squared(normal) > normal_epsilon ? math::normalize(normal)
                                                               : math::vector3f{0.0f, 1.0f, 0.0f};
    return normals;
}

std::uint64_t geometry_fingerprint(terrain_region_id id, const terrain_evaluated_surface& surface) noexcept
{
    stable_hash64 hash;
    append_region_identity(hash, id, surface.local_bounds);
    if (const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        hash.byte(0u);
        hash.u32(heightfield->sample_width);
        hash.u32(heightfield->sample_height);
        hash.f32(heightfield->width);
        hash.f32(heightfield->depth);
        for (const auto height : heightfield->heights)
            hash.f32(height);
    }
    else
    {
        hash.byte(1u);
        const auto& mesh = std::get<terrain_evaluated_mesh>(surface.geometry);
        hash.u64(mesh.positions.size());
        hash.u64(mesh.indices.size());
        for (const auto& position : mesh.positions)
        {
            hash.f32(position[0]);
            hash.f32(position[1]);
            hash.f32(position[2]);
        }
        for (const auto index : mesh.indices)
            hash.u32(index);
    }
    return hash.value();
}

std::uint64_t attribute_fingerprint(terrain_region_id id, const terrain_evaluated_surface& surface) noexcept
{
    stable_hash64 hash;
    append_region_identity(hash, id, surface.local_bounds);
    if (const auto* heightfield = std::get_if<terrain_evaluated_heightfield>(&surface.geometry))
    {
        hash.u32(heightfield->sample_width);
        hash.u32(heightfield->sample_height);
        for (const auto& weights : heightfield->material_weights)
            for (const auto weight : weights)
                hash.byte(weight);
    }
    else
    {
        hash.byte(0u);
    }
    return hash.value();
}

} // namespace

std::vector<terrain_render_region> build_terrain_render_regions(const terrain_surface_ir& surface,
                                                                double target_region_size)
{
    if (!validate_terrain_surface_ir(surface) || !std::isfinite(target_region_size) || target_region_size <= 0.0)
        return {};

    if (!std::holds_alternative<terrain_surface_heightfield_ir>(surface.geometry))
    {
        auto owned = copy_terrain_surface_ir(surface);
        if (!owned) return {};
        terrain_render_region region;
        region.id = {};
        region.surface = std::move(*owned);
        region.geometry_fingerprint = geometry_fingerprint(region.id, region.surface);
        region.attribute_fingerprint = attribute_fingerprint(region.id, region.surface);
        return {std::move(region)};
    }

    const auto& source = std::get<terrain_surface_heightfield_ir>(surface.geometry);
    const auto normals = source_normals(surface);
    if (normals.size() != source.heights.size()) return {};

    const auto total_quads_x = source.sample_width - 1u;
    const auto total_quads_z = source.sample_height - 1u;
    const auto quads_per_region_x = region_quad_span(source.width, total_quads_x, target_region_size);
    const auto quads_per_region_z = region_quad_span(source.depth, total_quads_z, target_region_size);
    const auto region_count_x = (total_quads_x + quads_per_region_x - 1u) / quads_per_region_x;
    const auto region_count_z = (total_quads_z + quads_per_region_z - 1u) / quads_per_region_z;

    std::vector<terrain_render_region> result;
    result.reserve(static_cast<std::size_t>(region_count_x) * region_count_z);

    const double full_extent_x = surface.local_bounds.max_x - surface.local_bounds.min_x;
    const double full_extent_z = surface.local_bounds.max_z - surface.local_bounds.min_z;
    for (std::uint32_t region_z = 0; region_z < region_count_z; ++region_z)
        for (std::uint32_t region_x = 0; region_x < region_count_x; ++region_x)
        {
            const auto start_x = region_x * quads_per_region_x;
            const auto start_z = region_z * quads_per_region_z;
            const auto end_x = std::min(start_x + quads_per_region_x, total_quads_x);
            const auto end_z = std::min(start_z + quads_per_region_z, total_quads_z);
            const auto sample_width = end_x - start_x + 1u;
            const auto sample_height = end_z - start_z + 1u;

            terrain_render_region region;
            region.id = {static_cast<std::int64_t>(region_x), static_cast<std::int64_t>(region_z)};
            region.surface.schema_version = surface.schema_version;
            region.surface.source_revision = surface.source_revision;
            region.surface.local_bounds.min_x =
                surface.local_bounds.min_x + full_extent_x * static_cast<double>(start_x) / total_quads_x;
            region.surface.local_bounds.max_x =
                surface.local_bounds.min_x + full_extent_x * static_cast<double>(end_x) / total_quads_x;
            region.surface.local_bounds.min_z =
                surface.local_bounds.min_z + full_extent_z * static_cast<double>(start_z) / total_quads_z;
            region.surface.local_bounds.max_z =
                surface.local_bounds.min_z + full_extent_z * static_cast<double>(end_z) / total_quads_z;

            terrain_evaluated_heightfield heightfield;
            heightfield.sample_width = sample_width;
            heightfield.sample_height = sample_height;
            heightfield.width = static_cast<float>(region.surface.local_bounds.max_x - region.surface.local_bounds.min_x);
            heightfield.depth = static_cast<float>(region.surface.local_bounds.max_z - region.surface.local_bounds.min_z);
            heightfield.heights.reserve(static_cast<std::size_t>(sample_width) * sample_height);
            heightfield.material_weights.reserve(static_cast<std::size_t>(sample_width) * sample_height);
            region.vertex_normals.reserve(static_cast<std::size_t>(sample_width) * sample_height);

            float minimum_height = std::numeric_limits<float>::max();
            float maximum_height = std::numeric_limits<float>::lowest();
            for (std::uint32_t z = start_z; z <= end_z; ++z)
                for (std::uint32_t x = start_x; x <= end_x; ++x)
                {
                    const auto source_index = static_cast<std::size_t>(z) * source.sample_width + x;
                    const auto height = source.heights[source_index];
                    heightfield.heights.push_back(height);
                    heightfield.material_weights.push_back(source.material_weights[source_index]);
                    region.vertex_normals.push_back(normals[source_index]);
                    minimum_height = std::min(minimum_height, height);
                    maximum_height = std::max(maximum_height, height);
                }

            region.surface.local_bounds.min_y = minimum_height;
            region.surface.local_bounds.max_y = maximum_height;
            region.surface.geometry = std::move(heightfield);
            region.geometry_fingerprint = geometry_fingerprint(region.id, region.surface);
            region.attribute_fingerprint = attribute_fingerprint(region.id, region.surface);
            result.push_back(std::move(region));
        }
    return result;
}

std::uint64_t terrain_render_region_instance_id(terrain_region_id id) noexcept
{
    stable_hash64 hash;
    hash.u64(static_cast<std::uint64_t>(id.x));
    hash.u64(static_cast<std::uint64_t>(id.z));
    return hash.value();
}

} // namespace arc::scene
''',
)

# Region geometry may supply full-source normals to keep duplicated seam vertices shading-compatible.
replace_once(
    "engine/scene/inc/arc/scene/terrain_render_geometry.h",
    "#include <optional>\n",
    "#include <optional>\n#include <span>\n",
)
replace_once(
    "engine/scene/inc/arc/scene/terrain_render_geometry.h",
    "build_terrain_render_geometry(const terrain_surface_ir& surface,\n"
    "                              const render::virtual_mesh_build_options& options = {});\n",
    "build_terrain_render_geometry(const terrain_surface_ir& surface,\n"
    "                              const render::virtual_mesh_build_options& options = {});\n\n"
    "/** Compile one render region using normals evaluated from its complete source surface. */\n"
    "[[nodiscard]] std::optional<render::virtual_mesh_data>\n"
    "build_terrain_render_region_geometry(const terrain_surface_ir& surface,\n"
    "                                     std::span<const math::vector3f> vertex_normals,\n"
    "                                     const render::virtual_mesh_build_options& options = {});\n",
)
write(
    "engine/scene/src/common/terrain_render_geometry.cpp",
    r'''#include <arc/scene/terrain_render_geometry.h>

#include <arc/render/mesh.h>

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace arc::scene
{
namespace
{

constexpr float normal_epsilon = 1.0e-12f;

math::vector3f stable_tangent(const math::vector3f& normal) noexcept
{
    const math::vector3f x_axis{1.0f, 0.0f, 0.0f};
    const math::vector3f z_axis{0.0f, 0.0f, 1.0f};
    math::vector3f tangent = math::sub(x_axis, math::mul(normal, math::dot(normal, x_axis)));
    if (math::length_squared(tangent) <= normal_epsilon)
        tangent = math::sub(z_axis, math::mul(normal, math::dot(normal, z_axis)));
    return math::length_squared(tangent) > normal_epsilon ? math::normalize(tangent) : x_axis;
}

std::optional<render::virtual_mesh_data>
build_geometry(const terrain_surface_ir& surface, std::span<const math::vector3f> supplied_normals,
               const render::virtual_mesh_build_options& options)
{
    const auto canonical = canonicalize_terrain_surface_geometry(surface);
    if (!canonical) return std::nullopt;
    if (!supplied_normals.empty() && supplied_normals.size() != canonical->positions.size()) return std::nullopt;

    render::mesh_data source;
    source.name = "terrain";
    source.usage = render::mesh_usage::static_gpu;
    source.vertices.resize(canonical->positions.size());
    source.indices = canonical->indices;

    std::vector<math::vector3f> accumulated_normals;
    if (supplied_normals.empty())
    {
        accumulated_normals.resize(canonical->positions.size());
        for (std::size_t index = 0; index + 2u < canonical->indices.size(); index += 3u)
        {
            const auto i0 = canonical->indices[index + 0u];
            const auto i1 = canonical->indices[index + 1u];
            const auto i2 = canonical->indices[index + 2u];
            const auto edge0 = math::sub(canonical->positions[i1], canonical->positions[i0]);
            const auto edge1 = math::sub(canonical->positions[i2], canonical->positions[i0]);
            const auto face = math::cross(edge0, edge1);
            if (math::length_squared(face) <= normal_epsilon) continue;
            accumulated_normals[i0] = math::add(accumulated_normals[i0], face);
            accumulated_normals[i1] = math::add(accumulated_normals[i1], face);
            accumulated_normals[i2] = math::add(accumulated_normals[i2], face);
        }
    }

    const float extent_x = static_cast<float>(surface.local_bounds.max_x - surface.local_bounds.min_x);
    const float extent_z = static_cast<float>(surface.local_bounds.max_z - surface.local_bounds.min_z);
    for (std::size_t index = 0; index < canonical->positions.size(); ++index)
    {
        const auto& position = canonical->positions[index];
        const auto normal_source = supplied_normals.empty() ? accumulated_normals[index] : supplied_normals[index];
        const auto normal = math::length_squared(normal_source) > normal_epsilon ? math::normalize(normal_source)
                                                                                : math::vector3f{0.0f, 1.0f, 0.0f};
        const auto tangent = stable_tangent(normal);
        auto& vertex = source.vertices[index];
        vertex.position[0] = position[0];
        vertex.position[1] = position[1];
        vertex.position[2] = position[2];
        vertex.normal[0] = normal[0];
        vertex.normal[1] = normal[1];
        vertex.normal[2] = normal[2];
        vertex.tangent[0] = tangent[0];
        vertex.tangent[1] = tangent[1];
        vertex.tangent[2] = tangent[2];
        vertex.tangent[3] = 1.0f;
        vertex.texcoord[0] = std::abs(extent_x) > 1.0e-8f
                                 ? (position[0] - static_cast<float>(surface.local_bounds.min_x)) / extent_x
                                 : 0.0f;
        vertex.texcoord[1] = std::abs(extent_z) > 1.0e-8f
                                 ? (position[2] - static_cast<float>(surface.local_bounds.min_z)) / extent_z
                                 : 0.0f;
    }

    auto result = render::build_virtual_mesh(source, options);
    if (result.clusters.empty() || result.root_nodes.empty() || result.pages.empty()) return std::nullopt;
    if (options.build_conventional_lods && result.conventional_lods.empty()) return std::nullopt;
    return result;
}

} // namespace

std::optional<render::virtual_mesh_data>
build_terrain_render_geometry(const terrain_surface_ir& surface, const render::virtual_mesh_build_options& options)
{
    return build_geometry(surface, {}, options);
}

std::optional<render::virtual_mesh_data>
build_terrain_render_region_geometry(const terrain_surface_ir& surface,
                                     std::span<const math::vector3f> vertex_normals,
                                     const render::virtual_mesh_build_options& options)
{
    return build_geometry(surface, vertex_normals, options);
}

} // namespace arc::scene
''',
)

# Regionized proxy ownership.
replace_once(
    "engine/scene/inc/arc/scene/terrain.h",
    "#include <arc/scene/terrain_surface_ir.h>\n",
    "#include <arc/scene/terrain_surface_ir.h>\n#include <arc/scene/terrain_render_regions.h>\n",
)
replace_once(
    "engine/scene/inc/arc/scene/terrain.h",
    "struct terrain_render_proxy\n"
    "{\n"
    "    /** Generic geometry realized from TerrainSurfaceIR and used by the M1 renderer path. */\n"
    "    render::geometry_resource_handle geometry{};\n"
    "    /** Per-surface RGBA8 terrain material weights, kept separate from generic mesh vertices. */\n"
    "    render::texture_handle surface_attribute_texture{};\n"
    "    geometric::box3f local_bounds{};\n"
    "    std::uint64_t synchronized_revision{};\n"
    "    render::material_handle material{};\n"
    "};",
    "struct terrain_render_region_proxy\n"
    "{\n"
    "    terrain_region_id id{};\n"
    "    render::geometry_resource_handle geometry{};\n"
    "    render::texture_handle surface_attribute_texture{};\n"
    "    geometric::box3f local_bounds{};\n"
    "    std::uint64_t geometry_fingerprint{};\n"
    "    std::uint64_t attribute_fingerprint{};\n"
    "};\n\n"
    "/** @brief Nonserialized renderer proxy state for one terrain entity. */\n"
    "struct terrain_render_proxy\n"
    "{\n"
    "    /** Stable independently replaceable world-space render regions. */\n"
    "    std::vector<terrain_render_region_proxy> regions;\n"
    "    std::uint64_t synchronized_revision{};\n"
    "    render::material_handle material{};\n"
    "};",
)

write(
    "engine/scene/src/common/terrain_geometry_proxy.cpp",
    r'''#include <arc/scene/terrain.h>

#include <arc/render/renderer.h>
#include <arc/scene/terrain_render_attributes.h>
#include <arc/scene/terrain_render_geometry.h>
#include <arc/scene/terrain_render_regions.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>

namespace arc::scene
{
namespace
{

std::uint32_t terrain_geometry_generation(std::uint64_t fingerprint) noexcept
{
    const auto folded = static_cast<std::uint32_t>(fingerprint) ^ static_cast<std::uint32_t>(fingerprint >> 32u);
    return folded == 0u ? 1u : folded;
}

geometric::box3f terrain_local_bounds(const terrain_world_bounds& bounds) noexcept
{
    return geometric::box3f{geometric::point3f{static_cast<float>(bounds.min_x), static_cast<float>(bounds.min_y),
                                               static_cast<float>(bounds.min_z)},
                            geometric::point3f{static_cast<float>(bounds.max_x), static_cast<float>(bounds.max_y),
                                               static_cast<float>(bounds.max_z)}};
}

bool geometry_alive(const terrain_render_region_proxy& proxy, const render::renderer& renderer)
{
    return proxy.geometry.valid() && renderer.mesh_alive(proxy.geometry.conventional);
}

bool attributes_alive(const terrain_render_region_proxy& proxy, const render::renderer& renderer)
{
    return proxy.surface_attribute_texture.valid() && renderer.texture_alive(proxy.surface_attribute_texture);
}

render::texture_data make_attribute_texture_data(const terrain_render_attributes& attributes)
{
    render::texture_data data;
    data.name = "terrain-material-weights";
    data.width = attributes.width;
    data.height = attributes.height;
    data.format = render::texture_format::rgba8_unorm;
    data.color_space = render::texture_color_space::linear;
    data.mip_levels = 1u;
    data.pixels.resize(attributes.material_weights.size() * sizeof(attributes.material_weights.front()));
    if (!data.pixels.empty()) std::memcpy(data.pixels.data(), attributes.material_weights.data(), data.pixels.size());
    return data;
}

render::texture_handle create_attribute_texture(const terrain_render_attributes& attributes, render::renderer& renderer)
{
    auto texture = renderer.create_texture(make_attribute_texture_data(attributes));
    if (!texture.valid() || !renderer.texture_alive(texture))
    {
        if (texture.valid()) renderer.destroy_texture(texture);
        return {};
    }
    return texture;
}

void destroy_region(terrain_render_region_proxy& proxy, render::renderer& renderer)
{
    if (proxy.geometry.conventional.valid() || proxy.geometry.virtualized.valid())
        (void)renderer.destroy_geometry_resource(proxy.geometry);
    proxy.geometry = {};
    if (renderer.texture_alive(proxy.surface_attribute_texture)) renderer.destroy_texture(proxy.surface_attribute_texture);
    proxy.surface_attribute_texture = {};
}

void destroy_proxy(terrain_render_proxy& proxy, render::renderer& renderer)
{
    for (auto& region : proxy.regions)
        destroy_region(region, renderer);
    proxy.regions.clear();
}

const terrain_render_region_proxy* find_region(const terrain_render_proxy& proxy, terrain_region_id id) noexcept
{
    const auto found = std::find_if(proxy.regions.begin(), proxy.regions.end(),
                                    [id](const terrain_render_region_proxy& value) { return value.id == id; });
    return found == proxy.regions.end() ? nullptr : &*found;
}

bool proxy_resources_alive(const terrain_render_proxy& proxy, const render::renderer& renderer)
{
    return !proxy.regions.empty() &&
           std::all_of(proxy.regions.begin(), proxy.regions.end(), [&](const terrain_render_region_proxy& region)
                       { return geometry_alive(region, renderer) && attributes_alive(region, renderer); });
}

bool geometry_reused(const std::vector<terrain_render_region_proxy>& regions,
                     render::geometry_resource_handle geometry)
{
    return std::any_of(regions.begin(), regions.end(), [&](const terrain_render_region_proxy& candidate)
                       { return candidate.geometry == geometry; });
}

bool texture_reused(const std::vector<terrain_render_region_proxy>& regions, render::texture_handle texture)
{
    return std::any_of(regions.begin(), regions.end(), [&](const terrain_render_region_proxy& candidate)
                       { return candidate.surface_attribute_texture == texture; });
}

void destroy_unreused_old_resources(terrain_render_proxy& previous,
                                    const std::vector<terrain_render_region_proxy>& replacement,
                                    render::renderer& renderer)
{
    for (auto& region : previous.regions)
    {
        if (!geometry_reused(replacement, region.geometry) &&
            (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!texture_reused(replacement, region.surface_attribute_texture) &&
            renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = {};
        region.surface_attribute_texture = {};
    }
}

void cleanup_staged_resources(const terrain_render_proxy& previous, std::vector<terrain_render_region_proxy>& staged,
                              render::renderer& renderer)
{
    for (auto& region : staged)
    {
        const bool geometry_owned_by_previous = geometry_reused(previous.regions, region.geometry);
        const bool attributes_owned_by_previous = texture_reused(previous.regions, region.surface_attribute_texture);
        if (!geometry_owned_by_previous && (region.geometry.conventional.valid() || region.geometry.virtualized.valid()))
            (void)renderer.destroy_geometry_resource(region.geometry);
        if (!attributes_owned_by_previous && renderer.texture_alive(region.surface_attribute_texture))
            renderer.destroy_texture(region.surface_attribute_texture);
        region.geometry = {};
        region.surface_attribute_texture = {};
    }
}

} // namespace

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_surface_ir& surface,
                                             const terrain_component& terrain, render::renderer& renderer,
                                             const terrain_dirty_region* dirty_region)
{
    (void)dirty_region;
    if (!guid.valid() || !validate_terrain_surface_ir(surface)) return false;

    auto& proxy = proxies_[guid];
    if (proxy.synchronized_revision == surface.source_revision && proxy_resources_alive(proxy, renderer))
    {
        proxy.material = terrain.material;
        return true;
    }

    auto source_regions = build_terrain_render_regions(surface);
    if (source_regions.empty()) return false;

    std::vector<terrain_render_region_proxy> staged;
    staged.reserve(source_regions.size());
    for (auto& source_region : source_regions)
    {
        const auto* previous = find_region(proxy, source_region.id);
        terrain_render_region_proxy next;
        next.id = source_region.id;
        next.local_bounds = terrain_local_bounds(source_region.surface.local_bounds);
        next.geometry_fingerprint = source_region.geometry_fingerprint;
        next.attribute_fingerprint = source_region.attribute_fingerprint;

        if (previous && previous->geometry_fingerprint == source_region.geometry_fingerprint &&
            geometry_alive(*previous, renderer))
        {
            next.geometry = previous->geometry;
        }
        else
        {
            const auto view = source_region.surface.view();
            auto artifact = source_region.vertex_normals.empty()
                                ? build_terrain_render_geometry(view)
                                : build_terrain_render_region_geometry(view, source_region.vertex_normals);
            if (!artifact)
            {
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
            next.geometry = renderer.create_geometry_resource(
                std::move(*artifact), terrain_geometry_generation(source_region.geometry_fingerprint));
            if (!next.geometry.valid() || !renderer.mesh_alive(next.geometry.conventional))
            {
                if (next.geometry.conventional.valid() || next.geometry.virtualized.valid())
                    (void)renderer.destroy_geometry_resource(next.geometry);
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
        }

        const auto view = source_region.surface.view();
        auto attributes = build_terrain_render_attributes(view);
        if (!attributes)
        {
            if (!previous || next.geometry != previous->geometry)
                (void)renderer.destroy_geometry_resource(next.geometry);
            cleanup_staged_resources(proxy, staged, renderer);
            return false;
        }

        if (previous && previous->attribute_fingerprint == source_region.attribute_fingerprint &&
            attributes_alive(*previous, renderer))
        {
            next.surface_attribute_texture = previous->surface_attribute_texture;
        }
        else if (previous && attributes_alive(*previous, renderer) &&
                 renderer.update_texture(previous->surface_attribute_texture, make_attribute_texture_data(*attributes)))
        {
            next.surface_attribute_texture = previous->surface_attribute_texture;
        }
        else
        {
            next.surface_attribute_texture = create_attribute_texture(*attributes, renderer);
            if (!next.surface_attribute_texture.valid())
            {
                if (!previous || next.geometry != previous->geometry)
                    (void)renderer.destroy_geometry_resource(next.geometry);
                cleanup_staged_resources(proxy, staged, renderer);
                return false;
            }
        }
        staged.push_back(next);
    }

    destroy_unreused_old_resources(proxy, staged, renderer);
    proxy.regions = std::move(staged);
    proxy.synchronized_revision = surface.source_revision;
    proxy.material = terrain.material;
    return true;
}

bool terrain_render_proxy_cache::synchronize(ecs::entity_guid guid, const terrain_component& terrain,
                                             render::renderer& renderer, const terrain_dirty_region* dirty_region)
{
    const auto surface = make_legacy_terrain_surface_ir(terrain);
    return surface && synchronize(guid, *surface, terrain, renderer, dirty_region);
}

bool terrain_render_proxy_cache::erase(ecs::entity_guid guid, render::renderer& renderer)
{
    const auto found = proxies_.find(guid);
    if (found == proxies_.end()) return false;
    destroy_proxy(found->second, renderer);
    proxies_.erase(found);
    return true;
}

void terrain_render_proxy_cache::release_missing(std::span<const ecs::entity_guid> active, render::renderer& renderer)
{
    for (auto found = proxies_.begin(); found != proxies_.end();)
    {
        if (std::find(active.begin(), active.end(), found->first) != active.end())
        {
            ++found;
            continue;
        }
        destroy_proxy(found->second, renderer);
        found = proxies_.erase(found);
    }
}

void terrain_render_proxy_cache::clear(render::renderer& renderer)
{
    for (auto& [guid, proxy] : proxies_)
    {
        (void)guid;
        destroy_proxy(proxy, renderer);
    }
    proxies_.clear();
}

} // namespace arc::scene
''',
)

# Submit every region while preserving the terrain entity ObjectID for picking.
path = "engine/scene/src/common/render_scene.cpp"
text = read(path)
start = text.index("    std::vector<ecs::entity_guid> active_terrain_guids;")
end_marker = "    if (terrain_proxies) terrain_proxies->release_missing(active_terrain_guids, renderer);\n"
end = text.index(end_marker, start) + len(end_marker)
replacement = r'''    std::vector<ecs::entity_guid> active_terrain_guids;
    scene.view<transform_component, terrain_component>().each(
        [&](entity value, const transform_component& transform, const terrain_component& terrain)
        {
            if (!environment_visibility.terrain || !entity_is_active(scene, value) || !terrain.enabled) return;
            if (!terrain_proxies) return;
            ecs::entity_guid guid{world_packet.gpu_scene_world_id,
                                  (static_cast<std::uint64_t>(value.generation) << 32u) | value.index};
            if (const auto* persistent = scene.try_get<ecs::persistent_id_component>(value)) guid = persistent->value;
            active_terrain_guids.push_back(guid);
            const auto surface = make_legacy_terrain_surface_ir(terrain);
            if (!surface || !terrain_proxies->synchronize(guid, *surface, terrain, renderer)) return;
            const auto* proxy = terrain_proxies->find(guid);
            if (!proxy || proxy->regions.empty()) return;

            const auto world = transform.dirty ? local_matrix(transform) : transform.world;
            const bool selected = entity_selected(scene, value);
            bool submitted_terrain = false;
            for (const auto& region : proxy->regions)
            {
                if (!renderer.mesh_alive(region.geometry.conventional)) continue;
                const auto renderer_bounds = transform_bounds(region.local_bounds, world);
                const auto instance_id = terrain_render_region_instance_id(region.id);
                if (renderer.resolved_config().features.virtual_geometry &&
                    renderer.virtual_mesh_alive(region.geometry.virtualized))
                {
                    const auto* virtual_mesh = renderer.virtual_mesh_data_for(region.geometry.virtualized);
                    if (!virtual_mesh) continue;
                    ++result.renderable_count;
                    if (selected) ++result.selected_count;
                    world_packet.virtual_items.push_back(
                        {.mesh = region.geometry.virtualized,
                         .material = proxy->material,
                         .material_attribute_texture = region.surface_attribute_texture,
                         .root_node = virtual_mesh->root_nodes.size() == 1 ? virtual_mesh->root_nodes.front()
                                                                           : render::invalid_virtual_geometry_index,
                         .model = world,
                         .previous_model = world,
                         .world_bounds = renderer_bounds,
                         .render_layer_mask = render_layer_mask(scene, value),
                         .instance_id = instance_id,
                         .object_id = render::make_render_object_id(value.index, value.generation),
                         .visible = true,
                         .selected = selected,
                         .casts_shadows = terrain.cast_shadows,
                         .receives_shadows = terrain.receive_shadows,
                         .mobility = entity_mobility(scene, value),
                         .shadow_lod_bias = terrain.shadow_lod_bias,
                         .maximum_shadow_distance = terrain.maximum_shadow_distance,
                         .geometry_error_scale = 1.0f,
                         .label = entity_label(scene, value)});
                    submitted_terrain = true;
                    continue;
                }

                const auto mesh = select_cooked_lod(region.geometry, world_packet.camera, renderer_bounds,
                                                    renderer.resolved_config().geometry_error_threshold, -1, 0.0f);
                if (!renderer.mesh_alive(mesh)) continue;

                append_mesh_item(scene, world_packet, result, value, transform, mesh, proxy->material, true, false, {},
                                 0, 1, math::vector4f::one, terrain.cast_shadows, terrain.receive_shadows,
                                 terrain.shadow_lod_bias, terrain.maximum_shadow_distance);
                if (!world_packet.items.empty())
                {
                    auto& item = world_packet.items.back();
                    item.world_bounds = renderer_bounds;
                    item.material_attribute_texture = region.surface_attribute_texture;
                    item.instance_id = instance_id;
                }
                submitted_terrain = true;
            }
            if (submitted_terrain) ++result.terrain_count;
        });
    if (terrain_proxies) terrain_proxies->release_missing(active_terrain_guids, renderer);
'''
write(path, text[:start] + replacement + text[end:])

replace_once(
    "engine/scene/inc/arc/scene/scene.h",
    "#include <arc/scene/terrain_render_geometry.h>\n",
    "#include <arc/scene/terrain_render_geometry.h>\n#include <arc/scene/terrain_render_regions.h>\n",
)

# Existing tests use a single small render region, so preserve their assertions through regions.front().
path = "engine/scene/tests/terrain_conventional_render_tests.cpp"
text = read(path)
text = text.replace("->geometry", "->regions.front().geometry")
text = text.replace("->surface_attribute_texture", "->regions.front().surface_attribute_texture")
write(path, text)

write(
    "engine/render/tests/gpu_scene_instance_identity_tests.cpp",
    r'''#include <arc/render/gpu_scene.h>
#include <arc/render/render_world.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("gpu scene supports multiple stable render instances for one object")
{
    arc::render::gpu_scene gpu_scene;
    arc::render::render_world_packet packet;
    packet.gpu_scene_world_id = 7u;
    packet.world_epoch = 1u;

    const arc::render::render_object_id object{.index = 42u, .generation = 3u};
    packet.items.push_back({.mesh = {.index = 1u, .generation = 1u}, .instance_id = 100u, .object_id = object});
    packet.items.push_back({.mesh = {.index = 2u, .generation = 1u}, .instance_id = 200u, .object_id = object});

    const auto first = gpu_scene.synchronize(packet, 1u);
    REQUIRE(first.active_instance_count == 2u);
    REQUIRE(packet.items[0].gpu_scene_instance.valid());
    REQUIRE(packet.items[1].gpu_scene_instance.valid());
    CHECK(packet.items[0].gpu_scene_instance != packet.items[1].gpu_scene_instance);

    const auto first_handle = packet.items[0].gpu_scene_instance;
    const auto second_handle = packet.items[1].gpu_scene_instance;
    packet.items[0].mesh = {.index = 3u, .generation = 2u};
    const auto second = gpu_scene.synchronize(packet, 2u);
    CHECK(second.active_instance_count == 2u);
    CHECK(packet.items[0].gpu_scene_instance == first_handle);
    CHECK(packet.items[1].gpu_scene_instance == second_handle);
}

TEST_CASE("gpu scene differentiates virtual roots from the same object by instance id")
{
    arc::render::gpu_scene gpu_scene;
    arc::render::render_world_packet packet;
    packet.gpu_scene_world_id = 9u;
    packet.world_epoch = 1u;

    const arc::render::render_object_id object{.index = 4u, .generation = 2u};
    packet.virtual_items.push_back(
        {.mesh = {.index = 10u, .generation = 1u}, .root_node = 0u, .instance_id = 11u, .object_id = object});
    packet.virtual_items.push_back(
        {.mesh = {.index = 11u, .generation = 1u}, .root_node = 0u, .instance_id = 12u, .object_id = object});

    const auto batch = gpu_scene.synchronize(packet, 1u);
    REQUIRE(batch.active_instance_count == 2u);
    CHECK(packet.virtual_items[0].gpu_scene_instance != packet.virtual_items[1].gpu_scene_instance);
}
''',
)

write(
    "engine/scene/tests/terrain_render_region_tests.cpp",
    r'''#include <arc/scene/scene.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <vector>

namespace
{

arc::scene::terrain_surface_ir make_region_test_surface(std::uint64_t revision = 1u)
{
    constexpr std::uint32_t sample_width = 5u;
    constexpr std::uint32_t sample_height = 5u;
    static std::vector<float> heights(sample_width * sample_height);
    static std::vector<std::array<std::uint8_t, 4>> weights(
        sample_width * sample_height, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < sample_height; ++z)
        for (std::uint32_t x = 0; x < sample_width; ++x)
            heights[static_cast<std::size_t>(z) * sample_width + x] = static_cast<float>(x + z) * 0.25f;

    arc::scene::terrain_surface_ir surface;
    surface.source_revision = revision;
    surface.local_bounds = {-256.0, 0.0, -256.0, 256.0, 2.0, 256.0};
    surface.geometry = arc::scene::terrain_surface_heightfield_ir{
        .sample_width = sample_width,
        .sample_height = sample_height,
        .width = 512.0f,
        .depth = 512.0f,
        .heights = heights,
        .material_weights = weights,
    };
    return surface;
}

arc::scene::terrain_component make_region_test_terrain()
{
    arc::scene::terrain_component terrain;
    terrain.size = 512.0f;
    terrain.subdivisions = 4u;
    terrain.content_revision = 1u;
    terrain.heights.resize(25u);
    terrain.layer_weights.assign(25u, std::array<std::uint8_t, 4>{255u, 0u, 0u, 0u});
    for (std::uint32_t z = 0; z < 5u; ++z)
        for (std::uint32_t x = 0; x < 5u; ++x)
            terrain.heights[static_cast<std::size_t>(z) * 5u + x] = static_cast<float>(x + z) * 0.25f;
    return terrain;
}

} // namespace

TEST_CASE("heightfield render partition is sample aligned and seam compatible")
{
    const auto surface = make_region_test_surface();
    const auto regions = arc::scene::build_terrain_render_regions(surface, 256.0);
    REQUIRE(regions.size() == 4u);
    CHECK(regions[0].id == arc::scene::terrain_region_id{0, 0});
    CHECK(regions[1].id == arc::scene::terrain_region_id{1, 0});
    CHECK(regions[2].id == arc::scene::terrain_region_id{0, 1});
    CHECK(regions[3].id == arc::scene::terrain_region_id{1, 1});

    const auto left = regions[0].surface.view();
    const auto right = regions[1].surface.view();
    const auto left_geometry = arc::scene::canonicalize_terrain_surface_geometry(left);
    const auto right_geometry = arc::scene::canonicalize_terrain_surface_geometry(right);
    REQUIRE(left_geometry.has_value());
    REQUIRE(right_geometry.has_value());

    const auto left_width = std::get<arc::scene::terrain_surface_heightfield_ir>(left.geometry).sample_width;
    const auto right_width = std::get<arc::scene::terrain_surface_heightfield_ir>(right.geometry).sample_width;
    REQUIRE(left_width == 3u);
    REQUIRE(right_width == 3u);
    for (std::uint32_t z = 0; z < 3u; ++z)
    {
        const auto& left_edge = left_geometry->positions[static_cast<std::size_t>(z) * left_width + 2u];
        const auto& right_edge = right_geometry->positions[static_cast<std::size_t>(z) * right_width];
        CHECK(left_edge[0] == Catch::Approx(right_edge[0]));
        CHECK(left_edge[1] == Catch::Approx(right_edge[1]));
        CHECK(left_edge[2] == Catch::Approx(right_edge[2]));
        CHECK(regions[0].vertex_normals[static_cast<std::size_t>(z) * left_width + 2u] ==
              regions[1].vertex_normals[static_cast<std::size_t>(z) * right_width]);
    }
}

TEST_CASE("terrain proxy replaces only the region whose geometry changed")
{
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    auto terrain = make_region_test_terrain();
    const auto guid = arc::ecs::generate_entity_guid();

    REQUIRE(cache.synchronize(guid, terrain, renderer));
    const auto* initial = cache.find(guid);
    REQUIRE(initial != nullptr);
    REQUIRE(initial->regions.size() == 4u);
    const auto region0 = initial->regions[0].geometry;
    const auto region1 = initial->regions[1].geometry;
    const auto region2 = initial->regions[2].geometry;
    const auto region3 = initial->regions[3].geometry;

    terrain.heights[6u] += 3.0f;
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .heights_changed = true};
    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));

    const auto* rebuilt = cache.find(guid);
    REQUIRE(rebuilt != nullptr);
    REQUIRE(rebuilt->regions.size() == 4u);
    CHECK(rebuilt->regions[0].geometry != region0);
    CHECK(rebuilt->regions[1].geometry == region1);
    CHECK(rebuilt->regions[2].geometry == region2);
    CHECK(rebuilt->regions[3].geometry == region3);
    CHECK_FALSE(renderer.mesh_alive(region0.conventional));
    CHECK(renderer.mesh_alive(region1.conventional));
    CHECK(renderer.mesh_alive(region2.conventional));
    CHECK(renderer.mesh_alive(region3.conventional));
}

TEST_CASE("terrain paint changes attributes without replacing region geometry")
{
    arc::render::renderer renderer;
    arc::scene::terrain_render_proxy_cache cache;
    auto terrain = make_region_test_terrain();
    const auto guid = arc::ecs::generate_entity_guid();

    REQUIRE(cache.synchronize(guid, terrain, renderer));
    const auto* initial = cache.find(guid);
    REQUIRE(initial != nullptr);
    REQUIRE(initial->regions.size() == 4u);
    const auto geometry0 = initial->regions[0].geometry;
    const auto attribute_fingerprint = initial->regions[0].attribute_fingerprint;

    terrain.layer_weights[6u] = {0u, 255u, 0u, 0u};
    ++terrain.content_revision;
    const arc::scene::terrain_dirty_region dirty{
        .min_x = 1u, .min_z = 1u, .max_x = 1u, .max_z = 1u, .valid = true, .weights_changed = true};
    REQUIRE(cache.synchronize(guid, terrain, renderer, &dirty));

    const auto* painted = cache.find(guid);
    REQUIRE(painted != nullptr);
    CHECK(painted->regions[0].geometry == geometry0);
    CHECK(painted->regions[0].attribute_fingerprint != attribute_fingerprint);
    CHECK(renderer.mesh_alive(geometry0.conventional));
}
''',
)

replace_once(
    "docs/terrain-roadmap.md",
    "## Milestone 1 - Compile Existing Heightfield Terrain into Virtual Geometry\n\n",
    "## Milestone 1 - Compile Existing Heightfield Terrain into Virtual Geometry\n\n"
    "**Status: complete.** Terrain now submits through ARC's generic conventional/virtual geometry path, "
    "with the dedicated terrain renderer removed after residency/fallback hardening.\n\n",
)
replace_once(
    "docs/terrain-roadmap.md",
    "## Milestone 2 - Terrain-Specific Virtual Geometry Streaming\n\n",
    "## Milestone 2 - Terrain-Specific Virtual Geometry Streaming\n\n"
    "**Status: in progress.** M2.1 establishes stable independently replaceable render-artifact regions; "
    "later M2 stages make their cooked payloads lazy, asynchronous, predictive, and budgeted.\n\n",
)

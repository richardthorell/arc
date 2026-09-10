#include <arc/scene/terrain_cooked_storage.h>

#include <arc/render/virtual_geometry_artifact.h>
#include <arc/scene/terrain_render_geometry.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

namespace arc::scene
{
namespace
{

constexpr std::uint32_t fallback_geometry_schema_version = 1u;
constexpr std::array<std::byte, 8> fallback_magic{
    static_cast<std::byte>('A'), static_cast<std::byte>('R'), static_cast<std::byte>('C'), static_cast<std::byte>('T'),
    static_cast<std::byte>('F'), static_cast<std::byte>('B'), static_cast<std::byte>('0'), static_cast<std::byte>('1')};

template <class T, bool = std::is_enum_v<T>> struct stored_type_for
{
    using type = T;
};

template <class T> struct stored_type_for<T, true>
{
    using type = std::underlying_type_t<T>;
};

template <class T> using stored_type_for_t = typename stored_type_for<T>::type;

class byte_writer
{
public:
    template <class T> void value(T input)
    {
        static_assert(std::is_integral_v<T> || std::is_enum_v<T> || std::is_floating_point_v<T>);
        using stored_type = stored_type_for_t<T>;
        if constexpr (std::is_floating_point_v<stored_type>)
        {
            using bits_type = std::conditional_t<sizeof(stored_type) == 4, std::uint32_t, std::uint64_t>;
            value(std::bit_cast<bits_type>(static_cast<stored_type>(input)));
        }
        else
        {
            using unsigned_type = std::make_unsigned_t<stored_type>;
            const auto bits = static_cast<unsigned_type>(static_cast<stored_type>(input));
            for (std::size_t index = 0; index < sizeof(stored_type); ++index)
                bytes_.push_back(static_cast<std::byte>((bits >> (index * 8u)) & 0xffu));
        }
    }

    void raw(std::span<const std::byte> bytes)
    {
        bytes_.insert(bytes_.end(), bytes.begin(), bytes.end());
    }

    [[nodiscard]] std::vector<std::byte> take() &&
    {
        return std::move(bytes_);
    }

private:
    std::vector<std::byte> bytes_;
};

std::uint32_t generation_for(std::uint64_t fingerprint) noexcept
{
    const auto folded = static_cast<std::uint32_t>(fingerprint) ^ static_cast<std::uint32_t>(fingerprint >> 32u);
    return folded == 0u ? 1u : folded;
}

std::string region_storage_key(terrain_region_id id, std::string_view product)
{
    return "terrain/regions/" + std::to_string(id.x) + "/" + std::to_string(id.z) + "/" + std::string(product);
}

std::vector<std::byte> encode_fallback_geometry(const render::virtual_mesh_data& geometry)
{
    byte_writer writer;
    writer.raw(fallback_magic);
    writer.value(fallback_geometry_schema_version);
    writer.value(static_cast<std::uint32_t>(geometry.conventional_lods.size()));
    for (const auto& lod : geometry.conventional_lods)
    {
        writer.value(lod.ratio);
        writer.value(lod.geometric_error);
        writer.value(static_cast<std::uint32_t>(lod.vertices.size()));
        writer.value(static_cast<std::uint32_t>(lod.indices.size()));
        for (const auto& vertex : lod.vertices)
        {
            for (const auto value : vertex.position) writer.value(value);
            for (const auto value : vertex.normal) writer.value(value);
            for (const auto value : vertex.tangent) writer.value(value);
            for (const auto value : vertex.texcoord) writer.value(value);
            for (const auto value : vertex.color) writer.value(value);
        }
        for (const auto index : lod.indices) writer.value(index);
    }
    return std::move(writer).take();
}

assets::cooked_artifact make_cooked_artifact(std::string name, std::string extension, assets::artifact_schema_id schema,
                                             std::uint32_t schema_version, std::vector<std::byte> bytes)
{
    assets::cooked_artifact artifact;
    artifact.name = std::move(name);
    artifact.extension = std::move(extension);
    artifact.schema = schema;
    artifact.schema_version = schema_version;
    artifact.bytes = std::move(bytes);
    artifact.hash = assets::hash_bytes(artifact.bytes);
    artifact.size = artifact.bytes.size();
    return artifact;
}

terrain_cooked_storage_result storage_failure(std::string message)
{
    return terrain_cooked_storage_result::failure({.message = std::move(message)});
}

} // namespace

terrain_cooked_storage_result build_terrain_cooked_storage(assets::asset_guid terrain, const terrain_surface_ir& surface,
                                                           std::uint64_t authoring_revision,
                                                           std::string_view target_profile,
                                                           double target_region_size)
{
    if (!terrain.valid() || authoring_revision == 0u || target_profile.empty() || !validate_terrain_surface_ir(surface) ||
        surface.source_revision == 0u)
        return storage_failure("invalid terrain cooked-storage build input");

    auto regions = build_terrain_render_regions(surface, target_region_size);
    if (regions.empty()) return storage_failure("terrain surface produced no cookable render regions");

    terrain_cooked_storage result;
    result.manifest.terrain = terrain;
    result.manifest.authoring_revision = authoring_revision;
    result.manifest.regions.reserve(regions.size());
    result.artifacts.reserve(regions.size() * 2u + 1u);

    for (auto& region : regions)
    {
        const auto view = region.surface.view();
        auto geometry = region.vertex_normals.empty() ? build_terrain_render_geometry(view)
                                                      : build_terrain_render_region_geometry(view, region.vertex_normals);
        if (!geometry || geometry->pages.empty() || geometry->conventional_lods.empty())
            return storage_failure("terrain region failed to produce virtual geometry and conventional fallback");

        const auto generation = generation_for(region.geometry_fingerprint);
        terrain_artifact_build_input key_input;
        key_input.region = region.id;
        key_input.surface_fingerprint = region.geometry_fingerprint;
        key_input.authoring_revision = authoring_revision;
        key_input.source_revision = surface.source_revision;
        key_input.target_profile = std::string(target_profile);

        render::virtual_geometry_artifact_source virtual_source{
            .name = "terrain-region", .material_index = 0u, .geometry = &*geometry};
        render::virtual_geometry_artifact_encode_options encode_options;
        encode_options.page_order = render::virtual_geometry_artifact_page_order::spatial_morton;
        auto encoded = render::encode_virtual_geometry_artifact(std::span(&virtual_source, 1u), 0u, encode_options);
        if (!encoded) return storage_failure("failed to encode terrain virtual-geometry artifact");
        auto virtual_bytes = std::move(encoded).value();
        auto inspected = render::inspect_virtual_geometry_artifact(virtual_bytes);
        if (!inspected || inspected.value().meshes.size() != 1u)
            return storage_failure("failed to inspect terrain virtual-geometry artifact");
        const auto& mesh_index = inspected.value().meshes.front();

        terrain_region_manifest manifest_region;
        manifest_region.region = region.id;
        manifest_region.bounds = region.surface.local_bounds;
        manifest_region.source_revision = surface.source_revision;
        manifest_region.compiled_revision = authoring_revision;

        terrain_artifact_reference render_reference;
        render_reference.kind = terrain_artifact_kind::render_geometry;
        render_reference.key = make_terrain_artifact_key(
            key_input, terrain_artifact_kind::render_geometry, render::virtual_geometry_artifact_schema_version);
        render_reference.compiler_version = render::virtual_geometry_artifact_schema_version;
        render_reference.storage_key = region_storage_key(region.id, "virtual-geometry");
        render_reference.generation = generation;
        render_reference.payload_size = virtual_bytes.size();
        render_reference.metadata_offset = mesh_index.metadata_offset;
        render_reference.metadata_size = mesh_index.metadata_size;
        render_reference.pages.reserve(mesh_index.pages.size());
        for (std::size_t page_index = 0; page_index < mesh_index.pages.size(); ++page_index)
        {
            const auto& page = mesh_index.pages[page_index];
            render_reference.pages.push_back({.index = static_cast<std::uint32_t>(page_index),
                                              .offset = page.offset,
                                              .stored_size = page.stored_size,
                                              .decoded_size = page.decoded_size,
                                              .content_hash = page.content_hash,
                                              .root = page.root});
        }
        result.artifacts.push_back(make_cooked_artifact(render_reference.storage_key, ".arcvg",
                                                        assets::artifact_schemas::virtual_geometry,
                                                        render::virtual_geometry_artifact_schema_version,
                                                        std::move(virtual_bytes)));
        manifest_region.artifacts.push_back(std::move(render_reference));

        auto fallback_bytes = encode_fallback_geometry(*geometry);
        terrain_artifact_reference fallback_reference;
        fallback_reference.kind = terrain_artifact_kind::fallback_geometry;
        fallback_reference.key = make_terrain_artifact_key(
            key_input, terrain_artifact_kind::fallback_geometry, fallback_geometry_schema_version);
        fallback_reference.compiler_version = fallback_geometry_schema_version;
        fallback_reference.storage_key = region_storage_key(region.id, "fallback-geometry");
        fallback_reference.generation = generation;
        fallback_reference.payload_size = fallback_bytes.size();
        result.artifacts.push_back(make_cooked_artifact(fallback_reference.storage_key, ".arctfb",
                                                        assets::artifact_schemas::mesh,
                                                        fallback_geometry_schema_version,
                                                        std::move(fallback_bytes)));
        manifest_region.artifacts.push_back(std::move(fallback_reference));
        result.manifest.regions.push_back(std::move(manifest_region));
    }

    if (!validate_terrain_cooked_manifest(result.manifest))
        return storage_failure("generated terrain cooked manifest failed validation");
    auto manifest_bytes = encode_terrain_cooked_manifest(result.manifest);
    if (!manifest_bytes) return storage_failure(manifest_bytes.error().message);
    result.artifacts.push_back(make_cooked_artifact("terrain/manifest", ".arctcm",
                                                    assets::artifact_schemas::terrain_manifest,
                                                    terrain_cooked_manifest::current_contract_version,
                                                    std::move(manifest_bytes).value()));
    return terrain_cooked_storage_result::success(std::move(result));
}

} // namespace arc::scene

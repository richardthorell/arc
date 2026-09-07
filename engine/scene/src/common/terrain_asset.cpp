#include <arc/scene/terrain_asset.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace arc::scene
{
namespace
{

bool finite(double value) noexcept
{
    return std::isfinite(value);
}

bool finite(float value) noexcept
{
    return std::isfinite(value);
}

bool finite(const math::vector3f& value) noexcept
{
    return finite(value[0]) && finite(value[1]) && finite(value[2]);
}

bool finite(const math::quatf& value) noexcept
{
    return finite(value.x) && finite(value.y) && finite(value.z) && finite(value.w);
}

bool valid_source_transform(const terrain_source_transform& transform) noexcept
{
    if (!finite(transform.translation) || !finite(transform.rotation) || !finite(transform.scale)) return false;
    constexpr float minimum_scale = 1.0e-6f;
    return std::abs(transform.scale[0]) >= minimum_scale && std::abs(transform.scale[1]) >= minimum_scale &&
           std::abs(transform.scale[2]) >= minimum_scale;
}

bool attribute_value_matches(terrain_attribute_type type, const terrain_attribute_value& value) noexcept
{
    switch (type)
    {
    case terrain_attribute_type::boolean:
        return std::holds_alternative<bool>(value);
    case terrain_attribute_type::signed_integer:
        return std::holds_alternative<std::int64_t>(value);
    case terrain_attribute_type::unsigned_integer:
        return std::holds_alternative<std::uint64_t>(value);
    case terrain_attribute_type::floating_point:
        return std::holds_alternative<double>(value);
    case terrain_attribute_type::vector4:
        return std::holds_alternative<math::vector4f>(value);
    case terrain_attribute_type::string:
        return std::holds_alternative<std::string>(value);
    }
    return false;
}

bool mutable_runtime(terrain_runtime_mutability value) noexcept
{
    return value != terrain_runtime_mutability::immutable;
}

bool fractureable_runtime(terrain_runtime_mutability value) noexcept
{
    return value == terrain_runtime_mutability::fractureable ||
           value == terrain_runtime_mutability::deformable_and_fractureable;
}

void add_issue(terrain_asset_validation_result& result, terrain_asset_validation_severity severity,
               terrain_asset_validation_code code, std::string message, terrain_stable_id subject = {})
{
    result.issues.push_back({severity, code, subject, std::move(message)});
}

} // namespace

terrain_stable_id generate_terrain_stable_id() noexcept
{
    const auto value = assets::generate_asset_guid();
    return {value.high, value.low};
}

std::string to_string(terrain_stable_id value)
{
    return core::to_string(value, core::uuid_text_format::hyphenated);
}

std::optional<terrain_stable_id> parse_terrain_stable_id(std::string_view text) noexcept
{
    return core::parse_uuid<terrain_stable_id_tag>(text);
}

bool terrain_world_bounds::valid() const noexcept
{
    return finite(min_x) && finite(min_y) && finite(min_z) && finite(max_x) && finite(max_y) && finite(max_z) &&
           min_x <= max_x && min_y <= max_y && min_z <= max_z;
}

terrain_region_id terrain_region_at(const terrain_coordinate_system& coordinates,
                                    const terrain_partition_settings& partition, double world_x,
                                    double world_z) noexcept
{
    if (!finite(partition.authoring_region_size) || partition.authoring_region_size <= 0.0 || !finite(world_x) ||
        !finite(world_z) || !finite(coordinates.origin_x) || !finite(coordinates.origin_z))
        return {};

    const auto x = std::floor((world_x - coordinates.origin_x) / partition.authoring_region_size);
    const auto z = std::floor((world_z - coordinates.origin_z) / partition.authoring_region_size);
    return {static_cast<std::int64_t>(x), static_cast<std::int64_t>(z)};
}

terrain_world_bounds terrain_region_bounds(const terrain_coordinate_system& coordinates,
                                           const terrain_partition_settings& partition,
                                           terrain_region_id region) noexcept
{
    if (!finite(partition.authoring_region_size) || partition.authoring_region_size <= 0.0) return {};
    const double min_x = coordinates.origin_x + static_cast<double>(region.x) * partition.authoring_region_size;
    const double min_z = coordinates.origin_z + static_cast<double>(region.z) * partition.authoring_region_size;
    return {min_x,
            coordinates.origin_y,
            min_z,
            min_x + partition.authoring_region_size,
            coordinates.origin_y,
            min_z + partition.authoring_region_size};
}

terrain_world_bounds expand_terrain_bounds(terrain_world_bounds bounds, double amount) noexcept
{
    if (!bounds.valid() || !finite(amount) || amount < 0.0) return bounds;
    bounds.min_x -= amount;
    bounds.min_y -= amount;
    bounds.min_z -= amount;
    bounds.max_x += amount;
    bounds.max_y += amount;
    bounds.max_z += amount;
    return bounds;
}

std::vector<terrain_region_id> terrain_regions_overlapping(const terrain_coordinate_system& coordinates,
                                                           const terrain_partition_settings& partition,
                                                           terrain_world_bounds bounds)
{
    std::vector<terrain_region_id> result;
    if (!bounds.valid() || !finite(partition.authoring_region_size) || partition.authoring_region_size <= 0.0)
        return result;

    const double max_x = bounds.max_x > bounds.min_x
                             ? std::nextafter(bounds.max_x, -std::numeric_limits<double>::infinity())
                             : bounds.max_x;
    const double max_z = bounds.max_z > bounds.min_z
                             ? std::nextafter(bounds.max_z, -std::numeric_limits<double>::infinity())
                             : bounds.max_z;
    const auto minimum = terrain_region_at(coordinates, partition, bounds.min_x, bounds.min_z);
    const auto maximum = terrain_region_at(coordinates, partition, max_x, max_z);

    if (maximum.x < minimum.x || maximum.z < minimum.z) return result;
    const auto width = static_cast<std::uint64_t>(maximum.x - minimum.x) + 1u;
    const auto depth = static_cast<std::uint64_t>(maximum.z - minimum.z) + 1u;
    if (width > 1'000'000u || depth > 1'000'000u || width * depth > 1'000'000u) return result;

    result.reserve(static_cast<std::size_t>(width * depth));
    for (std::int64_t z = minimum.z; z <= maximum.z; ++z)
        for (std::int64_t x = minimum.x; x <= maximum.x; ++x) result.push_back({x, z});
    return result;
}

bool terrain_asset_validation_result::valid() const noexcept
{
    return std::none_of(issues.begin(), issues.end(), [](const auto& issue)
                        { return issue.severity == terrain_asset_validation_severity::error; });
}

terrain_asset_validation_result validate_terrain_asset(const terrain_asset& asset)
{
    terrain_asset_validation_result result;

    if (asset.schema_version != terrain_asset::current_schema_version)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::unsupported_schema,
                  "terrain asset schema version is unsupported");

    if (!finite(asset.coordinates.origin_x) || !finite(asset.coordinates.origin_y) ||
        !finite(asset.coordinates.origin_z) || !finite(asset.coordinates.meters_per_unit) ||
        asset.coordinates.meters_per_unit <= 0.0)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_coordinates,
                  "terrain coordinate system must contain finite origins and a positive meters-per-unit scale");

    if (!finite(asset.partition.authoring_region_size) || asset.partition.authoring_region_size <= 0.0 ||
        !finite(asset.partition.dependency_halo) || asset.partition.dependency_halo < 0.0)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_partition,
                  "terrain partition must use a positive region size and non-negative dependency halo");

    if (!asset.source.id.valid() || asset.source.schema_version == 0u || !valid_source_transform(asset.source.transform))
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_source,
                  "terrain source requires a stable ID, positive schema version, and finite non-zero transform",
                  asset.source.id);

    if ((asset.source.kind == terrain_source_kind::heightfield || asset.source.kind == terrain_source_kind::mesh) &&
        !asset.source.asset.guid.valid() && asset.source.asset.path_hint.empty())
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_source,
                  "heightfield and mesh terrain sources require an asset reference", asset.source.id);

    if (asset.source.kind == terrain_source_kind::procedural && asset.source.generator_id.empty())
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_source,
                  "procedural terrain source requires a generator ID", asset.source.id);

    std::unordered_set<terrain_stable_id, core::uuid_hash<terrain_stable_id_tag>> ids;
    if (asset.source.id.valid()) ids.insert(asset.source.id);

    for (const auto& modifier : asset.modifiers)
    {
        if (!modifier.id.valid() || modifier.type_id.empty() || modifier.schema_version == 0u ||
            modifier.domains == terrain_domain::none || modifier.canonical_parameters.empty() ||
            (modifier.affected_bounds && !modifier.affected_bounds->valid()))
            add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_modifier,
                      "terrain modifier requires an ID, type, schema, domains, parameters, and valid optional bounds",
                      modifier.id);

        if (modifier.id.valid() && !ids.insert(modifier.id).second)
            add_issue(result, terrain_asset_validation_severity::error,
                      terrain_asset_validation_code::duplicate_stable_id,
                      "terrain source, modifiers, and attributes must not share stable IDs", modifier.id);
    }

    std::unordered_set<std::string> attribute_names;
    for (const auto& attribute : asset.attributes)
    {
        if (!attribute.id.valid() || attribute.name.empty() || attribute.schema_version == 0u ||
            !attribute_value_matches(attribute.type, attribute.default_value))
            add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_attribute,
                      "terrain attribute requires an ID, name, schema, and default value matching its declared type",
                      attribute.id);

        if (attribute.id.valid() && !ids.insert(attribute.id).second)
            add_issue(result, terrain_asset_validation_severity::error,
                      terrain_asset_validation_code::duplicate_stable_id,
                      "terrain source, modifiers, and attributes must not share stable IDs", attribute.id);

        if (!attribute.name.empty() && !attribute_names.insert(attribute.name).second)
            add_issue(result, terrain_asset_validation_severity::error,
                      terrain_asset_validation_code::duplicate_attribute_name,
                      "terrain attribute names must be unique within an asset", attribute.id);
    }

    if (!finite(asset.build.target_surface_error) || asset.build.target_surface_error <= 0.0f)
        add_issue(result, terrain_asset_validation_severity::error,
                  terrain_asset_validation_code::invalid_build_settings,
                  "terrain target surface error must be finite and positive");

    if (!mutable_runtime(asset.runtime.mutability) &&
        (asset.runtime.persistent_runtime_changes || asset.runtime.replicate_runtime_changes))
        add_issue(result, terrain_asset_validation_severity::error,
                  terrain_asset_validation_code::invalid_runtime_policy,
                  "immutable terrain cannot persist or replicate runtime terrain operations");

    if (fractureable_runtime(asset.runtime.mutability) && !asset.build.build_destruction)
        add_issue(result, terrain_asset_validation_severity::warning,
                  terrain_asset_validation_code::invalid_runtime_policy,
                  "fractureable terrain has destruction derived-data generation disabled");

    return result;
}

} // namespace arc::scene

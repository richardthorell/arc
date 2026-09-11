#include <arc/scene/terrain_asset.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

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
    return finite(value.x()) && finite(value.y()) && finite(value.z()) && finite(value.w());
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

constexpr std::uint32_t domain_bits(terrain_domain value) noexcept
{
    return static_cast<std::uint32_t>(value);
}

constexpr terrain_domain valid_domains(terrain_domain value) noexcept
{
    return static_cast<terrain_domain>(domain_bits(value) & domain_bits(terrain_domain::all));
}

bool region_less(terrain_region_id lhs, terrain_region_id rhs) noexcept
{
    return lhs.z < rhs.z || (lhs.z == rhs.z && lhs.x < rhs.x);
}

bool close_enough(double lhs, double rhs) noexcept
{
    const auto scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= scale * 1.0e-10;
}

bool same_bounds(const terrain_world_bounds& lhs, const terrain_world_bounds& rhs) noexcept
{
    return close_enough(lhs.min_x, rhs.min_x) && close_enough(lhs.min_y, rhs.min_y) &&
           close_enough(lhs.min_z, rhs.min_z) && close_enough(lhs.max_x, rhs.max_x) &&
           close_enough(lhs.max_y, rhs.max_y) && close_enough(lhs.max_z, rhs.max_z);
}

terrain_region_record* find_region(terrain_asset& asset, terrain_region_id region) noexcept
{
    const auto found = std::lower_bound(asset.regions.begin(), asset.regions.end(), region,
                                        [](const terrain_region_record& value, terrain_region_id id)
                                        { return region_less(value.id, id); });
    return found != asset.regions.end() && found->id == region ? &*found : nullptr;
}

const terrain_region_record* find_region(const terrain_asset& asset, terrain_region_id region) noexcept
{
    const auto found = std::lower_bound(asset.regions.begin(), asset.regions.end(), region,
                                        [](const terrain_region_record& value, terrain_region_id id)
                                        { return region_less(value.id, id); });
    return found != asset.regions.end() && found->id == region ? &*found : nullptr;
}

void merge_dependency(std::vector<terrain_region_dependency>& dependencies, terrain_region_dependency dependency)
{
    dependency.domains = valid_domains(dependency.domains);
    if (dependency.domains == terrain_domain::none) return;
    const auto found =
        std::find_if(dependencies.begin(), dependencies.end(),
                     [&](const terrain_region_dependency& value) { return value.region == dependency.region; });
    if (found == dependencies.end())
        dependencies.push_back(dependency);
    else
        found->domains |= dependency.domains;
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

terrain_modifier_sample_location terrain_modifier_sample_at(const terrain_coordinate_system& coordinates,
                                                            const terrain_partition_settings& partition, double world_x,
                                                            double world_z) noexcept
{
    if (!finite(partition.authoring_region_size) || partition.authoring_region_size <= 0.0 || !finite(world_x) ||
        !finite(world_z) || !finite(coordinates.origin_x) || !finite(coordinates.origin_z))
        return {};

    const auto region = terrain_region_at(coordinates, partition, world_x, world_z);
    const auto bounds = terrain_region_bounds(coordinates, partition, region);
    const auto quantize = [](double value, double minimum, double maximum)
    {
        const auto span = maximum - minimum;
        if (!finite(span) || span <= 0.0) return 0u;
        const auto normalized = std::clamp((value - minimum) / span, 0.0, 1.0);
        return static_cast<std::uint32_t>(
            std::llround(normalized * static_cast<double>(terrain_modifier_sample_coordinate_max)));
    };
    return {region, quantize(world_x, bounds.min_x, bounds.max_x), quantize(world_z, bounds.min_z, bounds.max_z)};
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

    const auto minimum = terrain_region_at(coordinates, partition, bounds.min_x, bounds.min_z);
    auto maximum = terrain_region_at(coordinates, partition, bounds.max_x, bounds.max_z);
    if (bounds.max_x > bounds.min_x)
    {
        const auto boundary = coordinates.origin_x + static_cast<double>(maximum.x) * partition.authoring_region_size;
        if (bounds.max_x == boundary && maximum.x != std::numeric_limits<std::int64_t>::min()) --maximum.x;
    }
    if (bounds.max_z > bounds.min_z)
    {
        const auto boundary = coordinates.origin_z + static_cast<double>(maximum.z) * partition.authoring_region_size;
        if (bounds.max_z == boundary && maximum.z != std::numeric_limits<std::int64_t>::min()) --maximum.z;
    }

    if (maximum.x < minimum.x || maximum.z < minimum.z) return result;
    const auto width = static_cast<std::uint64_t>(maximum.x - minimum.x) + 1u;
    const auto depth = static_cast<std::uint64_t>(maximum.z - minimum.z) + 1u;
    if (width > 1'000'000u || depth > 1'000'000u || width * depth > 1'000'000u) return result;

    result.reserve(static_cast<std::size_t>(width * depth));
    for (std::int64_t z = minimum.z; z <= maximum.z; ++z)
        for (std::int64_t x = minimum.x; x <= maximum.x; ++x)
            result.push_back({x, z});
    return result;
}

terrain_region_record& ensure_terrain_region(terrain_asset& asset, terrain_region_id region)
{
    const auto found = std::lower_bound(asset.regions.begin(), asset.regions.end(), region,
                                        [](const terrain_region_record& value, terrain_region_id id)
                                        { return region_less(value.id, id); });
    if (found != asset.regions.end() && found->id == region) return *found;

    terrain_region_record record;
    record.id = region;
    record.authoring_bounds = terrain_region_bounds(asset.coordinates, asset.partition, region);
    return *asset.regions.insert(found, std::move(record));
}

terrain_dirty_update mark_terrain_dirty(terrain_asset& asset, terrain_world_bounds bounds, terrain_domain domains)
{
    terrain_dirty_update result;
    domains = valid_domains(domains);
    if (!bounds.valid() || domains == terrain_domain::none) return result;

    result.regions = terrain_regions_overlapping(asset.coordinates, asset.partition, bounds);
    if (result.regions.empty()) return result;

    if (asset.authoring_revision != std::numeric_limits<std::uint64_t>::max()) ++asset.authoring_revision;
    result.revision = asset.authoring_revision;
    for (const auto region : result.regions)
    {
        auto& record = ensure_terrain_region(asset, region);
        record.dirty_revision = result.revision;
        record.dirty_domains |= domains;
    }
    return result;
}

bool mark_terrain_region_compiled(terrain_asset& asset, terrain_region_id region, terrain_domain domains,
                                  std::uint64_t build_revision) noexcept
{
    auto* record = find_region(asset, region);
    domains = valid_domains(domains);
    if (!record || domains == terrain_domain::none || build_revision != record->dirty_revision) return false;

    const auto completed = domain_bits(record->dirty_domains) & domain_bits(domains);
    if (completed == 0u) return false;
    record->dirty_domains = static_cast<terrain_domain>(domain_bits(record->dirty_domains) & ~completed);
    if (record->dirty_domains == terrain_domain::none) record->compiled_revision = build_revision;
    return true;
}

terrain_build_region_snapshot make_terrain_build_region_snapshot(const terrain_asset& asset, terrain_region_id region)
{
    terrain_build_region_snapshot snapshot;
    snapshot.target = region;
    snapshot.authoring_bounds = terrain_region_bounds(asset.coordinates, asset.partition, region);
    snapshot.evaluation_bounds = expand_terrain_bounds(snapshot.authoring_bounds, asset.partition.dependency_halo);
    snapshot.authoring_revision = asset.authoring_revision;
    if (const auto* record = find_region(asset, region)) snapshot.target_dirty_revision = record->dirty_revision;

    constexpr auto halo_domains = terrain_domain::geometry | terrain_domain::attributes | terrain_domain::topology;
    for (const auto dependency_region :
         terrain_regions_overlapping(asset.coordinates, asset.partition, snapshot.evaluation_bounds))
    {
        if (dependency_region != region) merge_dependency(snapshot.dependencies, {dependency_region, halo_domains});
    }

    if (const auto* record = find_region(asset, region))
        for (const auto& dependency : record->dependencies)
            if (dependency.region != region) merge_dependency(snapshot.dependencies, dependency);

    std::sort(snapshot.dependencies.begin(), snapshot.dependencies.end(),
              [](const terrain_region_dependency& lhs, const terrain_region_dependency& rhs)
              { return region_less(lhs.region, rhs.region); });
    return snapshot;
}

bool terrain_asset_validation_result::valid() const noexcept
{
    return std::none_of(issues.begin(), issues.end(),
                        [](const auto& issue) { return issue.severity == terrain_asset_validation_severity::error; });
}

terrain_asset_validation_result validate_terrain_asset(const terrain_asset& asset)
{
    terrain_asset_validation_result result;

    if (asset.schema_version != terrain_asset::current_schema_version)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::unsupported_schema,
                  "terrain asset schema version is unsupported");

    if (asset.authoring_revision == 0u)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_region,
                  "terrain authoring revision must be positive");

    if (!finite(asset.coordinates.origin_x) || !finite(asset.coordinates.origin_y) ||
        !finite(asset.coordinates.origin_z) || !finite(asset.coordinates.meters_per_unit) ||
        asset.coordinates.meters_per_unit <= 0.0)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_coordinates,
                  "terrain coordinate system must contain finite origins and a positive meters-per-unit scale");

    if (!finite(asset.partition.authoring_region_size) || asset.partition.authoring_region_size <= 0.0 ||
        !finite(asset.partition.dependency_halo) || asset.partition.dependency_halo < 0.0)
        add_issue(result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_partition,
                  "terrain partition must use a positive region size and non-negative dependency halo");

    if (!asset.source.id.valid() || asset.source.schema_version == 0u ||
        !valid_source_transform(asset.source.transform))
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
            valid_domains(modifier.domains) == terrain_domain::none ||
            modifier.domains != valid_domains(modifier.domains) || modifier.canonical_parameters.empty() ||
            (modifier.affected_bounds && !modifier.affected_bounds->valid()))
            add_issue(
                result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_modifier,
                "terrain modifier requires an ID, type, schema, valid domains, parameters, and valid optional bounds",
                modifier.id);

        if (!validate_terrain_modifier_payloads(modifier))
            add_issue(result, terrain_asset_validation_severity::error,
                      terrain_asset_validation_code::invalid_modifier_payload,
                      "terrain modifier sparse region payloads are malformed or incompatible with the modifier type",
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
            add_issue(result, terrain_asset_validation_severity::error,
                      terrain_asset_validation_code::invalid_attribute,
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

    std::vector<terrain_region_id> region_ids;
    region_ids.reserve(asset.regions.size());
    for (const auto& region : asset.regions)
    {
        const auto expected_bounds = terrain_region_bounds(asset.coordinates, asset.partition, region.id);
        bool valid_region = region.authoring_bounds.valid() && same_bounds(region.authoring_bounds, expected_bounds) &&
                            region.dirty_revision <= asset.authoring_revision &&
                            region.compiled_revision <= region.dirty_revision &&
                            region.dirty_domains == valid_domains(region.dirty_domains);
        if (region.dirty_revision > 0u && region.dirty_domains == terrain_domain::none &&
            region.compiled_revision != region.dirty_revision)
            valid_region = false;

        std::vector<terrain_region_id> dependency_ids;
        dependency_ids.reserve(region.dependencies.size());
        for (const auto& dependency : region.dependencies)
        {
            if (dependency.region == region.id || dependency.domains == terrain_domain::none ||
                dependency.domains != valid_domains(dependency.domains) ||
                std::find(dependency_ids.begin(), dependency_ids.end(), dependency.region) != dependency_ids.end())
                valid_region = false;
            dependency_ids.push_back(dependency.region);
        }

        if (std::find(region_ids.begin(), region_ids.end(), region.id) != region_ids.end()) valid_region = false;
        region_ids.push_back(region.id);
        if (!valid_region)
            add_issue(
                result, terrain_asset_validation_severity::error, terrain_asset_validation_code::invalid_region,
                "terrain region records require canonical bounds, valid revisions, domains, and unique dependencies");
    }

    return result;
}

} // namespace arc::scene

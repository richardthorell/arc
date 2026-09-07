#include <arc/scene/terrain_asset_io.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace arc::scene
{
namespace
{

using json = nlohmann::json;
constexpr std::uint32_t terrain_document_format_version = 1;

struct domain_name
{
    terrain_domain domain;
    std::string_view name;
};

constexpr std::array<domain_name, 6> domain_names{{
    {terrain_domain::geometry, "geometry"},
    {terrain_domain::attributes, "attributes"},
    {terrain_domain::topology, "topology"},
    {terrain_domain::collision, "collision"},
    {terrain_domain::navigation, "navigation"},
    {terrain_domain::destruction, "destruction"},
}};

template <class Result> Result failure(terrain_asset_io_error_code code, std::string message)
{
    return Result::failure({code, std::move(message)});
}

const char* source_kind_name(terrain_source_kind value) noexcept
{
    switch (value)
    {
        case terrain_source_kind::flat:
            return "flat";
        case terrain_source_kind::heightfield:
            return "heightfield";
        case terrain_source_kind::mesh:
            return "mesh";
        case terrain_source_kind::procedural:
            return "procedural";
    }
    return "flat";
}

std::optional<terrain_source_kind> parse_source_kind(std::string_view value) noexcept
{
    if (value == "flat") return terrain_source_kind::flat;
    if (value == "heightfield") return terrain_source_kind::heightfield;
    if (value == "mesh") return terrain_source_kind::mesh;
    if (value == "procedural") return terrain_source_kind::procedural;
    return std::nullopt;
}

const char* attribute_type_name(terrain_attribute_type value) noexcept
{
    switch (value)
    {
        case terrain_attribute_type::boolean:
            return "boolean";
        case terrain_attribute_type::signed_integer:
            return "signed-integer";
        case terrain_attribute_type::unsigned_integer:
            return "unsigned-integer";
        case terrain_attribute_type::floating_point:
            return "floating-point";
        case terrain_attribute_type::vector4:
            return "vector4";
        case terrain_attribute_type::string:
            return "string";
    }
    return "floating-point";
}

std::optional<terrain_attribute_type> parse_attribute_type(std::string_view value) noexcept
{
    if (value == "boolean") return terrain_attribute_type::boolean;
    if (value == "signed-integer") return terrain_attribute_type::signed_integer;
    if (value == "unsigned-integer") return terrain_attribute_type::unsigned_integer;
    if (value == "floating-point") return terrain_attribute_type::floating_point;
    if (value == "vector4") return terrain_attribute_type::vector4;
    if (value == "string") return terrain_attribute_type::string;
    return std::nullopt;
}

const char* attribute_semantic_name(terrain_attribute_semantic value) noexcept
{
    switch (value)
    {
        case terrain_attribute_semantic::custom:
            return "custom";
        case terrain_attribute_semantic::material_weight:
            return "material-weight";
        case terrain_attribute_semantic::physical_material:
            return "physical-material";
        case terrain_attribute_semantic::wetness:
            return "wetness";
        case terrain_attribute_semantic::snow:
            return "snow";
        case terrain_attribute_semantic::biome:
            return "biome";
        case terrain_attribute_semantic::foliage_density:
            return "foliage-density";
        case terrain_attribute_semantic::navigation_cost:
            return "navigation-cost";
        case terrain_attribute_semantic::destruction_strength:
            return "destruction-strength";
        case terrain_attribute_semantic::hardness:
            return "hardness";
        case terrain_attribute_semantic::acoustic_surface:
            return "acoustic-surface";
        case terrain_attribute_semantic::gameplay_tag:
            return "gameplay-tag";
    }
    return "custom";
}

std::optional<terrain_attribute_semantic> parse_attribute_semantic(std::string_view value) noexcept
{
    if (value == "custom") return terrain_attribute_semantic::custom;
    if (value == "material-weight") return terrain_attribute_semantic::material_weight;
    if (value == "physical-material") return terrain_attribute_semantic::physical_material;
    if (value == "wetness") return terrain_attribute_semantic::wetness;
    if (value == "snow") return terrain_attribute_semantic::snow;
    if (value == "biome") return terrain_attribute_semantic::biome;
    if (value == "foliage-density") return terrain_attribute_semantic::foliage_density;
    if (value == "navigation-cost") return terrain_attribute_semantic::navigation_cost;
    if (value == "destruction-strength") return terrain_attribute_semantic::destruction_strength;
    if (value == "hardness") return terrain_attribute_semantic::hardness;
    if (value == "acoustic-surface") return terrain_attribute_semantic::acoustic_surface;
    if (value == "gameplay-tag") return terrain_attribute_semantic::gameplay_tag;
    return std::nullopt;
}

const char* attribute_storage_name(terrain_attribute_storage value) noexcept
{
    switch (value)
    {
        case terrain_attribute_storage::sparse_tiles:
            return "sparse-tiles";
        case terrain_attribute_storage::dense_tiles:
            return "dense-tiles";
        case terrain_attribute_storage::procedural:
            return "procedural";
    }
    return "sparse-tiles";
}

std::optional<terrain_attribute_storage> parse_attribute_storage(std::string_view value) noexcept
{
    if (value == "sparse-tiles") return terrain_attribute_storage::sparse_tiles;
    if (value == "dense-tiles") return terrain_attribute_storage::dense_tiles;
    if (value == "procedural") return terrain_attribute_storage::procedural;
    return std::nullopt;
}

const char* interpolation_name(terrain_attribute_interpolation value) noexcept
{
    return value == terrain_attribute_interpolation::nearest ? "nearest" : "linear";
}

std::optional<terrain_attribute_interpolation> parse_interpolation(std::string_view value) noexcept
{
    if (value == "nearest") return terrain_attribute_interpolation::nearest;
    if (value == "linear") return terrain_attribute_interpolation::linear;
    return std::nullopt;
}

const char* geometry_quality_name(terrain_geometry_quality value) noexcept
{
    switch (value)
    {
        case terrain_geometry_quality::scalable:
            return "scalable";
        case terrain_geometry_quality::balanced:
            return "balanced";
        case terrain_geometry_quality::maximum:
            return "maximum";
    }
    return "balanced";
}

std::optional<terrain_geometry_quality> parse_geometry_quality(std::string_view value) noexcept
{
    if (value == "scalable") return terrain_geometry_quality::scalable;
    if (value == "balanced") return terrain_geometry_quality::balanced;
    if (value == "maximum") return terrain_geometry_quality::maximum;
    return std::nullopt;
}

const char* mutability_name(terrain_runtime_mutability value) noexcept
{
    switch (value)
    {
        case terrain_runtime_mutability::immutable:
            return "immutable";
        case terrain_runtime_mutability::deformable:
            return "deformable";
        case terrain_runtime_mutability::fractureable:
            return "fractureable";
        case terrain_runtime_mutability::deformable_and_fractureable:
            return "deformable-and-fractureable";
    }
    return "immutable";
}

std::optional<terrain_runtime_mutability> parse_mutability(std::string_view value) noexcept
{
    if (value == "immutable") return terrain_runtime_mutability::immutable;
    if (value == "deformable") return terrain_runtime_mutability::deformable;
    if (value == "fractureable") return terrain_runtime_mutability::fractureable;
    if (value == "deformable-and-fractureable") return terrain_runtime_mutability::deformable_and_fractureable;
    return std::nullopt;
}

json domains_json(terrain_domain domains)
{
    json result = json::array();
    for (const auto& entry : domain_names)
        if (terrain_domain_contains(domains, entry.domain)) result.push_back(entry.name);
    return result;
}

std::optional<terrain_domain> parse_domains(const json& value)
{
    if (!value.is_array()) return std::nullopt;
    terrain_domain result = terrain_domain::none;
    for (const auto& item : value)
    {
        if (!item.is_string()) return std::nullopt;
        const auto text = item.get<std::string>();
        const auto found = std::find_if(domain_names.begin(), domain_names.end(),
                                        [&](const domain_name& entry) { return entry.name == text; });
        if (found == domain_names.end()) return std::nullopt;
        result |= found->domain;
    }
    return result;
}

json reference_json(const assets::asset_reference& reference)
{
    json result = json::object();
    if (reference.guid.valid()) result["guid"] = assets::to_string(reference.guid);
    if (reference.expected_type.valid()) result["expectedType"] = assets::to_string(reference.expected_type);
    if (!reference.path_hint.empty()) result["pathHint"] = reference.path_hint;
    return result;
}

bool parse_reference(const json& value, assets::asset_reference& reference)
{
    if (!value.is_object()) return false;
    if (const auto found = value.find("guid"); found != value.end())
    {
        if (!found->is_string()) return false;
        const auto parsed = assets::parse_asset_guid(found->get<std::string>());
        if (!parsed) return false;
        reference.guid = *parsed;
    }
    if (const auto found = value.find("expectedType"); found != value.end())
    {
        if (!found->is_string()) return false;
        const auto parsed = assets::parse_asset_type_id(found->get<std::string>());
        if (!parsed) return false;
        reference.expected_type = *parsed;
    }
    if (const auto found = value.find("pathHint"); found != value.end())
    {
        if (!found->is_string()) return false;
        reference.path_hint = found->get<std::string>();
    }
    return true;
}

json bounds_json(const terrain_world_bounds& bounds)
{
    return {{"min", {bounds.min_x, bounds.min_y, bounds.min_z}}, {"max", {bounds.max_x, bounds.max_y, bounds.max_z}}};
}

bool parse_bounds(const json& value, terrain_world_bounds& bounds)
{
    if (!value.is_object() || !value.contains("min") || !value.contains("max") || !value["min"].is_array() ||
        !value["max"].is_array() || value["min"].size() != 3 || value["max"].size() != 3)
        return false;
    bounds = {value["min"][0].get<double>(), value["min"][1].get<double>(), value["min"][2].get<double>(),
              value["max"][0].get<double>(), value["max"][1].get<double>(), value["max"][2].get<double>()};
    return bounds.valid();
}

json attribute_default_json(const terrain_attribute_definition& attribute)
{
    switch (attribute.type)
    {
        case terrain_attribute_type::boolean:
            return std::get<bool>(attribute.default_value);
        case terrain_attribute_type::signed_integer:
            return std::get<std::int64_t>(attribute.default_value);
        case terrain_attribute_type::unsigned_integer:
            return std::get<std::uint64_t>(attribute.default_value);
        case terrain_attribute_type::floating_point:
            return std::get<double>(attribute.default_value);
        case terrain_attribute_type::vector4:
        {
            const auto& value = std::get<math::vector4f>(attribute.default_value);
            return json::array({value[0], value[1], value[2], value[3]});
        }
        case terrain_attribute_type::string:
            return std::get<std::string>(attribute.default_value);
    }
    return nullptr;
}

bool parse_attribute_default(terrain_attribute_type type, const json& value, terrain_attribute_value& output)
{
    switch (type)
    {
        case terrain_attribute_type::boolean:
            if (!value.is_boolean()) return false;
            output = value.get<bool>();
            return true;
        case terrain_attribute_type::signed_integer:
            if (!value.is_number_integer()) return false;
            output = value.get<std::int64_t>();
            return true;
        case terrain_attribute_type::unsigned_integer:
            if (!value.is_number_unsigned()) return false;
            output = value.get<std::uint64_t>();
            return true;
        case terrain_attribute_type::floating_point:
            if (!value.is_number()) return false;
            output = value.get<double>();
            return true;
        case terrain_attribute_type::vector4:
            if (!value.is_array() || value.size() != 4) return false;
            output = math::vector4f{value[0].get<float>(), value[1].get<float>(), value[2].get<float>(),
                                    value[3].get<float>()};
            return true;
        case terrain_attribute_type::string:
            if (!value.is_string()) return false;
            output = value.get<std::string>();
            return true;
    }
    return false;
}

void append_dependency(std::vector<assets::asset_reference>& output, assets::asset_reference dependency)
{
    if (!dependency.guid.valid() && dependency.path_hint.empty()) return;
    const auto found =
        std::find_if(output.begin(), output.end(),
                     [&](const assets::asset_reference& existing)
                     {
                         return dependency.guid.valid() && existing.guid.valid()
                                    ? dependency.guid == existing.guid
                                    : !dependency.path_hint.empty() && dependency.path_hint == existing.path_hint;
                     });
    if (found == output.end()) output.push_back(std::move(dependency));
}

void collect_parameter_dependencies(const json& value, std::vector<assets::asset_reference>& output)
{
    if (value.is_object())
    {
        if (value.contains("guid") || value.contains("pathHint"))
        {
            assets::asset_reference reference;
            if (parse_reference(value, reference)) append_dependency(output, std::move(reference));
        }
        for (const auto& [key, child] : value.items())
        {
            (void)key;
            collect_parameter_dependencies(child, output);
        }
    }
    else if (value.is_array())
    {
        for (const auto& child : value)
            collect_parameter_dependencies(child, output);
    }
}

std::vector<assets::asset_reference> terrain_dependencies(const terrain_asset& asset)
{
    std::vector<assets::asset_reference> result;
    append_dependency(result, asset.source.asset);
    append_dependency(result, asset.runtime.damage_profile);
    for (const auto& modifier : asset.modifiers)
    {
        const auto parameters = json::parse(modifier.canonical_parameters, nullptr, false);
        if (!parameters.is_discarded()) collect_parameter_dependencies(parameters, result);
    }
    return result;
}

class terrain_asset_importer final : public assets::asset_importer
{
public:
    terrain_asset_importer()
    {
        descriptor_.id = assets::importer_ids::terrain;
        descriptor_.name = "ARC Terrain";
        descriptor_.version = 1;
        descriptor_.settings_version = 1;
        descriptor_.extensions = {".terrain"};
        descriptor_.output_types = {assets::asset_types::terrain};
    }

    const assets::asset_importer_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }

    assets::asset_import_result import(const assets::asset_import_context& context) override
    {
        if (context.cancellation.stop_requested())
            return {.error = {.code = assets::asset_error_code::cancelled,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "Terrain asset import was cancelled"}};

        const std::string text(reinterpret_cast<const char*>(context.source_bytes.data()), context.source_bytes.size());
        auto decoded = read_terrain_asset_json(text);
        if (!decoded)
            return {.error = {.code = assets::asset_error_code::import_failed,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = decoded.error().message}};

        auto terrain = std::make_shared<terrain_asset>(std::move(decoded.value()));
        assets::asset_import_result result;
        result.dependencies = terrain_dependencies(*terrain);
        result.payload = assets::asset_payload::make<terrain_asset>(
            assets::asset_types::terrain, std::move(terrain), sizeof(terrain_asset) + context.source_bytes.size());
        return result;
    }

private:
    assets::asset_importer_descriptor descriptor_;
};

} // namespace

terrain_asset_json_result write_terrain_asset_json(const terrain_asset& asset, bool pretty)
{
    const auto validation = validate_terrain_asset(asset);
    if (!validation.valid())
        return failure<terrain_asset_json_result>(terrain_asset_io_error_code::invalid_asset,
                                                  "Cannot serialize an invalid terrain asset");

    try
    {
        json source{
            {"id", to_string(asset.source.id)},
            {"kind", source_kind_name(asset.source.kind)},
            {"schemaVersion", asset.source.schema_version},
            {"asset", reference_json(asset.source.asset)},
            {"generatorId", asset.source.generator_id},
            {"seed", asset.source.seed},
            {"transform",
             {{"translation",
               {asset.source.transform.translation[0], asset.source.transform.translation[1],
                asset.source.transform.translation[2]}},
              {"rotation",
               {asset.source.transform.rotation.x(), asset.source.transform.rotation.y(),
                asset.source.transform.rotation.z(), asset.source.transform.rotation.w()}},
              {"scale",
               {asset.source.transform.scale[0], asset.source.transform.scale[1], asset.source.transform.scale[2]}}}}};

        json modifiers = json::array();
        for (const auto& modifier : asset.modifiers)
        {
            auto parameters = json::parse(modifier.canonical_parameters, nullptr, false);
            if (parameters.is_discarded())
                return failure<terrain_asset_json_result>(terrain_asset_io_error_code::invalid_asset,
                                                          "Terrain modifier parameters are not valid JSON");
            json record{{"id", to_string(modifier.id)},
                        {"type", modifier.type_id},
                        {"name", modifier.name},
                        {"schemaVersion", modifier.schema_version},
                        {"enabled", modifier.enabled},
                        {"domains", domains_json(modifier.domains)},
                        {"parameters", std::move(parameters)}};
            if (modifier.affected_bounds) record["affectedBounds"] = bounds_json(*modifier.affected_bounds);
            modifiers.push_back(std::move(record));
        }

        json attributes = json::array();
        for (const auto& attribute : asset.attributes)
            attributes.push_back({{"id", to_string(attribute.id)},
                                  {"name", attribute.name},
                                  {"schemaVersion", attribute.schema_version},
                                  {"type", attribute_type_name(attribute.type)},
                                  {"semantic", attribute_semantic_name(attribute.semantic)},
                                  {"storage", attribute_storage_name(attribute.storage)},
                                  {"interpolation", interpolation_name(attribute.interpolation)},
                                  {"default", attribute_default_json(attribute)}});

        json regions = json::array();
        for (const auto& region : asset.regions)
        {
            json dependencies = json::array();
            for (const auto& dependency : region.dependencies)
                dependencies.push_back({{"x", dependency.region.x},
                                        {"z", dependency.region.z},
                                        {"domains", domains_json(dependency.domains)}});
            regions.push_back({{"x", region.id.x},
                               {"z", region.id.z},
                               {"authoringBounds", bounds_json(region.authoring_bounds)},
                               {"dirtyRevision", region.dirty_revision},
                               {"compiledRevision", region.compiled_revision},
                               {"dirtyDomains", domains_json(region.dirty_domains)},
                               {"dependencies", std::move(dependencies)}});
        }

        json terrain{{"schemaVersion", asset.schema_version},
                     {"authoringRevision", asset.authoring_revision},
                     {"coordinates",
                      {{"origin", {asset.coordinates.origin_x, asset.coordinates.origin_y, asset.coordinates.origin_z}},
                       {"metersPerUnit", asset.coordinates.meters_per_unit}}},
                     {"partition",
                      {{"authoringRegionSize", asset.partition.authoring_region_size},
                       {"dependencyHalo", asset.partition.dependency_halo}}},
                     {"source", std::move(source)},
                     {"modifiers", std::move(modifiers)},
                     {"attributes", std::move(attributes)},
                     {"build",
                      {{"geometryQuality", geometry_quality_name(asset.build.geometry_quality)},
                       {"targetSurfaceError", asset.build.target_surface_error},
                       {"renderGeometry", asset.build.build_render_geometry},
                       {"attributes", asset.build.build_attributes},
                       {"collision", asset.build.build_collision},
                       {"navigation", asset.build.build_navigation},
                       {"destruction", asset.build.build_destruction}}},
                     {"runtime",
                      {{"mutability", mutability_name(asset.runtime.mutability)},
                       {"persistentChanges", asset.runtime.persistent_runtime_changes},
                       {"replicateChanges", asset.runtime.replicate_runtime_changes},
                       {"damageProfile", reference_json(asset.runtime.damage_profile)}}},
                     {"regions", std::move(regions)}};

        json document{{"format", "arc.terrain"},
                      {"formatVersion", terrain_document_format_version},
                      {"terrain", std::move(terrain)}};
        return terrain_asset_json_result::success(document.dump(pretty ? 2 : -1) + (pretty ? "\n" : ""));
    }
    catch (const std::exception& exception)
    {
        return failure<terrain_asset_json_result>(terrain_asset_io_error_code::invalid_asset,
                                                  std::string("Failed to serialize terrain asset: ") +
                                                      exception.what());
    }
}

terrain_asset_decode_result read_terrain_asset_json(std::string_view text)
{
    try
    {
        const auto document = json::parse(text.begin(), text.end());
        if (!document.is_object() || document.value("format", std::string{}) != "arc.terrain")
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Document is not an ARC terrain asset");
        if (document.value("formatVersion", 0u) != terrain_document_format_version)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::unsupported_format_version,
                                                        "Terrain document format version is unsupported");
        if (!document.contains("terrain") || !document["terrain"].is_object())
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain document is missing its terrain object");

        const auto& value = document["terrain"];
        terrain_asset asset;
        asset.schema_version = value.value("schemaVersion", 0u);
        if (asset.schema_version != terrain_asset::current_schema_version)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::unsupported_schema_version,
                                                        "Terrain asset schema version is unsupported");
        asset.authoring_revision = value.value("authoringRevision", 1ull);

        const auto& coordinates = value.at("coordinates");
        if (!coordinates.is_object() || !coordinates.contains("origin") || !coordinates["origin"].is_array() ||
            coordinates["origin"].size() != 3)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain coordinates are malformed");
        asset.coordinates = {coordinates["origin"][0].get<double>(), coordinates["origin"][1].get<double>(),
                             coordinates["origin"][2].get<double>(), coordinates.value("metersPerUnit", 1.0)};

        const auto& partition = value.at("partition");
        asset.partition.authoring_region_size = partition.at("authoringRegionSize").get<double>();
        asset.partition.dependency_halo = partition.value("dependencyHalo", 0.0);

        const auto& source = value.at("source");
        const auto source_id = parse_terrain_stable_id(source.value("id", std::string{}));
        const auto source_kind = parse_source_kind(source.value("kind", std::string{}));
        if (!source_id || !source_kind)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain source ID or kind is invalid");
        asset.source.id = *source_id;
        asset.source.kind = *source_kind;
        asset.source.schema_version = source.value("schemaVersion", 0u);
        asset.source.generator_id = source.value("generatorId", std::string{});
        asset.source.seed = source.value("seed", 1ull);
        if (source.contains("asset") && !parse_reference(source["asset"], asset.source.asset))
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain source asset reference is invalid");
        const auto& transform = source.at("transform");
        if (!transform.contains("translation") || !transform.contains("rotation") || !transform.contains("scale") ||
            !transform["translation"].is_array() || transform["translation"].size() != 3 ||
            !transform["rotation"].is_array() || transform["rotation"].size() != 4 || !transform["scale"].is_array() ||
            transform["scale"].size() != 3)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain source transform is malformed");
        asset.source.transform.translation = {transform["translation"][0].get<float>(),
                                              transform["translation"][1].get<float>(),
                                              transform["translation"][2].get<float>()};
        asset.source.transform.rotation = {transform["rotation"][0].get<float>(), transform["rotation"][1].get<float>(),
                                           transform["rotation"][2].get<float>(),
                                           transform["rotation"][3].get<float>()};
        asset.source.transform.scale = {transform["scale"][0].get<float>(), transform["scale"][1].get<float>(),
                                        transform["scale"][2].get<float>()};

        if (const auto found = value.find("modifiers"); found != value.end())
        {
            if (!found->is_array())
                return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                            "Terrain modifiers must be an array");
            for (const auto& record : *found)
            {
                terrain_modifier_descriptor modifier;
                const auto modifier_id = parse_terrain_stable_id(record.value("id", std::string{}));
                const auto domains = parse_domains(record.value("domains", json::array()));
                if (!modifier_id || !domains || !record.contains("parameters"))
                    return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                "Terrain modifier record is malformed");
                modifier.id = *modifier_id;
                modifier.type_id = record.value("type", std::string{});
                modifier.name = record.value("name", std::string{});
                modifier.schema_version = record.value("schemaVersion", 0u);
                modifier.enabled = record.value("enabled", true);
                modifier.domains = *domains;
                modifier.canonical_parameters = record["parameters"].dump();
                if (const auto bounds = record.find("affectedBounds"); bounds != record.end())
                {
                    terrain_world_bounds parsed;
                    if (!parse_bounds(*bounds, parsed))
                        return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                    "Terrain modifier bounds are malformed");
                    modifier.affected_bounds = parsed;
                }
                asset.modifiers.push_back(std::move(modifier));
            }
        }

        if (const auto found = value.find("attributes"); found != value.end())
        {
            if (!found->is_array())
                return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                            "Terrain attributes must be an array");
            for (const auto& record : *found)
            {
                terrain_attribute_definition attribute;
                const auto attribute_id = parse_terrain_stable_id(record.value("id", std::string{}));
                const auto type = parse_attribute_type(record.value("type", std::string{}));
                const auto semantic = parse_attribute_semantic(record.value("semantic", std::string{}));
                const auto storage = parse_attribute_storage(record.value("storage", std::string{}));
                const auto interpolation = parse_interpolation(record.value("interpolation", std::string{}));
                if (!attribute_id || !type || !semantic || !storage || !interpolation || !record.contains("default"))
                    return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                "Terrain attribute record is malformed");
                attribute.id = *attribute_id;
                attribute.name = record.value("name", std::string{});
                attribute.schema_version = record.value("schemaVersion", 0u);
                attribute.type = *type;
                attribute.semantic = *semantic;
                attribute.storage = *storage;
                attribute.interpolation = *interpolation;
                if (!parse_attribute_default(attribute.type, record["default"], attribute.default_value))
                    return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                "Terrain attribute default value is malformed");
                asset.attributes.push_back(std::move(attribute));
            }
        }

        const auto& build = value.at("build");
        const auto quality = parse_geometry_quality(build.value("geometryQuality", std::string{}));
        if (!quality)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain geometry quality is invalid");
        asset.build.geometry_quality = *quality;
        asset.build.target_surface_error = build.at("targetSurfaceError").get<float>();
        asset.build.build_render_geometry = build.value("renderGeometry", true);
        asset.build.build_attributes = build.value("attributes", true);
        asset.build.build_collision = build.value("collision", true);
        asset.build.build_navigation = build.value("navigation", true);
        asset.build.build_destruction = build.value("destruction", false);

        const auto& runtime = value.at("runtime");
        const auto mutability = parse_mutability(runtime.value("mutability", std::string{}));
        if (!mutability)
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain runtime mutability is invalid");
        asset.runtime.mutability = *mutability;
        asset.runtime.persistent_runtime_changes = runtime.value("persistentChanges", false);
        asset.runtime.replicate_runtime_changes = runtime.value("replicateChanges", false);
        if (runtime.contains("damageProfile") &&
            !parse_reference(runtime["damageProfile"], asset.runtime.damage_profile))
            return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                        "Terrain damage-profile reference is invalid");

        if (const auto found = value.find("regions"); found != value.end())
        {
            if (!found->is_array())
                return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                            "Terrain regions must be an array");
            for (const auto& record : *found)
            {
                terrain_region_record region;
                region.id = {record.at("x").get<std::int64_t>(), record.at("z").get<std::int64_t>()};
                if (!parse_bounds(record.at("authoringBounds"), region.authoring_bounds))
                    return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                "Terrain region bounds are malformed");
                region.dirty_revision = record.value("dirtyRevision", 0ull);
                region.compiled_revision = record.value("compiledRevision", 0ull);
                const auto dirty_domains = parse_domains(record.value("dirtyDomains", json::array()));
                if (!dirty_domains)
                    return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                "Terrain region dirty domains are malformed");
                region.dirty_domains = *dirty_domains;
                if (const auto dependencies = record.find("dependencies"); dependencies != record.end())
                {
                    if (!dependencies->is_array())
                        return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                                    "Terrain region dependencies must be an array");
                    for (const auto& dependency : *dependencies)
                    {
                        const auto domains = parse_domains(dependency.value("domains", json::array()));
                        if (!domains)
                            return failure<terrain_asset_decode_result>(
                                terrain_asset_io_error_code::invalid_document,
                                "Terrain region dependency domains are malformed");
                        region.dependencies.push_back(
                            {{dependency.at("x").get<std::int64_t>(), dependency.at("z").get<std::int64_t>()},
                             *domains});
                    }
                }
                asset.regions.push_back(std::move(region));
            }
            std::sort(asset.regions.begin(), asset.regions.end(),
                      [](const terrain_region_record& lhs, const terrain_region_record& rhs)
                      { return lhs.id.z < rhs.id.z || (lhs.id.z == rhs.id.z && lhs.id.x < rhs.id.x); });
        }

        const auto validation = validate_terrain_asset(asset);
        if (!validation.valid())
        {
            const auto found = std::find_if(validation.issues.begin(), validation.issues.end(), [](const auto& issue)
                                            { return issue.severity == terrain_asset_validation_severity::error; });
            return failure<terrain_asset_decode_result>(
                terrain_asset_io_error_code::invalid_asset,
                found == validation.issues.end() ? "Terrain asset failed validation" : found->message);
        }
        return terrain_asset_decode_result::success(std::move(asset));
    }
    catch (const std::exception& exception)
    {
        return failure<terrain_asset_decode_result>(terrain_asset_io_error_code::invalid_document,
                                                    std::string("Invalid terrain asset JSON: ") + exception.what());
    }
}

std::unique_ptr<assets::asset_importer> make_terrain_asset_importer()
{
    return std::make_unique<terrain_asset_importer>();
}

bool register_terrain_asset_importer(assets::asset_manager& manager)
{
    return manager.register_importer(make_terrain_asset_importer());
}

} // namespace arc::scene

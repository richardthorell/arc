#pragma once

#include <arc/assets/assets.h>
#include <arc/assets/terrain_types.h>
#include <arc/core/id.h>
#include <arc/math/math.h>

#include <array>
#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace arc::scene
{

struct terrain_stable_id_tag;
using terrain_stable_id = core::uuid<terrain_stable_id_tag>;

[[nodiscard]] terrain_stable_id generate_terrain_stable_id() noexcept;
[[nodiscard]] std::string to_string(terrain_stable_id value);
[[nodiscard]] std::optional<terrain_stable_id> parse_terrain_stable_id(std::string_view text) noexcept;

/** @brief Stable authoring-region coordinate. Runtime streaming cells and render pages are intentionally separate. */
struct terrain_region_id
{
    std::int64_t x{};
    std::int64_t z{};

    friend constexpr auto operator<=>(const terrain_region_id&, const terrain_region_id&) noexcept = default;
};

/** @brief Double-precision world bounds used by authoring and build dependency tracking. */
struct terrain_world_bounds
{
    double min_x{};
    double min_y{};
    double min_z{};
    double max_x{};
    double max_y{};
    double max_z{};

    [[nodiscard]] bool valid() const noexcept;
};

/** @brief High-precision terrain origin with float-local evaluated geometry. */
struct terrain_coordinate_system
{
    double origin_x{};
    double origin_y{};
    double origin_z{};
    double meters_per_unit{1.0};
};

/** @brief Stable source-control/dirty-tracking partition. This is not a virtual-geometry page size. */
struct terrain_partition_settings
{
    double authoring_region_size{256.0};
    double dependency_halo{8.0};
};

[[nodiscard]] terrain_region_id terrain_region_at(const terrain_coordinate_system& coordinates,
                                                  const terrain_partition_settings& partition, double world_x,
                                                  double world_z) noexcept;
[[nodiscard]] terrain_world_bounds terrain_region_bounds(const terrain_coordinate_system& coordinates,
                                                         const terrain_partition_settings& partition,
                                                         terrain_region_id region) noexcept;
[[nodiscard]] terrain_world_bounds expand_terrain_bounds(terrain_world_bounds bounds, double amount) noexcept;
[[nodiscard]] std::vector<terrain_region_id> terrain_regions_overlapping(const terrain_coordinate_system& coordinates,
                                                                         const terrain_partition_settings& partition,
                                                                         terrain_world_bounds bounds);

enum class terrain_source_kind : std::uint8_t
{
    flat,
    heightfield,
    mesh,
    procedural
};

struct terrain_source_transform
{
    math::vector3f translation{};
    math::quatf rotation{};
    math::vector3f scale{math::vector3f::one};
};

/** @brief One initial source for a unified terrain asset. Source kind never determines runtime rendering. */
struct terrain_source_descriptor
{
    terrain_stable_id id{};
    terrain_source_kind kind{terrain_source_kind::flat};
    assets::asset_reference asset;
    std::string generator_id;
    std::uint64_t seed{1};
    terrain_source_transform transform;
    std::uint32_t schema_version{1};
};

enum class terrain_domain : std::uint32_t
{
    none = 0,
    geometry = 1u << 0u,
    attributes = 1u << 1u,
    topology = 1u << 2u,
    collision = 1u << 3u,
    navigation = 1u << 4u,
    destruction = 1u << 5u,
    all = 0x3fu
};

[[nodiscard]] constexpr terrain_domain operator|(terrain_domain lhs, terrain_domain rhs) noexcept
{
    return static_cast<terrain_domain>(static_cast<std::uint32_t>(lhs) | static_cast<std::uint32_t>(rhs));
}

[[nodiscard]] constexpr terrain_domain operator&(terrain_domain lhs, terrain_domain rhs) noexcept
{
    return static_cast<terrain_domain>(static_cast<std::uint32_t>(lhs) & static_cast<std::uint32_t>(rhs));
}

constexpr terrain_domain& operator|=(terrain_domain& lhs, terrain_domain rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

[[nodiscard]] constexpr bool terrain_domain_contains(terrain_domain mask, terrain_domain value) noexcept
{
    return (mask & value) == value;
}

namespace terrain_builtin_modifier_types
{
inline constexpr std::string_view sculpt_layer = "arc.terrain.sculpt_layer.v1";
inline constexpr std::string_view paint_layer = "arc.terrain.paint_layer.v1";
} // namespace terrain_builtin_modifier_types

/** Sparse accumulated height delta owned by one sculpt modifier. Coordinates are local to an authoring region. */
struct terrain_sculpt_sample_delta
{
    std::uint32_t x{};
    std::uint32_t z{};
    float delta{};

    friend constexpr bool operator==(const terrain_sculpt_sample_delta&,
                                     const terrain_sculpt_sample_delta&) noexcept = default;
};

/** Sparse accumulated material-weight delta owned by one paint modifier. */
struct terrain_paint_sample_delta
{
    std::uint32_t x{};
    std::uint32_t z{};
    std::array<std::int16_t, 4> delta{};

    friend constexpr bool operator==(const terrain_paint_sample_delta&,
                                     const terrain_paint_sample_delta&) noexcept = default;
};

struct terrain_sculpt_region_payload
{
    std::vector<terrain_sculpt_sample_delta> samples;
};

struct terrain_paint_region_payload
{
    std::vector<terrain_paint_sample_delta> samples;
};

using terrain_modifier_region_payload_data = std::variant<terrain_sculpt_region_payload, terrain_paint_region_payload>;

/** Persistent sparse edit payload for one modifier and stable authoring region. */
struct terrain_modifier_region_payload
{
    terrain_region_id region{};
    std::uint32_t schema_version{1};
    terrain_modifier_region_payload_data data;
};

/** @brief Versioned non-destructive operation. Parameters and sparse region payloads are authoring data only. */
struct terrain_modifier_descriptor
{
    terrain_stable_id id{};
    std::string type_id;
    std::string name;
    std::uint32_t schema_version{1};
    bool enabled{true};
    terrain_domain domains{terrain_domain::geometry};
    std::optional<terrain_world_bounds> affected_bounds;
    std::string canonical_parameters{"{}"};
    std::vector<terrain_modifier_region_payload> region_payloads;
};

enum class terrain_attribute_type : std::uint8_t
{
    boolean,
    signed_integer,
    unsigned_integer,
    floating_point,
    vector4,
    string
};

enum class terrain_attribute_semantic : std::uint8_t
{
    custom,
    material_weight,
    physical_material,
    wetness,
    snow,
    biome,
    foliage_density,
    navigation_cost,
    destruction_strength,
    hardness,
    acoustic_surface,
    gameplay_tag
};

enum class terrain_attribute_storage : std::uint8_t
{
    sparse_tiles,
    dense_tiles,
    procedural
};

enum class terrain_attribute_interpolation : std::uint8_t
{
    nearest,
    linear
};

using terrain_attribute_value = std::variant<bool, std::int64_t, std::uint64_t, double, math::vector4f, std::string>;

struct terrain_attribute_definition
{
    terrain_stable_id id{};
    std::string name;
    terrain_attribute_type type{terrain_attribute_type::floating_point};
    terrain_attribute_semantic semantic{terrain_attribute_semantic::custom};
    terrain_attribute_storage storage{terrain_attribute_storage::sparse_tiles};
    terrain_attribute_interpolation interpolation{terrain_attribute_interpolation::linear};
    terrain_attribute_value default_value{0.0};
    std::uint32_t schema_version{1};
};

enum class terrain_geometry_quality : std::uint8_t
{
    scalable,
    balanced,
    maximum
};

/** @brief Authoring-level build intent. Renderer-internal cluster/page settings deliberately do not live here. */
struct terrain_build_settings
{
    terrain_geometry_quality geometry_quality{terrain_geometry_quality::balanced};
    float target_surface_error{0.05f};
    bool build_render_geometry{true};
    bool build_attributes{true};
    bool build_collision{true};
    bool build_navigation{true};
    bool build_destruction{};
};

enum class terrain_runtime_mutability : std::uint8_t
{
    immutable,
    deformable,
    fractureable,
    deformable_and_fractureable
};

/** @brief Future-facing runtime mutation policy shared by heightfield-, mesh-, and volume-authored terrain. */
struct terrain_runtime_policy
{
    terrain_runtime_mutability mutability{terrain_runtime_mutability::immutable};
    bool persistent_runtime_changes{};
    bool replicate_runtime_changes{};
    assets::asset_reference damage_profile;
};

/** @brief One explicit cross-region authoring dependency and the domains read from it. */
struct terrain_region_dependency
{
    terrain_region_id region{};
    terrain_domain domains{terrain_domain::geometry};

    friend constexpr auto operator<=>(const terrain_region_dependency&,
                                      const terrain_region_dependency&) noexcept = default;
};

/**
 * @brief Persistent authoring/build state for one stable terrain region.
 *
 * Runtime streaming cells, virtual-geometry pages, collision partitions, and fracture pieces intentionally use their
 * own hierarchies instead of reusing this record.
 */
struct terrain_region_record
{
    terrain_region_id id{};
    terrain_world_bounds authoring_bounds{};
    std::vector<terrain_region_dependency> dependencies;
    std::uint64_t dirty_revision{};
    std::uint64_t compiled_revision{};
    terrain_domain dirty_domains{terrain_domain::none};
};

/** @brief Immutable dependency snapshot consumed by one incremental terrain-region build. */
struct terrain_build_region_snapshot
{
    terrain_region_id target{};
    terrain_world_bounds authoring_bounds{};
    terrain_world_bounds evaluation_bounds{};
    std::vector<terrain_region_dependency> dependencies;
    std::uint64_t authoring_revision{};
    std::uint64_t target_dirty_revision{};
};

/** @brief Result of one authoring edit being propagated to stable terrain regions. */
struct terrain_dirty_update
{
    std::uint64_t revision{};
    std::vector<terrain_region_id> regions;
};

/** @brief Unified authored terrain definition. Evaluated/cooked renderer data must never be serialized into this type.
 */
struct terrain_asset
{
    static constexpr std::uint32_t current_schema_version = 1;

    std::uint32_t schema_version{current_schema_version};
    std::uint64_t authoring_revision{1};
    terrain_coordinate_system coordinates;
    terrain_partition_settings partition;
    terrain_source_descriptor source;
    std::vector<terrain_modifier_descriptor> modifiers;
    std::vector<terrain_attribute_definition> attributes;
    terrain_build_settings build;
    terrain_runtime_policy runtime;
    std::vector<terrain_region_record> regions;
};

enum class terrain_asset_validation_severity : std::uint8_t
{
    warning,
    error
};

enum class terrain_asset_validation_code : std::uint8_t
{
    unsupported_schema,
    invalid_coordinates,
    invalid_partition,
    invalid_source,
    duplicate_stable_id,
    invalid_modifier,
    invalid_modifier_payload,
    invalid_attribute,
    duplicate_attribute_name,
    invalid_build_settings,
    invalid_runtime_policy,
    invalid_region
};

struct terrain_asset_validation_issue
{
    terrain_asset_validation_severity severity{terrain_asset_validation_severity::error};
    terrain_asset_validation_code code{terrain_asset_validation_code::invalid_source};
    terrain_stable_id subject{};
    std::string message;
};

struct [[nodiscard]] terrain_asset_validation_result
{
    std::vector<terrain_asset_validation_issue> issues;

    [[nodiscard]] bool valid() const noexcept;
};

/** @brief Validate persistent authoring invariants without evaluating or cooking terrain. */
[[nodiscard]] terrain_asset_validation_result validate_terrain_asset(const terrain_asset& asset);

/** Built-in non-destructive layer construction helpers. Empty layers do not invalidate compiled regions. */
terrain_modifier_descriptor& add_terrain_sculpt_layer(terrain_asset& asset, std::string name = "Sculpt Layer");
terrain_modifier_descriptor& add_terrain_paint_layer(terrain_asset& asset, std::string name = "Paint Layer");

[[nodiscard]] terrain_modifier_descriptor* find_terrain_modifier(terrain_asset& asset, terrain_stable_id id) noexcept;
[[nodiscard]] const terrain_modifier_descriptor* find_terrain_modifier(const terrain_asset& asset,
                                                                       terrain_stable_id id) noexcept;
[[nodiscard]] terrain_modifier_region_payload* find_terrain_modifier_payload(terrain_modifier_descriptor& modifier,
                                                                             terrain_region_id region) noexcept;
[[nodiscard]] const terrain_modifier_region_payload*
find_terrain_modifier_payload(const terrain_modifier_descriptor& modifier, terrain_region_id region) noexcept;

/** Replace one region's sparse sculpt payload and dirty only geometry for that authoring region. */
[[nodiscard]] terrain_dirty_update set_terrain_sculpt_region_samples(terrain_asset& asset, terrain_stable_id modifier,
                                                                     terrain_region_id region,
                                                                     std::vector<terrain_sculpt_sample_delta> samples);

/** Replace one region's sparse paint payload and dirty only attributes for that authoring region. */
[[nodiscard]] terrain_dirty_update set_terrain_paint_region_samples(terrain_asset& asset, terrain_stable_id modifier,
                                                                    terrain_region_id region,
                                                                    std::vector<terrain_paint_sample_delta> samples);

/** Validate built-in sparse payload ownership, schemas, values, and per-region uniqueness. */
[[nodiscard]] bool validate_terrain_modifier_payloads(const terrain_modifier_descriptor& modifier) noexcept;

/** @brief Return the persisted region record, creating it with canonical authoring bounds when necessary. */
terrain_region_record& ensure_terrain_region(terrain_asset& asset, terrain_region_id region);

/** @brief Mark all authoring regions overlapping bounds dirty for the supplied domains using one new revision. */
[[nodiscard]] terrain_dirty_update mark_terrain_dirty(terrain_asset& asset, terrain_world_bounds bounds,
                                                      terrain_domain domains);

/**
 * @brief Publish completed domains for a region only when the build was produced from its current dirty revision.
 * @return False for a missing region, stale build revision, or empty domain mask.
 */
bool mark_terrain_region_compiled(terrain_asset& asset, terrain_region_id region, terrain_domain domains,
                                  std::uint64_t build_revision) noexcept;

/** @brief Capture deterministic target, halo, and explicit region dependencies for an incremental build. */
[[nodiscard]] terrain_build_region_snapshot make_terrain_build_region_snapshot(const terrain_asset& asset,
                                                                               terrain_region_id region);

} // namespace arc::scene

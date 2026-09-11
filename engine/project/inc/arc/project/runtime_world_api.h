#pragma once

#include <cstddef>
#include <cstdint>

namespace arc::project
{

/** @brief ABI-safe transient entity handle used by project runtime callbacks. */
struct game_entity_v1
{
    std::uint32_t index{0xffffffffu};
    std::uint32_t generation{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return index != 0xffffffffu;
    }
};

/** @brief Entity handle returned for structural creation deferred to a scheduler phase boundary. */
struct game_deferred_entity_v1
{
    std::uint64_t buffer{};
    std::uint32_t ordinal{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return buffer != 0;
    }
};

/** @brief Target accepted by runtime structural operations; it may be immediate or deferred. */
struct game_entity_target_v1
{
    game_entity_v1 entity{};
    game_deferred_entity_v1 deferred{};
    bool is_deferred{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return is_deferred ? deferred.valid() : entity.valid();
    }
};

/** @brief Engine-owned components exposed through the stable project runtime API. */
enum class game_core_component_v1 : std::uint8_t
{
    name,
    transform,
    tag,
    active
};

/** @brief Scheduler access mode for a stable engine-owned component. */
enum class game_core_component_access_mode_v1 : std::uint8_t
{
    read,
    write
};

/** @brief Declared project-system access to one engine-owned component. */
struct game_core_component_access_v1
{
    game_core_component_v1 component{game_core_component_v1::transform};
    game_core_component_access_mode_v1 mode{game_core_component_access_mode_v1::read};
};

using game_core_component_mask_v1 = std::uint32_t;

[[nodiscard]] constexpr game_core_component_mask_v1
game_core_component_bit_v1(game_core_component_v1 component) noexcept
{
    return game_core_component_mask_v1{1u} << static_cast<std::uint8_t>(component);
}

/** @brief ABI-safe three-component vector. */
struct game_vector3_v1
{
    float x{};
    float y{};
    float z{};
};

/** @brief ABI-safe quaternion stored in `(x, y, z, w)` order. */
struct game_quaternion_v1
{
    float x{};
    float y{};
    float z{};
    float w{1.0f};
};

/** @brief ABI-safe local transform. Cached world data remains engine-owned. */
struct game_transform_v1
{
    game_vector3_v1 position{};
    game_quaternion_v1 rotation{};
    game_vector3_v1 scale{1.0f, 1.0f, 1.0f};
};

/** @brief Borrowed UTF-8 string view valid until the component is modified or the callback returns. */
struct game_string_view_v1
{
    const char* data{};
    std::size_t size{};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return data != nullptr;
    }
};

/** @brief Stable entity query over the exposed engine-owned component set. */
struct game_world_query_v1
{
    std::size_t structure_size{sizeof(game_world_query_v1)};
    game_core_component_mask_v1 required_core_components{};
    game_core_component_mask_v1 excluded_core_components{};
};

/** @brief Entity query visitor. Return false to stop visiting further matches. */
using game_visit_entity_v1 = bool (*)(void* user_data, game_entity_v1 entity);

/**
 * @brief Stable runtime-world API supplied to project callbacks.
 *
 * The table and its user data are borrowed for the duration of the owning callback. Structural changes requested
 * from an ECS system are recorded in that system's command buffer and become visible at the scheduler phase boundary.
 * BeginPlay and EndPlay operate directly because they execute outside scheduled system work.
 */
struct game_world_api_v1
{
    std::size_t structure_size{sizeof(game_world_api_v1)};
    void* user_data{};

    game_entity_target_v1 (*create_entity)(void* user_data){};
    bool (*destroy_entity)(void* user_data, game_entity_target_v1 entity){};
    bool (*entity_alive)(void* user_data, game_entity_v1 entity){};
    std::size_t (*entity_count)(void* user_data){};
    bool (*query_entities)(void* user_data, const game_world_query_v1* query, void* visitor_user_data,
                           game_visit_entity_v1 visitor){};

    bool (*has_core_component)(void* user_data, game_entity_v1 entity, game_core_component_v1 component){};
    bool (*remove_core_component)(void* user_data, game_entity_target_v1 entity, game_core_component_v1 component){};

    game_string_view_v1 (*read_name)(void* user_data, game_entity_v1 entity){};
    bool (*set_name)(void* user_data, game_entity_target_v1 entity, const char* value, std::size_t value_size){};
    bool (*read_transform)(void* user_data, game_entity_v1 entity, game_transform_v1* value){};
    bool (*set_transform)(void* user_data, game_entity_target_v1 entity, const game_transform_v1* value){};
    game_string_view_v1 (*read_tag)(void* user_data, game_entity_v1 entity){};
    bool (*set_tag)(void* user_data, game_entity_target_v1 entity, const char* value, std::size_t value_size){};
    bool (*read_active)(void* user_data, game_entity_v1 entity, bool* value){};
    bool (*set_active)(void* user_data, game_entity_target_v1 entity, bool value){};
};

} // namespace arc::project
